#!/usr/bin/env python3
"""
MLB Pitcher Strikeouts Analyzer v1.0
MLB Stats API (gratuit, no key) + The Odds API
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import numpy as np
from scipy import stats as scipy_stats
import os, math, time, re
from datetime import datetime
from collections import Counter

app = Flask(__name__)
CORS(app)

ODDS_API_KEY = os.environ.get('ODDS_API_KEY')
MLB_BASE  = "https://statsapi.mlb.com/api/v1"
ODDS_BASE = "https://api.the-odds-api.com/v4"

GAMELOG_CACHE  = {}
SCHEDULE_CACHE = {}

# ── Helpers ────────────────────────────────────────────────────────────────────

def to_python(obj):
    if isinstance(obj, dict):  return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [to_python(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return 0.0 if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, float): return 0.0 if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, np.bool_): return bool(obj)
    if isinstance(obj, np.ndarray): return [to_python(v) for v in obj.tolist()]
    return obj

def norm(name):
    return re.sub(r'[^a-z ]', '', name.lower().strip())

def names_match(a, b):
    na, nb = norm(a), norm(b)
    if na == nb: return True
    parts_a, parts_b = na.split(), nb.split()
    return (parts_a[-1] == parts_b[-1] and parts_a[0][0] == parts_b[0][0])

# ── MLB Stats API ──────────────────────────────────────────────────────────────

def get_probable_pitchers():
    today = datetime.now().strftime('%Y-%m-%d')
    if today in SCHEDULE_CACHE:
        return SCHEDULE_CACHE[today]
    try:
        resp = requests.get(f"{MLB_BASE}/schedule", params={
            'sportId': 1, 'date': today,
            'hydrate': 'probablePitcher', 'gameType': 'R'
        }, timeout=12)
        if resp.status_code != 200:
            return []
        pitchers = []
        for date_block in resp.json().get('dates', []):
            for game in date_block.get('games', []):
                home = game.get('teams', {}).get('home', {}).get('team', {}).get('name', '')
                away = game.get('teams', {}).get('away', {}).get('team', {}).get('name', '')
                gtime = game.get('gameDate', '')
                for side in ['home', 'away']:
                    pp = game.get('teams', {}).get(side, {}).get('probablePitcher', {})
                    if pp and pp.get('id'):
                        pitchers.append({
                            'id': pp['id'], 'name': pp.get('fullName', ''),
                            'home_team': home, 'away_team': away, 'game_time': gtime
                        })
        SCHEDULE_CACHE[today] = pitchers
        return pitchers
    except Exception as e:
        print(f"Schedule error: {e}")
        return []

def get_gamelog(player_id, season=2025):
    key = f"{player_id}_{season}"
    if key in GAMELOG_CACHE:
        return GAMELOG_CACHE[key]
    try:
        resp = requests.get(f"{MLB_BASE}/people/{player_id}/stats", params={
            'stats': 'gameLog', 'season': season,
            'group': 'pitching', 'gameType': 'R'
        }, timeout=15)
        if resp.status_code != 200:
            return None
        splits = resp.json().get('stats', [{}])[0].get('splits', [])
        if not splits:
            return None
        games = []
        for s in splits:
            st = s.get('stat', {})
            ip_str = str(st.get('inningsPitched', '0') or '0')
            try:
                parts = ip_str.split('.')
                ip = int(parts[0]) + (int(parts[1]) / 3 if len(parts) > 1 and parts[1] else 0)
            except:
                ip = 0.0
            games.append({
                'date': s.get('date', ''),
                'k': int(st.get('strikeOuts', 0) or 0),
                'ip': round(ip, 2),
                'er': int(st.get('earnedRuns', 0) or 0),
                'h': int(st.get('hits', 0) or 0),
                'bb': int(st.get('baseOnBalls', 0) or 0),
            })
        games.sort(key=lambda x: x['date'], reverse=True)
        GAMELOG_CACHE[key] = games
        return games
    except Exception as e:
        print(f"Gamelog error {player_id}: {e}")
        return None

# ── Analysis ──────────────────────────────────────────────────────────────────

def analyze(games, line):
    qualified = [g for g in games if g['ip'] >= 3.0] or games
    if len(qualified) < 4:
        return None

    vals = np.array([g['k'] for g in qualified], dtype=float)
    n = len(vals)

    mean = float(np.mean(vals))
    std  = float(np.std(vals))
    med  = float(np.median(vals))

    q1, q3 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
    iqr = q3 - q1
    clean = vals[(vals >= q1 - 1.5*iqr) & (vals <= q3 + 1.5*iqr)]
    cmean = float(np.mean(clean)) if len(clean) else mean
    cstd  = float(np.std(clean))  if len(clean) else std

    # Normal distribution probability (continuity correction +0.5)
    if cstd > 0:
        over_p  = float((1 - scipy_stats.norm.cdf((line + 0.5 - cmean) / cstd)) * 100)
    else:
        over_p = float(np.sum(vals > line) / n * 100)
    under_p = 100.0 - over_p

    implied = 52.38  # -110 vig
    eo = over_p  - implied
    eu = under_p - implied

    if eo >= eu and eo > 0:  rec, edge = 'OVER',  eo
    elif eu > 0:              rec, edge = 'UNDER', eu
    else:                     rec, edge = 'SKIP',  max(eo, eu)

    try:
        slope, _, r, _, _ = scipy_stats.linregress(np.arange(n), vals)
        r_sq = round(float(r**2), 3)
        slope = round(float(slope), 3)
    except:
        r_sq = slope = 0.0

    kelly = 0.0
    if rec != 'SKIP':
        prob = over_p if rec == 'OVER' else under_p
        if prob > implied:
            kelly = min(20, ((prob/100 - 0.5238) / 0.9091) * 100 * 0.25)

    over_count = int(np.sum(vals > line))

    recent = [{'date': g['date'][:10], 'stat': g['k'], 'ip': g['ip'], 'er': g['er']}
              for g in qualified[:10]]

    return {
        'n': n, 'mean': round(mean,1), 'cmean': round(cmean,1),
        'std': round(std,2), 'cstd': round(cstd,2), 'med': round(med,1),
        'l5': round(float(np.mean(vals[:5])), 1),
        'l10': round(float(np.mean(vals[:10])),1) if n>=10 else round(mean,1),
        'over_p': round(over_p,1), 'under_p': round(under_p,1),
        'over_n': over_count, 'under_n': int(n - over_count),
        'edge': round(edge,1), 'rec': rec,
        'r_sq': r_sq, 'slope': slope,
        'kelly': round(kelly,1),
        'cons': round(max(0, 100-(cstd/cmean*100)),1) if cmean>0 else 50.0,
        'recent': recent
    }

# ── Odds API ──────────────────────────────────────────────────────────────────

def get_mlb_props():
    if not ODDS_API_KEY:
        return {}, {}
    try:
        resp = requests.get(f"{ODDS_BASE}/sports/baseball_mlb/odds", params={
            'apiKey': ODDS_API_KEY, 'regions': 'us',
            'markets': 'h2h', 'oddsFormat': 'american'
        }, timeout=10)
        if resp.status_code != 200:
            return {}, {}
        games = resp.json()
    except:
        return {}, {}

    props, event_info = {}, {}

    for game in games[:15]:
        gid = game['id']
        event_info[gid] = {
            'home_team': game.get('home_team',''),
            'away_team': game.get('away_team',''),
            'time': game.get('commence_time','')
        }
        try:
            resp2 = requests.get(
                f"{ODDS_BASE}/sports/baseball_mlb/events/{gid}/odds",
                params={'apiKey': ODDS_API_KEY, 'regions': 'us',
                        'markets': 'pitcher_strikeouts', 'oddsFormat': 'american'},
                timeout=10
            )
            if resp2.status_code != 200:
                continue
            for bk in resp2.json().get('bookmakers', []):
                for mk in bk.get('markets', []):
                    if mk['key'] != 'pitcher_strikeouts':
                        continue
                    for oc in mk.get('outcomes', []):
                        player = oc.get('description','').strip()
                        point  = oc.get('point')
                        btype  = oc.get('name','')
                        if not player or point is None:
                            continue
                        if player not in props:
                            props[player] = {'game_id': gid, 'lines': []}
                        props[player]['lines'].append({
                            'book': bk['key'], 'line': float(point),
                            'price': int(oc.get('price',-110)), 'type': btype
                        })
            time.sleep(0.2)
        except:
            continue

    return props, event_info

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/api/daily-opportunities', methods=['GET'])
def daily_opportunities():
    try:
        min_edge = float(request.args.get('min_edge', 5))

        pitchers  = get_probable_pitchers()
        props, ev = get_mlb_props()

        opportunities = []
        analyzed = 0

        for pitcher in pitchers:
            pid  = pitcher['id']
            name = pitcher['name']

            # Match prop
            prop = next((v for k,v in props.items() if names_match(name, k)), None)
            if not prop:
                continue

            # Consensus line
            overs = [l for l in prop['lines'] if l['type'] == 'Over']
            if not overs:
                continue
            line_counts = Counter([l['line'] for l in overs])
            line  = line_counts.most_common(1)[0][0]
            best  = min(overs, key=lambda x: abs(x['line']-line))

            games = get_gamelog(pid)
            if not games or len(games) < 4:
                continue

            analyzed += 1
            a = analyze(games, line)
            if not a or a['rec'] == 'SKIP' or a['edge'] < min_edge:
                continue

            gi = ev.get(prop['game_id'], {
                'home_team': pitcher['home_team'],
                'away_team': pitcher['away_team'],
                'time': pitcher['game_time']
            })

            opportunities.append({
                'player': name,
                'game_info': {
                    'home_team': gi.get('home_team', pitcher['home_team']),
                    'away_team': gi.get('away_team', pitcher['away_team']),
                    'time': gi.get('time', pitcher['game_time'])
                },
                'line_analysis': {
                    'bookmaker_line': line,
                    'bookmaker': best['book'].upper(),
                    'recommendation': a['rec'],
                    'edge': a['edge'],
                    'over_probability': a['over_p'],
                    'under_probability': a['under_p'],
                    'kelly_criterion': a['kelly']
                },
                'deep_stats': {
                    'mean': a['mean'], 'clean_mean': a['cmean'],
                    'std': a['std'], 'avg_last_5': a['l5'],
                    'avg_last_10': a['l10'], 'consistency': a['cons'],
                    'games_analyzed': a['n'],
                    'over_count': a['over_n'], 'under_count': a['under_n'],
                    'r_squared': a['r_sq'], 'trend_slope': a['slope']
                },
                'recent_games': a['recent']
            })

        opportunities.sort(key=lambda x: x['line_analysis']['edge'], reverse=True)

        return jsonify(to_python({
            'status': 'SUCCESS',
            'sport': 'MLB', 'stat': 'Pitcher Strikeouts',
            'total_probable_pitchers': len(pitchers),
            'total_props_found': len(props),
            'analyzed': analyzed,
            'opportunities_found': len(opportunities),
            'opportunities': opportunities,
            'scan_time': datetime.now().strftime('%Y-%m-%d %H:%M')
        }))

    except Exception as e:
        import traceback
        return jsonify({'status': 'ERROR', 'message': str(e),
                        'trace': traceback.format_exc()[:500]}), 500

@app.route('/api/schedule', methods=['GET'])
def schedule():
    p = get_probable_pitchers()
    return jsonify({'status':'SUCCESS', 'pitchers': p, 'count': len(p)})

@app.route('/api/odds/usage', methods=['GET'])
def usage():
    try:
        r = requests.get(f"{ODDS_BASE}/sports", params={'apiKey': ODDS_API_KEY}, timeout=10)
        return jsonify({'used': r.headers.get('x-requests-used','N/A'),
                        'remaining': r.headers.get('x-requests-remaining','N/A')})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status':'healthy','version':'1.0','sport':'MLB','stat':'pitcher_strikeouts'})

@app.route('/')
def home():
    return jsonify({'app':'MLB Strikeouts Analyzer','version':'1.0'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
