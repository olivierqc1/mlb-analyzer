# ╔══════════════════════════════════════════════════════╗
# ║  mlb_analyzer.py — PARTIE 1/2                       ║
# ║  1. Crée mlb_analyzer.py sur GitHub                 ║
# ║  2. Colle ce fichier                                 ║
# ║  3. Colle PART 2 directement à la suite             ║
# ║     (pas de ligne vide entre les deux)              ║
# ╚══════════════════════════════════════════════════════╝
#!/usr/bin/env python3
"""
MLB Betting Analyzer v2.0
Stat types : pitcher_strikeouts, pitcher_earned_runs, batter_hits, batter_total_bases
Validation : Shapiro-Wilk (normalité), KS test, Chi-carré goodness-of-fit
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

GAMELOG_CACHE   = {}
SCHEDULE_CACHE  = {}
PLAYER_ID_CACHE = {}

# ── Stat config ───────────────────────────────────────────────────────────────
STAT_CONFIG = {
    'pitcher_strikeouts': {
        'group': 'pitching', 'col': 'strikeOuts',
        'label': 'K', 'min_ip': 3.0, 'dist': 'normal'
    },
    'pitcher_earned_runs': {
        'group': 'pitching', 'col': 'earnedRuns',
        'label': 'ER', 'min_ip': 3.0, 'dist': 'poisson'
    },
    'batter_hits': {
        'group': 'hitting', 'col': 'hits',
        'label': 'H', 'min_ip': None, 'dist': 'normal'
    },
    'batter_total_bases': {
        'group': 'hitting', 'col': 'totalBases',
        'label': 'TB', 'min_ip': None, 'dist': 'normal'
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def to_python(obj):
    if isinstance(obj, dict):    return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):    return [to_python(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return 0.0 if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, float):   return 0.0 if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, np.bool_): return bool(obj)
    if isinstance(obj, np.ndarray): return [to_python(v) for v in obj.tolist()]
    return obj

def norm_name(name):
    return re.sub(r'[^a-z ]', '', name.lower().strip())

def names_match(a, b):
    na, nb = norm_name(a), norm_name(b)
    if na == nb: return True
    pa, pb = na.split(), nb.split()
    if not pa or not pb: return False
    return pa[-1] == pb[-1] and pa[0][0] == pb[0][0]

# ── MLB Stats API ─────────────────────────────────────────────────────────────

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
                home  = game.get('teams',{}).get('home',{}).get('team',{}).get('name','')
                away  = game.get('teams',{}).get('away',{}).get('team',{}).get('name','')
                gtime = game.get('gameDate','')
                for side in ['home','away']:
                    pp = game.get('teams',{}).get(side,{}).get('probablePitcher',{})
                    if pp and pp.get('id'):
                        pitchers.append({
                            'id': pp['id'], 'name': pp.get('fullName',''),
                            'home_team': home, 'away_team': away, 'game_time': gtime
                        })
        SCHEDULE_CACHE[today] = pitchers
        return pitchers
    except Exception as e:
        print(f"Schedule error: {e}")
        return []

def search_batter(name):
    key = norm_name(name)
    if key in PLAYER_ID_CACHE:
        return PLAYER_ID_CACHE[key]
    last = name.split()[-1] if ' ' in name else name
    try:
        resp = requests.get(f"{MLB_BASE}/people/search",
            params={'names': last, 'sportIds': 1}, timeout=10)
        if resp.status_code == 200:
            for p in resp.json().get('people', []):
                if names_match(name, p.get('fullName','')):
                    PLAYER_ID_CACHE[key] = p['id']
                    return p['id']
        # Legacy lookup fallback
        resp2 = requests.get(
            "https://lookup-service-prod.mlb.com/json/named.search_player_all.bam",
            params={'sport_code': "'mlb'", 'active_sw': "'Y'",
                    'name_part': f"'{last}%'"}, timeout=10)
        if resp2.status_code == 200:
            rows = resp2.json().get('search_player_all',{}).get('queryResults',{}).get('row',[])
            if isinstance(rows, dict): rows = [rows]
            for r in rows:
                if names_match(name, r.get('name_display_first_last','')):
                    pid = int(r.get('player_id',0) or 0)
                    if pid:
                        PLAYER_ID_CACHE[key] = pid
                        return pid
    except Exception as e:
        print(f"Search error {name}: {e}")
    return None

def get_gamelog(player_id, stat_type):
    cfg = STAT_CONFIG[stat_type]
    cache_key = f"{player_id}_{stat_type}"
    if cache_key in GAMELOG_CACHE:
        return GAMELOG_CACHE[cache_key]
    try:
        resp = requests.get(f"{MLB_BASE}/people/{player_id}/stats", params={
            'stats': 'gameLog', 'season': 2025,
            'group': cfg['group'], 'gameType': 'R'
        }, timeout=15)
        if resp.status_code != 200:
            return None
        splits = resp.json().get('stats',[{}])[0].get('splits',[])
        if not splits:
            return None
        games = []
        for s in splits:
            st  = s.get('stat',{})
            val = st.get(cfg['col'])
            if val is None:
                continue
            entry = {'date': s.get('date',''), 'stat': int(val)}
            if cfg['group'] == 'pitching':
                ip_str = str(st.get('inningsPitched','0') or '0')
                try:
                    p = ip_str.split('.')
                    entry['ip'] = int(p[0]) + (int(p[1])/3 if len(p)>1 and p[1] else 0)
                except:
                    entry['ip'] = 0.0
                entry['er'] = int(st.get('earnedRuns',0) or 0)
            games.append(entry)
        games.sort(key=lambda x: x['date'], reverse=True)
        GAMELOG_CACHE[cache_key] = games
        return games or None
    except Exception as e:
        print(f"Gamelog error {player_id} {stat_type}: {e}")
        return None

# ── Statistical Validation ────────────────────────────────────────────────────

def normality_tests(vals):
    """Shapiro-Wilk + KS test. Returns verdict + confidence penalty."""
    results = {}

    # Shapiro-Wilk (optimal n < 50)
    try:
        stat, p = scipy_stats.shapiro(vals)
        results['shapiro_wilk'] = {
            'stat': round(float(stat),4), 'p_value': round(float(p),4),
            'is_normal': float(p) > 0.05,
            'label': 'Normal ✅' if float(p) > 0.05 else 'Non-normal ⚠️'
        }
    except:
        results['shapiro_wilk'] = None

    # KS test vs fitted normal
    try:
        mu, sigma = float(np.mean(vals)), float(np.std(vals))
        if sigma > 0:
            ks_stat, ks_p = scipy_stats.kstest(vals, 'norm', args=(mu, sigma))
            results['ks_test'] = {
                'stat': round(float(ks_stat),4), 'p_value': round(float(ks_p),4),
                'is_normal': float(ks_p) > 0.05,
                'label': 'Normal ✅' if float(ks_p) > 0.05 else 'Non-normal ⚠️'
            }
        else:
            results['ks_test'] = None
    except:
        results['ks_test'] = None

    sw_ok = results.get('shapiro_wilk') and results['shapiro_wilk']['is_normal']
    ks_ok = results.get('ks_test')      and results['ks_test']['is_normal']

    if sw_ok and ks_ok:
        verdict, label, penalty = 'NORMAL',      '✅ Distribution normale confirmée',          0
    elif sw_ok or ks_ok:
        verdict, label, penalty = 'BORDERLINE',  '⚠️ Distribution approximativement normale', 10
    else:
        verdict, label, penalty = 'NON_NORMAL',  '❌ Non-normale — edge pénalisé',            25

    results.update({'verdict': verdict, 'verdict_label': label, 'confidence_penalty': penalty})
    return results

def chi_gof(vals, line):
    """Chi-carré goodness-of-fit: fréquences observées vs théoriques (normale fittée)."""
    n = len(vals)
    if n < 8:
        return None
    try:
        mu, sigma = float(np.mean(vals)), float(np.std(vals))
        if sigma <= 0:
            return None
        over_obs  = int(np.sum(vals > line))
        under_obs = int(n - over_obs)
        p_over    = float(1 - scipy_stats.norm.cdf(line, mu, sigma))
        over_exp  = p_over * n
        under_exp = (1 - p_over) * n
        if over_exp < 1 or under_exp < 1:
            return None
        chi2, p = scipy_stats.chisquare([over_obs, under_obs], f_exp=[over_exp, under_exp])
        good = float(p) > 0.05
        return {
            'chi2': round(float(chi2),4), 'p_value': round(float(p),4),
            'is_good_fit': good,
            'label': '✅ Bon fit' if good else '⚠️ Fit imparfait',
            'observed': [over_obs, under_obs],
            'expected': [round(over_exp,1), round(under_exp,1)]
        }
    except Exception as e:
        print(f"Chi2 error: {e}")
        return None

# ── Core Analysis ─────────────────────────────────────────────────────────────

def analyze(games, line, stat_type):
    cfg = STAT_CONFIG[stat_type]

    qualified = [g for g in games if g.get('ip', cfg['min_ip'] or 0) >= (cfg['min_ip'] or 0)] if cfg['min_ip'] else games
    if len(qualified) < 4:
        qualified = games
    if len(qualified) < 4:
        return None

    vals  = np.array([g['stat'] for g in qualified], dtype=float)
    n     = len(vals)
    mean  = float(np.mean(vals))
    std   = float(np.std(vals))

    # IQR cleaning
    q1, q3 = float(np.percentile(vals,25)), float(np.percentile(vals,75))
    iqr     = q3 - q1
    clean   = vals[(vals >= q1-1.5*iqr) & (vals <= q3+1.5*iqr)]
    cmean   = float(np.mean(clean)) if len(clean) >= 3 else mean
    cstd    = float(np.std(clean))  if len(clean) >= 3 else std

    # Validation

# ╔══════════════════════════════════════════════════════╗
# ║  mlb_analyzer.py — PARTIE 2/2                       ║
# ║  Colle directement à la suite de PART 1             ║
# ╚══════════════════════════════════════════════════════╝
    norm_res = normality_tests(vals)
    gof      = chi_gof(vals, line)

    # Probability (dist-appropriate)
    if cfg['dist'] == 'poisson' and cmean > 0:
        over_p = float((1 - scipy_stats.poisson.cdf(int(line), cmean)) * 100)
    else:
        if cstd > 0:
            over_p = float((1 - scipy_stats.norm.cdf(line + 0.5, cmean, cstd)) * 100)
        else:
            over_p = float(np.sum(vals > line) / n * 100)
    under_p = 100.0 - over_p

    implied = 52.38
    eo, eu  = over_p - implied, under_p - implied

    if eo >= eu and eo > 0:  rec, raw_edge = 'OVER',  eo
    elif eu > 0:              rec, raw_edge = 'UNDER', eu
    else:                     rec, raw_edge = 'SKIP',  max(eo, eu)

    # Penalty for non-normality
    penalty       = norm_res['confidence_penalty']
    adj_edge      = max(0, raw_edge * (1 - penalty/100)) if rec != 'SKIP' else raw_edge

    # Trend
    try:
        slope, _, r, _, _ = scipy_stats.linregress(np.arange(n), vals)
        r_sq  = round(float(r**2), 3)
        slope = round(float(slope), 3)
    except:
        r_sq = slope = 0.0

    # Quarter Kelly, reduced if non-normal
    kelly = 0.0
    if rec != 'SKIP':
        prob = over_p if rec == 'OVER' else under_p
        if prob > implied:
            kelly = min(20, ((prob/100 - 0.5238) / 0.9091) * 100 * 0.25)
            if norm_res['verdict'] == 'NON_NORMAL': kelly *= 0.5
            elif norm_res['verdict'] == 'BORDERLINE': kelly *= 0.75

    recent = []
    for g in qualified[:10]:
        e = {'date': g['date'][:10] if g['date'] else '', 'stat': g['stat']}
        if 'ip' in g: e['ip'] = round(g['ip'],1)
        if 'er' in g: e['er'] = g['er']
        recent.append(e)

    return {
        'n': n, 'mean': round(mean,1), 'cmean': round(cmean,1),
        'std': round(std,2), 'cstd': round(cstd,2),
        'l5':  round(float(np.mean(vals[:5])),1),
        'l10': round(float(np.mean(vals[:10])),1) if n>=10 else round(mean,1),
        'over_p': round(over_p,1), 'under_p': round(under_p,1),
        'over_n': int(np.sum(vals > line)), 'under_n': int(np.sum(vals <= line)),
        'raw_edge': round(raw_edge,1), 'edge': round(adj_edge,1), 'rec': rec,
        'r_sq': r_sq, 'slope': slope, 'kelly': round(kelly,1),
        'cons': round(max(0, 100-(cstd/cmean*100)),1) if cmean>0 else 50.0,
        'normality': norm_res, 'chi_gof': gof,
        'dist_used': cfg['dist'], 'recent': recent
    }

# ── Odds API ──────────────────────────────────────────────────────────────────

def get_all_props(stat_type_filter=None):
    if not ODDS_API_KEY:
        return {}, {}
    markets = [stat_type_filter] if stat_type_filter else list(STAT_CONFIG.keys())
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

    all_props  = {m: {} for m in markets}
    event_info = {}

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
                        'markets': ','.join(markets), 'oddsFormat': 'american'},
                timeout=10)
            if resp2.status_code != 200:
                continue
            for bk in resp2.json().get('bookmakers',[]):
                for mk in bk.get('markets',[]):
                    mkey = mk['key']
                    if mkey not in all_props:
                        continue
                    for oc in mk.get('outcomes',[]):
                        player = oc.get('description','').strip()
                        point  = oc.get('point')
                        btype  = oc.get('name','')
                        if not player or point is None:
                            continue
                        if player not in all_props[mkey]:
                            all_props[mkey][player] = {'game_id': gid, 'lines': []}
                        all_props[mkey][player]['lines'].append({
                            'book': bk['key'], 'line': float(point),
                            'price': int(oc.get('price',-110)), 'type': btype
                        })
            time.sleep(0.3)
        except:
            continue

    return all_props, event_info

# ── Main scan ─────────────────────────────────────────────────────────────────

def scan(stat_type_filter=None, min_edge=5.0):
    pitchers  = get_probable_pitchers()
    all_props, ev = get_all_props(stat_type_filter)
    stat_types = [stat_type_filter] if stat_type_filter else list(STAT_CONFIG.keys())

    opportunities = []
    analyzed = 0

    for stat_type in stat_types:
        props = all_props.get(stat_type, {})
        is_pitcher = STAT_CONFIG[stat_type]['group'] == 'pitching'

        for prop_name, prop_data in props.items():
            overs = [l for l in prop_data['lines'] if l['type'] == 'Over']
            if not overs:
                continue
            line_counts = Counter([l['line'] for l in overs])
            line = line_counts.most_common(1)[0][0]
            best = min(overs, key=lambda x: abs(x['line'] - line))

            if is_pitcher:
                pitcher = next((p for p in pitchers if names_match(prop_name, p['name'])), None)
                if not pitcher:
                    continue
                player_id   = pitcher['id']
                gi_fallback = {'home_team': pitcher['home_team'],
                               'away_team': pitcher['away_team'],
                               'time': pitcher['game_time']}
            else:
                player_id = search_batter(prop_name)
                if not player_id:
                    continue
                gi_fallback = {'home_team':'','away_team':'','time':''}
                time.sleep(0.1)

            games = get_gamelog(player_id, stat_type)
            if not games or len(games) < 4:
                continue

            analyzed += 1
            a = analyze(games, line, stat_type)
            if not a or a['rec'] == 'SKIP' or a['edge'] < min_edge:
                continue

            gi = ev.get(prop_data['game_id'], gi_fallback)
            is_reliable = (
                a['normality']['verdict'] != 'NON_NORMAL' and
                (a['chi_gof'] is None or a['chi_gof']['is_good_fit'])
            )

            opportunities.append({
                'player': prop_name,
                'stat_type': stat_type,
                'stat_label': STAT_CONFIG[stat_type]['label'],
                'game_info': {
                    'home_team': gi.get('home_team',''),
                    'away_team': gi.get('away_team',''),
                    'time': gi.get('time','')
                },
                'line_analysis': {
                    'bookmaker_line': line,
                    'bookmaker': best['book'].upper(),
                    'recommendation': a['rec'],
                    'raw_edge': a['raw_edge'],
                    'edge': a['edge'],
                    'over_probability': a['over_p'],
                    'under_probability': a['under_p'],
                    'kelly_criterion': a['kelly'],
                    'distribution_used': a['dist_used']
                },
                'statistical_validation': {
                    'normality':   a['normality'],
                    'chi_gof':     a['chi_gof'],
                    'is_reliable': is_reliable
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

    # Reliable first, then by edge
    opportunities.sort(key=lambda x: (
        -int(x['statistical_validation']['is_reliable']),
        -x['line_analysis']['edge']
    ))

    return opportunities, analyzed, len(pitchers)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/api/daily-opportunities', methods=['GET'])
def daily_opportunities():
    try:
        min_edge  = float(request.args.get('min_edge', 5))
        stat_type = request.args.get('stat_type', None)
        if stat_type and stat_type not in STAT_CONFIG:
            return jsonify({'status':'ERROR',
                'message': f"stat_type must be one of {list(STAT_CONFIG.keys())}"}), 400
        opps, analyzed, n_pitchers = scan(stat_type, min_edge)
        return jsonify(to_python({
            'status': 'SUCCESS', 'sport': 'MLB',
            'stat_types_scanned': [stat_type] if stat_type else list(STAT_CONFIG.keys()),
            'total_probable_pitchers': n_pitchers,
            'players_analyzed': analyzed,
            'opportunities_found': len(opps),
            'opportunities': opps,
            'scan_time': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'note': 'edge penalisé si distribution non-normale (Shapiro-Wilk + KS)'
        }))
    except Exception as e:
        import traceback
        return jsonify({'status':'ERROR','message':str(e),
                        'trace':traceback.format_exc()[:800]}), 500

@app.route('/api/schedule', methods=['GET'])
def schedule():
    p = get_probable_pitchers()
    return jsonify({'status':'SUCCESS','pitchers':p,'count':len(p)})

@app.route('/api/stat-types', methods=['GET'])
def stat_types_route():
    return jsonify({'stat_types':[
        {'key': k, 'label': v['label'], 'group': v['group'], 'dist': v['dist']}
        for k, v in STAT_CONFIG.items()
    ]})

@app.route('/api/odds/usage', methods=['GET'])
def usage():
    try:
        r = requests.get(f"{ODDS_BASE}/sports",
            params={'apiKey': ODDS_API_KEY}, timeout=10)
        return jsonify({'used': r.headers.get('x-requests-used','N/A'),
                        'remaining': r.headers.get('x-requests-remaining','N/A')})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy', 'version': '2.0', 'sport': 'MLB',
        'stat_types': list(STAT_CONFIG.keys()),
        'validation': ['shapiro-wilk', 'ks-test', 'chi2-goodness-of-fit']
    })

@app.route('/')
def home():
    return jsonify({'app':'MLB Betting Analyzer','version':'2.0'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
