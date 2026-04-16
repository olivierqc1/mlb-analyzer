# ╔══════════════════════════════════════════════════════╗
# ║  app.py — PARTIE 1/3                                ║
# ║  Colle en premier, puis PARTIE 2 puis PARTIE 3      ║
# ║  Render: Start Command = python app.py              ║
# ╚══════════════════════════════════════════════════════╝
#!/usr/bin/env python3
"""
Multi-Sport Betting Analyzer v1.0
MLB (K, TB) | NHL (Goalie Saves) | Golf (Top 20) | Tennis (stub)
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

ODDS_API_KEY     = os.environ.get('ODDS_API_KEY')
ODDS_BASE        = "https://api.the-odds-api.com/v4"
MLB_BASE         = "https://statsapi.mlb.com/api/v1"
NHL_BASE         = "https://api-web.nhle.com/v1"
NHL_SEARCH_BASE  = "https://search.d3.nhle.com/api/v1/search"
LEAGUE_AVG_K_PCT = 0.225

# Caches
GAMELOG_CACHE    = {}
SCHEDULE_CACHE   = {}
PLAYER_ID_CACHE  = {}
TEAM_STATS_CACHE = {}
SPLITS_CACHE     = {}

# ── Unified Stat Config ────────────────────────────────────────────────────────
STAT_CONFIG = {
    # MLB
    'pitcher_strikeouts': {
        'sport': 'mlb', 'group': 'pitching', 'col': 'strikeOuts',
        'label': 'K', 'min_ip': 3.0, 'dist': 'normal',
        'bettable': True, 'min_games': 5,
        'odds_sport': 'baseball_mlb', 'odds_market': 'pitcher_strikeouts',
    },
    'batter_total_bases': {
        'sport': 'mlb', 'group': 'hitting', 'col': 'totalBases',
        'label': 'TB', 'min_ip': None, 'dist': 'normal',
        'bettable': True, 'min_games': 10,
        'odds_sport': 'baseball_mlb', 'odds_market': 'batter_total_bases',
    },
    # NHL
    'goalie_saves': {
        'sport': 'nhl', 'group': 'goalie', 'col': 'saves',
        'label': 'SV', 'min_ip': None, 'dist': 'normal',
        'bettable': True, 'min_games': 5,
        'odds_sport': 'icehockey_nhl', 'odds_market': 'player_saves',
    },
    # Golf
    'golf_top20': {
        'sport': 'golf', 'group': 'golf', 'col': 'scoring_avg',
        'label': 'T20', 'min_ip': None, 'dist': 'normal',
        'bettable': True, 'min_games': 5,
        'odds_sport': None, 'odds_market': 'top_20_finish',
    },
}

SPORT_STATS = {
    'mlb':    ['pitcher_strikeouts', 'batter_total_bases'],
    'nhl':    ['goalie_saves'],
    'golf':   ['golf_top20'],
    'tennis': [],
}

# ── Common Helpers ────────────────────────────────────────────────────────────

def to_python(obj):
    if isinstance(obj, dict):    return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):    return [to_python(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj); return 0.0 if (math.isnan(v) or math.isinf(v)) else v
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

def safe_request(url, params=None, timeout=12):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"Request error {url}: {e}")
    return None

# ── Statistical Validation ────────────────────────────────────────────────────

def normality_tests(vals):
    results = {}
    try:
        stat, p = scipy_stats.shapiro(vals)
        results['shapiro_wilk'] = {'stat': round(float(stat),4), 'p_value': round(float(p),4),
            'is_normal': float(p) > 0.05, 'label': 'Normal ✅' if float(p) > 0.05 else 'Non-normal ⚠️'}
    except: results['shapiro_wilk'] = None
    try:
        mu, sigma = float(np.mean(vals)), float(np.std(vals))
        if sigma > 0:
            ks_stat, ks_p = scipy_stats.kstest(vals, 'norm', args=(mu, sigma))
            results['ks_test'] = {'stat': round(float(ks_stat),4), 'p_value': round(float(ks_p),4),
                'is_normal': float(ks_p) > 0.05, 'label': 'Normal ✅' if float(ks_p) > 0.05 else 'Non-normal ⚠️'}
        else: results['ks_test'] = None
    except: results['ks_test'] = None
    sw_ok = results.get('shapiro_wilk') and results['shapiro_wilk']['is_normal']
    ks_ok = results.get('ks_test')      and results['ks_test']['is_normal']
    if sw_ok and ks_ok:     v, l, p = 'NORMAL',     '✅ Distribution normale confirmée',         0
    elif sw_ok or ks_ok:    v, l, p = 'BORDERLINE', '⚠️ Distribution approximativement normale', 10
    else:                   v, l, p = 'NON_NORMAL', '❌ Non-normale — edge pénalisé',            25
    results.update({'verdict': v, 'verdict_label': l, 'confidence_penalty': p})
    return results

def chi_gof(vals, line):
    n = len(vals)
    if n < 8: return None
    try:
        mu, sigma = float(np.mean(vals)), float(np.std(vals))
        if sigma <= 0: return None
        over_obs = int(np.sum(vals > line)); under_obs = int(n - over_obs)
        p_over = float(1 - scipy_stats.norm.cdf(line, mu, sigma))
        oe, ue = p_over*n, (1-p_over)*n
        if oe < 1 or ue < 1: return None
        chi2, p = scipy_stats.chisquare([over_obs, under_obs], f_exp=[oe, ue])
        good = float(p) > 0.05
        return {'chi2': round(float(chi2),4), 'p_value': round(float(p),4), 'is_good_fit': good,
                'label': '✅ Bon fit' if good else '⚠️ Fit imparfait',
                'observed': [over_obs, under_obs], 'expected': [round(oe,1), round(ue,1)]}
    except: return None

def bet_quality(a, cfg):
    score = 0; issues = []; pros = []
    if a['n'] >= 15: score += 2; pros.append(f"✅ {a['n']} matchs analysés (solide)")
    elif a['n'] >= 8: score += 1; pros.append(f"🟡 {a['n']} matchs (acceptable)")
    else: issues.append(f"❌ Seulement {a['n']} matchs — données insuffisantes")
    if a['cons'] >= 70: score += 2; pros.append(f"✅ Consistance {a['cons']}% (très stable)")
    elif a['cons'] >= 50: score += 1; pros.append(f"🟡 Consistance {a['cons']}%")
    else: issues.append(f"❌ Consistance {a['cons']}% — imprévisible")
    v = a['normality']['verdict']
    if v == 'NORMAL': score += 2; pros.append("✅ Distribution normale confirmée")
    elif v == 'BORDERLINE': score += 1; pros.append("🟡 Distribution approximativement normale")
    else: issues.append("❌ Distribution non-normale")
    gof = a.get('chi_gof')
    if gof and gof['is_good_fit']: score += 1; pros.append("✅ Chi² GOF cohérent")
    elif gof and not gof['is_good_fit']: issues.append("⚠️ Chi² : fréquences incohérentes")
    margin = abs(a['adj_mean'] - a['line']); sigma = a['cstd'] if a['cstd'] > 0 else 1
    z = margin / sigma
    if z >= 1.5: score += 2; pros.append(f"✅ Marge {margin:.1f} = {z:.1f}σ (signal fort)")
    elif z >= 0.8: score += 1; pros.append(f"🟡 Marge {margin:.1f} = {z:.1f}σ")
    else: issues.append(f"❌ Marge trop faible ({margin:.1f}) — ligne trop proche de la moyenne")
    rec = a['rec']; l5 = a['l5']; line = a['line']
    if rec == 'OVER' and l5 > line: score += 1; pros.append(f"✅ L5 ({l5}) confirme OVER ({line})")
    elif rec == 'UNDER' and l5 < line: score += 1; pros.append(f"✅ L5 ({l5}) confirme UNDER ({line})")
    elif rec == 'OVER': issues.append(f"⚠️ L5 ({l5}) contredit le OVER — forme récente douteuse")
    elif rec == 'UNDER': issues.append(f"⚠️ L5 ({l5}) contredit le UNDER — forme récente douteuse")
    if score >= 8 and not issues: grade, color, label = 'A', '#4ade80', '🟢 BET — Signal solide'
    elif score >= 6 and len(issues) <= 1: grade, color, label = 'B', '#86efac', '🟢 BET — Signal acceptable'
    elif score >= 4: grade, color, label = 'C', '#fbbf24', '🟡 PRUDENCE — Signal faible'
    else: grade, color, label = 'AVOID', '#f87171', '🔴 ÉVITER'
    return {'grade': grade, 'color': color, 'label': label, 'score': score, 'pros': pros, 'issues': issues}

# ── Core Analysis (sport-agnostic) ────────────────────────────────────────────

def analyze(games, line, stat_type, adj_mean_override=None):
    cfg = STAT_CONFIG[stat_type]
    if len(games) < 4: return None
    vals = np.array([g['stat'] for g in games], dtype=float)
    n = len(vals)
    mean = float(np.mean(vals)); std = float(np.std(vals))
    q1, q3 = float(np.percentile(vals,25)), float(np.percentile(vals,75))
    iqr = q3 - q1
    clean = vals[(vals >= q1-1.5*iqr) & (vals <= q3+1.5*iqr)]
    cmean = float(np.mean(clean)) if len(clean) >= 3 else mean
    cstd  = float(np.std(clean))  if len(clean) >= 3 else std
    adj_mean = adj_mean_override if adj_mean_override is not None else cmean
    norm_res = normality_tests(vals)
    gof = chi_gof(vals, line)
    if cstd > 0:
        over_p = float((1 - scipy_stats.norm.cdf(line + 0.5, adj_mean, cstd)) * 100)
    else:
        over_p = float(np.sum(vals > line) / n * 100)
    under_p = 100.0 - over_p
    implied = 52.38
    eo, eu = over_p - implied, under_p - implied
    if eo >= eu and eo > 0:  rec, raw_edge = 'OVER',  eo
    elif eu > 0:              rec, raw_edge = 'UNDER', eu
    else:                     rec, raw_edge = 'SKIP',  max(eo, eu)
    penalty  = norm_res['confidence_penalty']
    adj_edge = max(0, raw_edge * (1 - penalty/100)) if rec != 'SKIP' else raw_edge
    try:
        slope, _, r, _, _ = scipy_stats.linregress(np.arange(n), vals)
        r_sq = round(float(r**2),3); slope = round(float(slope),3)
    except: r_sq = slope = 0.0
    kelly = 0.0
    if rec != 'SKIP' and cfg.get('bettable'):
        prob = over_p if rec == 'OVER' else under_p
        if prob > implied:
            kelly = min(20, ((prob/100 - 0.5238) / 0.9091) * 100 * 0.25)
            if norm_res['verdict'] == 'NON_NORMAL':   kelly *= 0.5
            elif norm_res['verdict'] == 'BORDERLINE': kelly *= 0.75
    cons = round(max(0, 100-(cstd/cmean*100)),1) if cmean > 0 else 50.0
    recent = [{'date': g.get('date','')[:10], 'stat': g['stat']} for g in games[:10]]
    result = {
        'n': n, 'mean': round(mean,1), 'cmean': round(cmean,1), 'adj_mean': round(adj_mean,2),
        'total_adj': round(adj_mean - cmean, 2),
        'std': round(std,2), 'cstd': round(cstd,2),
        'l5': round(float(np.mean(vals[:5])),1),
        'l10': round(float(np.mean(vals[:10])),1) if n>=10 else round(mean,1),
        'over_p': round(over_p,1), 'under_p': round(under_p,1),
        'over_n': int(np.sum(vals > line)), 'under_n': int(np.sum(vals <= line)),
        'raw_edge': round(raw_edge,1), 'edge': round(adj_edge,1), 'rec': rec,
        'r_sq': r_sq, 'slope': slope, 'kelly': round(kelly,1), 'cons': cons,
        'normality': norm_res, 'chi_gof': gof, 'line': line, 'recent': recent
    }
    result['quality'] = bet_quality(result, cfg)
    return result

# ── MLB Data ──────────────────────────────────────────────────────────────────

def mlb_get_probable_pitchers():
    today = datetime.now().strftime('%Y-%m-%d')
    key = f"mlb_schedule_{today}"
    if key in SCHEDULE_CACHE: return SCHEDULE_CACHE[key]
    data = safe_request(f"{MLB_BASE}/schedule", params={
        'sportId': 1, 'date': today, 'hydrate': 'probablePitcher', 'gameType': 'R'})
    if not data: return []
    pitchers = []
    for db in data.get('dates', []):
        for game in db.get('games', []):
            home = game.get('teams',{}).get('home',{}).get('team',{}).get('name','')
            away = game.get('teams',{}).get('away',{}).get('team',{}).get('name','')
            gtime = game.get('gameDate','')
            for side in ['home','away']:
                pp = game.get('teams',{}).get(side,{}).get('probablePitcher',{})
                if pp and pp.get('id'):
                    pitchers.append({'id': pp['id'], 'name': pp.get('fullName',''),
                        'home_team': home, 'away_team': away,
                        'is_home': side=='home', 'opponent': away if side=='home' else home,
                        'game_time': gtime})
    SCHEDULE_CACHE[key] = pitchers
    return pitchers

def mlb_search_batter(name):
    key = norm_name(name)
    if key in PLAYER_ID_CACHE: return PLAYER_ID_CACHE[key]
    last = name.split()[-1] if ' ' in name else name
    data = safe_request(f"{MLB_BASE}/people/search", params={'names': last, 'sportIds': 1})
    if data:
        for p in data.get('people', []):
            if names_match(name, p.get('fullName','')):
                PLAYER_ID_CACHE[key] = p['id']; return p['id']
    return None

def mlb_get_gamelog(player_id, stat_type):
    cfg = STAT_CONFIG[stat_type]
    cache_key = f"mlb_{player_id}_{stat_type}"
    if cache_key in GAMELOG_CACHE: return GAMELOG_CACHE[cache_key]
    data = safe_request(f"{MLB_BASE}/people/{player_id}/stats", params={
        'stats': 'gameLog', 'season': 2025, 'group': cfg['group'], 'gameType': 'R'})
    if not data: return None
    splits = data.get('stats',[{}])[0].get('splits',[])
    if not splits: return None
    games = []
    for s in splits:
        st = s.get('stat',{}); val = st.get(cfg['col'])
        if val is None: continue
        entry = {'date': s.get('date',''), 'stat': int(val)}
        if cfg['group'] == 'pitching':
            ip_str = str(st.get('inningsPitched','0') or '0')
            try:
                p = ip_str.split('.')
                entry['ip'] = int(p[0]) + (int(p[1])/3 if len(p)>1 and p[1] else 0)
            except: entry['ip'] = 0.0
        games.append(entry)
    games.sort(key=lambda x: x['date'], reverse=True)
    GAMELOG_CACHE[cache_key] = games
    return games or None

def mlb_get_opponent_k_pct(team_name):
    key = norm_name(team_name) + '_kpct'
    if key in TEAM_STATS_CACHE: return TEAM_STATS_CACHE[key]
    teams_data = safe_request(f"{MLB_BASE}/teams", params={'sportId': 1, 'season': 2025})
    if not teams_data: return None
    team_id = None
    for t in teams_data.get('teams', []):
        if names_match(team_name, t.get('name','')):
            team_id = t['id']; break
    if not team_id: return None
    data = safe_request(f"{MLB_BASE}/teams/{team_id}/stats", params={
        'stats': 'season', 'group': 'hitting', 'season': 2025, 'gameType': 'R'})
    if not data: return None
    splits = data.get('stats',[{}])[0].get('splits',[{}])
    if not splits: return None
    st = splits[0].get('stat',{})
    k = int(st.get('strikeOuts',0) or 0); ab = int(st.get('atBats',1) or 1)
    bb = int(st.get('baseOnBalls',0) or 0); hbp = int(st.get('hitByPitch',0) or 0)
    pa = ab + bb + hbp
    if pa == 0: return None
    k_pct = k / pa; TEAM_STATS_CACHE[key] = k_pct
    return k_pct

def mlb_context(stat_type, cmean, opponent, is_home, player_id):
    adj = 0.0; adjustments = {}
    if stat_type == 'pitcher_strikeouts':
        opp_k = mlb_get_opponent_k_pct(opponent) if opponent else None
        if opp_k:
            a = round((opp_k - LEAGUE_AVG_K_PCT) / 0.01 * 0.3, 2)
            adjustments['opp_k_pct'] = {'value': round(opp_k*100,1), 'adjustment': a,
                'label': f"Lineup K% {round(opp_k*100,1)}% vs moy {round(LEAGUE_AVG_K_PCT*100,1)}%"}
            adj += a
    return round(cmean + adj, 2), round(adj, 2), adjustments


# ╔══════════════════════════════════════════════════════╗
# ║  app.py — PARTIE 2/3                                ║
# ║  Colle à la suite de PARTIE 1                       ║
# ╚══════════════════════════════════════════════════════╝

# ── NHL Data ──────────────────────────────────────────────────────────────────

def nhl_search_player(name):
    key = 'nhl_' + norm_name(name)
    if key in PLAYER_ID_CACHE: return PLAYER_ID_CACHE[key]
    last = name.split()[-1] if ' ' in name else name
    data = safe_request(NHL_SEARCH_BASE, params={'q': last, 'culture': 'en-us', 'isActive': 'true'})
    if data:
        for r in data:
            full = r.get('name', '')
            if names_match(name, full):
                pid = r.get('playerId')
                if pid:
                    PLAYER_ID_CACHE[key] = pid
                    return pid
    # Fallback: try NHL web API suggest
    data2 = safe_request(f"{NHL_BASE}/player-search", params={'q': last, 'culture': 'en-us', 'isActive': 'true'})
    if data2 and isinstance(data2, list):
        for r in data2:
            full = r.get('name','') or r.get('fullName','')
            if names_match(name, full):
                pid = r.get('playerId') or r.get('id')
                if pid:
                    PLAYER_ID_CACHE[key] = pid
                    return pid
    return None

def nhl_get_goalie_gamelog(player_id):
    """Get goalie saves per game — tries playoffs then regular season."""
    cache_key = f"nhl_{player_id}_saves"
    if cache_key in GAMELOG_CACHE: return GAMELOG_CACHE[cache_key]

    games = []
    for game_type in [3, 2]:  # 3=playoffs, 2=regular
        data = safe_request(f"{NHL_BASE}/player/{player_id}/game-log/20242025/{game_type}")
        if not data: continue
        # New NHL API structure
        gl = data.get('gameLog', [])
        for g in gl:
            saves = g.get('saves')
            if saves is None: continue
            games.append({
                'date': g.get('gameDate',''),
                'stat': int(saves),
                'shots_against': g.get('shotsAgainst', 0),
                'save_pct': g.get('savePctg', 0)
            })

    if not games:
        # Try older NHL API as fallback
        data = safe_request(f"https://statsapi.web.nhl.com/api/v1/people/{player_id}/stats",
            params={'stats': 'gameLog', 'season': '20242025'})
        if data:
            splits = data.get('stats',[{}])[0].get('splits',[])
            for s in splits:
                st = s.get('stat',{})
                saves = st.get('saves')
                if saves is None: continue
                games.append({
                    'date': s.get('date',''),
                    'stat': int(saves),
                    'shots_against': st.get('shotsAgainst', 0),
                    'save_pct': st.get('savePercentage', 0)
                })

    games.sort(key=lambda x: x['date'], reverse=True)
    if games:
        GAMELOG_CACHE[cache_key] = games
    return games or None

def nhl_get_schedule():
    """Get today's NHL games."""
    today = datetime.now().strftime('%Y-%m-%d')
    key = f"nhl_schedule_{today}"
    if key in SCHEDULE_CACHE: return SCHEDULE_CACHE[key]
    data = safe_request(f"{NHL_BASE}/schedule/now")
    if not data: return []
    games = []
    for week in data.get('gameWeek', []):
        for g in week.get('games', []):
            home = g.get('homeTeam', {})
            away = g.get('awayTeam', {})
            games.append({
                'id': g.get('id'),
                'home_team': home.get('placeName', {}).get('default','') + ' ' + home.get('commonName', {}).get('default',''),
                'away_team': away.get('placeName', {}).get('default','') + ' ' + away.get('commonName', {}).get('default',''),
                'home_abbrev': home.get('abbrev',''),
                'away_abbrev': away.get('abbrev',''),
                'time': g.get('startTimeUTC','')
            })
    SCHEDULE_CACHE[key] = games
    return games

# ── Odds API (unified) ────────────────────────────────────────────────────────

def get_odds_props(odds_sport, odds_market, max_games=15):
    """Fetch player props for any sport from The Odds API."""
    if not ODDS_API_KEY: return {}, {}
    # Get games
    data = safe_request(f"{ODDS_BASE}/sports/{odds_sport}/odds", params={
        'apiKey': ODDS_API_KEY, 'regions': 'us', 'markets': 'h2h', 'oddsFormat': 'american'})
    if not data: return {}, {}
    props = {}; event_info = {}
    for game in data[:max_games]:
        gid = game['id']
        event_info[gid] = {'home_team': game.get('home_team',''),
                           'away_team': game.get('away_team',''),
                           'time': game.get('commence_time','')}
        try:
            d2 = safe_request(f"{ODDS_BASE}/sports/{odds_sport}/events/{gid}/odds",
                params={'apiKey': ODDS_API_KEY, 'regions': 'us',
                        'markets': odds_market, 'oddsFormat': 'american'})
            if not d2: continue
            for bk in d2.get('bookmakers',[]):
                for mk in bk.get('markets',[]):
                    if mk['key'] != odds_market: continue
                    for oc in mk.get('outcomes',[]):
                        player = oc.get('description','').strip()
                        point  = oc.get('point')
                        btype  = oc.get('name','')
                        if not player or point is None: continue
                        if player not in props:
                            props[player] = {'game_id': gid, 'lines': []}
                        props[player]['lines'].append({
                            'book': bk['key'], 'line': float(point),
                            'price': int(oc.get('price',-110)), 'type': btype
                        })
            time.sleep(0.3)
        except: continue
    return props, event_info

# ── Main Scan (unified) ───────────────────────────────────────────────────────

def scan_sport(sport, stat_type_filter=None, min_edge=5.0):
    stat_types = [stat_type_filter] if stat_type_filter else SPORT_STATS.get(sport, [])
    if not stat_types:
        return [], 0, 0

    opportunities = []
    analyzed = 0
    n_games = 0

    if sport == 'mlb':
        pitchers = mlb_get_probable_pitchers()
        n_games = len(set(p['home_team'] for p in pitchers)) if pitchers else 0

        for stat_type in stat_types:
            cfg = STAT_CONFIG[stat_type]
            props, ev = get_odds_props(cfg['odds_sport'], cfg['odds_market'])
            is_pitcher = cfg['group'] == 'pitching'

            for prop_name, prop_data in props.items():
                overs = [l for l in prop_data['lines'] if l['type'] == 'Over']
                if not overs: continue
                line = Counter([l['line'] for l in overs]).most_common(1)[0][0]
                best = min(overs, key=lambda x: abs(x['line'] - line))

                if is_pitcher:
                    pitcher = next((p for p in pitchers if names_match(prop_name, p['name'])), None)
                    if not pitcher: continue
                    player_id = pitcher['id']
                    opponent  = pitcher.get('opponent','')
                    is_home   = pitcher.get('is_home', True)
                    gi_fb = {'home_team': pitcher['home_team'], 'away_team': pitcher['away_team'], 'time': pitcher['game_time']}
                else:
                    player_id = mlb_search_batter(prop_name)
                    if not player_id: continue
                    opponent = ''; is_home = True; gi_fb = {'home_team':'','away_team':'','time':''}
                    time.sleep(0.1)

                games = mlb_get_gamelog(player_id, stat_type)
                if not games or len(games) < cfg['min_games']: continue
                analyzed += 1

                # Context adjustment for K
                adj_mean = None
                if stat_type == 'pitcher_strikeouts':
                    am, _, _ = mlb_context(stat_type, np.mean([g['stat'] for g in games]), opponent, is_home, player_id)
                    adj_mean = am

                a = analyze(games, line, stat_type, adj_mean_override=adj_mean)
                if not a or a['rec'] == 'SKIP' or a['edge'] < min_edge: continue
                if a['quality']['grade'] == 'AVOID': continue

                gi = ev.get(prop_data['game_id'], gi_fb)
                opportunities.append(_build_opp(prop_name, stat_type, sport, line, best, gi,
                    opponent, is_home, a, adj_mean))

    elif sport == 'nhl':
        nhl_schedule = nhl_get_schedule()
        n_games = len(nhl_schedule)
        cfg = STAT_CONFIG['goalie_saves']
        props, ev = get_odds_props(cfg['odds_sport'], cfg['odds_market'])

        for prop_name, prop_data in props.items():
            overs = [l for l in prop_data['lines'] if l['type'] == 'Over']
            if not overs: continue
            line = Counter([l['line'] for l in overs]).most_common(1)[0][0]
            best = min(overs, key=lambda x: abs(x['line'] - line))

            player_id = nhl_search_player(prop_name)
            if not player_id: continue

            games = nhl_get_goalie_gamelog(player_id)
            if not games or len(games) < cfg['min_games']: continue
            analyzed += 1

            a = analyze(games, line, 'goalie_saves')
            if not a or a['rec'] == 'SKIP' or a['edge'] < min_edge: continue
            if a['quality']['grade'] == 'AVOID': continue

            gi = ev.get(prop_data['game_id'], {'home_team':'','away_team':'','time':''})
            opportunities.append(_build_opp(prop_name, 'goalie_saves', 'nhl', line, best, gi,
                '', True, a, None))

    elif sport == 'golf':
        opportunities = []  # Golf: à implémenter - nécessite accès stats PGA
        analyzed = 0
        n_games = 0

    # Sort by grade then edge
    grade_order = {'A': 0, 'B': 1, 'C': 2}
    opportunities.sort(key=lambda x: (grade_order.get(x['quality']['grade'],3), -x['line_analysis']['edge']))
    return opportunities, analyzed, n_games

def _build_opp(player, stat_type, sport, line, best, gi, opponent, is_home, a, adj_mean):
    cfg = STAT_CONFIG[stat_type]
    return {
        'player': player, 'sport': sport,
        'stat_type': stat_type, 'stat_label': cfg['label'],
        'game_info': {'home_team': gi.get('home_team',''), 'away_team': gi.get('away_team',''),
                      'time': gi.get('time',''), 'opponent': opponent, 'is_home': is_home},
        'quality': a['quality'],
        'line_analysis': {
            'bookmaker_line': line, 'bookmaker': best['book'].upper(),
            'recommendation': a['rec'], 'raw_edge': a['raw_edge'], 'edge': a['edge'],
            'over_probability': a['over_p'], 'under_probability': a['under_p'],
            'kelly_criterion': a['kelly']
        },
        'deep_stats': {
            'mean': a['mean'], 'clean_mean': a['cmean'], 'adjusted_mean': a['adj_mean'],
            'std': a['std'], 'avg_last_5': a['l5'], 'avg_last_10': a['l10'],
            'consistency': a['cons'], 'games_analyzed': a['n'],
            'over_count': a['over_n'], 'under_count': a['under_n'],
            'r_squared': a['r_sq'], 'trend_slope': a['slope']
        },
        'statistical_validation': {
            'normality': a['normality'], 'chi_gof': a['chi_gof'],
            'is_reliable': a['normality']['verdict'] != 'NON_NORMAL' and
                           (a['chi_gof'] is None or a['chi_gof']['is_good_fit'])
        },
        'recent_games': a['recent']
    }

# ╔══════════════════════════════════════════════════════╗
# ║  app.py — PARTIE 3/3                                ║
# ║  Colle à la suite de PARTIE 2                       ║
# ╚══════════════════════════════════════════════════════╝

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/api/daily-opportunities', methods=['GET'])
def daily_opportunities():
    try:
        sport     = request.args.get('sport', 'mlb').lower()
        stat_type = request.args.get('stat_type', None)
        min_edge  = float(request.args.get('min_edge', 5))

        if sport not in SPORT_STATS:
            return jsonify({'status':'ERROR', 'message': f"sport doit être: {list(SPORT_STATS.keys())}"}), 400
        if sport == 'tennis':
            return jsonify({'status': 'SUCCESS', 'sport': 'tennis',
                'opportunities': [], 'players_analyzed': 0,
                'message': '🎾 Tennis en développement — données ATP insuffisantes pour modélisation fiable. Priorité: NHL saves.',
                'scan_time': datetime.now().strftime('%Y-%m-%d %H:%M')})
        if sport == 'golf':
            return jsonify({'status': 'SUCCESS', 'sport': 'golf',
                'opportunities': [], 'players_analyzed': 0,
                'message': '⛳ Golf en développement — API PGA Tour stats nécessaire. Reviens dans la prochaine version.',
                'scan_time': datetime.now().strftime('%Y-%m-%d %H:%M')})

        opps, analyzed, n_games = scan_sport(sport, stat_type, min_edge)
        return jsonify(to_python({
            'status': 'SUCCESS', 'sport': sport, 'version': '1.0',
            'stat_types_scanned': [stat_type] if stat_type else SPORT_STATS[sport],
            'total_games': n_games, 'players_analyzed': analyzed,
            'opportunities_found': len(opps), 'opportunities': opps,
            'scan_time': datetime.now().strftime('%Y-%m-%d %H:%M')
        }))
    except Exception as e:
        import traceback
        return jsonify({'status':'ERROR','message':str(e),
                        'trace':traceback.format_exc()[:800]}), 500

@app.route('/api/sports', methods=['GET'])
def sports_info():
    return jsonify({'sports': [
        {'key': 'mlb', 'label': '⚾ MLB', 'status': 'active',
         'stats': ['pitcher_strikeouts (K)', 'batter_total_bases (TB)'],
         'data_source': 'MLB Stats API (gratuit)'},
        {'key': 'nhl', 'label': '🏒 NHL', 'status': 'active',
         'stats': ['goalie_saves (SV)'],
         'data_source': 'NHL API (gratuit)'},
        {'key': 'golf', 'label': '⛳ Golf', 'status': 'coming_soon',
         'stats': ['top_20_finish (T20)'],
         'data_source': 'PGA Tour API — en développement'},
        {'key': 'tennis', 'label': '🎾 Tennis', 'status': 'coming_soon',
         'stats': ['aces (ACE)'],
         'data_source': 'ATP Tour — en développement'},
    ]})

@app.route('/api/nhl/schedule', methods=['GET'])
def nhl_schedule():
    games = nhl_get_schedule()
    return jsonify({'status':'SUCCESS','games':games,'count':len(games)})

@app.route('/api/mlb/schedule', methods=['GET'])
def mlb_schedule():
    p = mlb_get_probable_pitchers()
    return jsonify({'status':'SUCCESS','pitchers':p,'count':len(p)})

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
    return jsonify({
        'status': 'healthy', 'version': '1.0',
        'sports': {
            'mlb': {'status': 'active', 'markets': ['pitcher_strikeouts','batter_total_bases']},
            'nhl': {'status': 'active', 'markets': ['goalie_saves']},
            'golf': {'status': 'coming_soon'},
            'tennis': {'status': 'coming_soon'}
        },
        'validation': ['shapiro-wilk','ks-test','chi2-gof','bet_quality_grade']
    })

@app.route('/')
def home():
    return jsonify({'app':'Multi-Sport Betting Analyzer','version':'1.0',
                    'sports':['mlb','nhl','golf (soon)','tennis (soon)']})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("⚾🏒⛳🎾 Multi-Sport Analyzer v1.0")
    app.run(host='0.0.0.0', port=port, debug=False)
