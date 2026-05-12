# soccer.py — Soccer Scanner automatique
# Flux : Odds API → matchs + cotes → API-Football → stats équipes → Dixon-Coles → EV
# Coût : 0$ (API-Football free tier ~8 calls/match)
from flask import Blueprint, jsonify, request
import os, math, time, re
import requests as req
from datetime import datetime
from stats import to_py, calc_ev, calc_kelly

soccer_bp = Blueprint('soccer', __name__)
ODDS_KEY = os.environ.get('ODDS_API_KEY','')
FB_KEY   = os.environ.get('FOOTBALL_API_KEY','')
ODDS_BASE = 'https://api.the-odds-api.com/v4'
FB_BASE   = 'https://v3.football.api-sports.io'
SEASON    = 2024
HOME_ADV  = 1.15

# Odds API sport keys → (API-Football league_id, avg_goals)
SOCCER_LEAGUES = {
    'soccer_spain_la_liga':         (140, 2.50),
    'soccer_epl':                   (39,  2.60),
    'soccer_italy_serie_a':         (135, 2.70),
    'soccer_germany_bundesliga':    (78,  3.10),
    'soccer_france_ligue_one':      (61,  2.60),
    'soccer_uefa_champs_league':    (2,   2.80),
    'soccer_argentina_primera_division': (128, 2.40),
    'soccer_usa_mls':               (253, 2.90),
    'soccer_england_championship':  (40,  2.40),
    'soccer_belgium_first_div':     (144, 2.90),
}

_team_cache  = {}
_stats_cache = {}

# ── API-Football ──────────────────────────────────────────────────────────────
def fb(endpoint, params):
    if not FB_KEY: return None
    p = {'season': SEASON}; p.update(params)
    try:
        r = req.get(f'{FB_BASE}/{endpoint}',
                    params=p, headers={'x-apisports-key': FB_KEY}, timeout=12)
        if r.status_code == 200:
            return r.json().get('response', [])
    except Exception as e: print(f'FB error: {e}')
    return None

def get_team_id(name, league_id):
    key = f'{name}_{league_id}'
    if key in _team_cache: return _team_cache[key]
    res = fb('teams', {'name': name, 'league': league_id}) or fb('teams', {'name': name}) or []
    if not res: return None
    tid = res[0]['team']['id']
    _team_cache[key] = tid; return tid

def get_team_stats(team_id, league_id):
    key = f'{team_id}_{league_id}'
    if key in _stats_cache: return _stats_cache[key]
    res = fb('teams/statistics', {'team': team_id, 'league': league_id})
    s   = res if isinstance(res, dict) else (res[0] if res else {})
    try:
        gf  = s.get('goals',{}).get('for',{})
        ga  = s.get('goals',{}).get('against',{})
        pl  = s.get('fixtures',{}).get('played',{}).get('total',1) or 1
        af  = gf.get('average',{}).get('total')
        aag = ga.get('average',{}).get('total')
        avg_f = float(af)  if af  else (gf.get('total',{}).get('total',0) or 0)/pl
        avg_a = float(aag) if aag else (ga.get('total',{}).get('total',0) or 0)/pl
        result = (round(avg_f,3), round(avg_a,3))
        _stats_cache[key] = result; return result
    except: return (None, None)

# ── Dixon-Coles ───────────────────────────────────────────────────────────────
def pmf(k, lam):
    if lam <= 0 or k < 0: return 0.0
    try:
        lp = -lam + k*math.log(lam)
        for i in range(1,k+1): lp -= math.log(i)
        return math.exp(lp)
    except: return 0.0

def tau(i, j, lh, la, rho=-0.13):
    if i==0 and j==0: return max(0.01, 1-lh*la*rho)
    if i==1 and j==0: return 1+la*rho
    if i==0 and j==1: return 1+lh*rho
    if i==1 and j==1: return 1-rho
    return 1.0

def dixon_coles(lh, la):
    lh = max(0.3, min(4.5, float(lh)))
    la = max(0.3, min(4.5, float(la)))
    ph=pd=pa=btts=o15=o25=o35=0.0
    for i in range(9):
        for j in range(9):
            p = pmf(i,lh)*pmf(j,la)*tau(i,j,lh,la)
            if i>j: ph+=p
            elif i==j: pd+=p
            else: pa+=p
            if i>0 and j>0: btts+=p
            if i+j>1: o15+=p
            if i+j>2: o25+=p
            if i+j>3: o35+=p
    return {'home':round(ph,4),'draw':round(pd,4),'away':round(pa,4),
            'btts_yes':round(btts,4),'btts_no':round(1-btts,4),
            'over15':round(o15,4),'under15':round(1-o15,4),
            'over25':round(o25,4),'under25':round(1-o25,4),
            'over35':round(o35,4),'under35':round(1-o35,4)}

# ── Odds API soccer ───────────────────────────────────────────────────────────
MARKET_KEYS = ['h2h','totals','btts_yes_no']

def get_soccer_events(sport_key, max_events=8):
    """Retourne les événements avec toutes leurs cotes."""
    if not ODDS_KEY: return []
    try:
        r = req.get(f'{ODDS_BASE}/sports/{sport_key}/odds', params={
            'apiKey':ODDS_KEY,'regions':'eu,us','markets':','.join(MARKET_KEYS),
            'oddsFormat':'american','dateFormat':'iso'}, timeout=12)
        if r.status_code != 200: return []
        return r.json()[:max_events]
    except Exception as e:
        print(f'Soccer odds error {sport_key}: {e}'); return []

def parse_odds(event):
    """Extrait les meilleures cotes par outcome depuis un event Odds API."""
    best = {}
    for bk in event.get('bookmakers',[]):
        for mk in bk.get('markets',[]):
            mkey = mk['key']
            for oc in mk.get('outcomes',[]):
                name  = oc.get('name','')
                price = oc.get('price', -110)
                point = oc.get('point')


                # Clé unique par outcome
                if mkey == 'h2h':
                    home = event.get('home_team','')
                    away = event.get('away_team','')
                    if name == home:    k = 'home'
                    elif name == away:  k = 'away'
                    elif name == 'Draw': k = 'draw'
                    else: continue
                elif mkey == 'totals':
                    if point is None: continue
                    k = f'over25' if (name=='Over'  and abs(point-2.5)<0.1) else \
                        f'under25' if (name=='Under' and abs(point-2.5)<0.1) else \
                        f'over15'  if (name=='Over'  and abs(point-1.5)<0.1) else \
                        f'under15' if (name=='Under' and abs(point-1.5)<0.1) else \
                        f'over35'  if (name=='Over'  and abs(point-3.5)<0.1) else \
                        f'under35' if (name=='Under' and abs(point-3.5)<0.1) else None
                    if k is None: continue
                elif mkey == 'btts_yes_no':
                    k = 'btts_yes' if name=='Yes' else 'btts_no' if name=='No' else None
                    if k is None: continue
                else: continue
                # Garde la meilleure cote (plus haute = plus favorable)
                if k not in best or price > best[k]:
                    best[k] = price
    return best

# ── Scanner principal ─────────────────────────────────────────────────────────
def scan_soccer(min_ev=3.0):
    opps = []
    analyzed = 0
    ngames   = 0

    for sport_key, (league_id, lg_avg) in SOCCER_LEAGUES.items():
        events = get_soccer_events(sport_key, max_events=6)
        if not events: continue
        ngames += len(events)

        for event in events:
            home_name = event.get('home_team','')
            away_name = event.get('away_team','')
            game_time = event.get('commence_time','')
            odds      = parse_odds(event)
            if not odds: continue

            # Récupère les IDs équipes
            home_id = get_team_id(home_name, league_id); time.sleep(0.3)
            away_id = get_team_id(away_name, league_id); time.sleep(0.3)
            if not home_id or not away_id: continue

            # Stats équipes
            hs, hc = get_team_stats(home_id, league_id); time.sleep(0.3)
            as_, ac = get_team_stats(away_id, league_id); time.sleep(0.3)
            half = lg_avg/2
            hs = hs or half; hc = hc or half
            as_ = as_ or half; ac = ac or half

            # Dixon-Coles xG
            xg_h = round((hs/half)*(ac/half)*HOME_ADV*half, 3)
            xg_a = round((as_/half)*(hc/half)*(1/HOME_ADV)*half, 3)
            probs = dixon_coles(xg_h, xg_a)
            analyzed += 1

            gi = {'home_team':home_name,'away_team':away_name,
                  'time':game_time,'xgHome':xg_h,'xgAway':xg_a}

            # Cherche les opportunités +EV
            market_labels = {
                'home':f'{home_name} Win','draw':'Draw','away':f'{away_name} Win',
                'btts_yes':'BTTS Oui','btts_no':'BTTS Non',
                'over15':'Over 1.5','under15':'Under 1.5',
                'over25':'Over 2.5','under25':'Under 2.5',
                'over35':'Over 3.5','under35':'Under 3.5',
            }
            for k, model_prob in probs.items():
                if k not in odds or k not in market_labels: continue
                american = odds[k]
                ev = calc_ev(model_prob, american)
                if ev is None or ev < min_ev: continue
                kelly = calc_kelly(model_prob, american)
                implied = abs(american)/(abs(american)+100) if american<0 else 100/(american+100)

                # Grade basé sur EV et confiance des stats
                if ev >= 8 and model_prob >= 0.55:   grade,color = 'A','#4ade80'
                elif ev >= 5 and model_prob >= 0.45: grade,color = 'B','#86efac'
                else:                                 grade,color = 'C','#fbbf24'

                opps.append({
                    'player': f'{home_name} vs {away_name}',
                    'sport': 'soccer',
                    'stat_type': k,
                    'stat_label': market_labels[k],
                    'game_info': gi,
                    'quality': {
                        'grade': grade, 'color': color,
                        'label': f'{"🟢 BET" if grade in ("A","B") else "🟡 PRUDENCE"} — EV {ev:+.1f}%',
                        'pros': [f'✅ Modèle: {round(model_prob*100,1)}% vs Book: {round(implied*100,1)}%',
                                 f'✅ xG: {xg_h} - {xg_a}'],
                        'issues': []
                    },
                    'line_analysis': {
                        'bookmaker_line': 0,
                        'bookmaker': 'BEST',
                        'recommendation': 'BET',
                        'edge': round(ev, 1),
                        'over_probability': round(model_prob*100, 1),
                        'under_probability': round((1-model_prob)*100, 1),
                        'kelly_criterion': kelly,
                        'actual_odds': american,
                    },
                    'deep_stats': {
                        'mean': xg_h, 'weighted_mean': xg_a,
                        'avg_last_5': round(model_prob*100,1),
                        'avg_last_10': round(implied*100,1),
                        'hit_rate': round(model_prob*100,1),
                        'over_count': 0, 'under_count': 0,
                        'games_analyzed': 0, 'trend': 'stable',
                    },
                    'statistical_validation': {'is_reliable': grade in ('A','B')},
                    'recent_games': [],
                })

        time.sleep(0.5)  # Entre chaque ligue

    opps.sort(key=lambda x: ({'A':0,'B':1,'C':2}.get(x['quality']['grade'],3),
                               -x['line_analysis']['edge']))
    return opps, analyzed, ngames

@soccer_bp.route('/api/soccer/health', methods=['GET'])
def soccer_health():
    return jsonify({'status':'healthy','model':'Dixon-Coles scanner',
                    'football_api_key':'ok' if FB_KEY else 'MISSING',
                    'odds_api_key':'ok' if ODDS_KEY else 'MISSING'})
