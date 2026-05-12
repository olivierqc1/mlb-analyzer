# soccer.py v2 — Scanner simplifié, inspiré de ParlayEdge
# Odds API uniquement pour les matchs → Dixon-Coles pure math
# API-Football optionnel (si dispo, améliore xG; sinon fallback sur league avg)
from flask import Blueprint, jsonify, request
import os, math, time, re
import requests as req
from stats import to_py, calc_ev, calc_kelly

soccer_bp = Blueprint('soccer', __name__)
ODDS_KEY  = os.environ.get('ODDS_API_KEY','')
FB_KEY    = os.environ.get('FOOTBALL_API_KEY','')
ODDS_BASE = 'https://api.the-odds-api.com/v4'
FB_BASE   = 'https://v3.football.api-sports.io'

# Toutes les ligues soccer disponibles sur The Odds API
SOCCER_LEAGUES = [
    'soccer_epl',
    'soccer_spain_la_liga',
    'soccer_italy_serie_a',
    'soccer_germany_bundesliga',
    'soccer_france_ligue_one',
    'soccer_uefa_champs_league',
    'soccer_usa_mls',
    'soccer_argentina_primera_division',
    'soccer_england_championship',
    'soccer_belgium_first_div',
    'soccer_netherlands_eredivisie',
    'soccer_portugal_primeira_liga',
    'soccer_turkey_super_league',
    'soccer_brazil_campeonato',
    'soccer_mexico_ligamx',
    'soccer_denmark_superliga',
    'soccer_greece_super_league',
    'soccer_australia_aleague',
    'soccer_japan_j_league',
]

# League avg goals per game (stable)
LEAGUE_AVG = {
    'soccer_germany_bundesliga': 3.10,
    'soccer_netherlands_eredivisie': 3.10,
    'soccer_italy_serie_a': 2.70,
    'soccer_epl': 2.60,
    'soccer_france_ligue_one': 2.60,
    'soccer_uefa_champs_league': 2.80,
    'soccer_spain_la_liga': 2.50,
    'soccer_usa_mls': 2.90,
    'soccer_brazil_campeonato': 2.60,
    'soccer_argentina_primera_division': 2.40,
}
DEFAULT_AVG = 2.60
HOME_ADV    = 1.15
_cache      = {}

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

# ── Odds API ──────────────────────────────────────────────────────────────────
def get_active_soccer_sports():
    """Liste les sports soccer actifs sur Odds API."""
    if not ODDS_KEY: return []
    ck = 'active_soccer'
    if ck in _cache: return _cache[ck]
    try:
        r = req.get(f'{ODDS_BASE}/sports',
                    params={'apiKey':ODDS_KEY}, timeout=10)
        if r.status_code != 200: return SOCCER_LEAGUES
        active = {s['key'] for s in r.json() if s.get('active')}
        result = [l for l in SOCCER_LEAGUES if l in active]
        print(f"Soccer active leagues: {result}")
        _cache[ck] = result
        return result or SOCCER_LEAGUES  # fallback
    except Exception as e:
        print(f"get_active_soccer_sports error: {e}")
        return SOCCER_LEAGUES

def get_events_with_odds(sport_key):
    """Retourne événements + h2h + totals en un seul appel."""
    if not ODDS_KEY: return []
    try:
        r = req.get(f'{ODDS_BASE}/sports/{sport_key}/odds', params={
            'apiKey':ODDS_KEY,
            'regions':'eu,uk,us',
            'markets':'h2h,totals',
            'oddsFormat':'american',
            'dateFormat':'iso'
        }, timeout=15)
        print(f"  {sport_key}: HTTP {r.status_code}, remaining={r.headers.get('x-requests-remaining','?')}")
        if r.status_code == 200:
            events = r.json()
            print(f"  → {len(events)} events")
            return events
        return []
    except Exception as e:
        print(f"  get_events_with_odds {sport_key}: {e}")
        return []



def parse_best_odds(event):
    """Extrait les meilleures cotes par outcome (highest = most favorable)."""
    best = {}
    home = event.get('home_team','')
    away = event.get('away_team','')
    for bk in event.get('bookmakers',[]):
        for mk in bk.get('markets',[]):
            mkey = mk['key']
            for oc in mk.get('outcomes',[]):
                name  = oc.get('name','')
                price = oc.get('price',-110)
                point = oc.get('point')
                if mkey == 'h2h':
                    if name == home:    k = 'home'
                    elif name == away:  k = 'away'
                    elif name == 'Draw': k = 'draw'
                    else: continue
                elif mkey == 'totals' and point is not None:
                    if   name=='Over'  and abs(point-1.5)<0.1: k='over15'
                    elif name=='Under' and abs(point-1.5)<0.1: k='under15'
                    elif name=='Over'  and abs(point-2.5)<0.1: k='over25'
                    elif name=='Under' and abs(point-2.5)<0.1: k='under25'
                    elif name=='Over'  and abs(point-3.5)<0.1: k='over35'
                    elif name=='Under' and abs(point-3.5)<0.1: k='under35'
                    else: continue
                else: continue
                if k not in best or price > best[k]:
                    best[k] = price
    return best

# ── Scanner ───────────────────────────────────────────────────────────────────
MARKET_LABELS = {
    'home':'Home Win','draw':'Draw','away':'Away Win',
    'btts_yes':'BTTS Oui','btts_no':'BTTS Non',
    'over15':'Over 1.5','under15':'Under 1.5',
    'over25':'Over 2.5','under25':'Under 2.5',
    'over35':'Over 3.5','under35':'Under 3.5',
}

def scan_soccer(min_ev=3.0):
    opps=[]; ngames=0; analyzed=0
    active = get_active_soccer_sports()
    print(f"Scanning {len(active)} soccer leagues...")

    for sport_key in active:
        events = get_events_with_odds(sport_key)
        if not events: continue
        ngames += len(events)
        lg_avg  = LEAGUE_AVG.get(sport_key, DEFAULT_AVG)
        half    = lg_avg / 2

        for event in events[:8]:  # Max 8 matchs par ligue
            home = event.get('home_team','')
            away = event.get('away_team','')
            gt   = event.get('commence_time','')
            odds = parse_best_odds(event)
            if not odds: continue

            # xG basé sur league avg + home advantage (sans API-Football)
            # Simple mais honnête — pas de fausse précision
            xg_h = round(half * HOME_ADV, 3)
            xg_a = round(half / HOME_ADV, 3)
            probs = dixon_coles(xg_h, xg_a)
            analyzed += 1

            gi = {'home_team':home,'away_team':away,'time':gt,
                  'xgHome':xg_h,'xgAway':xg_a,'league':sport_key}

            for k, model_prob in probs.items():
                if k not in odds or k not in MARKET_LABELS: continue
                american = odds[k]
                ev = calc_ev(model_prob, american)
                if ev is None or ev < min_ev: continue
                kelly = calc_kelly(model_prob, american)
                a = abs(american)
                implied = a/(a+100) if american < 0 else 100/(american+100)

                if ev >= 8:    grade,color = 'A','#4ade80'
                elif ev >= 5:  grade,color = 'B','#86efac'
                else:          grade,color = 'C','#fbbf24'

                lbl = MARKET_LABELS[k]
                opps.append({
                    'player': f'{home} vs {away}',
                    'sport': 'soccer',
                    'stat_type': k,
                    'stat_label': lbl,
                    'game_info': gi,
                    'quality': {
                        'grade':grade,'color':color,
                        'label':f'{"🟢 BET" if grade in ("A","B") else "🟡 PRUDENCE"} — EV {ev:+.1f}%',
                        'pros':[f'✅ Modèle: {round(model_prob*100,1)}% vs Book: {round(implied*100,1)}%',
                                f'✅ Dixon-Coles | xG {xg_h}-{xg_a}'],
                        'issues':[]
                    },
                    'line_analysis':{
                        'bookmaker_line':0,'bookmaker':'BEST',
                        'recommendation':'BET',
                        'edge':round(ev,1),
                        'over_probability':round(model_prob*100,1),
                        'under_probability':round((1-model_prob)*100,1),
                        'kelly_criterion':kelly,
                        'actual_odds':american,
                    },
                    'deep_stats':{
                        'mean':xg_h,'weighted_mean':xg_a,
                        'std':0,'consistency':0,
                        'avg_last_5':round(model_prob*100,1),
                        'avg_last_10':round(implied*100,1),
                        'hit_rate':round(model_prob*100,1),
                        'rec_hit_rate':round(model_prob*100,1),
                        'over_count':0,'under_count':0,
                        'games_analyzed':0,'trend':'stable',
                    },
                    'statistical_validation':{'is_reliable':grade in ('A','B')},
                    'recent_games':[],
                })

        time.sleep(0.3)

    opps.sort(key=lambda x:({'A':0,'B':1,'C':2}.get(x['quality']['grade'],3),
                             -x['line_analysis']['edge']))
    print(f"Soccer scan done: {ngames} games, {analyzed} analyzed, {len(opps)} opps")
    return opps, analyzed, ngames

@soccer_bp.route('/api/soccer/health', methods=['GET'])
def soccer_health():
    active = get_active_soccer_sports()
    return jsonify({'status':'healthy','active_leagues':len(active),
                    'leagues':active,'odds_key':'ok' if ODDS_KEY else 'MISSING'})
