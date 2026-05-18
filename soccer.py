# soccer.py — Soccer Player Props (manuel, API-Football free tier)
# Tu entres le joueur + ligne vue sur BetOnline → modèle calcule EV + Kelly
# ~17 calls par joueur, 100 calls/jour gratuit = 5 joueurs/jour
from flask import Blueprint, jsonify, request
import os, math, time, re
import requests as req
from stats import to_py, calc_ev, calc_kelly

soccer_bp = Blueprint('soccer', __name__)
FB_KEY   = os.environ.get('FOOTBALL_API_KEY','')
FB_BASE  = 'https://v3.football.api-sports.io'

LEAGUES = {
    'EPL':39,'La Liga':140,'Serie A':135,
    'Bundesliga':78,'Ligue 1':61,'MLS':253,
    'Champions League':2,'Eredivisie':88,
}
SEASON = 2024
_cache = {}

def fb(endpoint, params, timeout=12):
    if not FB_KEY:
        raise ValueError('FOOTBALL_API_KEY manquant sur Render')
    p = {'season': SEASON}; p.update(params)
    r = req.get(f'{FB_BASE}/{endpoint}', params=p,
                headers={'x-apisports-key': FB_KEY}, timeout=timeout)
    rem = r.headers.get('x-ratelimit-requests-remaining','?')
    print(f'  FB /{endpoint} | HTTP {r.status_code} | rem={rem}')
    if r.status_code == 200:
        return r.json().get('response', [])
    return []

# ── Cherche joueur + son équipe ───────────────────────────────────────────────
def find_player(name, league_id):
    key = f'player_{name}_{league_id}'
    if key in _cache: return _cache[key]
    # Try current season first, then previous, then without season
    for params in [
        {'search': name, 'league': league_id},
        {'search': name.split()[-1], 'league': league_id},
        {'search': name},  # no league filter
        {'search': name.split()[-1]},
    ]:
        results = fb('players', params)
        if results: break
    if not results:
        _cache[key] = None; return None
    # Best match
    best = None
    for r in results:
        pname = r.get('player',{}).get('name','').lower()
        if name.lower().split()[-1] in pname:
            best = r; break
    if not best: best = results[0]
    _cache[key] = best
    return best

# ── Récupère les derniers matchs d'un joueur ──────────────────────────────────
def get_player_fixtures(player_id, team_id, league_id, n=15):
    key = f'fix_{player_id}_{league_id}'
    if key in _cache: return _cache[key]

    # Derniers matchs de l'équipe
    fixtures = fb('fixtures', {'team': team_id, 'league': league_id, 'last': n})
    if not fixtures:
        _cache[key] = []
        return []

    fixture_ids = [f['fixture']['id'] for f in fixtures]
    games = []

    for fid in fixture_ids:
        fkey = f'fp_{fid}_{player_id}'
        if fkey in _cache:
            entry = _cache[fkey]
            if entry: games.append(entry)
            continue

        # Stats joueur pour ce match
        stats = fb('fixtures/players', {'fixture': fid})
        time.sleep(0.15)

        found = False
        for team in stats:
            for p in team.get('players', []):
                if p['player']['id'] == player_id:
                    ps = p.get('statistics', [{}])[0]
                    mins = ps.get('games',{}).get('minutes') or 0
                    if not mins or mins < 15:
                        _cache[fkey] = None
                        found = True; break
                    sot = ps.get('shots',{}).get('on') or 0
                    shots = ps.get('shots',{}).get('total') or 0
                    entry = {
                        'date':  next((f['fixture']['date'][:10] for f in fixtures if f['fixture']['id']==fid), ''),
                        'stat':  int(sot),
                        'shots': int(shots),
                        'mins':  int(mins),
                    }
                    _cache[fkey] = entry
                    games.append(entry)
                    found = True; break
            if found: break
        if not found:
            _cache[fkey] = None

    games.sort(key=lambda x: x['date'], reverse=True)
    _cache[key] = games
    return games



# ── Analyze (empirique comme MLB) ─────────────────────────────────────────────
def analyze_player(games, line):
    from stats import analyze
    if len(games) < 5: return None
    return analyze(games, line)

# ── EV block helper ───────────────────────────────────────────────────────────
def ev_block(prob, american, bankroll):
    ev  = calc_ev(prob, american)
    k   = calc_kelly(prob, american)
    bet = round(bankroll * k / 100, 2)
    return {'ev': ev, 'kelly': k, 'bet_size': bet}

# ── Route principale ──────────────────────────────────────────────────────────
@soccer_bp.route('/api/soccer/analyze', methods=['POST'])
def analyze_soccer():
    try:
        d        = request.get_json()
        name     = d.get('player','').strip()
        league   = d.get('league','EPL')
        market   = d.get('market','Shots On Goal')
        line     = float(d.get('line', 1.5))
        odds_ov  = d.get('odds_over')
        odds_un  = d.get('odds_under')
        bankroll = float(d.get('bankroll', 200))

        if not name:
            return jsonify({'status':'ERROR','message':'Joueur requis'}), 400

        lid = LEAGUES.get(league)
        if not lid:
            return jsonify({'status':'ERROR','message':f'Ligue inconnue: {league}'}), 400

        # 1. Trouve le joueur (1-2 calls)
        player_data = find_player(name, lid)
        if not player_data:
            return jsonify({'status':'ERROR',
                'message':f'"{name}" non trouvé en {league}. Vérifie le nom.'}), 404

        pid        = player_data['player']['id']
        pname_full = player_data['player']['name']
        team_id    = player_data.get('statistics',[{}])[0].get('team',{}).get('id')

        if not team_id:
            return jsonify({'status':'ERROR',
                'message':f'Équipe non trouvée pour {pname_full}'}), 404

        # 2. Récupère les 15 derniers matchs (~15 calls)
        games = get_player_fixtures(pid, team_id, lid, n=15)
        if len(games) < 5:
            return jsonify({'status':'ERROR',
                'message':f'Pas assez de matchs avec 15+ minutes pour {pname_full} '
                          f'({len(games)} trouvés, 5 minimum)'}), 404

        # 3. Analyse empirique (même modèle que MLB)
        a = analyze_player(games, line)
        if not a:
            return jsonify({'status':'ERROR','message':'Analyse impossible'}), 500

        # 4. EV + Kelly
        over_p  = a['hit_rate']
        under_p = 1 - over_p
        rec     = a['rec']

        result = {
            'status':    'SUCCESS',
            'player':    pname_full,
            'league':    league,
            'market':    market,
            'line':      line,
            'games_analyzed': a['n'],
            'avg_last_5':     a['l5'],
            'avg_overall':    a['mean'],
            'hit_rate_over':  round(over_p*100, 1),
            'hit_rate_under': round(under_p*100, 1),
            'recommendation': rec,
            'rec_hit_rate':   round(a['rec_prob']*100, 1),
            'grade':          a['grade'],
            'grade_color':    a['color'],
            'grade_label':    a['label'],
            'trend':          a['trend'],
            'pros':           a['quality']['pros'],
            'issues':         a['quality']['issues'],
            'recent_games':   a['recent'],
        }

        # EV si cotes fournies
        if odds_ov:
            result['over_ev']  = ev_block(over_p, float(odds_ov), bankroll)
        if odds_un:
            result['under_ev'] = ev_block(under_p, float(odds_un), bankroll)

        return jsonify(to_py(result))

    except Exception as e:
        import traceback
        return jsonify({'status':'ERROR','message':str(e),
                        'trace':traceback.format_exc()[:500]}), 500

@soccer_bp.route('/api/soccer/health', methods=['GET'])
def soccer_health():
    return jsonify({'status':'healthy','mode':'manual','fb_key':'ok' if FB_KEY else 'MISSING'})

# scan_soccer stub pour app.py
def scan_soccer(min_ev=3.0):
    return [], 0, 0
