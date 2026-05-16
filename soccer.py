# soccer.py — Soccer Model : Dixon-Coles + Poisson props
# Données : API-Football (FOOTBALL_API_KEY, déjà sur Render)
# Coût : 0$ — free tier 100 calls/day amplement suffisant
# Intégration app.py :
#   En haut  → from soccer import soccer_bp
#   Avant __main__ → app.register_blueprint(soccer_bp)

from flask import Blueprint, request, jsonify
import os, math, time
import requests as req

soccer_bp = Blueprint('soccer', __name__)
FB_KEY   = os.environ.get('FOOTBALL_API_KEY', '')
FB_BASE  = 'https://v3.football.api-sports.io'

# League name → (api_football_id, season)
LEAGUE_MAP = {
    'La Liga': 140, 'EPL': 39, 'Serie A': 135, 'Bundesliga': 78,
    'Ligue 1': 61, 'Champions League': 2, 'Belgium Pro': 144,
    'Denmark Superliga': 119, 'Italy Serie B': 136, 'Greece SL': 197,
    'Argentina LPF': 128, 'Saudi Pro': 307, 'J.League': 98,
    'MLS': 253, 'Portugal Primeira': 94, 'Eng Championship': 40,
}
# Historical league average goals per game (stable across seasons)
LEAGUE_AVG_GOALS = {
    140: 2.50, 39: 2.60, 135: 2.70, 78: 3.10, 61: 2.60,
    2: 2.80, 144: 2.90, 119: 2.80, 136: 2.50, 197: 2.60,
    128: 2.40, 307: 2.80, 98: 2.60, 253: 2.90, 94: 2.60, 40: 2.40,
}
SEASON = 2024
HOME_ADV = 1.15  # home advantage factor

TEAM_CACHE   = {}
PLAYER_CACHE = {}

# ── API helper ────────────────────────────────────────────────────────────────
def fb(endpoint, params=None):
    if not FB_KEY:
        raise ValueError('FOOTBALL_API_KEY manquant sur Render')
    params = params or {}
    params['season'] = params.get('season', SEASON)
    try:
        r = req.get(f'{FB_BASE}/{endpoint.lstrip("/")}',
                    params=params,
                    headers={'x-apisports-key': FB_KEY},
                    timeout=12)
        r.raise_for_status()
        data = r.json()
        remaining = r.headers.get('x-ratelimit-requests-remaining', '?')
        print(f"  FB /{endpoint}: {len(data.get('response',[]))} results | remaining={remaining}")
        return data.get('response', [])
    except Exception as e:
        print(f"  FB error /{endpoint}: {e}")
        return []

# ── Team helpers ──────────────────────────────────────────────────────────────
def get_team_id(name, league_id):
    key = f'{name}_{league_id}'
    if key in TEAM_CACHE: return TEAM_CACHE[key]
    results = fb('teams', {'name': name, 'league': league_id})
    if not results:
        # Try without league filter
        results = fb('teams', {'name': name})
    if not results: return None
    tid = results[0]['team']['id']
    TEAM_CACHE[key] = tid
    return tid

def get_team_stats(team_id, league_id):
    """Returns (goals_scored_avg, goals_conceded_avg) per game."""
    key = f'stats_{team_id}_{league_id}'
    if key in TEAM_CACHE: return TEAM_CACHE[key]
    results = fb('teams/statistics', {'team': team_id, 'league': league_id})
    if not results: return None, None
    s = results  # teams/statistics returns a single object, not a list
    # If response is a list of 1
    if isinstance(results, list): s = results[0] if results else {}
    try:
        goals_for  = s.get('goals', {}).get('for', {})
        goals_ag   = s.get('goals', {}).get('against', {})
        # Prefer 'average' total, fallback to computing from total/played
        avg_for = goals_for.get('average', {}).get('total')
        avg_ag  = goals_ag.get('average', {}).get('total')
        if avg_for is None:
            total_for    = goals_for.get('total', {}).get('total', 0) or 0
            fixtures_pl  = s.get('fixtures', {}).get('played', {}).get('total', 1) or 1
            avg_for = total_for / fixtures_pl
        else:
            avg_for = float(avg_for)
        if avg_ag is None:
            total_ag    = goals_ag.get('total', {}).get('total', 0) or 0
            fixtures_pl = s.get('fixtures', {}).get('played', {}).get('total', 1) or 1
            avg_ag = total_ag / fixtures_pl
        else:
            avg_ag = float(avg_ag)
        TEAM_CACHE[key] = (round(avg_for, 3), round(avg_ag, 3))
        return TEAM_CACHE[key]
    except Exception as e:
        print(f"  get_team_stats parse error: {e}")
        return None, None

def get_recent_form(team_id, league_id, n=5):
    """Returns form string like WWDLW (most recent first)."""
    results = fb('fixtures', {'team': team_id, 'league': league_id, 'last': n})
    if not results: return '-----'
    form = []
    for fix in sorted(results, key=lambda x: x.get('fixture',{}).get('date',''), reverse=True):
        teams  = fix.get('teams', {})
        goals  = fix.get('goals', {})
        home   = teams.get('home', {})
        is_home = home.get('id') == team_id
        gf = goals.get('home') if is_home else goals.get('away')
        ga = goals.get('away') if is_home else goals.get('home')
        if gf is None or ga is None: continue
        form.append('W' if gf > ga else ('D' if gf == ga else 'L'))
    return ''.join(form) if form else '-----'

# ── Dixon-Coles ───────────────────────────────────────────────────────────────
def _pmf(k, lam):
    if lam <= 0 or k < 0: return 0.0
    try:
        log_p = -lam + k * math.log(lam)
        for i in range(1, k + 1): log_p -= math.log(i)
        return math.exp(log_p)
    except: return 0.0

def _tau(i, j, lh, la, rho=-0.13):
    if i == 0 and j == 0: return max(0.01, 1 - lh * la * rho)
    if i == 1 and j == 0: return 1 + la * rho
    if i == 0 and j == 1: return 1 + lh * rho
    if i == 1 and j == 1: return 1 - rho
    return 1.0

def dixon_coles(lh, la):
    lh = max(0.3, min(4.5, float(lh)))
    la = max(0.3, min(4.5, float(la)))
    ph = pd = pa = btts = o15 = o25 = o35 = 0.0
    for i in range(9):
        for j in range(9):
            p = _pmf(i, lh) * _pmf(j, la) * _tau(i, j, lh, la)
            if i > j:    ph += p
            elif i == j: pd += p
            else:        pa += p
            if i > 0 and j > 0: btts += p
            if i + j > 1: o15 += p
            if i + j > 2: o25 += p
            if i + j > 3: o35 += p
    return {
        'pHome': round(ph, 4),      'pDraw': round(pd, 4),      'pAway': round(pa, 4),
        'pBTTS': round(btts, 4),    'pBTTSNo': round(1-btts, 4),
        'pOver15': round(o15, 4),   'pUnder15': round(1-o15, 4),
        'pOver25': round(o25, 4),   'pUnder25': round(1-o25, 4),
        'pOver35': round(o35, 4),   'pUnder35': round(1-o35, 4),
        'pHomeOrDraw': round(ph+pd, 4), 'pAwayOrDraw': round(pa+pd, 4),
    }

# ── Player helpers ────────────────────────────────────────────────────────────
MARKET_STAT = {
    'Shots On Goal': ('shots', 'on'),
    'Shots':         ('shots', 'total'),
    'Goals':         ('goals', 'total'),
    'Assists':       ('goals', 'assists'),
    'Cards':         ('cards', 'yellow'),
}

def get_player_avg(name, league_id, market):
    key = f'{name}_{league_id}_{market}'
    if key in PLAYER_CACHE: return PLAYER_CACHE[key]
    results = fb('players', {'search': name, 'league': league_id})
    if not results:
        results = fb('players', {'search': name.split()[-1], 'league': league_id})
    if not results: return None, None, None
    # Find best name match
    player_data = None
    for r in results:
        pname = r.get('player', {}).get('name', '').lower()
        if name.lower().split()[-1] in pname:
            player_data = r; break
    if not player_data: player_data = results[0]
    player_id   = player_data['player']['id']
    player_name = player_data['player']['name']
    stats = player_data.get('statistics', [{}])[0] if player_data.get('statistics') else {}
    stat_keys = MARKET_STAT.get(market, ('shots', 'on'))
    cat  = stats.get(stat_keys[0], {})
    val  = cat.get(stat_keys[1]) if cat else None
    apps = stats.get('games', {}).get('appearences') or stats.get('games', {}).get('appearances')
    if val is None or not apps or apps == 0:
        PLAYER_CACHE[key] = (player_name, None, apps)
        return player_name, None, apps
    avg = round(float(val) / float(apps), 3)
    PLAYER_CACHE[key] = (player_name, avg, int(apps))
    return player_name, avg, int(apps)

def poisson_over(avg, line):
    """P(X >= ceil(line)) using Poisson distribution."""
    if not avg or avg <= 0: return 0.5
    threshold = math.ceil(line)  # Over 1.5 means >= 2
    p_under = sum(_pmf(k, avg) for k in range(threshold))
    return round(max(0.02, min(0.98, 1 - p_under)), 4)

# ── EV + Kelly ────────────────────────────────────────────────────────────────
def calc_ev(prob, american):
    try:
        a   = float(american)
        dec = a/100+1 if a >= 0 else 100/abs(a)+1
        return round((prob * dec - 1) * 100, 2)
    except: return None

def calc_kelly(prob, american, fraction=0.25):
    try:
        a   = float(american)
        dec = a/100+1 if a >= 0 else 100/abs(a)+1
        b   = dec - 1; q = 1 - prob
        k   = (prob * b - q) / b
        return round(max(0, k * fraction) * 100, 2)
    except: return 0.0

# ── Routes ────────────────────────────────────────────────────────────────────
@soccer_bp.route('/api/soccer/game', methods=['POST'])
def soccer_game():
    try:
        d      = request.get_json()
        home   = d.get('home', '').strip()
        away   = d.get('away', '').strip()
        league = d.get('league', 'La Liga').strip()
        if not home or not away:
            return jsonify({'status': 'ERROR', 'message': 'home et away requis'}), 400
        lid = LEAGUE_MAP.get(league)
        if not lid:
            return jsonify({'status': 'ERROR', 'message': f'Ligue inconnue: {league}'}), 400
        league_avg = LEAGUE_AVG_GOALS.get(lid, 2.6)

        # Fetch team IDs
        home_id = get_team_id(home, lid)
        time.sleep(0.3)
        away_id = get_team_id(away, lid)
        time.sleep(0.3)
        if not home_id or not away_id:
            return jsonify({'status': 'ERROR',
                'message': f'Équipe non trouvée: {"home" if not home_id else "away"}. Vérifie l\'orthographe.'}), 404

        # Fetch team stats
        home_scored, home_conceded = get_team_stats(home_id, lid)
        time.sleep(0.3)
        away_scored, away_conceded = get_team_stats(away_id, lid)
        time.sleep(0.3)

        # Fallback to league average if stats missing
        home_scored   = home_scored   or league_avg / 2
        home_conceded = home_conceded or league_avg / 2
        away_scored   = away_scored   or league_avg / 2
        away_conceded = away_conceded or league_avg / 2

        # Dixon-Coles expected goals
        # λ_home = attack_home * defense_away * home_advantage * league_avg/2
        # Strengths are relative to league average (each side ~league_avg/2)
        half_avg      = league_avg / 2
        attack_home   = home_scored   / half_avg
        defense_home  = home_conceded / half_avg
        attack_away   = away_scored   / half_avg
        defense_away  = away_conceded / half_avg

        xg_home = round(attack_home * defense_away * HOME_ADV * half_avg, 3)
        xg_away = round(attack_away * defense_home * (1/HOME_ADV) * half_avg, 3)
        probs   = dixon_coles(xg_home, xg_away)

        # Recent form (optional, 2 more API calls)
        form_home = get_recent_form(home_id, lid)
        time.sleep(0.2)
        form_away = get_recent_form(away_id, lid)

        confidence = 'high' if home_scored and away_scored else 'low'

        return jsonify({
            'status': 'SUCCESS',
            'home': home, 'away': away, 'league': league,
            'xgHome': xg_home, 'xgAway': xg_away,
            'homeStats': {'scored': home_scored, 'conceded': home_conceded},
            'awayStats': {'scored': away_scored, 'conceded': away_conceded},
            'formHome': form_home, 'formAway': form_away,
            'confidence': confidence, 'probs': probs,
            'model': 'Dixon-Coles',
            'keyFactors': [
                f'{home}: {home_scored:.2f} buts/match, {home_conceded:.2f} encaissés',
                f'{away}: {away_scored:.2f} buts/match, {away_conceded:.2f} encaissés',
                f'xG calculé avec avantage domicile x{HOME_ADV}'
            ]
        })
    except Exception as e:
        import traceback
        return jsonify({'status': 'ERROR', 'message': str(e),
                        'trace': traceback.format_exc()[:400]}), 500

@soccer_bp.route('/api/soccer/player', methods=['POST'])
def soccer_player():
    try:
        d      = request.get_json()
        player = d.get('player', '').strip()
        market = d.get('market', 'Shots On Goal').strip()
        line   = float(d.get('line', 1.5))
        league = d.get('league', 'La Liga').strip()
        if not player:
            return jsonify({'status': 'ERROR', 'message': 'player requis'}), 400
        lid = LEAGUE_MAP.get(league)
        if not lid:
            return jsonify({'status': 'ERROR', 'message': f'Ligue inconnue: {league}'}), 400

        player_name, avg, apps = get_player_avg(player, lid, market)

        if avg is None:
            return jsonify({'status': 'ERROR',
                'message': f'Stat "{market}" non trouvée pour {player_name or player}. '
                           f'Essaie un autre marché ou vérifie le nom.'}), 404

        # Poisson model
        over_p  = poisson_over(avg, line)
        under_p = round(1 - over_p, 4)

        # Confidence based on appearances
        if apps and apps >= 20: conf = 'HIGH'
        elif apps and apps >= 10: conf = 'MEDIUM'
        else: conf = 'LOW'

        # Expected hit count
        hit_count_est = round(over_p * (apps or 0))

        return jsonify({
            'status': 'SUCCESS',
            'player': player_name or player,
            'market': market, 'line': line,
            'seasonAvg': avg, 'appearances': apps,
            'hitCountEst': hit_count_est,
            'overProb': over_p, 'underProb': under_p,
            'confidence': conf,
            'model': 'Poisson',
            'note': f'Poisson(λ={avg}) sur {apps} apparitions — P(>={math.ceil(line)})={round(over_p*100,1)}%'
        })
    except Exception as e:
        import traceback
        return jsonify({'status': 'ERROR', 'message': str(e),
                        'trace': traceback.format_exc()[:400]}), 500

@soccer_bp.route('/api/soccer/ev', methods=['POST'])
def soccer_ev():
    try:
        d   = request.get_json()
        prob = float(d['prob'])
        odds = d['odds']
        bk   = float(d.get('bankroll', 200))
        e    = calc_ev(prob, odds)
        k    = calc_kelly(prob, odds)
        a    = float(odds)
        ip   = abs(a)/(abs(a)+100) if a < 0 else 100/(a+100)
        return jsonify({
            'status': 'SUCCESS',
            'ev': e, 'kelly_pct': k,
            'bet_size': round(bk * k / 100, 2),
            'implied_prob': round(ip, 4),
            'should_bet': e is not None and e > 2.0
        })
    except Exception as e:
        return jsonify({'status': 'ERROR', 'message': str(e)}), 500

@soccer_bp.route('/api/soccer/health', methods=['GET'])
def soccer_health():
    key_ok = bool(FB_KEY)
    return jsonify({
        'status': 'healthy',
        'model': 'Dixon-Coles + Poisson',
        'data_source': 'API-Football',
        'football_api_key': 'configured' if key_ok else 'MISSING — ajoute FOOTBALL_API_KEY sur Render',
        'cost': '$0 — free tier 100 calls/day',
        'calls_per_game_analysis': '~8',
        'calls_per_player_analysis': '~2',
    })

