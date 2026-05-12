# nba.py — Blueprint NBA (Pts, Reb, Ast)
# API : nba_api (gratuit, same as original)
from flask import Blueprint, jsonify, request
import requests as req, os, time, re
from collections import Counter
from datetime import datetime
from stats import to_py, analyze, build_opp, calc_ev

nba_bp   = Blueprint('nba', __name__)
ODDS_KEY = os.environ.get('ODDS_API_KEY','')
ODDS_BASE= 'https://api.the-odds-api.com/v4'
NBA_SEASON     = '2025-26'
MIN_MINUTES    = 15
_cache = {}

from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players as nba_players
from nba_api.stats.static import teams as nba_teams

def _norm(name): return re.sub(r'[^a-z ]','',name.lower().strip())
def _match(a,b):
    na,nb = _norm(a),_norm(b)
    if na == nb: return True
    pa,pb = na.split(),nb.split()
    return bool(pa and pb and pa[-1]==pb[-1] and pa[0][0]==pb[0][0])

# ── Player search ─────────────────────────────────────────────────────────────
def search_player(name):
    key = 'nba_'+_norm(name)
    if key in _cache: return _cache[key]
    for fn in [nba_players.find_players_by_full_name,
               nba_players.find_players_by_last_name,
               nba_players.find_players_by_first_name]:
        try:
            arg = name if fn == nba_players.find_players_by_full_name else name.split()[-1 if fn == nba_players.find_players_by_last_name else 0]
            res = fn(arg)
            if res:
                active = [p for p in res if p.get('is_active')]
                pid = (active or res)[0]['id']
                _cache[key] = pid; return pid
        except: continue
    return None

def _parse_min(s):
    if not s: return 0.0
    try:
        p = str(s).split(':')
        return float(p[0]) + (float(p[1])/60 if len(p)>1 else 0)
    except: return 0.0

def _opp_abbr(matchup):
    return str(matchup or '').strip().split()[-1].upper()

# ── Game log ──────────────────────────────────────────────────────────────────
def get_gamelog(player_id, stat_col, opp_abbr=None):
    nba_col = {'pts':'PTS','reb':'REB','ast':'AST'}.get(stat_col, stat_col.upper())
    key = f'nba_{player_id}_{stat_col}_{opp_abbr or "all"}'
    if key in _cache: return _cache[key]
    raw = []
    for season_type, weight in [('Playoffs',2),('Regular Season',1)]:
        try:
            gl = playergamelog.PlayerGameLog(
                player_id=player_id, season=NBA_SEASON,
                season_type_all_star=season_type, timeout=30)
            df = gl.get_data_frames()[0]
            if df.empty: continue
            for _,row in df.iterrows():
                mins = _parse_min(row.get('MIN',0))
                if mins < MIN_MINUTES: continue
                val = row.get(nba_col)
                if val is None: continue
                try: val_i = int(float(val))
                except: continue
                if val_i == 0: continue
                try:
                    from datetime import datetime as dt
                    gdate = dt.strptime(str(row.get('GAME_DATE','')), '%b %d, %Y').strftime('%Y-%m-%d')
                except: gdate = str(row.get('GAME_DATE',''))[:10]
                matchup = str(row.get('MATCHUP',''))
                game_opp = _opp_abbr(matchup)
                is_same  = bool(opp_abbr and game_opp == opp_abbr.upper())
                w        = 3 if is_same else weight
                raw.append({'date':gdate,'stat':val_i,'opp':game_opp,'weight':w,'is_same_opp':is_same})
            time.sleep(0.6)
        except Exception as e:
            print(f'nba_api error {player_id} {season_type}: {e}'); continue
    if not raw: return None
    raw.sort(key=lambda x:(x['date'],-x['weight']), reverse=True)
    seen,uniq = set(),[]
    for g in raw:
        if g['date'] not in seen: seen.add(g['date']); uniq.append(g)
    uniq = uniq[:30]
    if uniq: _cache[key] = uniq
    return uniq or None

# ── Opponent abbreviation ─────────────────────────────────────────────────────
def get_opp_abbr(player_name, home_team, away_team):
    try:
        from nba_api.stats.endpoints import commonplayerinfo
        pid = search_player(player_name)
        if not pid: return None
        info = commonplayerinfo.CommonPlayerInfo(player_id=pid, timeout=10)
        df   = info.get_data_frames()[0]
        time.sleep(0.3)
        abbr = str(df['TEAM_ABBREVIATION'].iloc[0]).strip().upper()
        if not abbr or abbr == 'NAN': return None
        abbr_kw = {
            'OKC':'thunder','DEN':'nuggets','MIN':'timberwolves','MEM':'grizzlies',
            'GSW':'warriors','HOU':'rockets','DAL':'mavericks','LAC':'clippers',
            'LAL':'lakers','PHX':'suns','SAC':'kings','POR':'blazers',
            'BOS':'celtics','NYK':'knicks','MIA':'heat','MIL':'bucks',
            'CLE':'cavaliers','IND':'pacers','PHI':'76ers','ORL':'magic',
            'ATL':'hawks','CHA':'hornets','CHI':'bulls','DET':'pistons',
            'BKN':'nets','TOR':'raptors','WAS':'wizards','NOP':'pelicans',
            'SAS':'spurs','UTA':'jazz',
        }
        kw  = abbr_kw.get(abbr, abbr.lower())
        opp = away_team if kw in home_team.lower() or abbr.lower() in home_team.lower() else home_team
        for t in nba_teams.get_teams():
            if _norm(t['full_name']) in _norm(opp) or _norm(t['nickname']) in _norm(opp):
                return t['abbreviation']
        return None
    except Exception as e:
        print(f'get_opp_abbr error: {e}'); return None

# ── Odds API ──────────────────────────────────────────────────────────────────
def get_odds_props(market):
    if not ODDS_KEY: return {},{}
    try:
        base = req.get(f'{ODDS_BASE}/sports/basketball_nba/odds', params={
            'apiKey':ODDS_KEY,'regions':'eu','markets':'h2h',
            'bookmakers':'pinnacle','oddsFormat':'american'}, timeout=10)
        if base.status_code != 200: return {},{}
        props,ev = {},{}
        for game in base.json()[:20]:
            gid = game['id']
            ev[gid] = {'home_team':game.get('home_team',''),'away_team':game.get('away_team',''),'time':game.get('commence_time','')}
            data = req.get(f'{ODDS_BASE}/sports/basketball_nba/events/{gid}/odds', params={
                'apiKey':ODDS_KEY,'regions':'eu','markets':market,
                'bookmakers':'pinnacle','oddsFormat':'american'}, timeout=10)
            if data.status_code != 200: continue
            for bk in data.json().get('bookmakers',[]):
                for mk in bk.get('markets',[]):
                    if mk['key'] != market: continue
                    for oc in mk.get('outcomes',[]):
                        player = oc.get('description','').strip()
                        point  = oc.get('point')
                        if not player or point is None: continue
                        if player not in props: props[player] = {'game_id':gid,'lines':[]}
                        props[player]['lines'].append({'book':bk['key'],'line':float(point),'price':int(oc.get('price',-110)),'type':oc.get('name','')})
            time.sleep(0.2)
        return props,ev
    except Exception as e:
        print(f'NBA odds error: {e}'); return {},{}

# ── Stat config ───────────────────────────────────────────────────────────────
STAT_MARKETS = {
    'nba_points':   ('pts', 'Pts', 'player_points'),
    'nba_rebounds': ('reb', 'Reb', 'player_rebounds'),
    'nba_assists':  ('ast', 'Ast', 'player_assists'),
}

# ── Scan ──────────────────────────────────────────────────────────────────────
def scan_nba(stat_filter=None, min_ev=3.0):
    stats = [stat_filter] if stat_filter else list(STAT_MARKETS.keys())
    opps,analyzed,ngames = [],0,0
    pid_cache  = {}
    opp_cache  = {}

    for st in stats:
        col,label,market = STAT_MARKETS[st]
        props,ev = get_odds_props(market)
        ngames = max(ngames, len(ev))
        if not props: continue

        for pname,pd in props.items():
            overs = [l for l in pd['lines'] if l['type']=='Over']
            if not overs: continue
            line = Counter([l['line'] for l in overs]).most_common(1)[0][0]
            best = min(overs, key=lambda x:abs(x['line']-line))
            gi   = ev.get(pd['game_id'], {'home_team':'','away_team':'','time':''})
            home_team = gi.get('home_team','')
            away_team = gi.get('away_team','')

            # Player ID
            pk = _norm(pname)
            if pk not in pid_cache:
                pid_cache[pk] = search_player(pname); time.sleep(0.2)
            pid = pid_cache[pk]
            if not pid: continue

            # Opponent abbreviation
            ok = f'{pk}_{_norm(home_team)}_{_norm(away_team)}'
            if ok not in opp_cache:
                opp_cache[ok] = get_opp_abbr(pname, home_team, away_team); time.sleep(0.15)
            opp_abbr = opp_cache.get(ok)
            if opp_abbr is None:
                print(f'TEAM FAIL {pname}: skip'); continue

            games = get_gamelog(pid, col, opp_abbr=opp_abbr)
            if not games or len(games) < 8: continue

            raw_mean = sum(g['stat'] for g in games) / len(games)
            if line > 0 and (raw_mean/line < 0.35 or raw_mean/line > 3.5):
                print(f'SANITY FAIL {pname}: raw={raw_mean:.1f} line={line}'); continue

            analyzed += 1
            a = analyze(games, line)
            if not a or a['grade'] == 'AVOID': continue
            ev_val = calc_ev(a['rec_prob'], best['price'])
            if ev_val is None or ev_val < min_ev: continue
            opps.append(build_opp(pname,'nba',st,label,line,best['book'].upper(),best['price'],gi,a))

    opps.sort(key=lambda x:({'A':0,'B':1,'C':2}.get(x['quality']['grade'],3)))
    return opps, analyzed, ngames

# ── Routes ────────────────────────────────────────────────────────────────────
def nba_opportunities():
    try:
        sport = request.args.get('sport','').lower()
        if sport != 'nba': return jsonify({'status':'SKIP'}),200
        sf   = request.args.get('stat_type') or None
        me   = float(request.args.get('min_edge',3))
        opps,analyzed,ngames = scan_nba(sf, me)
        if not ngames:
            return jsonify({'status':'INFO','sport':'nba',
                'message':'🏀 NBA: Aucun marché trouvé. Hors-saison ou playoffs terminés.',
                'opportunities':[]})
        return jsonify(to_py({'status':'SUCCESS','sport':'nba','version':'4.0',
            'total_games':ngames,'players_analyzed':analyzed,
            'opportunities_found':len(opps),'opportunities':opps,
            'scan_time':datetime.now().strftime('%Y-%m-%d %H:%M')}))
    except Exception as e:
        import traceback
        return jsonify({'status':'ERROR','message':str(e),'trace':traceback.format_exc()[:400]}),500

def nba_actual_result(player, stat_type, date_str):
    col = {'nba_points':'pts','nba_rebounds':'reb','nba_assists':'ast'}.get(stat_type,'pts')
    pid = search_player(player)
    if not pid: return jsonify({'status':'NOT_FOUND','message':f'Joueur non trouvé: {player}'}),404
    games = get_gamelog(pid, col)
    if not games: return jsonify({'status':'NOT_FOUND','message':f'Pas de données pour {player}'}),404
    g = next((x for x in games if x['date'][:10]==date_str[:10]),games[0]) if date_str else games[0]
    return jsonify({'status':'SUCCESS','player':player,'stat_type':stat_type,'date':g['date'],'actual_value':g['stat']})
