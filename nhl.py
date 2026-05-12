# nhl.py — Blueprint NHL (Shots On Goal)
# API : api-web.nhle.com (gratuit, aucune clé)
from flask import Blueprint, jsonify, request
import requests as req, os, time, re
from collections import Counter
from datetime import datetime
from stats import to_py, analyze, build_opp, calc_ev

nhl_bp    = Blueprint('nhl', __name__)
NHL_BASE  = 'https://api-web.nhle.com/v1'
NHL_SRCH  = 'https://search.d3.nhle.com/api/v1/search'
ODDS_KEY  = os.environ.get('ODDS_API_KEY','')
ODDS_BASE = 'https://api.the-odds-api.com/v4'
_cache = {}

def _norm(name): return re.sub(r'[^a-z ]','',name.lower().strip())
def _match(a,b):
    na,nb = _norm(a),_norm(b)
    if na == nb: return True
    pa,pb = na.split(),nb.split()
    return bool(pa and pb and pa[-1]==pb[-1] and pa[0][0]==pb[0][0])

def _safe(url, params=None):
    try:
        r = req.get(url, params=params, timeout=12)
        if r.status_code == 200: return r.json()
    except Exception as e: print(f'NHL req error: {e}')
    return None

def search_player(name):
    key = 'nhl_'+_norm(name)
    if key in _cache: return _cache[key]
    last = name.split()[-1]
    data = _safe(NHL_SRCH, params={'q':last,'culture':'en-us','isActive':'true'})
    for r in (data if isinstance(data,list) else []):
        if _match(name, r.get('name','')):
            pid = r.get('playerId')
            if pid: _cache[key]=pid; return pid
    return None

def get_gamelog(player_id):
    key = f'nhl_{player_id}'
    if key in _cache: return _cache[key]
    games = []
    for season, gt in [('20252026',3),('20252026',2),('20242025',2)]:
        data = _safe(f'{NHL_BASE}/player/{player_id}/game-log/{season}/{gt}')
        for g in (data or {}).get('gameLog',[]):
            shots = g.get('shots') or g.get('shotsOnGoal')
            if shots is None: continue
            games.append({'date':g.get('gameDate',''),'stat':int(shots)})
    games.sort(key=lambda x:x['date'], reverse=True)
    seen,uniq = set(),[]
    for g in games:
        d=g['date'][:10]
        if d not in seen: seen.add(d); uniq.append(g)
    if uniq: _cache[key] = uniq
    return uniq or None

def get_odds_props():
    if not ODDS_KEY: return {},{}
    for sport_key in ['icehockey_nhl','icehockey_nhl_championship']:
        base = _safe(f'{ODDS_BASE}/sports/{sport_key}/odds', params={
            'apiKey':ODDS_KEY,'regions':'us','markets':'h2h','oddsFormat':'american'})
        if not base: continue
        props,ev = {},{}
        for game in base[:15]:
            gid = game['id']
            ev[gid] = {'home_team':game.get('home_team',''),'away_team':game.get('away_team',''),'time':game.get('commence_time','')}
            data = _safe(f'{ODDS_BASE}/sports/{sport_key}/events/{gid}/odds', params={
                'apiKey':ODDS_KEY,'regions':'us','markets':'player_shots_on_goal','oddsFormat':'american'})
            for bk in (data or {}).get('bookmakers',[]):
                for mk in bk.get('markets',[]):
                    if mk['key']!='player_shots_on_goal': continue
                    for oc in mk.get('outcomes',[]):
                        player = oc.get('description','').strip()
                        point  = oc.get('point')
                        if not player or point is None: continue
                        if player not in props: props[player]={'game_id':gid,'lines':[]}
                        props[player]['lines'].append({'book':bk['key'],'line':float(point),'price':int(oc.get('price',-110)),'type':oc.get('name','')})
            time.sleep(0.25)
        if props: return props,ev
    return {},{}

def scan_nhl(min_ev=3.0):
    props,ev = get_odds_props()
    if not props:
        return [{'_info':True,'message':'🏒 NHL: Aucun marché SOG trouvé.'}],0,0
    opps,analyzed = [],0
    for pname,pd in props.items():
        overs = [l for l in pd['lines'] if l['type']=='Over']
        if not overs: continue
        line = Counter([l['line'] for l in overs]).most_common(1)[0][0]
        best = min(overs,key=lambda x:abs(x['line']-line))
        pid  = search_player(pname)
        if not pid: continue
        games = get_gamelog(pid)
        if not games or len(games) < 7: continue
        analyzed += 1
        a = analyze(games, line)
        if not a or a['grade']=='AVOID': continue
        ev_val = calc_ev(a['rec_prob'], best['price'])
        if ev_val is None or ev_val < min_ev: continue
        gi = ev.get(pd['game_id'],{'home_team':'','away_team':'','time':''})
        opps.append(build_opp(pname,'nhl','skater_shots','SOG',line,best['book'].upper(),best['price'],gi,a))
    opps.sort(key=lambda x:({'A':0,'B':1,'C':2}.get(x['quality']['grade'],3)))
    return opps, analyzed, len(ev)

@nhl_bp.route('/api/daily-opportunities', methods=['GET'])
def nhl_opportunities():
    try:
        sport = request.args.get('sport','').lower()
        if sport != 'nhl': return jsonify({'status':'SKIP'}),200
        me   = float(request.args.get('min_edge',3))
        opps,analyzed,ngames = scan_nhl(me)
        if opps and isinstance(opps[0],dict) and opps[0].get('_info'):
            return jsonify({'status':'INFO','sport':'nhl','message':opps[0]['message'],'opportunities':[]})
        return jsonify(to_py({'status':'SUCCESS','sport':'nhl','version':'4.0',
            'total_games':ngames,'players_analyzed':analyzed,
            'opportunities_found':len(opps),'opportunities':opps,
            'scan_time':datetime.now().strftime('%Y-%m-%d %H:%M')}))
    except Exception as e:
        import traceback; return jsonify({'status':'ERROR','message':str(e),'trace':traceback.format_exc()[:400]}),500

def nhl_actual_result(player, stat_type, date_str):
    pid = search_player(player)
    if not pid: return jsonify({'status':'NOT_FOUND','message':f'Joueur non trouvé: {player}'}),404
    games = get_gamelog(pid)
    if not games: return jsonify({'status':'NOT_FOUND','message':f'Pas de données pour {player}'}),404
    g = next((x for x in games if x['date'][:10]==date_str[:10]),games[0]) if date_str else games[0]
    return jsonify({'status':'SUCCESS','player':player,'stat_type':stat_type,'date':g['date'],'actual_value':g['stat']})
