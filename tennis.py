# tennis.py — Blueprint Tennis (Aces)
# API : Jeff Sackmann GitHub ATP CSVs (gratuit, aucune clé)
from flask import Blueprint, jsonify, request
import requests as req, os, io, csv, re, time
from collections import Counter
from datetime import datetime
from stats import to_py, analyze, build_opp, calc_ev

tennis_bp  = Blueprint('tennis', __name__)
SACKMANN   = 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master'
ODDS_KEY   = os.environ.get('ODDS_API_KEY','')
ODDS_BASE  = 'https://api.the-odds-api.com/v4'
_cache = {}

def _norm(name): return re.sub(r'[^a-z ]','',name.lower().strip())
def _match(a,b):
    na,nb = _norm(a),_norm(b)
    if na == nb: return True
    pa,pb = na.split(),nb.split()
    return bool(pa and pb and pa[-1]==pb[-1] and pa[0][0]==pb[0][0])

def get_aces(player_name, surface=None):
    key = f'tennis_{_norm(player_name)}_{surface or "all"}'
    if key in _cache: return _cache[key]
    games = []
    for year in [2026,2025,2024]:
        try:
            r = req.get(f'{SACKMANN}/atp_matches_{year}.csv', timeout=15)
            if r.status_code != 200: continue
            for row in csv.DictReader(io.StringIO(r.text)):
                surf = row.get('surface','')
                if surface and surf.lower() != surface.lower(): continue
                date = row.get('tourney_date','')
                for role,ace_col in [('winner_name','w_ace'),('loser_name','l_ace')]:
                    pname = row.get(role,'')
                    if pname and _match(player_name, pname):
                        val = (row.get(ace_col) or '').strip()
                        if val:
                            try: games.append({'date':str(date),'stat':int(float(val)),'surface':surf})
                            except: pass
        except Exception as e: print(f'Tennis CSV {year}: {e}')
    games.sort(key=lambda x:x['date'], reverse=True)
    if games: _cache[key] = games
    return games or None

def get_odds_props():
    if not ODDS_KEY: return {},{}
    for sport_key in ['tennis_atp','tennis_atp_french_open','tennis_atp_wimbledon',
                      'tennis_atp_us_open','tennis_atp_australian_open']:
        try:
            base = req.get(f'{ODDS_BASE}/sports/{sport_key}/odds', params={
                'apiKey':ODDS_KEY,'regions':'us','markets':'h2h','oddsFormat':'american'}, timeout=10)
            if base.status_code != 200: continue
            props,ev = {},{}
            for game in base.json()[:20]:
                gid = game['id']
                ev[gid] = {'home_team':game.get('home_team',''),'away_team':game.get('away_team',''),'time':game.get('commence_time','')}
                data = req.get(f'{ODDS_BASE}/sports/{sport_key}/events/{gid}/odds', params={
                    'apiKey':ODDS_KEY,'regions':'us','markets':'player_aces','oddsFormat':'american'}, timeout=10)
                if data.status_code != 200: continue
                for bk in data.json().get('bookmakers',[]):
                    for mk in bk.get('markets',[]):
                        if mk['key']!='player_aces': continue
                        for oc in mk.get('outcomes',[]):
                            player = oc.get('description','').strip()
                            point  = oc.get('point')
                            if not player or point is None: continue
                            if player not in props: props[player]={'game_id':gid,'lines':[]}
                            props[player]['lines'].append({'book':bk['key'],'line':float(point),'price':int(oc.get('price',-110)),'type':oc.get('name','')})
                time.sleep(0.2)
            if props: return props,ev
        except Exception as e: print(f'Tennis odds {sport_key}: {e}')
    return {},{}

def scan_tennis(min_ev=3.0):
    props,ev = get_odds_props()
    if not props:
        return [{'_info':True,'message':'🎾 Tennis: Aucun marché player_aces actif en ce moment.'}],0,0
    opps,analyzed = [],0
    for pname,pd in props.items():
        overs = [l for l in pd['lines'] if l['type']=='Over']
        if not overs: continue
        line = Counter([l['line'] for l in overs]).most_common(1)[0][0]
        best = min(overs,key=lambda x:abs(x['line']-line))
        games = get_aces(pname)
        if not games or len(games) < 7: continue
        analyzed += 1
        a = analyze(games, line)
        if not a or a['grade']=='AVOID': continue
        ev_val = calc_ev(a['rec_prob'], best['price'])
        if ev_val is None or ev_val < min_ev: continue
        gi = ev.get(pd['game_id'],{'home_team':'','away_team':'','time':''})
        opps.append(build_opp(pname,'tennis','tennis_aces','ACE',line,best['book'].upper(),best['price'],gi,a))
    opps.sort(key=lambda x:({'A':0,'B':1,'C':2}.get(x['quality']['grade'],3)))
    return opps, analyzed, len(props)

def tennis_opportunities():
    try:
        sport = request.args.get('sport','').lower()
        if sport != 'tennis': return jsonify({'status':'SKIP'}),200
        me   = float(request.args.get('min_edge',3))
        opps,analyzed,ngames = scan_tennis(me)
        if opps and isinstance(opps[0],dict) and opps[0].get('_info'):
            return jsonify({'status':'INFO','sport':'tennis','message':opps[0]['message'],'opportunities':[]})
        return jsonify(to_py({'status':'SUCCESS','sport':'tennis','version':'4.0',
            'total_games':ngames,'players_analyzed':analyzed,
            'opportunities_found':len(opps),'opportunities':opps,
            'scan_time':datetime.now().strftime('%Y-%m-%d %H:%M')}))
    except Exception as e:
        import traceback; return jsonify({'status':'ERROR','message':str(e),'trace':traceback.format_exc()[:400]}),500

def tennis_actual_result(player, stat_type, date_str):
    games = get_aces(player)
    if not games: return jsonify({'status':'NOT_FOUND','message':f'Pas de données pour {player}'}),404
    g = next((x for x in games if x['date'][:10]==date_str[:10]),games[0]) if date_str else games[0]
    return jsonify({'status':'SUCCESS','player':player,'stat_type':stat_type,'date':g['date'],'actual_value':g['stat']})
