# mlb.py — Blueprint MLB (pitcher K, batter TB)
# API : statsapi.mlb.com (gratuit, aucune clé)
from flask import Blueprint, jsonify, request
import requests as req, os, time, re
from collections import Counter
from datetime import datetime
from stats import to_py, analyze, build_opp, calc_ev

mlb_bp = Blueprint('mlb', __name__)
MLB      = 'https://statsapi.mlb.com/api/v1'
ODDS_KEY = os.environ.get('ODDS_API_KEY','')
ODDS_BASE= 'https://api.the-odds-api.com/v4'
MLB_SEASON = 2026
K_LEAGUE_AVG = 0.225
_cache = {}

# ── Helpers ───────────────────────────────────────────────────────────────────
def _safe(url, params=None, timeout=12):
    try:
        r = req.get(url, params=params, timeout=timeout)
        if r.status_code == 200: return r.json()
    except Exception as e: print(f'MLB req error: {e}')
    return None

def _norm(name): return re.sub(r'[^a-z ]','',name.lower().strip())
def _match(a,b):
    na,nb = _norm(a),_norm(b)
    if na == nb: return True
    pa,pb = na.split(),nb.split()
    return bool(pa and pb and pa[-1]==pb[-1] and pa[0][0]==pb[0][0])

def get_pitchers():
    today = datetime.now().strftime('%Y-%m-%d')
    if today in _cache: return _cache[today]
    data = _safe(f'{MLB}/schedule', params={
        'sportId':1,'date':today,'hydrate':'probablePitcher',
        'gameType':'R','season':MLB_SEASON})
    pitchers = []
    for db in (data or {}).get('dates',[]):
        for game in db.get('games',[]):
            home = game['teams']['home']['team']['name']
            away = game['teams']['away']['team']['name']
            gt   = game.get('gameDate','')
            for side in ['home','away']:
                pp = game['teams'][side].get('probablePitcher',{})
                if pp and pp.get('id'):
                    pitchers.append({'id':pp['id'],'name':pp.get('fullName',''),
                        'home_team':home,'away_team':away,
                        'is_home':side=='home',
                        'opponent':away if side=='home' else home,
                        'game_time':gt})
    _cache[today] = pitchers; return pitchers

def get_gamelog(pid, stat_type):
    cfg = {'pitcher_strikeouts':('pitching','strikeOuts'),
           'batter_total_bases':('hitting','totalBases')}
    group, col = cfg.get(stat_type, ('pitching','strikeOuts'))
    key = f'{pid}_{stat_type}'
    if key in _cache: return _cache[key]
    games = []
    for season in [MLB_SEASON, MLB_SEASON-1]:
        data = _safe(f'{MLB}/people/{pid}/stats', params={
            'stats':'gameLog','season':season,'group':group,'gameType':'R'})
        for s in (data or {}).get('stats',[{}])[0].get('splits',[]):
            val = s.get('stat',{}).get(col)
            if val is None: continue
            entry = {'date':s.get('date',''),'stat':int(val)}
            if group == 'pitching':
                ip_str = str(s.get('stat',{}).get('inningsPitched','0') or '0')
                try: p = ip_str.split('.'); entry['ip'] = int(p[0])+(int(p[1])/3 if len(p)>1 and p[1] else 0)
                except: entry['ip'] = 0.0
            games.append(entry)
    games.sort(key=lambda x:x['date'], reverse=True)
    seen,uniq = set(),[]
    for g in games:
        d = g['date'][:10]
        if d not in seen: seen.add(d); uniq.append(g)
    _cache[key] = uniq; return uniq

def search_batter(name):
    key = 'bat_'+_norm(name)
    if key in _cache: return _cache[key]
    last = name.split()[-1] if ' ' in name else name
    data = _safe(f'{MLB}/people/search', params={'names':last,'sportIds':1})
    for p in (data or {}).get('people',[]):
        if _match(name, p.get('fullName','')):
            _cache[key] = p['id']; return p['id']
    return None

def opp_k_pct(team):
    key = 'kpct_'+_norm(team)
    if key in _cache: return _cache[key]
    td = _safe(f'{MLB}/teams', params={'sportId':1,'season':MLB_SEASON})
    tid = next((t['id'] for t in (td or {}).get('teams',[]) if _match(team,t.get('name',''))), None)
    if not tid: return None
    data = _safe(f'{MLB}/teams/{tid}/stats', params={'stats':'season','group':'hitting','season':MLB_SEASON,'gameType':'R'})
    sp = (data or {}).get('stats',[{}])[0].get('splits',[])
    if not sp: return None
    st = sp[0].get('stat',{}); pa = int(st.get('atBats',1))+int(st.get('baseOnBalls',0))+int(st.get('hitByPitch',0))
    if not pa: return None
    kp = int(st.get('strikeOuts',0)) / pa
    _cache[key] = kp; return kp

def get_odds_props(market):
    if not ODDS_KEY: return {},{}
    base = _safe(f'{ODDS_BASE}/sports/baseball_mlb/odds', params={
        'apiKey':ODDS_KEY,'regions':'us','markets':'h2h','oddsFormat':'american'})
    props,ev = {},{}
    for game in (base or [])[:15]:
        gid = game['id']
        ev[gid] = {'home_team':game.get('home_team',''),'away_team':game.get('away_team',''),'time':game.get('commence_time','')}
        data = _safe(f'{ODDS_BASE}/sports/baseball_mlb/events/{gid}/odds', params={
            'apiKey':ODDS_KEY,'regions':'us','markets':market,'oddsFormat':'american'})
        for bk in (data or {}).get('bookmakers',[]):
            for mk in bk.get('markets',[]):
                if mk['key'] != market: continue
                for oc in mk.get('outcomes',[]):
                    player = oc.get('description','').strip()
                    point  = oc.get('point')
                    if not player or point is None: continue
                    if player not in props: props[player] = {'game_id':gid,'lines':[]}
                    props[player]['lines'].append({'book':bk['key'],'line':float(point),'price':int(oc.get('price',-110)),'type':oc.get('name','')})
        time.sleep(0.25)
    return props, ev

# ── Scan ──────────────────────────────────────────────────────────────────────
def scan_mlb(stat_filter=None, min_ev=3.0):
    stats = [stat_filter] if stat_filter else ['pitcher_strikeouts','batter_total_bases']
    pitchers = get_pitchers()
    opps,analyzed = [],0
    for st in stats:
        market_key = 'pitcher_strikeouts' if st=='pitcher_strikeouts' else 'batter_total_bases'
        props,ev = get_odds_props(market_key)
        is_p = st == 'pitcher_strikeouts'
        label = 'K' if st=='pitcher_strikeouts' else 'TB'
        for pname, pd in props.items():
            overs = [l for l in pd['lines'] if l['type']=='Over']
            if not overs: continue
            line  = Counter([l['line'] for l in overs]).most_common(1)[0][0]
            best  = min(overs, key=lambda x:abs(x['line']-line))
            if is_p:
                pitcher = next((p for p in pitchers if _match(pname,p['name'])),None)
                if not pitcher: continue
                pid = pitcher['id']; opp = pitcher['opponent']; ih = pitcher['is_home']
                gi  = {'home_team':pitcher['home_team'],'away_team':pitcher['away_team'],'time':pitcher['game_time']}
                games = get_gamelog(pid, st)
                if not games or len(games) < 6: continue
                # Qualifier: IP >= 3 for K
                games_q = [g for g in games if g.get('ip',99) >= 3.0]
                if len(games_q) < 5: games_q = games
                analyzed += 1
                a = analyze(games_q, line)
                if not a or a['grade'] == 'AVOID': continue
                ev_val = calc_ev(a['rec_prob'], best['price'])
                if ev_val is None or ev_val < min_ev: continue
                opps.append(build_opp(pname, 'mlb', st, label, line, best['book'].upper(), best['price'], ev.get(pd['game_id'],gi), a))
            else:
                pid = search_batter(pname)
                if not pid: continue
                games = get_gamelog(pid, st)
                if not games or len(games) < 8: continue
                if sum(g['stat'] for g in games[:10])/min(10,len(games)) < 1.0: continue
                analyzed += 1; time.sleep(0.1)
                a = analyze(games, line)
                if not a or a['grade'] == 'AVOID': continue
                ev_val = calc_ev(a['rec_prob'], best['price'])
                if ev_val is None or ev_val < min_ev: continue
                gi = ev.get(pd['game_id'],{'home_team':'','away_team':'','time':''})
                opps.append(build_opp(pname, 'mlb', st, label, line, best['book'].upper(), best['price'], gi, a))
    opps.sort(key=lambda x:({'A':0,'B':1,'C':2}.get(x['quality']['grade'],3)))
    return opps, analyzed, len(pitchers)

# ── Routes ────────────────────────────────────────────────────────────────────
def mlb_opportunities():
    try:
        sport = request.args.get('sport','').lower()
        if sport != 'mlb': return jsonify({'status':'SKIP'}),200
        sf  = request.args.get('stat_type') or None
        me  = float(request.args.get('min_edge',3))
        opps,analyzed,ngames = scan_mlb(sf, me)
        return jsonify(to_py({'status':'SUCCESS','sport':'mlb','version':'4.0',
            'total_games':ngames,'players_analyzed':analyzed,
            'opportunities_found':len(opps),'opportunities':opps,
            'scan_time':datetime.now().strftime('%Y-%m-%d %H:%M')}))
    except Exception as e:
        import traceback; return jsonify({'status':'ERROR','message':str(e),'trace':traceback.format_exc()[:400]}),500

def mlb_actual_result(player, stat_type, date_str):
    cfg = {'pitcher_strikeouts':('pitching','strikeOuts'),'batter_total_bases':('hitting','totalBases')}
    if stat_type not in cfg: return jsonify({'status':'ERROR','message':'stat_type invalide'}),400
    pid = next((p['id'] for p in get_pitchers() if _match(player,p['name'])),None) if stat_type=='pitcher_strikeouts' else search_batter(player)
    if not pid: pid = search_batter(player)
    games = get_gamelog(pid, stat_type) if pid else None
    if not games: return jsonify({'status':'NOT_FOUND','message':f'Aucune donnée pour {player}'}),404
    g = next((x for x in games if x['date'][:10]==date_str[:10]),games[0]) if date_str else games[0]
    return jsonify({'status':'SUCCESS','player':player,'stat_type':stat_type,'date':g['date'][:10],'actual_value':g['stat']})
