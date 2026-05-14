# nba.py — NBA avec vrai modèle scipy (restauré depuis app.py v3.1 qui marchait)
from flask import Blueprint, jsonify, request
import requests as req, os, time, re, math
import numpy as np
from scipy import stats as scipy_stats
from collections import Counter
from datetime import datetime
from stats import to_py, calc_ev

nba_bp   = Blueprint('nba', __name__)
ODDS_KEY = os.environ.get('ODDS_API_KEY','')
ODDS_BASE= 'https://api.the-odds-api.com/v4'
NBA_SEASON  = '2025-26'
MIN_MINUTES = 15
_cache = {}

from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players as nba_players
from nba_api.stats.static import teams as nba_teams

def _norm(name): return re.sub(r'[^a-z ]','',name.lower().strip())
def _match(a,b):
    na,nb=_norm(a),_norm(b)
    if na==nb: return True
    pa,pb=na.split(),nb.split()
    return bool(pa and pb and pa[-1]==pb[-1] and pa[0][0]==pb[0][0])

# ── Scipy analysis (le vrai modèle qui marchait) ──────────────────────────────
def normality_tests(vals):
    r={}
    try:
        s,p=scipy_stats.shapiro(vals)
        r['shapiro_wilk']={'stat':round(float(s),4),'p_value':round(float(p),4),
            'is_normal':float(p)>0.02,
            'label':'Normal ✅' if float(p)>0.02 else 'Non-normal ⚠️'}
    except: r['shapiro_wilk']=None
    try:
        mu,sigma=float(np.mean(vals)),float(np.std(vals))
        if sigma>0:
            ks,kp=scipy_stats.kstest(vals,'norm',args=(mu,sigma))
            r['ks_test']={'stat':round(float(ks),4),'p_value':round(float(kp),4),
                'is_normal':float(kp)>0.05,
                'label':'Normal ✅' if float(kp)>0.05 else 'Non-normal ⚠️'}
        else: r['ks_test']=None
    except: r['ks_test']=None
    sw=r.get('shapiro_wilk') and r['shapiro_wilk']['is_normal']
    ks=r.get('ks_test') and r['ks_test']['is_normal']
    if sw and ks:   v,l,p='NORMAL','✅ Distribution normale confirmée',0
    elif sw or ks:  v,l,p='BORDERLINE','⚠️ Distribution approximativement normale',10
    else:           v,l,p='NON_NORMAL','❌ Non-normale — edge pénalisé',25
    r.update({'verdict':v,'verdict_label':l,'confidence_penalty':p})
    return r

def analyze_scipy(games, line, stat_type='nba'):
    if len(games) < 5: return None
    vals  = np.array([g['stat'] for g in games], dtype=float)
    n     = len(vals)
    mean  = float(np.mean(vals))
    std   = float(np.std(vals))
    q1,q3 = float(np.percentile(vals,25)),float(np.percentile(vals,75))
    iqr   = q3-q1
    clean = vals[(vals>=q1-1.0*iqr)&(vals<=q3+1.0*iqr)]
    cmean = float(np.mean(clean)) if len(clean)>=3 else mean
    cstd  = float(np.std(clean))  if len(clean)>=3 else std

    # Weighted mean (opponent games weighted x3)
    ws    = [g.get('weight',1) for g in games]
    total_w = sum(ws)
    wmean = sum(g['stat']*w for g,w in zip(games,ws))/total_w if total_w else mean

    norm_res = normality_tests(vals)
    if cstd > 0:
        over_p = float((1-scipy_stats.norm.cdf(line+0.5, wmean, cstd))*100)
    else:
        over_p = float(np.sum(vals>line)/n*100)
    under_p = 100.0-over_p

    implied = 52.38
    eo,eu   = over_p-implied, under_p-implied
    if eo>=eu and eo>0:  rec,raw_edge='OVER',eo
    elif eu>0:            rec,raw_edge='UNDER',eu
    else:                 rec,raw_edge='SKIP',max(eo,eu)

    penalty  = norm_res['confidence_penalty']
    adj_edge = max(0,raw_edge*(1-penalty/100)) if rec!='SKIP' else raw_edge

    kelly=0.0
    if rec!='SKIP':
        prob=over_p if rec=='OVER' else under_p
        if prob>implied:
            kelly=min(20,((prob/100-0.5238)/0.9091)*100*0.25)
            if norm_res['verdict']=='NON_NORMAL':   kelly*=0.5
            elif norm_res['verdict']=='BORDERLINE': kelly*=0.75

    cons = round(max(0,100-(cstd/cmean*100)),1) if cmean>0 else 50.0
    l5   = round(float(np.mean(vals[:5])),1)
    l10  = round(float(np.mean(vals[:min(10,n)])),1)
    over_n = int(np.sum(vals>line))

    # Sanity check: si ligne trop loin de la moyenne, éviter
    if line > 0 and (wmean/line < 0.35 or wmean/line > 3.5):
        return None

    # Grade
    margin = abs(wmean-line); sigma_z = margin/cstd if cstd>0 else 0
    issues=[]; pros=[]
    if n>=15:   pros.append(f'✅ {n} matchs analysés')
    elif n>=8:  pros.append(f'🟡 {n} matchs')
    else:       issues.append(f'❌ Seulement {n} matchs')
    if cons>=65: pros.append(f'✅ Consistance {cons}%')
    elif cons>=45: pros.append(f'🟡 Consistance {cons}%')
    else:        issues.append(f'❌ Consistance {cons}%')
    if norm_res['verdict']=='NORMAL':    pros.append('✅ Distribution normale')
    elif norm_res['verdict']=='BORDERLINE': pros.append('🟡 Distribution approx. normale')
    else: issues.append('❌ Distribution non-normale')
    if sigma_z>=1.2: pros.append(f'✅ Marge {round(margin,1)} = {round(sigma_z,1)}σ')
    elif sigma_z>=0.7: pros.append(f'🟡 Marge {round(margin,1)} = {round(sigma_z,1)}σ')
    else: issues.append(f'❌ Marge trop faible ({round(margin,1)})')
    if rec=='OVER' and l5>line:   pros.append(f'✅ L5 ({l5}) confirme OVER')
    elif rec=='UNDER' and l5<line: pros.append(f'✅ L5 ({l5}) confirme UNDER')
    elif rec=='OVER':  issues.append(f'⚠️ L5 ({l5}) contredit OVER')
    elif rec=='UNDER': issues.append(f'⚠️ L5 ({l5}) contredit UNDER')

    score=len(pros)*2-len(issues)
    if score>=8 and not issues:   grade,color='A','#4ade80'
    elif score>=5 and len(issues)<=1: grade,color='B','#86efac'
    elif score>=3: grade,color='C','#fbbf24'
    else: grade,color='AVOID','#f87171'
    labels={'A':'🟢 BET — Signal solide','B':'🟢 BET — Signal acceptable',
            'C':'🟡 PRUDENCE — Signal faible','AVOID':'🔴 ÉVITER'}

    rec_hr = round((over_n/n)*100,1) if rec=='OVER' else round((1-over_n/n)*100,1)

    return {
        'n':n,'mean':round(mean,2),'wmean':round(wmean,2),
        'cmean':round(cmean,2),'cstd':round(cstd,2),'std':round(std,2),
        'cons':cons,'l5':l5,'l10':l10,
        'hit_rate':round(over_n/n,4),'rec_hit_rate':round(rec_hr/100,4),
        'over_n':over_n,'under_n':n-over_n,
        'rec':rec,'rec_prob':round((over_p if rec=='OVER' else under_p)/100,4),
        'edge':round(adj_edge,1),'raw_edge':round(raw_edge,1),
        'kelly':round(kelly,1),'normality':norm_res,'line':line,
        'trend':'stable',
        'quality':{'grade':grade,'color':color,'label':labels[grade],
                   'pros':pros,'issues':issues},
        'recent':[{'date':g.get('date','')[:10],'stat':g['stat']} for g in games[:10]]
    }

def build_opp_nba(player, stat_type, label, line, book, price, gi, a):
    rec  = a['rec']; prob = a['rec_prob']
    ev   = calc_ev(prob, price)
    kelly= a['kelly']
    return {
        'player':player,'sport':'nba',
        'stat_type':stat_type,'stat_label':label,
        'game_info':gi,'quality':a['quality'],
        'line_analysis':{
            'bookmaker_line':line,'bookmaker':book,
            'recommendation':rec,
            'edge':ev if ev is not None else a['edge'],
            'over_probability':round(a['hit_rate']*100,1),
            'under_probability':round((1-a['hit_rate'])*100,1),
            'kelly_criterion':kelly,'actual_odds':price,
        },
        'deep_stats':{
            'mean':a['cmean'],'weighted_mean':a['wmean'],
            'std':a['cstd'],'consistency':a['cons'],
            'avg_last_5':a['l5'],'avg_last_10':a['l10'],
            'hit_rate':round(a['hit_rate']*100,1),
            'rec_hit_rate':a['rec_hit_rate'],
            'over_count':a['over_n'],'under_count':a['under_n'],
            'games_analyzed':a['n'],'trend':a['trend'],
        },
        'statistical_validation':{
            'normality':a['normality'],


            'is_reliable':a['normality']['verdict']!='NON_NORMAL'
        },
        'recent_games':a['recent'],
    }

# ── Player search ─────────────────────────────────────────────────────────────
def search_player(name):
    key='nba_'+_norm(name)
    if key in _cache: return _cache[key]
    for fn in [nba_players.find_players_by_full_name,
               nba_players.find_players_by_last_name,
               nba_players.find_players_by_first_name]:
        try:
            arg=name if fn==nba_players.find_players_by_full_name else name.split()[-1 if fn==nba_players.find_players_by_last_name else 0]
            res=fn(arg)
            if res:
                active=[p for p in res if p.get('is_active')]
                pid=(active or res)[0]['id']
                _cache[key]=pid; return pid
        except: continue
    return None

def _parse_min(s):
    if not s: return 0.0
    try:
        p=str(s).split(':')
        return float(p[0])+(float(p[1])/60 if len(p)>1 else 0)
    except: return 0.0

def get_gamelog(player_id, stat_col, opp_abbr=None):
    nba_col={'pts':'PTS','reb':'REB','ast':'AST'}.get(stat_col,stat_col.upper())
    key=f'nba_{player_id}_{stat_col}_{opp_abbr or "all"}'
    if key in _cache: return _cache[key]
    raw=[]
    for season_type,weight in [('Playoffs',2),('Regular Season',1)]:
        try:
            gl=playergamelog.PlayerGameLog(player_id=player_id,season=NBA_SEASON,
                season_type_all_star=season_type,timeout=30)
            df=gl.get_data_frames()[0]
            if df.empty: continue
            for _,row in df.iterrows():
                mins=_parse_min(row.get('MIN',0))
                if mins<MIN_MINUTES: continue
                val=row.get(nba_col)
                if val is None: continue
                try: val_i=int(float(val))
                except: continue
                if val_i==0: continue
                try:
                    from datetime import datetime as dt
                    gdate=dt.strptime(str(row.get('GAME_DATE','')), '%b %d, %Y').strftime('%Y-%m-%d')
                except: gdate=str(row.get('GAME_DATE',''))[:10]
                matchup=str(row.get('MATCHUP',''))
                game_opp=matchup.strip().split()[-1].upper()
                is_same=bool(opp_abbr and game_opp==opp_abbr.upper())
                w=3 if is_same else weight
                raw.append({'date':gdate,'stat':val_i,'opp':game_opp,'weight':w,'is_same_opp':is_same})
            time.sleep(0.6)
        except Exception as e:
            print(f'nba_api error {player_id} {season_type}: {e}'); continue
    if not raw: return None
    raw.sort(key=lambda x:(x['date'],-x['weight']),reverse=True)
    seen,uniq=set(),[]
    for g in raw:
        if g['date'] not in seen: seen.add(g['date']); uniq.append(g)
    uniq=uniq[:30]
    if uniq: _cache[key]=uniq
    return uniq or None

def get_opp_abbr(player_name, home_team, away_team):
    try:
        from nba_api.stats.endpoints import commonplayerinfo
        pid=search_player(player_name)
        if not pid: return None
        info=commonplayerinfo.CommonPlayerInfo(player_id=pid,timeout=10)
        df=info.get_data_frames()[0]
        time.sleep(0.3)
        abbr=str(df['TEAM_ABBREVIATION'].iloc[0]).strip().upper()
        if not abbr or abbr=='NAN': return None
        abbr_kw={'OKC':'thunder','DEN':'nuggets','MIN':'timberwolves','MEM':'grizzlies',
            'GSW':'warriors','HOU':'rockets','DAL':'mavericks','LAC':'clippers',
            'LAL':'lakers','PHX':'suns','SAC':'kings','POR':'blazers',
            'BOS':'celtics','NYK':'knicks','MIA':'heat','MIL':'bucks',
            'CLE':'cavaliers','IND':'pacers','PHI':'76ers','ORL':'magic',
            'ATL':'hawks','CHA':'hornets','CHI':'bulls','DET':'pistons',
            'BKN':'nets','TOR':'raptors','WAS':'wizards','NOP':'pelicans',
            'SAS':'spurs','UTA':'jazz'}
        kw=abbr_kw.get(abbr,abbr.lower())
        opp=away_team if kw in home_team.lower() or abbr.lower() in home_team.lower() else home_team
        for t in nba_teams.get_teams():
            if _norm(t['full_name']) in _norm(opp) or _norm(t['nickname']) in _norm(opp):
                return t['abbreviation']
        return None
    except Exception as e:
        print(f'get_opp_abbr error: {e}'); return None

def get_odds_props(market):
    if not ODDS_KEY: return {},{}
    try:
        base=req.get(f'{ODDS_BASE}/sports/basketball_nba/odds',params={
            'apiKey':ODDS_KEY,'regions':'eu','markets':'h2h',
            'bookmakers':'pinnacle','oddsFormat':'american'},timeout=10)
        if base.status_code!=200: return {},{}
        props,ev={},{}
        for game in base.json()[:20]:
            gid=game['id']
            ev[gid]={'home_team':game.get('home_team',''),'away_team':game.get('away_team',''),'time':game.get('commence_time','')}
            data=req.get(f'{ODDS_BASE}/sports/basketball_nba/events/{gid}/odds',params={
                'apiKey':ODDS_KEY,'regions':'eu','markets':market,
                'bookmakers':'pinnacle','oddsFormat':'american'},timeout=10)
            if data.status_code!=200: continue
            for bk in data.json().get('bookmakers',[]):
                for mk in bk.get('markets',[]):
                    if mk['key']!=market: continue
                    for oc in mk.get('outcomes',[]):
                        player=oc.get('description','').strip()
                        point=oc.get('point')
                        if not player or point is None: continue
                        if player not in props: props[player]={'game_id':gid,'lines':[]}
                        props[player]['lines'].append({'book':bk['key'],'line':float(point),'price':int(oc.get('price',-110)),'type':oc.get('name','')})
            time.sleep(0.2)
        return props,ev
    except Exception as e:
        print(f'NBA odds error: {e}'); return {},{}

STAT_MARKETS={'nba_points':('pts','Pts','player_points'),
              'nba_rebounds':('reb','Reb','player_rebounds'),
              'nba_assists':('ast','Ast','player_assists')}

def scan_nba(stat_filter=None, min_ev=3.0):
    stats=[stat_filter] if stat_filter else list(STAT_MARKETS.keys())
    opps,analyzed,ngames=[],0,0
    pid_cache={}; opp_cache={}
    for st in stats:
        col,label,market=STAT_MARKETS[st]
        props,ev=get_odds_props(market)
        ngames=max(ngames,len(ev))
        if not props: continue
        for pname,pd in props.items():
            overs=[l for l in pd['lines'] if l['type']=='Over']
            if not overs: continue
            line=Counter([l['line'] for l in overs]).most_common(1)[0][0]
            best=min(overs,key=lambda x:abs(x['line']-line))
            gi=ev.get(pd['game_id'],{'home_team':'','away_team':'','time':''})
            pk=_norm(pname)
            if pk not in pid_cache:
                pid_cache[pk]=search_player(pname); time.sleep(0.2)
            pid=pid_cache[pk]
            if not pid: continue
            ok=f'{pk}_{_norm(gi.get("home_team",""))}_{_norm(gi.get("away_team",""))}'
            if ok not in opp_cache:
                opp_cache[ok]=get_opp_abbr(pname,gi.get('home_team',''),gi.get('away_team','')); time.sleep(0.15)
            opp_abbr=opp_cache.get(ok)
            if opp_abbr is None:
                print(f'TEAM FAIL {pname}: skip'); continue
            games=get_gamelog(pid,col,opp_abbr=opp_abbr)
            if not games or len(games)<8: continue
            analyzed+=1
            a=analyze_scipy(games,line)
            if not a or a['quality']['grade']=='AVOID': continue
            ev_val=calc_ev(a['rec_prob'],best['price'])
            if ev_val is None or ev_val<min_ev: continue
            opps.append(build_opp_nba(pname,st,label,line,best['book'].upper(),best['price'],gi,a))
    opps.sort(key=lambda x:({'A':0,'B':1,'C':2}.get(x['quality']['grade'],3)))
    return opps,analyzed,ngames

def nba_actual_result(player, stat_type, date_str):
    col={'nba_points':'pts','nba_rebounds':'reb','nba_assists':'ast'}.get(stat_type,'pts')
    pid=search_player(player)
    if not pid: return __import__('flask').jsonify({'status':'NOT_FOUND','message':f'Joueur non trouvé: {player}'}),404
    games=get_gamelog(pid,col)
    if not games: return __import__('flask').jsonify({'status':'NOT_FOUND','message':f'Pas de données pour {player}'}),404
    g=next((x for x in games if x['date'][:10]==date_str[:10]),games[0]) if date_str else games[0]
    return __import__('flask').jsonify({'status':'SUCCESS','player':player,'stat_type':stat_type,'date':g['date'],'actual_value':g['stat']})
