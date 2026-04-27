# ╔══════════════════════════════════════════════════════╗
# ║  app.py — PARTIE 1/3   (colle en premier)           ║
# ║  v2.4: NBA Points/Rebounds/Assists ajouté           ║
# ╚══════════════════════════════════════════════════════╝
#!/usr/bin/env python3
"""Multi-Sport Betting Analyzer v2.4"""
from flask import Flask, jsonify, request
from flask_cors import CORS
import requests, numpy as np, os, math, time, re, io, csv
from scipy import stats as scipy_stats
from datetime import datetime
from collections import Counter

app = Flask(__name__)
CORS(app)

ODDS_API_KEY       = os.environ.get('ODDS_API_KEY')
DATAGOLF_KEY       = os.environ.get('DATAGOLF_KEY','')
BDL_KEY            = os.environ.get('BDL_KEY','d59ef355-67fa-4b6c-afbc-0aaf1f231c6a')
ODDS_BASE          = "https://api.the-odds-api.com/v4"
MLB_BASE           = "https://statsapi.mlb.com/api/v1"
NHL_BASE           = "https://api-web.nhle.com/v1"
NHL_SEARCH_BASE    = "https://search.d3.nhle.com/api/v1/search"
SACKMANN_BASE      = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
DATAGOLF_BASE      = "https://feeds.datagolf.com"
BDL_BASE           = "https://api.balldontlie.io/v1"
DK_BASE            = "https://sportsbook.draftkings.com/api/odds/v1"
DK_HEADERS         = {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36','Accept':'application/json','Referer':'https://sportsbook.draftkings.com/'}
LEAGUE_AVG_K_PCT   = 0.225
CURRENT_MLB_SEASON = 2026
NBA_SEASON         = 2025   # 2024 = saison 2024-25
NBA_MIN_MINUTES    = 20     # ignore garbage time

GAMELOG_CACHE={}; SCHEDULE_CACHE={}; PLAYER_ID_CACHE={}
TEAM_STATS_CACHE={}; TENNIS_CACHE={}; NBA_CACHE={}

STAT_CONFIG = {
    'pitcher_strikeouts': {'sport':'mlb','group':'pitching','col':'strikeOuts','label':'K','min_ip':3.0,'bettable':True,'min_games':5,'odds_sport':'baseball_mlb','odds_market':'pitcher_strikeouts'},
    'batter_total_bases': {'sport':'mlb','group':'hitting','col':'totalBases','label':'TB','min_ip':None,'bettable':True,'min_games':10,'odds_sport':'baseball_mlb','odds_market':'batter_total_bases'},
    'skater_shots':       {'sport':'nhl','group':'skater','col':'shots','label':'SOG','min_ip':None,'bettable':True,'min_games':8,'odds_sport':'icehockey_nhl','odds_market':'player_shots_on_goal'},
    'tennis_aces':        {'sport':'tennis','group':'tennis','col':'aces','label':'ACE','min_ip':None,'bettable':True,'min_games':8,'odds_sport':'tennis_atp','odds_market':'player_aces'},
    'golf_scoring':       {'sport':'golf','group':'golf','col':'sg_total','label':'SG','min_ip':None,'bettable':True,'min_games':6,'odds_sport':'golf_pga','odds_market':'golfer_top_20_finish'},
    'nba_points':         {'sport':'nba','group':'nba','col':'pts','label':'PTS','min_ip':None,'bettable':True,'min_games':10,'odds_sport':'basketball_nba','odds_market':'player_points'},
    'nba_rebounds':       {'sport':'nba','group':'nba','col':'reb','label':'REB','min_ip':None,'bettable':True,'min_games':10,'odds_sport':'basketball_nba','odds_market':'player_rebounds'},
    'nba_assists':        {'sport':'nba','group':'nba','col':'ast','label':'AST','min_ip':None,'bettable':True,'min_games':10,'odds_sport':'basketball_nba','odds_market':'player_assists'},
}
SPORT_STATS = {
    'mlb':    ['pitcher_strikeouts','batter_total_bases'],
    'nhl':    ['skater_shots'],
    'nba':    ['nba_points','nba_rebounds','nba_assists'],
    'tennis': ['tennis_aces'],
    'golf':   ['golf_scoring'],
}

# ── Utils ─────────────────────────────────────────────────────────────────────
def to_python(obj):
    if isinstance(obj,dict): return {k:to_python(v) for k,v in obj.items()}
    if isinstance(obj,list): return [to_python(v) for v in obj]
    if isinstance(obj,(np.integer,)): return int(obj)
    if isinstance(obj,(np.floating,)):
        v=float(obj); return 0.0 if(math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj,float): return 0.0 if(math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj,np.bool_): return bool(obj)
    if isinstance(obj,np.ndarray): return [to_python(v) for v in obj.tolist()]
    return obj

def norm_name(name): return re.sub(r'[^a-z ]','',name.lower().strip())
def names_match(a,b):
    na,nb=norm_name(a),norm_name(b)
    if na==nb: return True
    pa,pb=na.split(),nb.split()
    if not pa or not pb: return False
    if pa[-1]==pb[-1] and pa[0][0]==pb[0][0]: return True
    if len(pa)>=2 and len(pb)>=2 and pa[0]==pb[-1] and pa[-1]==pb[0]: return True
    return False

def safe_req(url,params=None,timeout=12):
    try:
        r=requests.get(url,params=params,timeout=timeout)
        if r.status_code==200: return r.json()
    except Exception as e: print(f"Req error {url[:60]}: {e}")
    return None

def safe_req_dk(url,params=None):
    try:
        r=requests.get(url,params=params,headers=DK_HEADERS,timeout=15)
        if r.status_code==200: return r.json()
        print(f"DK HTTP {r.status_code}: {url[:80]}")
    except Exception as e: print(f"DK error: {e}")
    return None

def safe_req_bdl(endpoint,params=None):
    """balldontlie.io — nécessite Authorization header"""
    try:
        r=requests.get(f"{BDL_BASE}{endpoint}",params=params,
            headers={'Authorization':BDL_KEY},timeout=15)
        if r.status_code==200: return r.json()
        print(f"BDL HTTP {r.status_code}: {endpoint}")
    except Exception as e: print(f"BDL error: {e}")
    return None

# ── Stats ─────────────────────────────────────────────────────────────────────
def normality_tests(vals):
    r={}
    try:
        s,p=scipy_stats.shapiro(vals)
        r['shapiro_wilk']={'stat':round(float(s),4),'p_value':round(float(p),4),'is_normal':float(p)>0.05,'label':'Normal ✅' if float(p)>0.05 else 'Non-normal ⚠️'}
    except: r['shapiro_wilk']=None
    try:
        mu,sigma=float(np.mean(vals)),float(np.std(vals))
        if sigma>0:
            ks,kp=scipy_stats.kstest(vals,'norm',args=(mu,sigma))
            r['ks_test']={'stat':round(float(ks),4),'p_value':round(float(kp),4),'is_normal':float(kp)>0.05,'label':'Normal ✅' if float(kp)>0.05 else 'Non-normal ⚠️'}
        else: r['ks_test']=None
    except: r['ks_test']=None
    sw=r.get('shapiro_wilk') and r['shapiro_wilk']['is_normal']
    ks=r.get('ks_test') and r['ks_test']['is_normal']
    if sw and ks:   v,l,p='NORMAL','✅ Distribution normale confirmée',0
    elif sw or ks:  v,l,p='BORDERLINE','⚠️ Distribution approximativement normale',10
    else:           v,l,p='NON_NORMAL','❌ Non-normale — edge pénalisé',25
    r.update({'verdict':v,'verdict_label':l,'confidence_penalty':p}); return r

def chi_gof(vals,line):
    n=len(vals)
    if n<8: return None
    try:
        mu,sigma=float(np.mean(vals)),float(np.std(vals))
        if sigma<=0: return None
        oo=int(np.sum(vals>line)); uo=int(n-oo)
        po=float(1-scipy_stats.norm.cdf(line,mu,sigma)); oe,ue=po*n,(1-po)*n
        if oe<1 or ue<1: return None
        chi2,p=scipy_stats.chisquare([oo,uo],f_exp=[oe,ue]); good=float(p)>0.05
        return {'chi2':round(float(chi2),4),'p_value':round(float(p),4),'is_good_fit':good,'label':'✅ Bon fit' if good else '⚠️ Fit imparfait','observed':[oo,uo],'expected':[round(oe,1),round(ue,1)]}
    except: return None

def bet_quality(a,stat_type=''):
    score=0; issues=[]; pros=[]
    if stat_type=='batter_total_bases' and a['cmean']<1.5:
        return {'grade':'AVOID','color':'#f87171','label':'🔴 ÉVITER — TB moyen trop bas','score':0,'pros':[],'issues':['❌ Moyenne TB < 1.5']}
    if a['n']>=15:    score+=2; pros.append(f"✅ {a['n']} matchs analysés")
    elif a['n']>=8:   score+=1; pros.append(f"🟡 {a['n']} matchs")
    else: issues.append(f"❌ Seulement {a['n']} matchs")
    if a['cons']>=70:  score+=2; pros.append(f"✅ Consistance {a['cons']}%")
    elif a['cons']>=50: score+=1; pros.append(f"🟡 Consistance {a['cons']}%")
    else: issues.append(f"❌ Consistance {a['cons']}%")
    v=a['normality']['verdict']
    if v=='NORMAL':       score+=2; pros.append("✅ Distribution normale confirmée")
    elif v=='BORDERLINE': score+=1; pros.append("🟡 Distribution approximativement normale")
    else: issues.append("❌ Distribution non-normale")
    gof=a.get('chi_gof')
    if gof and gof['is_good_fit']:      score+=1; pros.append("✅ Chi² GOF cohérent")
    elif gof and not gof['is_good_fit']: issues.append("⚠️ Chi² incohérent")
    margin=abs(a['adj_mean']-a['line']); sigma=a['cstd'] if a['cstd']>0 else 1; z=margin/sigma
    if z>=1.5:   score+=2; pros.append(f"✅ Marge {margin:.1f} = {z:.1f}σ")
    elif z>=0.8: score+=1; pros.append(f"🟡 Marge {margin:.1f} = {z:.1f}σ")
    else: issues.append(f"❌ Marge trop faible ({margin:.1f})")
    if a['rec']=='OVER'  and a['l5']>a['line']:  score+=1; pros.append(f"✅ L5 ({a['l5']}) confirme OVER")
    elif a['rec']=='UNDER' and a['l5']<a['line']: score+=1; pros.append(f"✅ L5 ({a['l5']}) confirme UNDER")
    elif a['rec']=='OVER':  issues.append(f"⚠️ L5 ({a['l5']}) contredit le OVER")
    elif a['rec']=='UNDER': issues.append(f"⚠️ L5 ({a['l5']}) contredit le UNDER")
    min_ok=z>=0.5 and margin>=0.7
    if not min_ok:   grade,color,label='AVOID','#f87171','🔴 ÉVITER — Marge insuffisante'
    elif score>=8 and not issues and z>=1.5: grade,color,label='A','#4ade80','🟢 BET — Signal solide'
    elif score>=6 and len(issues)<=1:        grade,color,label='B','#86efac','🟢 BET — Signal acceptable'
    elif score>=4:                           grade,color,label='C','#fbbf24','🟡 PRUDENCE — Signal faible'
    else:                                    grade,color,label='AVOID','#f87171','🔴 ÉVITER'
    return {'grade':grade,'color':color,'label':label,'score':score,'pros':pros,'issues':issues}

def analyze(games,line,stat_type,adj_mean_override=None):
    if len(games)<4: return None
    vals=np.array([g['stat'] for g in games],dtype=float); n=len(vals)
    mean=float(np.mean(vals)); std=float(np.std(vals))
    q1,q3=float(np.percentile(vals,25)),float(np.percentile(vals,75)); iqr=q3-q1
    clean=vals[(vals>=q1-1.5*iqr)&(vals<=q3+1.5*iqr)]
    cmean=float(np.mean(clean)) if len(clean)>=3 else mean
    cstd=float(np.std(clean)) if len(clean)>=3 else std
    adj_mean=adj_mean_override if adj_mean_override is not None else cmean
    norm_res=normality_tests(vals); gof=chi_gof(vals,line)
    if cstd>0: over_p=float((1-scipy_stats.norm.cdf(line+0.5,adj_mean,cstd))*100)
    else:      over_p=float(np.sum(vals>line)/n*100)
    under_p=100.0-over_p; implied=52.38; eo,eu=over_p-implied,under_p-implied
    if eo>=eu and eo>0: rec,raw_edge='OVER',eo
    elif eu>0:           rec,raw_edge='UNDER',eu
    else:                rec,raw_edge='SKIP',max(eo,eu)
    penalty=norm_res['confidence_penalty']
    adj_edge=max(0,raw_edge*(1-penalty/100)) if rec!='SKIP' else raw_edge
    try:
        slope,_,r,_,_=scipy_stats.linregress(np.arange(n),vals)
        r_sq=round(float(r**2),3); slope=round(float(slope),3)
    except: r_sq=slope=0.0
    kelly=0.0
    if rec!='SKIP':
        prob=over_p if rec=='OVER' else under_p
        if prob>implied:
            kelly=min(20,((prob/100-0.5238)/0.9091)*100*0.25)
            if norm_res['verdict']=='NON_NORMAL':   kelly*=0.5
            elif norm_res['verdict']=='BORDERLINE': kelly*=0.75
    cons=round(max(0,100-(cstd/cmean*100)),1) if cmean>0 else 50.0
    recent=[{'date':g.get('date','')[:10],'stat':g['stat']} for g in games[:10]]
    a={'n':n,'mean':round(mean,1),'cmean':round(cmean,1),'adj_mean':round(adj_mean,2),
       'total_adj':round(adj_mean-cmean,2),'std':round(std,2),'cstd':round(cstd,2),
       'l5':round(float(np.mean(vals[:5])),1),'l10':round(float(np.mean(vals[:10])),1) if n>=10 else round(mean,1),
       'over_p':round(over_p,1),'under_p':round(under_p,1),
       'over_n':int(np.sum(vals>line)),'under_n':int(np.sum(vals<=line)),
       'raw_edge':round(raw_edge,1),'edge':round(adj_edge,1),'rec':rec,
       'r_sq':r_sq,'slope':slope,'kelly':round(kelly,1),'cons':cons,
       'normality':norm_res,'chi_gof':gof,'line':line,'recent':recent}
    a['quality']=bet_quality(a,stat_type); return a

def backtest(games,stat_type,min_train=10,stake=10.0):
    if len(games)<min_train+3: return None
    chrono=list(reversed(games)); results=[]; profit=0.0
    for i in range(min_train,len(chrono)):
        train=list(reversed(chrono[:i])); actual=chrono[i]['stat']
        tvals=np.array([g['stat'] for g in train]); sim_line=float(np.median(tvals))
        a=analyze(train,sim_line,stat_type)
        if not a or a['rec']=='SKIP': continue
        correct=(a['rec']=='OVER' and actual>sim_line) or (a['rec']=='UNDER' and actual<sim_line)
        pnl=stake*(100/110) if correct else -stake; profit+=pnl
        results.append({'game_num':i,'date':chrono[i].get('date','')[:10],'actual':actual,
            'predicted':round(a['adj_mean'],1),'sim_line':round(sim_line,1),'recommendation':a['rec'],
            'correct':correct,'quality_grade':a['quality']['grade'],'edge':a['edge'],'pnl':round(pnl,2)})
    if not results: return None
    n=len(results); wins=sum(1 for r in results if r['correct'])
    ab=[r for r in results if r['quality_grade'] in ('A','B')]; ab_wins=sum(1 for r in ab if r['correct'])
    return {'total_bets':n,'wins':wins,'losses':n-wins,'hit_rate':round(wins/n*100,1),
        'profit_usd':round(profit,2),'roi_pct':round(profit/(n*stake)*100,1),
        'mae':round(float(np.mean([abs(r['actual']-r['predicted']) for r in results])),2),
        'avg_edge':round(float(np.mean([r['edge'] for r in results])),1),
        'grade_ab_bets':len(ab),'grade_ab_hit_rate':round(ab_wins/len(ab)*100,1) if ab else 0,
        'grade_ab_profit':round(sum(r['pnl'] for r in ab),2),'per_bet':results[-20:]}

# ── MLB ───────────────────────────────────────────────────────────────────────
def mlb_get_pitchers():
    today=datetime.now().strftime('%Y-%m-%d')
    tomorrow=(__import__('datetime').date.today()+__import__('datetime').timedelta(days=1)).strftime('%Y-%m-%d')
    key=f"mlb_{today}"
    if key in SCHEDULE_CACHE: return SCHEDULE_CACHE[key]
    data=safe_req(f"{MLB_BASE}/schedule",params={'sportId':1,'startDate':today,'endDate':tomorrow,'hydrate':'probablePitcher','gameType':'R','season':CURRENT_MLB_SEASON})
    if not data: return []
    pitchers=[]
    for db in data.get('dates',[]):
        for game in db.get('games',[]):
            home=game.get('teams',{}).get('home',{}).get('team',{}).get('name','')
            away=game.get('teams',{}).get('away',{}).get('team',{}).get('name','')
            gtime=game.get('gameDate','')
            for side in ['home','away']:
                pp=game.get('teams',{}).get(side,{}).get('probablePitcher',{})
                if pp and pp.get('id'):
                    pitchers.append({'id':pp['id'],'name':pp.get('fullName',''),'home_team':home,'away_team':away,'is_home':side=='home','opponent':away if side=='home' else home,'game_time':gtime})
    SCHEDULE_CACHE[key]=pitchers; return pitchers

def mlb_get_gamelog(player_id,stat_type):
    cfg=STAT_CONFIG[stat_type]; ck=f"mlb_{player_id}_{stat_type}"
    if ck in GAMELOG_CACHE: return GAMELOG_CACHE[ck]
    games=[]
    for season in [CURRENT_MLB_SEASON,CURRENT_MLB_SEASON-1]:
        data=safe_req(f"{MLB_BASE}/people/{player_id}/stats",params={'stats':'gameLog','season':season,'group':cfg['group'],'gameType':'R'})
        if not data: continue
        splits=(data.get('stats') or [{}])[0].get('splits',[])
        for s in splits:
            st=s.get('stat',{}); val=st.get(cfg['col'])
            if val is None: continue
            entry={'date':s.get('date',''),'stat':int(val)}
            if cfg['group']=='pitching':
                ip_str=str(st.get('inningsPitched','0') or '0')
                try:
                    p=ip_str.split('.'); entry['ip']=int(p[0])+(int(p[1])/3 if len(p)>1 and p[1] else 0)
                except: entry['ip']=0.0
            games.append(entry)
    games.sort(key=lambda x:x['date'],reverse=True)
    seen=set(); uniq=[]
    for g in games:
        d=g['date'][:10]
        if d not in seen: seen.add(d); uniq.append(g)
    GAMELOG_CACHE[ck]=uniq; return uniq or None

def mlb_search_player(name):
    key=norm_name(name)
    if key in PLAYER_ID_CACHE: return PLAYER_ID_CACHE[key]
    last=name.split()[-1] if ' ' in name else name
    data=safe_req(f"{MLB_BASE}/people/search",params={'names':last,'sportIds':1})
    if data:
        for p in data.get('people',[]):
            if names_match(name,p.get('fullName','')): PLAYER_ID_CACHE[key]=p['id']; return p['id']
    return None

def mlb_opp_k_pct(team):
    key=norm_name(team)+'_kpct'
    if key in TEAM_STATS_CACHE: return TEAM_STATS_CACHE[key]
    td=safe_req(f"{MLB_BASE}/teams",params={'sportId':1,'season':CURRENT_MLB_SEASON})
    if not td: return None
    tid=next((t['id'] for t in td.get('teams',[]) if names_match(team,t.get('name',''))),None)
    if not tid: return None
    d=safe_req(f"{MLB_BASE}/teams/{tid}/stats",params={'stats':'season','group':'hitting','season':CURRENT_MLB_SEASON,'gameType':'R'})
    if not d: return None
    sp=(d.get('stats') or [{}])[0].get('splits',[])
    if not sp: return None
    st=sp[0].get('stat',{})
    k=int(st.get('strikeOuts',0) or 0); ab=int(st.get('atBats',1) or 1)
    bb=int(st.get('baseOnBalls',0) or 0); hbp=int(st.get('hitByPitch',0) or 0); pa=ab+bb+hbp
    if pa==0: return None
    kp=k/pa; TEAM_STATS_CACHE[key]=kp; return kp


# ╔══════════════════════════════════════════════════════╗
# ║  app.py — PARTIE 2/3   (colle à la suite de P1)     ║
# ║  v2.7: nba_api (stats.nba.com) pour les gamelogs   ║
# ╚══════════════════════════════════════════════════════╝

from nba_api.stats.endpoints import playergamelog, commonallplayers
from nba_api.stats.static import players as nba_players_static

# ── NBA — nba_api (stats.nba.com) ────────────────────────────────────────────
def nba_search_player(name):
    """Cherche un joueur NBA via nba_api static data"""
    key='nba_'+norm_name(name)
    if key in PLAYER_ID_CACHE: return PLAYER_ID_CACHE[key]
    # Recherche exacte d'abord
    results=nba_players_static.find_players_by_full_name(name)
    if not results:
        # Essaie juste le nom de famille
        last=name.split()[-1]
        results=nba_players_static.find_players_by_last_name(last)
    if not results:
        # Essaie prénom
        first=name.split()[0]
        results=nba_players_static.find_players_by_first_name(first)
    if results:
        # Prend le joueur actif en priorité
        active=[p for p in results if p.get('is_active',False)]
        chosen=active[0] if active else results[0]
        pid=chosen['id']
        PLAYER_ID_CACHE[key]=pid
        return pid
    return None

def nba_get_gamelog(player_id, stat_col):
    """
    Récupère le gamelog NBA via nba_api.
    stat_col: 'PTS', 'REB', 'AST'
    Retourne liste de {'date':str, 'stat':int}
    """
    # Map nos colonnes internes → colonnes nba_api
    col_map={'pts':'PTS','reb':'REB','ast':'AST'}
    nba_col=col_map.get(stat_col, stat_col.upper())
    ck=f"nba_{player_id}_{stat_col}"
    if ck in NBA_CACHE: return NBA_CACHE[ck]

    games=[]
    # Saison régulière + playoffs 2025-26
    for season_type in ['Regular Season','Playoffs']:
        try:
            gl=playergamelog.PlayerGameLog(
                player_id=player_id,
                season='2025-26',
                season_type_all_star=season_type,
                timeout=30
            )
            df=gl.get_data_frames()[0]
            if df.empty: continue
            for _,row in df.iterrows():
                # Filtre garbage time (MIN < 15)
                try:
                    mins=float(str(row.get('MIN',0)).split(':')[0])
                    if mins < 15: continue
                except: pass
                val=row.get(nba_col)
                if val is None: continue
                game_date=str(row.get('GAME_DATE',''))
                # Convertit "APR 26, 2026" → "2026-04-26"
                try:
                    from datetime import datetime as dt
                    gdate=dt.strptime(game_date,'%b %d, %Y').strftime('%Y-%m-%d')
                except:
                    gdate=game_date
                games.append({'date':gdate,'stat':int(val)})
            time.sleep(0.6)  # rate limit nba_api
        except Exception as e:
            print(f"nba_api gamelog error {player_id} {season_type}: {e}")
            continue

    games.sort(key=lambda x:x['date'],reverse=True)
    seen=set(); uniq=[]
    for g in games:
        if g['date'] not in seen: seen.add(g['date']); uniq.append(g)
    if uniq: NBA_CACHE[ck]=uniq
    return uniq or None

# ── Odds API ──────────────────────────────────────────────────────────────────
def get_odds_props(odds_sport, odds_market, max_games=20):
    if not ODDS_API_KEY: return {},{}
    data=safe_req(f"{ODDS_BASE}/sports/{odds_sport}/odds",
        params={'apiKey':ODDS_API_KEY,'regions':'us','markets':'h2h','oddsFormat':'american'})
    if not data:
        print(f"Odds API: aucun event pour {odds_sport}")
        return {},{}
    props={}; ev={}
    print(f"Odds API: {len(data)} events pour {odds_sport}")
    for game in data[:max_games]:
        gid=game['id']
        ev[gid]={'home_team':game.get('home_team',''),'away_team':game.get('away_team',''),
                 'time':game.get('commence_time','')}
        try:
            d2=safe_req(f"{ODDS_BASE}/sports/{odds_sport}/events/{gid}/odds",
                params={'apiKey':ODDS_API_KEY,'regions':'us','markets':odds_market,'oddsFormat':'american'})
            if not d2: continue
            for bk in d2.get('bookmakers',[]):
                for mk in bk.get('markets',[]):
                    if mk['key']!=odds_market: continue
                    for oc in mk.get('outcomes',[]):
                        player=oc.get('description','').strip()
                        point=oc.get('point'); btype=oc.get('name','')
                        if not player or point is None: continue
                        if player not in props: props[player]={'game_id':gid,'lines':[]}
                        props[player]['lines'].append({
                            'book':bk['key'],'line':float(point),
                            'price':int(oc.get('price',-110)),'type':btype})
            time.sleep(0.25)
        except Exception as e:
            print(f"Odds API event error: {e}"); continue
    print(f"Odds API {odds_market}: {len(props)} joueurs")
    return props,ev

def nba_get_odds(stat_type):
    """Direct Odds API — plan $30 supporte player props NBA"""
    cfg=STAT_CONFIG[stat_type]
    return get_odds_props(cfg['odds_sport'],cfg['odds_market'],max_games=20)

# ── NHL ───────────────────────────────────────────────────────────────────────
def nhl_search_player(name):
    key='nhl_'+norm_name(name)
    if key in PLAYER_ID_CACHE: return PLAYER_ID_CACHE[key]
    last=name.split()[-1] if ' ' in name else name
    data=safe_req(NHL_SEARCH_BASE,params={'q':last,'culture':'en-us','isActive':'true'})
    if data:
        for r in (data if isinstance(data,list) else []):
            if names_match(name,r.get('name','')):
                pid=r.get('playerId')
                if pid: PLAYER_ID_CACHE[key]=pid; return pid
    return None

def nhl_get_skater_gamelog(player_id):
    ck=f"nhl_{player_id}_sog"
    if ck in GAMELOG_CACHE: return GAMELOG_CACHE[ck]
    games=[]
    for season,gt in [('20252026',3),('20252026',2),('20242025',2)]:
        data=safe_req(f"{NHL_BASE}/player/{player_id}/game-log/{season}/{gt}")
        if not data: continue
        for g in data.get('gameLog',[]):
            shots=g.get('shots') or g.get('shotsOnGoal')
            if shots is None: continue
            games.append({'date':g.get('gameDate',''),'stat':int(shots)})
    games.sort(key=lambda x:x['date'],reverse=True)
    seen=set(); uniq=[]
    for g in games:
        d=g['date'][:10]
        if d not in seen: seen.add(d); uniq.append(g)
    if uniq: GAMELOG_CACHE[ck]=uniq
    return uniq or None

# ── Tennis ────────────────────────────────────────────────────────────────────
def tennis_get_aces(player_name, surface=None):
    cache_key=f"tennis_{norm_name(player_name)}_{surface or 'all'}"
    if cache_key in TENNIS_CACHE: return TENNIS_CACHE[cache_key]
    games=[]
    for year in [2026,2025,2024]:
        try:
            resp=requests.get(f"{SACKMANN_BASE}/atp_matches_{year}.csv",timeout=15)
            if resp.status_code!=200: continue
            for row in csv.DictReader(io.StringIO(resp.text)):
                surf=row.get('surface',''); date=row.get('tourney_date','')
                if surface and surf.lower()!=surface.lower(): continue
                for role,ace_col in [('winner_name','w_ace'),('loser_name','l_ace')]:
                    pname=row.get(role,'')
                    if pname and names_match(player_name,pname):
                        val=row.get(ace_col,'').strip()
                        if val:
                            try: games.append({'date':str(date),'stat':int(float(val)),'surface':surf})
                            except: pass
        except Exception as e: print(f"Tennis CSV {year}: {e}")
    games.sort(key=lambda x:x['date'],reverse=True)
    if games: TENNIS_CACHE[cache_key]=games
    return games or None

# ── Golf ──────────────────────────────────────────────────────────────────────
def golf_get_stats(player_name):
    if not DATAGOLF_KEY: return None
    ck=f"golf_{norm_name(player_name)}"
    if ck in GAMELOG_CACHE: return GAMELOG_CACHE[ck]
    data=safe_req(f"{DATAGOLF_BASE}/preds/player-decompositions",
        params={'file_format':'json','key':DATAGOLF_KEY,'tour':'pga'})
    if not data: return None
    games=[]
    for p in (data.get('players') or []):
        if names_match(player_name,p.get('player_name','')):
            sg=p.get('sg_total')
            if sg is not None:
                games.append({'date':datetime.now().strftime('%Y-%m-%d'),'stat':round(float(sg),3)})
    if games: GAMELOG_CACHE[ck]=games
    return games or None

# ── DraftKings NHL SOG ────────────────────────────────────────────────────────
def safe_req_dk(url, params=None):
    try:
        r=requests.get(url,params=params,headers=DK_HEADERS,timeout=15)
        if r.status_code==200: return r.json()
        print(f"DK HTTP {r.status_code}: {url[:80]}")
    except Exception as e: print(f"DK error: {e}")
    return None

def dk_nhl_get_sog():
    props={}; ev={}
    cats_data=safe_req_dk(f"{DK_BASE}/leagues/42/categories")
    if not cats_data: return {},{}
    cat_id=None; subcat_id=None
    for cat in cats_data.get('eventGroup',{}).get('offerCategories',[]):
        for desc in cat.get('offerSubcategoryDescriptors',[]):
            if 'shot' in desc.get('name','').lower():
                cat_id=cat.get('offerCategoryId'); subcat_id=desc.get('subcategoryId')
                print(f"DK NHL SOG: {desc.get('name')}"); break
        if cat_id: break
    if not cat_id: return {},{}
    data=safe_req_dk(f"{DK_BASE}/leagues/42/categories/{cat_id}/subcategories/{subcat_id}")
    if not data: return {},{}
    for event in data.get('eventGroup',{}).get('events',[]):
        eid=str(event.get('eventId',''))
        ev[eid]={'home_team':event.get('teamName1',''),'away_team':event.get('teamName2',''),'time':event.get('startDate','')}
    for cat in data.get('eventGroup',{}).get('offerCategories',[]):
        for desc in cat.get('offerSubcategoryDescriptors',[]):
            if 'shot' not in desc.get('name','').lower(): continue
            for offer in desc.get('offerSubcategory',{}).get('offers',[]):
                eid=str(offer.get('eventId','')); player=None; lines=[]
                for oc in offer.get('outcomes',[]):
                    p=(oc.get('participant') or oc.get('label') or '').strip()
                    if p and p not in ('Over','Under','Yes','No'): player=p
                    btype=oc.get('label','')
                    if btype not in ('Over','Under'): continue
                    line_val=oc.get('line') or oc.get('handicap')
                    try: price=int(str(oc.get('oddsAmerican','-110')).replace('+',''))
                    except: price=-110
                    if line_val is not None:
                        lines.append({'book':'DRAFTKINGS','line':float(line_val),'price':price,'type':btype})
                if player and lines:
                    if player not in props: props[player]={'game_id':eid,'lines':[]}
                    props[player]['lines'].extend(lines)
    print(f"DK NHL SOG: {len(props)} joueurs"); return props,ev

def nhl_get_odds():
    props,ev=dk_nhl_get_sog()
    if props: return props,ev
    for sk in ['icehockey_nhl','icehockey_nhl_championship']:
        p,e=get_odds_props(sk,'player_shots_on_goal')
        if p: return p,e
    return {},{}

def _build_opp(player,stat_type,sport,line,best,gi,opponent,is_home,a):
    cfg=STAT_CONFIG[stat_type]
    return {'player':player,'sport':sport,'stat_type':stat_type,'stat_label':cfg['label'],
        'game_info':{'home_team':gi.get('home_team',''),'away_team':gi.get('away_team',''),
                     'time':gi.get('time',''),'opponent':opponent,'is_home':is_home},
        'quality':a['quality'],
        'line_analysis':{'bookmaker_line':line,'bookmaker':best['book'].upper(),
            'recommendation':a['rec'],'raw_edge':a['raw_edge'],'edge':a['edge'],
            'over_probability':a['over_p'],'under_probability':a['under_p'],'kelly_criterion':a['kelly']},
        'deep_stats':{'mean':a['mean'],'clean_mean':a['cmean'],'adjusted_mean':a['adj_mean'],
            'std':a['std'],'avg_last_5':a['l5'],'avg_last_10':a['l10'],'consistency':a['cons'],
            'games_analyzed':a['n'],'over_count':a['over_n'],'under_count':a['under_n'],
            'r_squared':a['r_sq'],'trend_slope':a['slope']},
        'statistical_validation':{'normality':a['normality'],'chi_gof':a['chi_gof'],
            'is_reliable':a['normality']['verdict']!='NON_NORMAL' and
                          (a['chi_gof'] is None or a['chi_gof']['is_good_fit'])},
        'recent_games':a['recent']}


# ╔══════════════════════════════════════════════════════╗
# ║  app.py — PARTIE 3/3   (colle à la suite de P2)     ║
# ║  v2.6: debug-nba endpoint + fix NameError           ║
# ╚══════════════════════════════════════════════════════╝

def scan_sport(sport, stat_type_filter=None, min_edge=5.0):
    stat_types=[stat_type_filter] if stat_type_filter else SPORT_STATS.get(sport,[])
    if not stat_types: return [],0,0
    opps=[]; analyzed=0; n_games=0

    if sport=='mlb':
        pitchers=mlb_get_pitchers(); n_games=len(set(p['home_team'] for p in pitchers))
        for st in stat_types:
            cfg=STAT_CONFIG[st]; props,ev=get_odds_props(cfg['odds_sport'],cfg['odds_market'])
            is_p=cfg['group']=='pitching'
            for pname,pd in props.items():
                overs=[l for l in pd['lines'] if l['type']=='Over']
                if not overs: continue
                line=Counter([l['line'] for l in overs]).most_common(1)[0][0]
                best=min(overs,key=lambda x:abs(x['line']-line))
                if is_p:
                    pitcher=next((p for p in pitchers if names_match(pname,p['name'])),None)
                    if not pitcher: continue
                    pid=pitcher['id']; opp=pitcher.get('opponent',''); ih=pitcher.get('is_home',True)
                    gi={'home_team':pitcher['home_team'],'away_team':pitcher['away_team'],'time':pitcher['game_time']}
                else:
                    pid=mlb_search_player(pname)
                    if not pid: continue
                    opp=''; ih=True; gi={'home_team':'','away_team':'','time':''}; time.sleep(0.1)
                games=mlb_get_gamelog(pid,st)
                if not games or len(games)<cfg['min_games']: continue
                analyzed+=1; adj=None
                if st=='pitcher_strikeouts':
                    kp=mlb_opp_k_pct(opp)
                    if kp: adj=round(float(np.mean([g['stat'] for g in games]))+(kp-LEAGUE_AVG_K_PCT)/0.01*0.3,2)
                a=analyze(games,line,st,adj_mean_override=adj)
                if not a or a['rec']=='SKIP' or a['edge']<min_edge or a['quality']['grade']=='AVOID': continue
                opps.append(_build_opp(pname,st,'mlb',line,best,ev.get(pd['game_id'],gi),opp,ih,a))

    elif sport=='nba':
        nba_stat_map={'nba_points':'pts','nba_rebounds':'reb','nba_assists':'ast'}
        all_player_ids={}; n_games=0
        for st in stat_types:
            props,ev=nba_get_odds(st)
            if not props: continue
            n_games=max(n_games,len(ev))
            stat_col=nba_stat_map.get(st,'pts')
            for pname,pd in props.items():
                overs=[l for l in pd['lines'] if l['type']=='Over']
                if not overs: continue
                line=Counter([l['line'] for l in overs]).most_common(1)[0][0]
                best=min(overs,key=lambda x:abs(x['line']-line))
                pkey=norm_name(pname)
                if pkey not in all_player_ids:
                    all_player_ids[pkey]=nba_search_player(pname); time.sleep(0.1)
                pid=all_player_ids[pkey]
                if not pid: continue
                games=nba_get_gamelog(pid,stat_col)
                if not games or len(games)<STAT_CONFIG[st]['min_games']: continue
                analyzed+=1
                a=analyze(games,line,st)
                if not a or a['rec']=='SKIP' or a['edge']<min_edge or a['quality']['grade']=='AVOID': continue
                gi=ev.get(pd['game_id'],{'home_team':'','away_team':'','time':''})
                opps.append(_build_opp(pname,st,'nba',line,best,gi,'',True,a))
        if not n_games:
            return [{'_no_key':True,'message':'🏀 NBA: Aucun marché trouvé. Vérifie /api/debug-nba.'}],0,0

    elif sport=='nhl':
        n_games=10; props,ev=nhl_get_odds()
        if not props:
            return [{'_no_key':True,'message':'🏒 NHL: Aucun marché SOG trouvé.'}],0,0
        cfg=STAT_CONFIG['skater_shots']
        for pname,pd in props.items():
            overs=[l for l in pd['lines'] if l['type']=='Over']
            if not overs: continue
            line=Counter([l['line'] for l in overs]).most_common(1)[0][0]
            best=min(overs,key=lambda x:abs(x['line']-line))
            pid=nhl_search_player(pname)
            if not pid: continue
            games=nhl_get_skater_gamelog(pid)
            if not games or len(games)<cfg['min_games']: continue
            analyzed+=1
            a=analyze(games,line,'skater_shots')
            if not a or a['rec']=='SKIP' or a['edge']<min_edge or a['quality']['grade']=='AVOID': continue
            gi=ev.get(pd['game_id'],{'home_team':'','away_team':'','time':''})
            opps.append(_build_opp(pname,'skater_shots','nhl',line,best,gi,'',True,a))

    elif sport=='tennis':
        props,ev=get_odds_props('tennis_atp','player_aces')
        if not props: props,ev=get_odds_props('tennis_atp_french_open','player_aces')
        n_games=len(props)
        if not props:
            return [{'_no_key':True,'message':'🎾 Tennis: Aucun marché player_aces actif. Prochain: Roland Garros fin mai.'}],0,0
        cfg=STAT_CONFIG['tennis_aces']
        for pname,pd in props.items():
            overs=[l for l in pd['lines'] if l['type']=='Over']
            if not overs: continue
            line=Counter([l['line'] for l in overs]).most_common(1)[0][0]
            best=min(overs,key=lambda x:abs(x['line']-line))
            games=tennis_get_aces(pname)
            if not games or len(games)<cfg['min_games']: continue
            analyzed+=1
            a=analyze(games,line,'tennis_aces')
            if not a or a['rec']=='SKIP' or a['edge']<min_edge or a['quality']['grade']=='AVOID': continue
            gi=ev.get(pd['game_id'],{'home_team':'','away_team':'','time':''})
            opps.append(_build_opp(pname,'tennis_aces','tennis',line,best,gi,'',True,a))

    elif sport=='golf':
        if not DATAGOLF_KEY:
            return [{'_no_key':True,'message':'⛳ Golf: DATAGOLF_KEY manquant.'}],0,0
        cfg=STAT_CONFIG['golf_scoring']; props,ev=get_odds_props(cfg['odds_sport'],cfg['odds_market'])
        n_games=len(props)
        for pname,pd in props.items():
            overs=[l for l in pd['lines'] if l['type']=='Over']
            if not overs: continue
            line=Counter([l['line'] for l in overs]).most_common(1)[0][0]
            best=min(overs,key=lambda x:abs(x['line']-line))
            games=golf_get_stats(pname)
            if not games or len(games)<cfg['min_games']: continue
            analyzed+=1
            a=analyze(games,line,'golf_scoring')
            if not a or a['rec']=='SKIP' or a['edge']<min_edge or a['quality']['grade']=='AVOID': continue
            gi=ev.get(pd['game_id'],{'home_team':'','away_team':'','time':''})
            opps.append(_build_opp(pname,'golf_scoring','golf',line,best,gi,'',True,a))

    opps.sort(key=lambda x:({'A':0,'B':1,'C':2}.get(x['quality']['grade'],3),-x['line_analysis']['edge']))
    return opps,analyzed,n_games


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/api/debug-nba', methods=['GET'])
def debug_nba():
    """Teste balldontlie + Odds API NBA depuis Render"""
    import traceback
    results={}
    # Test balldontlie
    try:
        r=safe_req_bdl('/players',params={'search':'lebron james','per_page':3})
        if r:
            results['balldontlie']={'status':'OK','data':r.get('data',[])}
        else:
            results['balldontlie']={'status':'FAIL','response':None}
    except Exception as e:
        results['balldontlie']={'status':'ERROR','error':str(e),'trace':traceback.format_exc()[:300]}
    # Test Odds API NBA player_points sur 1 event
    try:
        props,ev=get_odds_props('basketball_nba','player_points',max_games=1)
        first_players=list(props.keys())[:5]
        results['odds_api_nba']={
            'status':'OK' if props else 'EMPTY',
            'events_found':len(ev),
            'props_found':len(props),
            'sample_players':first_players
        }
    except Exception as e:
        results['odds_api_nba']={'status':'ERROR','error':str(e),'trace':traceback.format_exc()[:300]}
    return jsonify(results)


@app.route('/api/debug-dk', methods=['GET'])
def debug_dk():
    """Debug DraftKings NHL connectivity"""
    import traceback
    results={}
    try:
        r=requests.get(f"{DK_BASE}/leagues/42/categories",
            headers=DK_HEADERS,timeout=15)
        results['nhl_categories']={'status':r.status_code,'body':r.text[:400]}
    except Exception as e:
        results['nhl_categories']={'error':str(e),'trace':traceback.format_exc()[:300]}
    return jsonify(results)


@app.route('/api/daily-opportunities', methods=['GET'])
def daily_opportunities():
    try:
        sport=request.args.get('sport','mlb').lower()
        stat_type=request.args.get('stat_type',None)
        min_edge=float(request.args.get('min_edge',5))
        if sport not in SPORT_STATS:
            return jsonify({'status':'ERROR','message':f"sport: {list(SPORT_STATS.keys())}"}),400
        opps,analyzed,n_games=scan_sport(sport,stat_type,min_edge)
        if opps and isinstance(opps[0],dict) and opps[0].get('_no_key'):
            return jsonify({'status':'INFO','sport':sport,'message':opps[0]['message'],
                'opportunities':[],'players_analyzed':0,'scan_time':datetime.now().strftime('%Y-%m-%d %H:%M')})
        return jsonify(to_python({'status':'SUCCESS','sport':sport,'version':'2.6',
            'stat_types_scanned':[stat_type] if stat_type else SPORT_STATS[sport],
            'total_games':n_games,'players_analyzed':analyzed,
            'opportunities_found':len(opps),'opportunities':opps,
            'scan_time':datetime.now().strftime('%Y-%m-%d %H:%M')}))
    except Exception as e:
        import traceback
        return jsonify({'status':'ERROR','message':str(e),'trace':traceback.format_exc()[:1000]}),500


@app.route('/api/actual-result', methods=['GET'])
def actual_result():
    try:
        player_name=request.args.get('player',''); stat_type=request.args.get('stat_type','')
        sport=request.args.get('sport','mlb'); date_str=request.args.get('date','')
        if not player_name or not stat_type:
            return jsonify({'status':'ERROR','message':'player et stat_type requis'}),400
        games=None
        if sport=='mlb':
            cfg=STAT_CONFIG.get(stat_type)
            if not cfg: return jsonify({'status':'ERROR','message':'stat_type invalide'}),400
            if cfg['group']=='pitching':
                pitcher=next((p for p in mlb_get_pitchers() if names_match(player_name,p['name'])),None)
                if pitcher: games=mlb_get_gamelog(pitcher['id'],stat_type)
                else:
                    pid=mlb_search_player(player_name)
                    if pid: games=mlb_get_gamelog(pid,stat_type)
            else:
                pid=mlb_search_player(player_name)
                if pid: games=mlb_get_gamelog(pid,stat_type)
        elif sport=='nhl':
            pid=nhl_search_player(player_name)
            if pid: games=nhl_get_skater_gamelog(pid)
        elif sport=='nba':
            stat_col={'nba_points':'pts','nba_rebounds':'reb','nba_assists':'ast'}.get(stat_type,'pts')
            pid=nba_search_player(player_name)
            if pid: games=nba_get_gamelog(pid,stat_col)
        elif sport=='tennis':
            games=tennis_get_aces(player_name)
        if not games:
            return jsonify({'status':'NOT_FOUND','message':f'Aucune donnée pour {player_name}'}),404
        if date_str:
            target=date_str[:10]
            exact=[g for g in games if g.get('date','')[:10]==target]
            if exact:
                return jsonify(to_python({'status':'SUCCESS','player':player_name,
                    'stat_type':stat_type,'date':target,'actual_value':exact[0]['stat'],'found':'exact'}))
            for g in sorted(games,key=lambda x:x.get('date','')):
                if g.get('date','')[:10]>=target:
                    return jsonify(to_python({'status':'SUCCESS','player':player_name,
                        'stat_type':stat_type,'date':g['date'][:10],'actual_value':g['stat'],'found':'closest'}))
        return jsonify(to_python({'status':'SUCCESS','player':player_name,
            'stat_type':stat_type,'date':games[0].get('date','')[:10],
            'actual_value':games[0]['stat'],'found':'latest'}))
    except Exception as e:
        import traceback
        return jsonify({'status':'ERROR','message':str(e),'trace':traceback.format_exc()[:500]}),500


@app.route('/api/backtest', methods=['GET'])
def run_backtest():
    try:
        player_name=request.args.get('player','')
        stat_type=request.args.get('stat_type','pitcher_strikeouts')
        sport=request.args.get('sport','mlb')
        if not player_name: return jsonify({'status':'ERROR','message':'player requis'}),400
        if stat_type not in STAT_CONFIG: return jsonify({'status':'ERROR','message':'stat_type invalide'}),400
        games=None
        if sport=='mlb':
            cfg=STAT_CONFIG[stat_type]
            if cfg['group']=='pitching':
                pitcher=next((p for p in mlb_get_pitchers() if names_match(player_name,p['name'])),None)
                if pitcher: games=mlb_get_gamelog(pitcher['id'],stat_type)
                else:
                    pid=mlb_search_player(player_name)
                    if pid: games=mlb_get_gamelog(pid,stat_type)
            else:
                pid=mlb_search_player(player_name)
                if pid: games=mlb_get_gamelog(pid,stat_type)
        elif sport=='nhl':
            pid=nhl_search_player(player_name)
            if pid: games=nhl_get_skater_gamelog(pid)
        elif sport=='nba':
            stat_col={'nba_points':'pts','nba_rebounds':'reb','nba_assists':'ast'}.get(stat_type,'pts')
            pid=nba_search_player(player_name)
            if pid: games=nba_get_gamelog(pid,stat_col)
        elif sport=='tennis':
            games=tennis_get_aces(player_name)
        if not games:
            return jsonify({'status':'ERROR','message':f'Aucune donnée pour {player_name}'}),404
        result=backtest(games,stat_type)
        if not result:
            return jsonify({'status':'ERROR','message':'Pas assez de données (min 13 matchs)'}),400
        return jsonify(to_python({'status':'SUCCESS','player':player_name,'stat_type':stat_type,
            'sport':sport,'total_games_available':len(games),'backtest':result}))
    except Exception as e:
        import traceback
        return jsonify({'status':'ERROR','message':str(e),'trace':traceback.format_exc()[:500]}),500


@app.route('/api/mlb/schedule', methods=['GET'])
def mlb_schedule():
    p=mlb_get_pitchers()
    return jsonify({'status':'SUCCESS','pitchers':p,'count':len(p)})

@app.route('/api/odds/usage', methods=['GET'])
def usage():
    try:
        r=requests.get(f"{ODDS_BASE}/sports",params={'apiKey':ODDS_API_KEY},timeout=10)
        return jsonify({'used':r.headers.get('x-requests-used','N/A'),
                        'remaining':r.headers.get('x-requests-remaining','N/A')})
    except Exception as e: return jsonify({'error':str(e)}),500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status':'healthy','version':'2.6','mlb_season':CURRENT_MLB_SEASON,
        'nba_season':NBA_SEASON,'sports':list(SPORT_STATS.keys()),
        'debug':'/api/debug-nba + /api/debug-dk'})

@app.route('/', methods=['GET'])
def home():
    return jsonify({'app':'Multi-Sport Analyzer','version':'2.6',
                    'sports':list(SPORT_STATS.keys())})

if __name__ == '__main__':
    port=int(os.environ.get('PORT',5000))
    print("🎰 Multi-Sport Analyzer v2.6 — MLB|NBA|NHL|Tennis|Golf",flush=True)
    app.run(host='0.0.0.0',port=port,debug=False)
