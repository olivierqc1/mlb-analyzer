# stats.py — Fonctions d'analyse partagées (hit rate empirique, EV, Kelly)
# Remplace scipy : plus simple, plus honnête, pas moins fiable
import math, numpy as np

def to_py(obj):
    if isinstance(obj, dict):  return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [to_py(v) for v in obj]
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj); return 0.0 if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.bool_): return bool(obj)
    return obj

def analyze(games, line):
    """
    games : list[{'date': str, 'stat': float}] triée récent→ancien
    Retourne dict compatible frontend ou None si pas assez de données.
    Modèle : hit rate empirique + moyenne pondérée + trend
    """
    if len(games) < 5: return None
    vals = [float(g['stat']) for g in games]
    n    = len(vals)
    line = float(line)

    # Hit rate (Over = stat > line, pas >=)
    over_n  = sum(1 for v in vals if v > line)
    under_n = n - over_n
    hr      = over_n / n

    # Moyennes
    mean = sum(vals) / n
    l5   = sum(vals[:5]) / min(5, n)
    l10  = sum(vals[:min(10,n)]) / min(10, n)

    # Moyenne pondérée exponentielle (récent = plus de poids)
    ws    = [0.90 ** i for i in range(n)]
    wmean = sum(v*w for v,w in zip(vals,ws)) / sum(ws)

    # Trend : derniers 5 vs global
    l5_hr = sum(1 for v in vals[:5] if v > line) / min(5, n)
    if   l5_hr > hr + 0.10: trend = 'up'
    elif l5_hr < hr - 0.10: trend = 'down'
    else:                   trend = 'stable'

    # Recommandation
    rec      = 'OVER' if hr >= 0.5 else 'UNDER'
    rec_prob = hr if rec == 'OVER' else (1 - hr)

    # Edge vs -110 (baseline)
    implied = 0.5238
    edge    = round((rec_prob - implied) * 100, 1)

    # Grade (basé sur sample size + margin + trend)
    margin   = abs(hr - 0.5)
    trend_ok = not (rec == 'OVER' and trend == 'down') \
               and not (rec == 'UNDER' and trend == 'up')

    if   n >= 15 and margin >= 0.15 and trend_ok: grade, color = 'A', '#4ade80'
    elif n >= 10 and margin >= 0.10:               grade, color = 'B', '#86efac'
    elif n >= 7  and margin >= 0.07:               grade, color = 'C', '#fbbf24'
    else:                                          grade, color = 'AVOID', '#f87171'

    # Label
    labels = {'A':'🟢 BET — Signal solide', 'B':'🟢 BET — Signal acceptable',
              'C':'🟡 PRUDENCE — Signal faible', 'AVOID':'🔴 ÉVITER'}

    # Pros/issues pour le frontend
    pros, issues = [], []
    if n >= 15: pros.append(f'✅ {n} matchs analysés')
    elif n >= 10: pros.append(f'🟡 {n} matchs')
    else: issues.append(f'❌ Seulement {n} matchs')
    hr_pct = round(hr*100,1)
    rec_pct = round((hr if rec=='OVER' else 1-hr)*100,1)
    rec_n   = over_n if rec=='OVER' else under_n
    if margin >= 0.15: pros.append(f'✅ Hit rate {rec}  {rec_pct}% ({rec_n}/{n})')
    elif margin >= 0.07: pros.append(f'🟡 Hit rate {rec}  {rec_pct}% ({rec_n}/{n})')
    else: issues.append(f'❌ Hit rate trop proche de 50% ({rec_pct}%)')
    if trend_ok and trend != 'stable': pros.append(f'✅ Trend récent confirme ({trend})')
    elif not trend_ok: issues.append(f'⚠️ Trend récent contredit la reco ({trend})')
    if grade == 'AVOID': issues.append('❌ Signal insuffisant')

    # Kelly 25%
    kelly = round(max(0, (rec_prob - implied) / (1 - implied)) * 25, 1) \
            if rec_prob > implied else 0.0

    # Standard deviation + consistency
    import math as _math
    std = _math.sqrt(sum((v-mean)**2 for v in vals)/n) if n>1 else 0.0
    cons = round(max(0, 100-(std/mean*100)),1) if mean>0 else 50.0

    return {
        'n': n, 'mean': round(mean,2), 'wmean': round(wmean,2),
        'std': round(std,2), 'cons': round(cons,1),
        'l5': round(l5,2), 'l10': round(l10,2),
        'hit_rate': round(hr,4),
        'rec_hit_rate': round(hr if rec=='OVER' else 1-hr, 4),
        'over_n': over_n, 'under_n': under_n,
        'rec': rec, 'rec_prob': round(rec_prob,4),
        'trend': trend, 'l5_hr': round(l5_hr,4),
        'grade': grade, 'color': color, 'label': labels[grade],
        'edge': edge, 'kelly': kelly,
        'line': line,
        'quality': {'grade':grade, 'color':color, 'label':labels[grade],
                    'pros':pros, 'issues':issues},
        'recent': [{'date':g.get('date','')[:10],'stat':g['stat']} for g in games[:10]]
    }

def calc_ev(prob, american):
    try:
        a = float(american)
        dec = a/100+1 if a >= 0 else 100/abs(a)+1
        return round((prob * dec - 1) * 100, 2)
    except: return None

def calc_kelly(prob, american, fraction=0.25):
    try:
        a = float(american)
        dec = a/100+1 if a >= 0 else 100/abs(a)+1
        b = dec-1; q = 1-prob
        return round(max(0,(prob*b-q)/b)*fraction*100, 2)
    except: return 0.0

def build_opp(player, sport, stat_type, stat_label, line, book, odds_price,
              game_info, a):
    """Construit le dict opportunity standard pour le frontend."""
    rec   = a['rec']
    prob  = a['rec_prob']
    ev    = calc_ev(prob, odds_price)
    kelly = calc_kelly(prob, odds_price)
    return {
        'player': player, 'sport': sport,
        'stat_type': stat_type, 'stat_label': stat_label,
        'game_info': game_info,
        'quality': a['quality'],
        'line_analysis': {
            'bookmaker_line': line, 'bookmaker': book,
            'recommendation': rec,
            'edge': ev if ev is not None else a['edge'],
            'over_probability': round(a['hit_rate']*100,1),
            'under_probability': round((1-a['hit_rate'])*100,1),
            'kelly_criterion': kelly,
            'actual_odds': odds_price,
        },
        'deep_stats': {
            'mean': a['mean'], 'weighted_mean': a['wmean'],
            'std': a.get('std',0), 'consistency': a.get('cons',50),
            'avg_last_5': a['l5'], 'avg_last_10': a['l10'],
            'hit_rate': round(a['hit_rate']*100,1),
            'rec_hit_rate': round(a['rec_hit_rate']*100,1),
            'over_count': a['over_n'], 'under_count': a['under_n'],
            'games_analyzed': a['n'], 'trend': a['trend'],
        },
        'statistical_validation': {'is_reliable': a['grade'] in ('A','B')},
        'recent_games': a['recent'],
    }


def walk_forward_backtest(games, min_train=12, stake=10.0):
    """
    Walk-forward backtest sur données historiques.
    games : list triée récent→ancien (index 0 = plus récent)
    On inverse pour avoir chrono, train sur premiers N, prédit N+1.
    """
    if len(games) < min_train + 3:
        return None

    chrono = list(reversed(games))  # ancien → récent
    results = []
    profit  = 0.0

    for i in range(min_train, len(chrono)):
        train  = list(reversed(chrono[:i]))   # récent→ancien pour analyze
        actual = chrono[i]['stat']

        # Ligne simulée = médiane du train set (faute de lignes historiques)
        vals_train = sorted([g['stat'] for g in train])
        n_t = len(vals_train)
        sim_line = (vals_train[n_t//2-1]+vals_train[n_t//2])/2 if n_t%2==0 else float(vals_train[n_t//2])

        a = analyze(train, sim_line)
        if not a or a['rec'] == 'SKIP' or a['grade'] == 'AVOID':
            continue

        correct = (a['rec'] == 'OVER'  and actual > sim_line) or                   (a['rec'] == 'UNDER' and actual < sim_line)
        pnl = stake*(100/110) if correct else -stake
        profit += pnl

        results.append({
            'game_num':   i,
            'date':       chrono[i].get('date','')[:10],
            'actual':     actual,
            'sim_line':   round(sim_line, 1),
            'rec':        a['rec'],
            'correct':    correct,
            'grade':      a['grade'],
            'hit_rate':   round(a['hit_rate']*100, 1),
            'pnl':        round(pnl, 2),
        })

    if not results:
        return None

    n     = len(results)
    wins  = sum(1 for r in results if r['correct'])
    ab    = [r for r in results if r['grade'] in ('A','B')]
    ab_w  = sum(1 for r in ab if r['correct'])

    return {
        'total_bets':       n,
        'wins':             wins,
        'losses':           n - wins,
        'hit_rate':         round(wins/n*100, 1),
        'profit_usd':       round(profit, 2),
        'roi_pct':          round(profit/(n*stake)*100, 1),
        'grade_ab_bets':    len(ab),
        'grade_ab_hit_rate':round(ab_w/len(ab)*100,1) if ab else 0,
        'grade_ab_profit':  round(sum(r['pnl'] for r in ab), 2),
        'breakeven':        52.4,
        'per_bet':          results[-15:],  # 15 derniers
    }
