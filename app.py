# app.py v4.3 — Soccer intégré comme 5ème sport dans le scanner
from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from datetime import datetime
from stats import to_py

from mlb    import scan_mlb,    mlb_actual_result, get_pitchers, get_gamelog as mlb_gamelog
from stats import walk_forward_backtest
from nba    import scan_nba,    nba_actual_result
from nhl    import scan_nhl,    nhl_actual_result
from tennis import scan_tennis, tennis_actual_result
from soccer import soccer_bp,   scan_soccer

app = Flask(__name__)
CORS(app)
app.register_blueprint(soccer_bp)

@app.route('/api/daily-opportunities', methods=['GET'])
def daily_opportunities():
    try:
        sport    = request.args.get('sport','mlb').lower()
        sf       = request.args.get('stat_type') or None
        min_ev   = float(request.args.get('min_edge', 3))
        scan_time = datetime.now().strftime('%Y-%m-%d %H:%M')

        if sport == 'mlb':
            opps, analyzed, ngames = scan_mlb(sf, min_ev)
        elif sport == 'nba':
            opps, analyzed, ngames = scan_nba(sf, min_ev)
            if not ngames:
                return jsonify({'status':'INFO','sport':'nba',
                    'message':'🏀 NBA: Aucun marché trouvé.','opportunities':[]})
        elif sport == 'nhl':
            opps, analyzed, ngames = scan_nhl(min_ev)
            if opps and isinstance(opps[0],dict) and opps[0].get('_info'):
                return jsonify({'status':'INFO','sport':'nhl',
                    'message':opps[0]['message'],'opportunities':[]})
        elif sport == 'tennis':
            opps, analyzed, ngames = scan_tennis(min_ev)
            if opps and isinstance(opps[0],dict) and opps[0].get('_info'):
                return jsonify({'status':'INFO','sport':'tennis',
                    'message':opps[0]['message'],'opportunities':[]})
        elif sport == 'soccer':
            opps, analyzed, ngames = scan_soccer(min_ev)
            if not ngames:
                return jsonify({'status':'INFO','sport':'soccer',
                    'message':'⚽ Soccer: Aucun match trouvé sur The Odds API.','opportunities':[]})
        else:
            return jsonify({'status':'ERROR',
                'message':f'Sport inconnu: {sport}'}), 400

        return jsonify(to_py({
            'status':'SUCCESS','sport':sport,'version':'4.3',
            'total_games':ngames,'players_analyzed':analyzed,
            'opportunities_found':len(opps),'opportunities':opps,
            'scan_time':scan_time
        }))
    except Exception as e:
        import traceback
        return jsonify({'status':'ERROR','message':str(e),
                        'trace':traceback.format_exc()[:600]}), 500

@app.route('/api/actual-result', methods=['GET'])
def actual_result():
    try:
        player    = request.args.get('player','').strip()
        stat_type = request.args.get('stat_type','')
        sport     = request.args.get('sport','mlb').lower()
        date_str  = request.args.get('date','')
        if not player or not stat_type:
            return jsonify({'status':'ERROR','message':'player et stat_type requis'}), 400
        dispatch = {'mlb':mlb_actual_result,'nba':nba_actual_result,
                    'nhl':nhl_actual_result,'tennis':tennis_actual_result}
        fn = dispatch.get(sport)
        if not fn:
            return jsonify({'status':'ERROR','message':f'Sport non supporté: {sport}'}), 400
        return fn(player, stat_type, date_str)
    except Exception as e:
        return jsonify({'status':'ERROR','message':str(e)}), 500

@app.route('/api/odds/usage', methods=['GET'])
def odds_usage():
    import requests as req
    try:
        r = req.get('https://api.the-odds-api.com/v4/sports',
                    params={'apiKey':os.environ.get('ODDS_API_KEY','')}, timeout=8)
        return jsonify({'used':r.headers.get('x-requests-used','?'),
                        'remaining':r.headers.get('x-requests-remaining','?')})
    except Exception as e:
        return jsonify({'error':str(e)}), 500

@app.route('/api/backtest', methods=['GET'])
def run_backtest():
    try:
        sport     = request.args.get('sport','mlb').lower()
        stat_type = request.args.get('stat_type','pitcher_strikeouts')
        player    = request.args.get('player','').strip()

        if sport != 'mlb':
            return jsonify({'status':'ERROR','message':'Backtest MLB seulement pour linstant'}), 400

        results_all = []
        pitchers = get_pitchers()

        # Si joueur précis demandé
        targets = []
        if player:
            import re
            def _norm(n): return re.sub(r'[^a-z ]','',n.lower().strip())
            def _match(a,b):
                na,nb=_norm(a),_norm(b)
                if na==nb: return True
                pa,pb=na.split(),nb.split()
                return bool(pa and pb and pa[-1]==pb[-1] and pa[0][0]==pb[0][0])
            p = next((p for p in pitchers if _match(player,p['name'])),None)
            if p: targets = [p]
        else:
            targets = pitchers[:12]  # Top 12 lanceurs du jour

        for p in targets:
            games = mlb_gamelog(p['id'], stat_type)
            if not games or len(games) < 15: continue
            bt = walk_forward_backtest(games)
            if not bt: continue
            results_all.append({
                'player':  p['name'],
                'games_available': len(games),
                'backtest': bt
            })

        if not results_all:
            return jsonify({'status':'INFO',
                'message':'Pas assez de données (min 15 matchs par lanceur).'})

        # Agrégat global
        total_bets = sum(r['backtest']['total_bets'] for r in results_all)
        total_wins = sum(r['backtest']['wins'] for r in results_all)
        total_profit = sum(r['backtest']['profit_usd'] for r in results_all)

        return jsonify(to_py({
            'status':'SUCCESS','sport':sport,'stat_type':stat_type,
            'players_backtested': len(results_all),
            'aggregate': {
                'total_bets':  total_bets,
                'total_wins':  total_wins,
                'hit_rate':    round(total_wins/total_bets*100,1) if total_bets else 0,
                'profit_usd':  round(total_profit,2),
                'roi_pct':     round(total_profit/(total_bets*10)*100,1) if total_bets else 0,
                'breakeven':   52.4,
            },
            'players': results_all
        }))
    except Exception as e:
        import traceback
        return jsonify({'status':'ERROR','message':str(e),
                        'trace':traceback.format_exc()[:600]}),500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status':'healthy','version':'4.3',
                    'sports':['mlb','nba','nhl','tennis','soccer']})

@app.route('/')
def home():
    return jsonify({'app':'Multi-Sport Analyzer','version':'4.3'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print('Multi-Sport Analyzer v4.3', flush=True)
    app.run(host='0.0.0.0', port=port, debug=False)
