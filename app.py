# app.py — Orchestrateur slim v4.0
from flask import Flask, jsonify, request
from flask_cors import CORS
import os, time
from stats import to_py
from mlb     import mlb_bp,     mlb_actual_result
from nba     import nba_bp,     nba_actual_result
from nhl     import nhl_bp,     nhl_actual_result
from tennis  import tennis_bp,  tennis_actual_result
from soccer  import soccer_bp

app = Flask(__name__)
CORS(app)

app.register_blueprint(mlb_bp)
app.register_blueprint(nba_bp)
app.register_blueprint(nhl_bp)
app.register_blueprint(tennis_bp)
app.register_blueprint(soccer_bp)

@app.route('/api/actual-result', methods=['GET'])
def actual_result():
    """Route partagée pour résoudre les bets sauvegardés."""
    try:
        player    = request.args.get('player','').strip()
        stat_type = request.args.get('stat_type','')
        sport     = request.args.get('sport','mlb').lower()
        date_str  = request.args.get('date','')
        if not player or not stat_type:
            return jsonify({'status':'ERROR','message':'player et stat_type requis'}),400

        dispatch = {
            'mlb':    mlb_actual_result,
            'nba':    nba_actual_result,
            'nhl':    nhl_actual_result,
            'tennis': tennis_actual_result,
        }
        fn = dispatch.get(sport)
        if not fn:
            return jsonify({'status':'ERROR','message':f'Sport non supporté: {sport}'}),400
        return fn(player, stat_type, date_str)
    except Exception as e:
        return jsonify({'status':'ERROR','message':str(e)}),500

@app.route('/api/odds/usage', methods=['GET'])
def odds_usage():
    import requests as req, os
    try:
        r = req.get('https://api.the-odds-api.com/v4/sports',
                    params={'apiKey': os.environ.get('ODDS_API_KEY','')}, timeout=8)
        return jsonify({'used': r.headers.get('x-requests-used','?'),
                        'remaining': r.headers.get('x-requests-remaining','?')})
    except Exception as e:
        return jsonify({'error': str(e)}),500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status':'healthy','version':'4.0',
                    'sports':['mlb','nba','nhl','tennis','soccer'],
                    'model':'Empirical hit rate + EV + Kelly 25%',
                    'removed':['scipy','nba_api','normality_tests']})

@app.route('/')
def home():
    return jsonify({'app':'Multi-Sport Analyzer','version':'4.0'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT',5000))
    print('Multi-Sport Analyzer v4.0', flush=True)
    app.run(host='0.0.0.0', port=port, debug=False)
