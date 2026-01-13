from flask import Flask, request, jsonify
from pathlib import Path
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.api.analysis import ChessAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize analyzer
try:
    analyzer = ChessAnalyzer()
    logger.info("Chess Analyzer initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize analyzer: {e}")
    analyzer = None


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    if analyzer is None:
        return jsonify({
            'status': 'unhealthy',
            'error': 'Analyzer not initialized'
        }), 500
    
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'style_classifier': analyzer.style_predictor is not None,
            'position_classifier': analyzer.position_classifier is not None
        }
    })


@app.route('/analyze-lichess', methods=['POST'])
def analyze_lichess():
    """
    Analyze a Lichess player's games.
    
    Request body:
    {
        "username": "DrNykterstein",
        "max_games": 50
    }
    """
    if analyzer is None:
        return jsonify({'error': 'Analyzer not initialized'}), 500
    
    data = request.get_json()
    
    if not data or 'username' not in data:
        return jsonify({'error': 'Username is required'}), 400
    
    username = data['username']
    max_games = data.get('max_games', 50)
    
    try:
        logger.info(f"Analyzing Lichess player: {username}")
        result = analyzer.analyze_lichess_player(username, max_games)
        
        if 'error' in result:
            return jsonify(result), 404
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error analyzing {username}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/analyze-chesscom', methods=['POST'])
def analyze_chesscom():
    """
    Analyze a Chess.com player's games.
    
    Request body:
    {
        "username": "hikaru",
        "year": 2024,
        "month": 1,
        "max_games": 50
    }
    """
    if analyzer is None:
        return jsonify({'error': 'Analyzer not initialized'}), 500
    
    data = request.get_json()
    
    if not data or 'username' not in data:
        return jsonify({'error': 'Username is required'}), 400
    
    username = data['username']
    year = data.get('year', 2024)
    month = data.get('month', 1)
    max_games = data.get('max_games', 50)
    
    try:
        logger.info(f"Analyzing Chess.com player: {username}")
        result = analyzer.analyze_chesscom_player(username, year, month, max_games)
        
        if 'error' in result:
            return jsonify(result), 404
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error analyzing {username}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/analyze-pgn', methods=['POST'])
def analyze_pgn():
    """
    Analyze games from uploaded PGN file.
    
    Form data:
    - file: PGN file
    - player_name: Name of player to analyze (optional)
    """
    if analyzer is None:
        return jsonify({'error': 'Analyzer not initialized'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    player_name = request.form.get('player_name', 'Player')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.pgn'):
        return jsonify({'error': 'File must be a PGN file'}), 400
    
    try:
        # Save uploaded file temporarily
        temp_dir = Path('temp')
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / file.filename
        file.save(temp_path)
        
        logger.info(f"Analyzing PGN file: {file.filename}")
        result = analyzer.analyze_pgn_file(str(temp_path), player_name)
        
        # Clean up
        temp_path.unlink()
        
        if 'error' in result:
            return jsonify(result), 404
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error analyzing PGN: {e}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)