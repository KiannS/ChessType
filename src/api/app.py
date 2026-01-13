from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
import sys
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app with static folder configuration
STATIC_FOLDER = PROJECT_ROOT / 'static'
TEMPLATE_FOLDER = PROJECT_ROOT / 'templates'

app = Flask(__name__,
            static_folder=str(STATIC_FOLDER),
            static_url_path='/static',
            template_folder=str(TEMPLATE_FOLDER) if TEMPLATE_FOLDER.exists() else None)

# Try to import the analyzer - try different possible locations
analyzer = None
try:
    # Try importing from analysis.py if ChessAnalyzer is there
    from src.api.analysis import ChessAnalyzer
    analyzer = ChessAnalyzer()
    logger.info("Chess Analyzer initialized successfully from src.api.analysis")
except ImportError:
    try:
        # Try importing from inference folder
        from src.inference.chess_analyzer import ChessAnalyzer
        analyzer = ChessAnalyzer()
        logger.info("Chess Analyzer initialized successfully from src.inference.chess_analyzer")
    except ImportError as e:
        logger.warning(f"ChessAnalyzer not found: {e}")
        logger.warning("Running in demo mode - analysis endpoints will return mock data")
except Exception as e:
    logger.error(f"Failed to initialize Chess Analyzer: {e}", exc_info=True)

# Log static folder configuration
if STATIC_FOLDER.exists():
    logger.info(f"Static folder found: {STATIC_FOLDER}")
    logger.info(f"Files: {list(STATIC_FOLDER.glob('*'))}")
else:
    logger.warning(f"Static folder not found: {STATIC_FOLDER}")


@app.route('/')
def home():
    """Serve the main HTML page"""
    try:
        # Try to serve index.html from static folder
        if (STATIC_FOLDER / 'index.html').exists():
            return send_from_directory(str(STATIC_FOLDER), 'index.html')
        else:
            return jsonify({
                "message": "ChessType API is running",
                "status": "healthy",
                "mode": "demo" if not analyzer else "full",
                "endpoints": {
                    "analyze_lichess": "POST /analyze-lichess",
                    "analyze_chesscom": "POST /analyze-chesscom",
                    "analyze": "GET/POST /analyze",
                    "health": "GET /health"
                }
            })
    except Exception as e:
        logger.error(f"Error serving home page: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/static/<path:filename>')
def serve_static(filename):
    """Explicitly serve static files"""
    try:
        return send_from_directory(str(STATIC_FOLDER), filename)
    except Exception as e:
        logger.error(f"Error serving static file {filename}: {e}")
        return jsonify({"error": "File not found"}), 404


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "analyzer_loaded": analyzer is not None,
        "mode": "demo" if not analyzer else "full",
        "static_folder": str(STATIC_FOLDER),
        "static_folder_exists": STATIC_FOLDER.exists()
    })


def get_mock_analysis(username, platform):
    """Return mock analysis data when analyzer is not available"""
    return {
        "username": username,
        "platform": platform,
        "games_analyzed": 50,
        "style_analysis": {
            "predicted_style": "Tactical",
            "confidence": 0.75,
            "style_probabilities": {
                "Tactical": 0.35,
                "Aggressive": 0.25,
                "Positional": 0.20,
                "Solid": 0.15,
                "Balanced": 0.05
            },
            "description": "You prefer sharp, tactical positions with concrete calculations.",
            "strengths": ["Pattern recognition", "Calculation", "Tactics"],
            "famous_players": ["Mikhail Tal", "Garry Kasparov"]
        },
        "position_analysis": {
            "performance_by_type": {
                "Tactical": {"win_rate": 0.58, "positions_encountered": 150, "rating": "Strong"},
                "Sharp": {"win_rate": 0.52, "positions_encountered": 120, "rating": "Average"},
                "Quiet": {"win_rate": 0.63, "positions_encountered": 80, "rating": "Excellent"}
            },
            "best_position_type": {"type": "Quiet", "win_rate": 0.63},
            "worst_position_type": {"type": "Sharp", "win_rate": 0.52},
            "position_distribution": {
                "Tactical": 0.30,
                "Sharp": 0.25,
                "Quiet": 0.20,
                "Chaotic": 0.15,
                "Endgame": 0.10
            }
        },
        "recommendations": [
            "Focus on improving tactical pattern recognition",
            "Practice endgame positions - high win rate suggests strength",
            "Work on handling chaotic positions more effectively"
        ],
        "statistics": {
            "total_moves_analyzed": 2500,
            "avg_game_length": 50,
            "avg_captures_per_game": 12,
            "early_castling_rate": 75,
            "aggressive_openings_pct": 45,
            "favorite_openings": ["Sicilian Defense", "King's Indian Defense", "Queen's Gambit"]
        },
        "note": "This is MOCK DATA for demonstration purposes"
    }


@app.route('/analyze-lichess', methods=['POST'])
def analyze_lichess():
    """Analyze a Lichess player's games"""
    try:
        data = request.get_json()
        username = data.get('username')
        max_games = data.get('max_games', 50)
        
        if not username:
            return jsonify({"error": "Username is required"}), 400
        
        logger.info(f"Analyzing Lichess player: {username}")
        
        if analyzer:
            # Use the actual analyzer - check which method it has
            if hasattr(analyzer, 'analyze_lichess_player'):
                result = analyzer.analyze_lichess_player(username, max_games)
            elif hasattr(analyzer, 'analyze_player'):
                result = analyzer.analyze_player(username, platform='lichess', max_games=max_games)
            else:
                result = get_mock_analysis(username, 'lichess')
        else:
            result = get_mock_analysis(username, 'lichess')
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error analyzing Lichess player: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/analyze-chesscom', methods=['POST'])
def analyze_chesscom():
    """Analyze a Chess.com player's games"""
    try:
        data = request.get_json()
        username = data.get('username')
        year = data.get('year')
        month = data.get('month')
        max_games = data.get('max_games', 50)
        
        if not username:
            return jsonify({"error": "Username is required"}), 400
        
        logger.info(f"Analyzing Chess.com player: {username}")
        
        if analyzer:
            # Use the actual analyzer - check which method it has
            if hasattr(analyzer, 'analyze_chesscom_player'):
                if not year or not month:
                    from datetime import datetime
                    now = datetime.now()
                    year = year or now.year
                    month = month or now.month
                result = analyzer.analyze_chesscom_player(username, year, month, max_games)
            elif hasattr(analyzer, 'analyze_player'):
                result = analyzer.analyze_player(
                    username, 
                    platform='chesscom', 
                    year=year,
                    month=month,
                    max_games=max_games
                )
            else:
                result = get_mock_analysis(username, 'chesscom')
        else:
            result = get_mock_analysis(username, 'chesscom')
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error analyzing Chess.com player: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/analyze', methods=['POST', 'GET'])
def analyze():
    """Universal analyze endpoint that handles both platforms"""
    try:
        # Handle GET requests (from URL parameters)
        if request.method == 'GET':
            username = request.args.get('username')
            platform = request.args.get('platform', 'lichess')
            max_games = int(request.args.get('max_games', 50))
            year = request.args.get('year')
            month = request.args.get('month')
        # Handle POST requests (from JSON body)
        else:
            data = request.get_json()
            username = data.get('username')
            platform = data.get('platform', 'lichess')
            max_games = data.get('max_games', 50)
            year = data.get('year')
            month = data.get('month')
        
        if not username:
            return jsonify({"error": "Username is required"}), 400
        
        logger.info(f"Analyzing {platform} player: {username}")
        
        # Route to appropriate endpoint based on platform
        if platform.lower() == 'lichess':
            if analyzer and hasattr(analyzer, 'analyze_lichess_player'):
                result = analyzer.analyze_lichess_player(username, max_games)
            elif analyzer and hasattr(analyzer, 'analyze_player'):
                result = analyzer.analyze_player(username, platform='lichess', max_games=max_games)
            else:
                result = get_mock_analysis(username, 'lichess')
        else:
            if analyzer and hasattr(analyzer, 'analyze_chesscom_player'):
                if not year or not month:
                    from datetime import datetime
                    now = datetime.now()
                    year = year or now.year
                    month = month or now.month
                result = analyzer.analyze_chesscom_player(username, year, month, max_games)
            elif analyzer and hasattr(analyzer, 'analyze_player'):
                result = analyzer.analyze_player(
                    username,
                    platform='chesscom',
                    year=year,
                    month=month,
                    max_games=max_games
                )
            else:
                result = get_mock_analysis(username, 'chesscom')
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error analyzing player: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    logger.info("Starting ChessType API server...")
    logger.info(f"Mode: {'DEMO (mock data)' if not analyzer else 'FULL (real analysis)'}")
    app.run(debug=True, host='0.0.0.0', port=8000)