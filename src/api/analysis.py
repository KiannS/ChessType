import tempfile
from pathlib import Path
from typing import Dict, List, Optional
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data_collection.fetch_games import GameFetcher
from src.data_collection.parse_pgn import PGNParser
from src.feature_engineering.game_features import GameFeatureExtractor
from src.feature_engineering.position_features import PositionFeatureExtractor
from src.inference.style_predictor import StylePredictor
from src.inference.position_classifier import PositionClassifier
from src.api.recommendations import RecommendationEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChessAnalyzer:
    """Main analyzer that orchestrates the analysis pipeline."""
    
    def __init__(self):
        """Initialize all components."""
        self.game_fetcher = GameFetcher()
        self.parser = PGNParser()
        self.game_extractor = GameFeatureExtractor()
        self.position_extractor = PositionFeatureExtractor()
        
        try:
            self.style_predictor = StylePredictor()
            self.position_classifier = PositionClassifier()
        except FileNotFoundError as e:
            logger.error(f"Models not found: {e}")
            raise
        
        self.recommendation_engine = RecommendationEngine()
    
    def analyze_lichess_player(self, username: str, max_games: int = 50) -> Dict:
        """Analyze a Lichess player."""
        logger.info(f"Fetching games for {username} from Lichess...")
        
        # Fetch games to temporary file
        with tempfile.TemporaryDirectory() as temp_dir:
            fetcher = GameFetcher(output_dir=temp_dir)
            pgn_path = fetcher.fetch_lichess_games(username, max_games)
            
            if not pgn_path:
                return {'error': f'Could not fetch games for {username}'}
            
            return self.analyze_pgn_file(pgn_path, username)
    
    def analyze_chesscom_player(
        self,
        username: str,
        year: Optional[int] = None,
        month: Optional[int] = None,
        max_games: int = 50
    ) -> Dict:
        """
        Analyze a Chess.com player.
        
        Args:
            username: Chess.com username
            year: Year (e.g., 2024) - None or 0 for all-time
            month: Month (1-12) - None or 0 for all-time
            max_games: Maximum number of games to analyze
            
        Returns:
            Analysis results dictionary
        """
        # Check if this is an all-time request
        is_all_time = (year is None or year == 0 or 
                       month is None or month == 0 or
                       str(year).lower() == 'all' or str(month).lower() == 'all')
        
        if is_all_time:
            logger.info(f"Fetching ALL-TIME games for {username} from Chess.com...")
        else:
            logger.info(f"Fetching games for {username} from Chess.com ({year}-{month:02d})...")
        
        # Fetch games to temporary file
        with tempfile.TemporaryDirectory() as temp_dir:
            fetcher = GameFetcher(output_dir=temp_dir)
            
            # FIXED: Use all-time method when year/month not specified
            if is_all_time:
                pgn_path = fetcher.fetch_chesscom_all_games(username, max_games)
            else:
                pgn_path = fetcher.fetch_chesscom_games(username, year, month, max_games)
            
            if not pgn_path:
                return {'error': f'Could not fetch games for {username}. Please check the username is correct.'}
            
            return self.analyze_pgn_file(pgn_path, username)
    
    def analyze_pgn_file(self, pgn_path: str, player_name: str) -> Dict:
        """
        Main analysis pipeline for a PGN file.
        
        Args:
            pgn_path: Path to PGN file
            player_name: Name of player to analyze
            
        Returns:
            Complete analysis results
        """
        logger.info(f"Parsing games from {pgn_path}...")
        games = self.parser.parse_pgn_file(pgn_path)
        
        if not games:
            return {'error': 'No games found in file'}
        
        logger.info(f"Analyzing {len(games)} games for {player_name}...")
        
        # Extract game-level features
        game_features_list = self.game_extractor.extract_features_from_games(
            games, player_name
        )
        
        if not game_features_list:
            return {'error': f'Could not extract features for {player_name}'}
        
        # Aggregate game features
        aggregated_features = self.game_extractor.aggregate_features(game_features_list)
        
        # Predict playing style
        style_result = self.style_predictor.predict(aggregated_features)
        style_description = self.style_predictor.get_style_description(
            style_result['predicted_style']
        )
        
        # Analyze positions
        position_analysis = self._analyze_positions(games, player_name)
        
        # Generate recommendations
        recommendations = self.recommendation_engine.generate_recommendations(
            style_result['predicted_style'],
            position_analysis
        )
        
        # Compile statistics
        statistics = self._compile_statistics(games, game_features_list, player_name)
        
        # Build response
        response = {
            'username': player_name,
            'games_analyzed': len(games),
            
            'style_analysis': {
                'predicted_style': style_result['predicted_style'],
                'confidence': style_result['confidence'],
                'style_probabilities': style_result['style_probabilities'],
                'description': style_description['description'],
                'strengths': style_description['strengths'],
                'famous_players': style_description['famous_players']
            },
            
            'position_analysis': position_analysis,
            'recommendations': recommendations,
            'statistics': statistics
        }
        
        return response
    
    def _analyze_positions(self, games: List, player_name: str) -> Dict:
        """Analyze position types and performance."""
        all_classifications = []
        position_performance = {}
        
        for game in games:
            # Determine player color
            white = game.headers.get('White', '')
            black = game.headers.get('Black', '')
            result = game.headers.get('Result', '*')
            
            if player_name.lower() not in white.lower() and \
               player_name.lower() not in black.lower():
                continue
            
            player_color = 'white' if player_name.lower() in white.lower() else 'black'
            
            # Determine outcome
            if result == '1-0':
                outcome = 'win' if player_color == 'white' else 'loss'
            elif result == '0-1':
                outcome = 'win' if player_color == 'black' else 'loss'
            elif result == '1/2-1/2':
                outcome = 'draw'
            else:
                continue
            
            # Extract and classify positions
            positions = self.parser.get_positions(game)
            
            # Sample positions (not all to save time)
            sampled_positions = positions[::max(1, len(positions) // 10)]  # Sample ~10 per game
            
            game_position_types = set()
            
            for board in sampled_positions:
                try:
                    features = self.position_extractor.extract_features(board)
                    classification = self.position_classifier.classify_position(features)
                    
                    pos_type = classification['position_type']
                    all_classifications.append(classification)
                    game_position_types.add(pos_type)
                    
                    # Track performance by type
                    if pos_type not in position_performance:
                        position_performance[pos_type] = {
                            'wins': 0, 'losses': 0, 'draws': 0, 'count': 0
                        }
                    
                    position_performance[pos_type]['count'] += 1
                except Exception as e:
                    logger.warning(f"Error classifying position: {e}")
                    continue
            
            # Add game outcome to position types encountered
            for pos_type in game_position_types:
                if outcome == 'win':
                    position_performance[pos_type]['wins'] += 1
                elif outcome == 'loss':
                    position_performance[pos_type]['losses'] += 1
                else:
                    position_performance[pos_type]['draws'] += 1
        
        # Calculate metrics
        performance_by_type = {}
        for pos_type, stats in position_performance.items():
            total_games = stats['wins'] + stats['losses'] + stats['draws']
            if total_games == 0:
                continue
            
            win_rate = stats['wins'] / total_games
            
            # Rating: Excellent (>0.65), Strong (>0.55), Average (>0.45), Weak (<0.45)
            if win_rate > 0.65:
                rating = "Excellent"
            elif win_rate > 0.55:
                rating = "Strong"
            elif win_rate > 0.45:
                rating = "Average"
            else:
                rating = "Weak"
            
            performance_by_type[pos_type] = {
                'win_rate': win_rate,
                'positions_encountered': stats['count'],
                'rating': rating
            }
        
        # Find best and worst
        if performance_by_type:
            best_type = max(performance_by_type.items(), key=lambda x: x[1]['win_rate'])
            worst_type = min(performance_by_type.items(), key=lambda x: x[1]['win_rate'])
        else:
            best_type = worst_type = None
        
        # Position distribution
        distribution = self.position_classifier.analyze_position_distribution(
            all_classifications
        )
        
        return {
            'performance_by_type': performance_by_type,
            'best_position_type': {
                'type': best_type[0],
                'win_rate': best_type[1]['win_rate']
            } if best_type else None,
            'worst_position_type': {
                'type': worst_type[0],
                'win_rate': worst_type[1]['win_rate']
            } if worst_type else None,
            'position_distribution': distribution
        }
    
    def _compile_statistics(self, games: List, game_features: List, player_name: str) -> Dict:
        """Compile game statistics."""
        import numpy as np
        
        total_moves = 0
        openings = {}
        
        for game in games:
            # FIXED: Try multiple headers for opening information
            opening = (
                game.headers.get('Opening') or 
                game.headers.get('ECO') or
                game.headers.get('Event', '')
            )
            
            # Clean up opening name
            if opening and opening.strip() and opening != '?':
                # Remove ECO codes (like "B12") from the beginning
                import re
                opening_clean = re.sub(r'^[A-E]\d{2}\s*', '', opening).strip()
                
                # If we still have something meaningful, use it
                if opening_clean and len(opening_clean) > 2:
                    openings[opening_clean] = openings.get(opening_clean, 0) + 1
            
            moves = self.parser.get_moves_list(game)
            total_moves += len(moves)
        
        # Top 5 openings
        if openings:
            top_openings = sorted(openings.items(), key=lambda x: x[1], reverse=True)[:5]
            favorite_openings = [opening for opening, _ in top_openings]
        else:
            favorite_openings = []
        
        # Average features
        avg_stats = {
            'total_moves_analyzed': total_moves,
            'avg_game_length': total_moves / len(games) if games else 0,
            'avg_captures_per_game': np.mean([f['captures_per_game'] for f in game_features]),
            'early_castling_rate': np.mean([f['castles_early_pct'] for f in game_features]) * 100,
            'aggressive_openings_pct': np.mean([f['aggressive_openings_pct'] for f in game_features]) * 100,
            'favorite_openings': favorite_openings
        }
        
        return avg_stats