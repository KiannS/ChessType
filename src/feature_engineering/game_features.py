"""
Extract features from chess games for style classification.
"""
import chess
import chess.pgn
from typing import Dict, List
import numpy as np


class GameFeatureExtractor:
    """Extract game-level features for playing style classification."""
    
    # Opening classifications (simplified)
    AGGRESSIVE_OPENINGS = {
        'King\'s Gambit', 'Danish Gambit', 'Smith-Morra Gambit',
        'Sicilian Dragon', 'Sicilian Najdorf', 'King\'s Indian Attack'
    }
    
    CENTER_SQUARES = [chess.E4, chess.E5, chess.D4, chess.D5]
    
    def extract_features(self, game: chess.pgn.Game, player_color: chess.Color) -> Dict:
        """
        Extract all features from a game for a specific player.
        
        Args:
            game: chess.pgn.Game object
            player_color: chess.WHITE or chess.BLACK
            
        Returns:
            Dictionary of features
        """
        board = game.board()
        moves_played = []
        
        # Track statistics
        piece_count_sum = 0
        pawn_moves = 0
        total_moves = 0
        captures = 0
        checks = 0
        castled = False
        castle_move_number = 100
        complex_positions = 0
        center_control_sum = 0
        
        move_number = 0
        
        for move in game.mainline_moves():
            move_number += 1
            
            # Only count moves by the specified player
            if board.turn == player_color:
                total_moves += 1
                moves_played.append(move)
                
                # Pawn moves
                if board.piece_at(move.from_square).piece_type == chess.PAWN:
                    pawn_moves += 1
                
                # Captures
                if board.is_capture(move):
                    captures += 1
                
                # Checks
                board.push(move)
                if board.is_check():
                    checks += 1
                board.pop()
                
                # Castling
                if board.is_castling(move) and not castled:
                    castled = True
                    castle_move_number = move_number
                
                # Piece count (complexity)
                piece_count = len(board.piece_map())
                piece_count_sum += piece_count
                if piece_count >= 20:
                    complex_positions += 1
                
                # Center control
                center_control = self._calculate_center_control(board, player_color)
                center_control_sum += center_control
            
            board.push(move)
        
        # Calculate aggregate features
        if total_moves == 0:
            total_moves = 1  # Avoid division by zero
        
        features = {
            'avg_piece_activity': piece_count_sum / total_moves,
            'pawn_moves_ratio': pawn_moves / total_moves,
            'avg_piece_trades': captures / (total_moves / 10),  # Captures per 10 moves
            'avg_center_control': center_control_sum / total_moves,
            'captures_per_game': captures,
            'checks_per_game': checks,
            'castles_early_pct': 1.0 if castle_move_number <= 10 else 0.0,
            'complex_positions_pct': complex_positions / total_moves,
            'aggressive_openings_pct': self._is_aggressive_opening(game)
        }
        
        return features
    
    def _calculate_center_control(self, board: chess.Board, color: chess.Color) -> float:
        """Calculate control of center squares."""
        control = 0
        for square in self.CENTER_SQUARES:
            attackers = board.attackers(color, square)
            control += len(attackers)
        return control
    
    def _is_aggressive_opening(self, game: chess.pgn.Game) -> float:
        """Check if opening is aggressive (1.0 or 0.0)."""
        opening = game.headers.get('Opening', '')
        for aggressive in self.AGGRESSIVE_OPENINGS:
            if aggressive.lower() in opening.lower():
                return 1.0
        return 0.0
    
    def extract_features_from_games(
        self,
        games: List[chess.pgn.Game],
        player_name: str
    ) -> List[Dict]:
        """
        Extract features from multiple games.
        
        Args:
            games: List of chess.pgn.Game objects
            player_name: Name of player to analyze
            
        Returns:
            List of feature dictionaries
        """
        all_features = []
        
        for game in games:
            # Determine player color
            white = game.headers.get('White', '')
            black = game.headers.get('Black', '')
            
            if player_name.lower() in white.lower():
                color = chess.WHITE
            elif player_name.lower() in black.lower():
                color = chess.BLACK
            else:
                continue  # Skip if player not found
            
            features = self.extract_features(game, color)
            all_features.append(features)
        
        return all_features
    
    def aggregate_features(self, feature_list: List[Dict]) -> Dict:
        """
        Aggregate features across multiple games.
        
        Args:
            feature_list: List of feature dictionaries
            
        Returns:
            Dictionary with aggregated features
        """
        if not feature_list:
            return {}
        
        # Calculate means
        aggregated = {}
        feature_names = feature_list[0].keys()
        
        for feature in feature_names:
            values = [f[feature] for f in feature_list]
            aggregated[feature] = np.mean(values)
        
        return aggregated


def main():
    """Test feature extraction."""
    from src.data_collection.parse_pgn import PGNParser
    
    parser = PGNParser()
    extractor = GameFeatureExtractor()
    
    # Load test game
    test_file = "data/training_data/DrNykterstein_lichess.pgn"
    games = parser.parse_pgn_file(test_file)
    
    if games:
        # Extract features from first 10 games
        features = extractor.extract_features_from_games(games[:10], "DrNykterstein")
        
        if features:
            # Aggregate
            aggregated = extractor.aggregate_features(features)
            
            print("\n=== Aggregated Features ===")
            for key, value in aggregated.items():
                print(f"{key}: {value:.3f}")


if __name__ == "__main__":
    main()