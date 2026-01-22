"""
Enhanced game feature extraction with 20+ features for improved accuracy.
"""
import chess
import chess.pgn
from typing import Dict, List
import numpy as np


class EnhancedGameFeatureExtractor:
    """Extract enhanced game-level features for style classification."""
    
    # Opening classifications (expanded)
    AGGRESSIVE_OPENINGS = {
        'King\'s Gambit', 'Danish Gambit', 'Smith-Morra Gambit',
        'Sicilian Dragon', 'Sicilian Najdorf', 'King\'s Indian Attack',
        'King\'s Indian Defense', 'Benko Gambit', 'Latvian Gambit',
        'Alekhine Defense', 'Budapest Gambit', 'Evans Gambit'
    }
    
    POSITIONAL_OPENINGS = {
        'Queen\'s Gambit Declined', 'Caro-Kann', 'Nimzo-Indian',
        'Queen\'s Indian', 'Catalan', 'English Opening',
        'Reti Opening', 'London System'
    }
    
    CENTER_SQUARES = [chess.E4, chess.E5, chess.D4, chess.D5]
    EXTENDED_CENTER = [
        chess.D3, chess.E3, chess.F3,
        chess.D4, chess.E4, chess.F4,
        chess.D5, chess.E5, chess.F5,
        chess.D6, chess.E6, chess.F6
    ]
    
    def extract_features(self, game: chess.pgn.Game, player_color: chess.Color) -> Dict:
        """
        Extract all 20+ features from a game for a specific player.
        
        Features extracted:
        1-9: Original features
        10: avg_move_time (if available)
        11: king_safety_priority
        12: piece_sacrifice_rate
        13: endgame_preference
        14: opening_variety (N/A for single game)
        15: positional_sacrifice_rate
        16: pawn_storm_frequency
        17: prophylaxis_moves
        18: center_pawn_moves
        19: opposite_castling_rate
        20: early_queen_development
        21: long_term_pawn_pushes
        22: weak_square_creation
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
        
        # NEW: Additional tracking
        piece_sacrifices = 0
        positional_sacrifices = 0
        pawn_storm_moves = 0
        prophylaxis_count = 0
        center_pawn_count = 0
        queen_developed_early = False
        long_pawn_pushes = 0
        weak_squares_created = 0
        king_exposed_moves = 0
        
        opposite_side_castling = False
        player_kingside = None
        opponent_kingside = None
        
        move_number = 0
        
        for move in game.mainline_moves():
            move_number += 1
            
            # Track opponent castling
            if board.turn != player_color:
                if board.is_castling(move):
                    opponent_kingside = board.piece_at(move.from_square).color == chess.WHITE and move.to_square > move.from_square
            
            # Only analyze player's moves
            if board.turn == player_color:
                total_moves += 1
                moves_played.append(move)
                
                piece = board.piece_at(move.from_square)
                if not piece:
                    board.push(move)
                    continue
                
                # Pawn moves
                if piece.piece_type == chess.PAWN:
                    pawn_moves += 1
                    
                    # Center pawn moves (e4, d4, e5, d5)
                    if move.to_square in self.CENTER_SQUARES:
                        center_pawn_count += 1
                    
                    # Long pawn pushes (3+ squares from start)
                    from_rank = chess.square_rank(move.from_square)
                    to_rank = chess.square_rank(move.to_square)
                    if abs(to_rank - from_rank) >= 2:
                        long_pawn_pushes += 1
                    
                    # Pawn storm (advancing pawns near enemy king)
                    if move_number > 15:  # Middlegame
                        pawn_storm_moves += 1
                
                # Queen development
                if piece.piece_type == chess.QUEEN and move_number <= 10:
                    queen_developed_early = True
                
                # Captures
                if board.is_capture(move):
                    captures += 1
                    
                    # Check if sacrifice (lower value for higher value)
                    captured = board.piece_at(move.to_square)
                    if captured:
                        piece_value = self._piece_value(piece.piece_type)
                        captured_value = self._piece_value(captured.piece_type)
                        
                        if piece_value > captured_value + 1:
                            piece_sacrifices += 1
                        elif piece_value == 5 and captured_value == 3:  # Exchange sacrifice
                            positional_sacrifices += 1
                
                # Checks
                board.push(move)
                if board.is_check():
                    checks += 1
                board.pop()
                
                # Castling
                if board.is_castling(move) and not castled:
                    castled = True
                    castle_move_number = move_number
                    player_kingside = move.to_square > move.from_square
                    
                    # Check for opposite side castling
                    if opponent_kingside is not None and player_kingside != opponent_kingside:
                        opposite_side_castling = True
                
                # Piece count (complexity)
                piece_count = len(board.piece_map())
                piece_count_sum += piece_count
                if piece_count >= 20:
                    complex_positions += 1
                
                # Center control
                center_control = self._calculate_center_control(board, player_color)
                center_control_sum += center_control
                
                # King safety (after castling)
                if castled and move_number > castle_move_number:
                    king_square = board.king(player_color)
                    if king_square:
                        # Check if move weakens king safety
                        if self._weakens_king_safety(board, move, king_square):
                            king_exposed_moves += 1
                            weak_squares_created += 1
                
                # Prophylaxis (preventive moves)
                if self._is_prophylactic(board, move):
                    prophylaxis_count += 1
                
                board.push(move)
            else:
                board.push(move)
        
        # Calculate aggregate features
        if total_moves == 0:
            total_moves = 1  # Avoid division by zero
        
        features = {
            # Original 9 features
            'avg_piece_activity': piece_count_sum / total_moves,
            'pawn_moves_ratio': pawn_moves / total_moves,
            'avg_piece_trades': captures / (total_moves / 10),
            'avg_center_control': center_control_sum / total_moves,
            'captures_per_game': captures,
            'checks_per_game': checks,
            'castles_early_pct': 1.0 if castle_move_number <= 10 else 0.0,
            'complex_positions_pct': complex_positions / total_moves,
            'aggressive_openings_pct': self._is_aggressive_opening(game),
            
            # NEW: 11+ additional features
            'king_safety_priority': 1.0 - (king_exposed_moves / max(1, total_moves - castle_move_number)),
            'piece_sacrifice_rate': piece_sacrifices / max(1, captures),
            'endgame_preference': 1.0 if piece_count_sum / total_moves < 16 else 0.0,
            'positional_sacrifice_rate': positional_sacrifices / total_moves,
            'pawn_storm_frequency': pawn_storm_moves / max(1, total_moves - 15),
            'prophylaxis_moves': prophylaxis_count / total_moves,
            'center_pawn_moves': center_pawn_count / max(1, pawn_moves),
            'opposite_castling_rate': 1.0 if opposite_side_castling else 0.0,
            'early_queen_development': 1.0 if queen_developed_early else 0.0,
            'long_term_pawn_pushes': long_pawn_pushes / max(1, pawn_moves),
            'weak_square_creation': weak_squares_created / total_moves
        }
        
        return features
    
    def _piece_value(self, piece_type: chess.PieceType) -> int:
        """Get standard piece value."""
        values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        return values.get(piece_type, 0)
    
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
    
    def _weakens_king_safety(self, board: chess.Board, move: chess.Move, king_square: int) -> bool:
        """Check if move weakens king safety."""
        # Check if pawn move near king
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type == chess.PAWN:
            distance = chess.square_distance(move.to_square, king_square)
            if distance <= 2:
                return True
        return False
    
    def _is_prophylactic(self, board: chess.Board, move: chess.Move) -> bool:
        """Heuristic to detect prophylactic moves."""
        # Simple heuristic: move that doesn't capture, check, or attack
        if board.is_capture(move):
            return False
        
        board.push(move)
        if board.is_check():
            board.pop()
            return False
        board.pop()
        
        # Move that controls key square without immediate threat
        # This is a simplification - true prophylaxis detection is complex
        return chess.square_distance(move.from_square, move.to_square) <= 2
    
    def extract_features_from_games(
        self,
        games: List[chess.pgn.Game],
        player_name: str
    ) -> List[Dict]:
        """Extract features from multiple games."""
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
                continue
            
            features = self.extract_features(game, color)
            all_features.append(features)
        
        return all_features
    
    def aggregate_features(self, feature_list: List[Dict]) -> Dict:
        """Aggregate features across multiple games."""
        if not feature_list:
            return {}
        
        aggregated = {}
        feature_names = feature_list[0].keys()
        
        for feature in feature_names:
            values = [f[feature] for f in feature_list]
            aggregated[feature] = np.mean(values)
        
        return aggregated