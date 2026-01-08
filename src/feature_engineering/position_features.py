"""
Extract features from chess positions for position classification.
"""
import chess
from typing import Dict, List


class PositionFeatureExtractor:
    """Extract position-level features for clustering/classification."""
    
    # Piece values for material calculation
    PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    
    CENTER_SQUARES = [chess.E4, chess.E5, chess.D4, chess.D5]
    EXTENDED_CENTER = [
        chess.D3, chess.E3, chess.F3,
        chess.D4, chess.E4, chess.F4,
        chess.D5, chess.E5, chess.F5,
        chess.D6, chess.E6, chess.F6
    ]
    
    def extract_features(self, board: chess.Board) -> Dict:
        """
        Extract all position features.
        
        Args:
            board: chess.Board object
            
        Returns:
            Dictionary of 20 features
        """
        features = {
            # Basic material
            'piece_count': len(board.piece_map()),
            'white_material': self._calculate_material(board, chess.WHITE),
            'black_material': self._calculate_material(board, chess.BLACK),
            'material_imbalance': abs(
                self._calculate_material(board, chess.WHITE) - 
                self._calculate_material(board, chess.BLACK)
            ),
            
            # Tactical complexity
            'num_legal_moves': board.legal_moves.count(),
            'num_captures': sum(1 for m in board.legal_moves if board.is_capture(m)),
            'num_checks': sum(1 for m in board.legal_moves if board.gives_check(m)),
            'hanging_pieces': self._count_hanging_pieces(board),
            'pins_present': self._count_pins(board),
            
            # Pawn structure
            'pawn_islands': self._count_pawn_islands(board),
            'doubled_pawns': self._count_doubled_pawns(board),
            'passed_pawns': self._count_passed_pawns(board),
            'pawn_tension': self._count_pawn_tension(board),
            
            # King safety
            'king_exposure_white': self._calculate_king_exposure(board, chess.WHITE),
            'king_exposure_black': self._calculate_king_exposure(board, chess.BLACK),
            'opposite_castling': self._opposite_side_castling(board),
            
            # Position dynamics
            'piece_mobility': board.legal_moves.count(),
            'center_control': self._calculate_center_control(board),
            'position_stability': self._calculate_stability(board),
            'open_files': self._count_open_files(board)
        }
        
        return features
    
    def _calculate_material(self, board: chess.Board, color: chess.Color) -> int:
        """Calculate total material value for a color."""
        material = 0
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            material += len(board.pieces(piece_type, color)) * self.PIECE_VALUES[piece_type]
        return material
    
    def _count_hanging_pieces(self, board: chess.Board) -> int:
        """Count undefended pieces that can be captured."""
        hanging = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
            
            # Check if piece is attacked
            if board.is_attacked_by(not piece.color, square):
                # Check if piece is defended
                if not board.is_attacked_by(piece.color, square):
                    hanging += 1
        
        return hanging
    
    def _count_pins(self, board: chess.Board) -> int:
        """Count pinned pieces."""
        pins = 0
        for color in [chess.WHITE, chess.BLACK]:
            king_square = board.king(color)
            if king_square is not None:
                pins += bin(board.pin_mask(color, king_square)).count('1')
        return pins
    
    def _count_pawn_islands(self, board: chess.Board) -> int:
        """Count pawn islands for both sides."""
        islands = 0
        for color in [chess.WHITE, chess.BLACK]:
            files_with_pawns = set()
            for square in board.pieces(chess.PAWN, color):
                files_with_pawns.add(chess.square_file(square))
            
            # Count groups of consecutive files
            if files_with_pawns:
                sorted_files = sorted(files_with_pawns)
                island_count = 1
                for i in range(1, len(sorted_files)):
                    if sorted_files[i] - sorted_files[i-1] > 1:
                        island_count += 1
                islands += island_count
        
        return islands
    
    def _count_doubled_pawns(self, board: chess.Board) -> int:
        """Count doubled pawns."""
        doubled = 0
        for color in [chess.WHITE, chess.BLACK]:
            file_counts = [0] * 8
            for square in board.pieces(chess.PAWN, color):
                file_counts[chess.square_file(square)] += 1
            doubled += sum(max(0, count - 1) for count in file_counts)
        return doubled
    
    def _count_passed_pawns(self, board: chess.Board) -> int:
        """Count passed pawns (simplified)."""
        passed = 0
        for color in [chess.WHITE, chess.BLACK]:
            for pawn_square in board.pieces(chess.PAWN, color):
                file = chess.square_file(pawn_square)
                rank = chess.square_rank(pawn_square)
                
                # Check if any enemy pawns ahead
                enemy_pawns_ahead = False
                for enemy_pawn in board.pieces(chess.PAWN, not color):
                    enemy_file = chess.square_file(enemy_pawn)
                    enemy_rank = chess.square_rank(enemy_pawn)
                    
                    # Adjacent files or same file
                    if abs(enemy_file - file) <= 1:
                        if color == chess.WHITE and enemy_rank > rank:
                            enemy_pawns_ahead = True
                        elif color == chess.BLACK and enemy_rank < rank:
                            enemy_pawns_ahead = True
                
                if not enemy_pawns_ahead:
                    passed += 1
        
        return passed
    
    def _count_pawn_tension(self, board: chess.Board) -> int:
        """Count pawns that attack enemy pawns."""
        tension = 0
        for color in [chess.WHITE, chess.BLACK]:
            for pawn_square in board.pieces(chess.PAWN, color):
                # Get pawn attacks
                attacks = board.attacks(pawn_square)
                for attack_square in attacks:
                    piece = board.piece_at(attack_square)
                    if piece and piece.piece_type == chess.PAWN and piece.color != color:
                        tension += 1
        return tension
    
    def _calculate_king_exposure(self, board: chess.Board, color: chess.Color) -> float:
        """Calculate how exposed the king is (0-10 scale)."""
        king_square = board.king(color)
        if king_square is None:
            return 0.0
        
        exposure = 0.0
        
        # Count attackers near king
        king_zone = [king_square]
        for attack_square in board.attacks(king_square):
            king_zone.append(attack_square)
        
        for square in king_zone:
            attackers = board.attackers(not color, square)
            exposure += len(attackers)
        
        # Pawn shield
        pawn_shield = 0
        rank = chess.square_rank(king_square)
        file = chess.square_file(king_square)
        
        if color == chess.WHITE:
            shield_rank = rank + 1
        else:
            shield_rank = rank - 1
        
        if 0 <= shield_rank < 8:
            for f in [file - 1, file, file + 1]:
                if 0 <= f < 8:
                    shield_square = chess.square(f, shield_rank)
                    piece = board.piece_at(shield_square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        pawn_shield += 1
        
        exposure -= pawn_shield
        
        return max(0.0, min(10.0, exposure))
    
    def _opposite_side_castling(self, board: chess.Board) -> float:
        """Check if kings castled on opposite sides (0 or 1)."""
        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)
        
        if white_king is None or black_king is None:
            return 0.0
        
        white_file = chess.square_file(white_king)
        black_file = chess.square_file(black_king)
        
        # Queenside: files 0-2, Kingside: files 5-7
        white_queenside = white_file <= 2
        white_kingside = white_file >= 5
        black_queenside = black_file <= 2
        black_kingside = black_file >= 5
        
        if (white_queenside and black_kingside) or (white_kingside and black_queenside):
            return 1.0
        
        return 0.0
    
    def _calculate_center_control(self, board: chess.Board) -> float:
        """Calculate control of center squares."""
        control = 0
        for square in self.EXTENDED_CENTER:
            white_attackers = len(board.attackers(chess.WHITE, square))
            black_attackers = len(board.attackers(chess.BLACK, square))
            control += abs(white_attackers - black_attackers)
        return control
    
    def _calculate_stability(self, board: chess.Board) -> float:
        """
        Calculate position stability (0-100).
        High stability = few hanging pieces, balanced material.
        """
        hanging = self._count_hanging_pieces(board)
        captures_available = sum(1 for m in board.legal_moves if board.is_capture(m))
        checks_available = sum(1 for m in board.legal_moves if board.gives_check(m))
        
        instability = hanging * 10 + captures_available * 2 + checks_available * 3
        stability = max(0, 100 - instability)
        
        return stability
    
    def _count_open_files(self, board: chess.Board) -> int:
        """Count files with no pawns."""
        open_files = 0
        for file in range(8):
            has_pawn = False
            for rank in range(8):
                square = chess.square(file, rank)
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN:
                    has_pawn = True
                    break
            if not has_pawn:
                open_files += 1
        return open_files


def main():
    """Test position feature extraction."""
    extractor = PositionFeatureExtractor()
    
    # Test positions
    positions = [
        chess.Board(),  # Starting position
        chess.Board("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"),  # Open game
        chess.Board("8/5k2/3p4/1p1Pp3/pP2Pp2/P4P2/8/6K1 w - - 0 1")  # Endgame
    ]
    
    for i, board in enumerate(positions):
        print(f"\n=== Position {i + 1} ===")
        print(board)
        print("\nFeatures:")
        features = extractor.extract_features(board)
        for key, value in features.items():
            print(f"  {key}: {value:.2f}")


if __name__ == "__main__":
    main()