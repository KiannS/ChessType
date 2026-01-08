"""
Parse PGN files and extract games.
"""
import chess.pgn
from pathlib import Path
from typing import List, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PGNParser:
    """Parse and extract information from PGN files."""
    
    @staticmethod
    def parse_pgn_file(pgn_path: str) -> List[chess.pgn.Game]:
        """
        Parse PGN file and return list of games.
        
        Args:
            pgn_path: Path to PGN file
            
        Returns:
            List of chess.pgn.Game objects
        """
        games = []
        
        try:
            with open(pgn_path, 'r', encoding='utf-8') as pgn_file:
                while True:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break
                    games.append(game)
            
            logger.info(f"Parsed {len(games)} games from {pgn_path}")
            return games
            
        except Exception as e:
            logger.error(f"Error parsing {pgn_path}: {e}")
            return []
    
    @staticmethod
    def extract_game_info(game: chess.pgn.Game) -> Dict:
        """
        Extract basic information from a game.
        
        Args:
            game: chess.pgn.Game object
            
        Returns:
            Dictionary with game metadata
        """
        headers = game.headers
        
        return {
            'event': headers.get('Event', 'Unknown'),
            'site': headers.get('Site', 'Unknown'),
            'date': headers.get('Date', 'Unknown'),
            'round': headers.get('Round', 'Unknown'),
            'white': headers.get('White', 'Unknown'),
            'black': headers.get('Black', 'Unknown'),
            'result': headers.get('Result', '*'),
            'white_elo': headers.get('WhiteElo', 'Unknown'),
            'black_elo': headers.get('BlackElo', 'Unknown'),
            'opening': headers.get('Opening', 'Unknown'),
            'eco': headers.get('ECO', 'Unknown'),
            'time_control': headers.get('TimeControl', 'Unknown'),
        }
    
    @staticmethod
    def get_moves_list(game: chess.pgn.Game) -> List[str]:
        """
        Extract list of moves in SAN notation.
        
        Args:
            game: chess.pgn.Game object
            
        Returns:
            List of moves as strings
        """
        moves = []
        board = game.board()
        
        for move in game.mainline_moves():
            moves.append(board.san(move))
            board.push(move)
        
        return moves
    
    @staticmethod
    def get_positions(game: chess.pgn.Game) -> List[chess.Board]:
        """
        Extract all positions from a game.
        
        Args:
            game: chess.pgn.Game object
            
        Returns:
            List of board positions
        """
        positions = []
        board = game.board()
        
        # Add starting position
        positions.append(board.copy())
        
        for move in game.mainline_moves():
            board.push(move)
            positions.append(board.copy())
        
        return positions


def main():
    """Test PGN parsing."""
    parser = PGNParser()
    
    # Test with a sample file
    test_file = "data/training_data/DrNykterstein_lichess.pgn"
    
    if Path(test_file).exists():
        games = parser.parse_pgn_file(test_file)
        
        if games:
            # Print info about first game
            game = games[0]
            info = parser.extract_game_info(game)
            moves = parser.get_moves_list(game)
            
            print("\n=== Sample Game Info ===")
            print(f"White: {info['white']} ({info['white_elo']})")
            print(f"Black: {info['black']} ({info['black_elo']})")
            print(f"Opening: {info['opening']}")
            print(f"Result: {info['result']}")
            print(f"Moves: {len(moves)}")
            print(f"First 10 moves: {' '.join(moves[:10])}")
    else:
        print(f"Test file not found: {test_file}")
        print("Run fetch_games.py first to download training data")


if __name__ == "__main__":
    main()