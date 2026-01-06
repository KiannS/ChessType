"""
Fetch the chess games from Lichess and Chess.com APIs.

"""

import requests
import time
import json
from pathlib import Path
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GameFetcher:
    """Fetch games from chess platforms."""
    
    LICHESS_API = "https://lichess.org/api"
    CHESSCOM_API = "https://api.chess.com/pub"
    
    def __init__(self, output_dir: str = "data/training_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_lichess_games(
        self, 
        username: str, 
        max_games: int = 100,
        rated: bool = True,
        perf_type: str = "blitz"
    ) -> str:
        """
        Fetch games from Lichess.
        
        Args:
            username: Lichess username
            max_games: Maximum number of games to fetch
            rated: Only fetch rated games
            perf_type: Game type (bullet, blitz, rapid, classical)
            
        Returns:
            Path to saved PGN file
        """
        url = f"{self.LICHESS_API}/games/user/{username}"
        params = {
            "max": max_games,
            "rated": "true" if rated else "false",
            "perfType": perf_type,
            "clocks": "false",
            "evals": "false",
            "opening": "true"
        }
        
        headers = {"Accept": "application/x-chess-pgn"}
        
        logger.info(f"Fetching {max_games} games for {username} from Lichess...")
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            
            pgn_content = response.text
            
            if not pgn_content.strip():
                logger.warning(f"No games found for {username}")
                return None
            
            # Save PGN file
            output_path = self.output_dir / f"{username}_lichess.pgn"
            output_path.write_text(pgn_content, encoding='utf-8')
            
            # Count games
            game_count = pgn_content.count('[Event ')
            logger.info(f"Successfully fetched {game_count} games for {username}")
            
            return str(output_path)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching games for {username}: {e}")
            return None
    
    def fetch_chesscom_games(
        self,
        username: str,
        year: int,
        month: int,
        max_games: Optional[int] = None
    ) -> str:
        """
        Fetch games from Chess.com.
        
        Args:
            username: Chess.com username
            year: Year (e.g., 2024)
            month: Month (1-12)
            max_games: Maximum number of games (None = all)
            
        Returns:
            Path to saved PGN file
        """
        url = f"{self.CHESSCOM_API}/player/{username}/games/{year}/{month:02d}"
        
        logger.info(f"Fetching games for {username} from Chess.com ({year}-{month:02d})...")
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            games = data.get('games', [])
            
            if not games:
                logger.warning(f"No games found for {username}")
                return None
            
            # Limit games if specified
            if max_games:
                games = games[:max_games]
            
            # Extract PGN from each game
            pgn_content = "\n\n".join([game.get('pgn', '') for game in games])
            
            # Save PGN file
            output_path = self.output_dir / f"{username}_chesscom_{year}_{month:02d}.pgn"
            output_path.write_text(pgn_content, encoding='utf-8')
            
            logger.info(f"Successfully fetched {len(games)} games for {username}")
            
            return str(output_path)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching games for {username}: {e}")
            return None
    
    def fetch_training_dataset(self) -> Dict[str, List[str]]:
        """
        Fetch games from famous players for training.
        
        Returns:
            Dictionary mapping style labels to PGN file paths
        """
        # Famous players by style
        players = {
            'Positional': ['DrNykterstein', 'LyonBeast'],  # Magnus, Levon
            'Aggressive': ['Hikaru', 'GMHIKARU'],  # Hikaru Nakamura
            'Tactical': ['GothamChess', 'GingerGM'],  # Levy, Simon
            'Solid': ['FabianoCaruana', 'RapidShooter'],  # Fabiano
            'Balanced': ['DanielNaroditsky', 'Giri']  # Danya, Anish
        }
        
        dataset = {style: [] for style in players.keys()}
        
        for style, usernames in players.items():
            logger.info(f"\n=== Fetching {style} players ===")
            for username in usernames:
                pgn_path = self.fetch_lichess_games(username, max_games=100)
                if pgn_path:
                    dataset[style].append(pgn_path)
                time.sleep(1)  # Rate limiting
        
        # Save dataset metadata
        metadata_path = self.output_dir / "training_metadata.json"
        metadata_path.write_text(json.dumps(dataset, indent=2))
        
        logger.info(f"\nTraining dataset saved to {self.output_dir}")
        return dataset


def main():
    """Main execution function."""
    fetcher = GameFetcher()
    
    # Fetch training dataset
    dataset = fetcher.fetch_training_dataset()

    print("\n=== Dataset Summary ===")
    for style, paths in dataset.items():
        print(f"{style}: {len(paths)} players")


if __name__ == "__main__":
    main()

