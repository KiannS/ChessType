"""
Download and process Lichess Elite Database (2400+ rated players).
This provides high-quality training data from elite players.

Requirements:
    pip install zstandard tqdm

Usage:
    # Download one month of elite games
    python src/data_collection/download_elite_database.py --year 2024 --month 1
    
    # Download and extract specific players
    python src/data_collection/download_elite_database.py --year 2024 --month 1 --extract-players
    
    # Process only (if already downloaded)
    python src/data_collection/download_elite_database.py --process-only
"""

import requests
import zstandard as zstd
import chess.pgn
import io
import sys
import os
from pathlib import Path
from typing import List, Dict, Set
import logging
from tqdm import tqdm
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LichessEliteDownloader:
    """Download and process Lichess Elite database."""
    
    BASE_URL = "https://database.lichess.org/elite"
    
    # Known elite players with documented playing styles
    ELITE_PLAYERS_BY_STYLE = {
        'Positional': {
            'DrNykterstein',      # Magnus Carlsen
            'penguingim1',        # Penguin GM
            'LyonBeast',          # Levon Aronian
            'Msb2',               # Matthias Bluebaum
            'wonderfultime',      # Tania Sachdev
            'GMWSO',              # Wesley So
            'Jospem',             # Jose Martinez
        },
        'Aggressive': {
            'Hikaru',             # Hikaru Nakamura
            'GMHIKARU',           # Hikaru alt
            'DanielNaroditskyGM', # Daniel Naroditsky (aggressive variant)
            'GM_dmitrij',         # Dmitry Andreikin
            'STL_Inverarity',     # Ben Finegold
            'Night-King96',       # Alireza Firouzja
            'aladdin65',          # Alireza alt
        },
        'Tactical': {
            'GothamChess',        # Levy Rozman
            'GingerGM',           # Simon Williams
            'TigrVShlyape',       # Daniil Dubov
            'shield-wall',        # Shield Wall
            'BillieKimbah',       # IM Levy alt
            'Polish_fighter3000', # Jan-Krzysztof Duda
        },
        'Solid': {
            'FabianoCaruana',     # Fabiano Caruana
            'RapidShooter',       # Rapid Shooter
            'Giri',               # Anish Giri
            'chessexplained',     # Chess Explained
            'ArtemTrushko',       # Artem Trushko
        },
        'Balanced': {
            'DanielNaroditsky',   # Daniel Naroditsky
            'chessexplained',     # Chess Explained
            'Jospem',             # Jose Martinez
        }
    }
    
    def __init__(self, output_dir: str = "data/elite_database"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.training_dir = Path("data/training_data")
        self.training_dir.mkdir(parents=True, exist_ok=True)
    
    def download_elite_database(self, year: int, month: int) -> Path:
        """
        Download Lichess Elite database for a specific month.
        
        Args:
            year: Year (e.g., 2024)
            month: Month (1-12)
            
        Returns:
            Path to downloaded file
        """
        filename = f"lichess_elite_{year}-{month:02d}.pgn.zst"
        url = f"{self.BASE_URL}/{filename}"
        output_path = self.output_dir / filename
        
        # Check if already downloaded
        if output_path.exists():
            logger.info(f"File already exists: {output_path}")
            return output_path
        
        logger.info(f"Downloading: {url}")
        logger.info("This file is ~500MB compressed, ~2GB uncompressed")
        logger.info("This may take 5-15 minutes depending on your connection...")
        
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Get total file size
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress bar
            with open(output_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            logger.info(f"✅ Downloaded: {output_path}")
            return output_path
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Download failed: {e}")
            if output_path.exists():
                output_path.unlink()  # Delete partial file
            raise
    
    def decompress_zst_file(self, zst_path: Path) -> Path:
        """
        Decompress .zst file to .pgn.
        
        Args:
            zst_path: Path to .zst file
            
        Returns:
            Path to decompressed .pgn file
        """
        pgn_path = zst_path.with_suffix('')  # Remove .zst extension
        
        if pgn_path.exists():
            logger.info(f"Decompressed file already exists: {pgn_path}")
            return pgn_path
        
        logger.info(f"Decompressing {zst_path.name}...")
        logger.info("This may take 10-20 minutes...")
        
        try:
            dctx = zstd.ZstdDecompressor()
            
            with open(zst_path, 'rb') as compressed:
                with open(pgn_path, 'wb') as decompressed:
                    # Decompress in chunks with progress
                    file_size = zst_path.stat().st_size
                    with tqdm(total=file_size, unit='B', unit_scale=True, desc="Decompressing") as pbar:
                        reader = dctx.stream_reader(compressed)
                        while True:
                            chunk = reader.read(16384)
                            if not chunk:
                                break
                            decompressed.write(chunk)
                            pbar.update(len(chunk))
            
            logger.info(f"✅ Decompressed: {pgn_path}")
            return pgn_path
            
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            if pgn_path.exists():
                pgn_path.unlink()  # Delete partial file
            raise
    
    def extract_players_from_database(
        self,
        pgn_path: Path,
        target_players: Set[str],
        max_games_per_player: int = 500
    ) -> Dict[str, List[chess.pgn.Game]]:
        """
        Extract games for specific players from the database.
        
        Args:
            pgn_path: Path to PGN file
            target_players: Set of usernames to extract
            max_games_per_player: Maximum games per player
            
        Returns:
            Dictionary mapping player names to their games
        """
        logger.info(f"Extracting games for {len(target_players)} target players...")
        logger.info("This may take 20-30 minutes for a full database file...")
        
        player_games = {player: [] for player in target_players}
        games_found = {player: 0 for player in target_players}
        total_games_processed = 0
        
        try:
            with open(pgn_path, 'r', encoding='utf-8') as pgn_file:
                with tqdm(desc="Processing games", unit=" games") as pbar:
                    while True:
                        game = chess.pgn.read_game(pgn_file)
                        if game is None:
                            break
                        
                        total_games_processed += 1
                        pbar.update(1)
                        
                        # Check if either player is in our target list
                        white = game.headers.get('White', '').lower()
                        black = game.headers.get('Black', '').lower()
                        
                        for player in target_players:
                            player_lower = player.lower()
                            
                            # Check if player is white or black
                            if player_lower in white or player_lower in black:
                                if games_found[player] < max_games_per_player:
                                    player_games[player].append(game)
                                    games_found[player] += 1
                        
                        # Stop if we have enough games for all players
                        if all(count >= max_games_per_player for count in games_found.values()):
                            logger.info("Reached max games for all players!")
                            break
                        
                        # Progress update every 10000 games
                        if total_games_processed % 10000 == 0:
                            pbar.set_postfix({
                                'found': sum(games_found.values()),
                                'target': len(target_players) * max_games_per_player
                            })
        
        except Exception as e:
            logger.error(f"Error processing PGN: {e}")
            raise
        
        logger.info(f"\n✅ Processing complete!")
        logger.info(f"Total games processed: {total_games_processed:,}")
        logger.info(f"Games extracted: {sum(len(games) for games in player_games.values())}")
        logger.info("\nGames per player:")
        for player, games in player_games.items():
            if games:
                logger.info(f"  {player}: {len(games)} games")
        
        return player_games
    
    def save_player_games_as_pgn(
        self,
        player_games: Dict[str, List[chess.pgn.Game]],
        style: str
    ) -> List[Path]:
        """
        Save extracted games to PGN files.
        
        Args:
            player_games: Dictionary mapping players to their games
            style: Playing style label
            
        Returns:
            List of paths to created PGN files
        """
        saved_files = []
        
        for player, games in player_games.items():
            if not games:
                continue
            
            output_path = self.training_dir / f"{player}_lichess_elite.pgn"
            
            logger.info(f"Saving {len(games)} games for {player}...")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for game in games:
                    # Write game to file
                    exporter = chess.pgn.FileExporter(f)
                    game.accept(exporter)
                    f.write('\n\n')
            
            saved_files.append(output_path)
            logger.info(f"✅ Saved: {output_path}")
        
        return saved_files
    
    def process_elite_database(
        self,
        year: int,
        month: int,
        max_games_per_player: int = 500,
        extract_players: bool = True
    ) -> Dict:
        """
        Complete pipeline: download, decompress, and extract games.
        
        Args:
            year: Year to download
            month: Month to download
            max_games_per_player: Maximum games per player
            extract_players: Whether to extract specific players
            
        Returns:
            Summary dictionary
        """
        summary = {
            'year': year,
            'month': month,
            'max_games_per_player': max_games_per_player,
            'by_style': {}
        }
        
        logger.info("\n" + "="*70)
        logger.info("LICHESS ELITE DATABASE PROCESSING")
        logger.info("="*70)
        logger.info(f"Year: {year}, Month: {month}")
        logger.info(f"Max games per player: {max_games_per_player}")
        logger.info("="*70 + "\n")
        
        # Step 1: Download
        zst_path = self.download_elite_database(year, month)
        
        # Step 2: Decompress
        pgn_path = self.decompress_zst_file(zst_path)
        
        # Step 3: Extract players (if requested)
        if extract_players:
            for style, players in self.ELITE_PLAYERS_BY_STYLE.items():
                logger.info(f"\n{'='*70}")
                logger.info(f"Extracting {style} players")
                logger.info(f"{'='*70}")
                
                player_games = self.extract_players_from_database(
                    pgn_path,
                    players,
                    max_games_per_player
                )
                
                saved_files = self.save_player_games_as_pgn(player_games, style)
                
                summary['by_style'][style] = {
                    'players': list(players),
                    'games_extracted': sum(len(games) for games in player_games.values()),
                    'files': [str(f) for f in saved_files]
                }
        
        # Save summary
        summary_path = self.output_dir / f"elite_extraction_summary_{year}_{month:02d}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("\n" + "="*70)
        logger.info("PROCESSING COMPLETE!")
        logger.info("="*70)
        if extract_players:
            total_games = sum(
                style_data['games_extracted']
                for style_data in summary['by_style'].values()
            )
            logger.info(f"Total games extracted: {total_games}")
            logger.info("\nGames by style:")
            for style, data in summary['by_style'].items():
                logger.info(f"  {style}: {data['games_extracted']} games")
        logger.info(f"\nSummary saved: {summary_path}")
        logger.info("="*70 + "\n")
        
        return summary


def main():
    """Main execution."""
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(
        description='Download and process Lichess Elite database'
    )
    parser.add_argument(
        '--year',
        type=int,
        default=datetime.now().year,
        help='Year to download (default: current year)'
    )
    parser.add_argument(
        '--month',
        type=int,
        default=datetime.now().month - 1,  # Previous month
        help='Month to download (default: previous month)'
    )
    parser.add_argument(
        '--max-games',
        type=int,
        default=500,
        help='Maximum games per player (default: 500)'
    )
    parser.add_argument(
        '--process-only',
        action='store_true',
        help='Skip download, only process existing file'
    )
    parser.add_argument(
        '--extract-players',
        action='store_true',
        default=True,
        help='Extract specific players for training'
    )
    
    args = parser.parse_args()
    
    # Validate month
    if args.month < 1 or args.month > 12:
        logger.error("Month must be between 1 and 12")
        sys.exit(1)
    
    # Check if zstandard is installed
    try:
        import zstandard
    except ImportError:
        logger.error("zstandard package not installed!")
        logger.error("Install it with: pip install zstandard")
        sys.exit(1)
    
    # Process database
    downloader = LichessEliteDownloader()
    
    if args.process_only:
        # Process existing file
        filename = f"lichess_elite_{args.year}-{args.month:02d}.pgn"
        pgn_path = downloader.output_dir / filename
        
        if not pgn_path.exists():
            logger.error(f"File not found: {pgn_path}")
            logger.error("Download it first without --process-only flag")
            sys.exit(1)
        
        if args.extract_players:
            for style, players in downloader.ELITE_PLAYERS_BY_STYLE.items():
                player_games = downloader.extract_players_from_database(
                    pgn_path,
                    players,
                    args.max_games
                )
                downloader.save_player_games_as_pgn(player_games, style)
    else:
        # Full pipeline
        summary = downloader.process_elite_database(
            args.year,
            args.month,
            args.max_games,
            args.extract_players
        )
        
        logger.info("\n✅ Elite database processing complete!")
        logger.info("\nNext steps:")
        logger.info("1. python src/models/train_style_classifier.py")
        logger.info("2. python src/models/train_position_clusters.py")


if __name__ == "__main__":
    main()