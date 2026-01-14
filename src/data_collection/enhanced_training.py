"""
Enhanced training data collection script.
Run this to collect a much larger and better training dataset.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data_collection.fetch_games import GameFetcher
import time
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# EXPANDED player list with known styles
ENHANCED_PLAYER_DATABASE = {
    'Positional': {
        'lichess': ['DrNykterstein', 'LyonBeast', 'penguingim1', 'wonderfultime', 'Msb2'],
        'chesscom': ['magnuscarlsen', 'levon_aronian', 'chessnetwork']
    },
    'Aggressive': {
        'lichess': ['Hikaru', 'GMHIKARU', 'STL_Inverarity', 'GM_dmitrij', 'DanielNaroditskyGM'],
        'chesscom': ['hikaru', 'firouzja2003', 'nihalsarin']
    },
    'Tactical': {
        'lichess': ['GothamChess', 'GingerGM', 'shield-wall', 'TigrVShlyape', 'BillieKimbah'],
        'chesscom': ['ghandeevam2003', 'lachesisq', 'champ2005']
    },
    'Solid': {
        'lichess': ['FabianoCaruana', 'RapidShooter', 'Night-King96', 'aladdin65', 'Polish_fighter3000'],
        'chesscom': ['fabianocaruana', 'viditchess', 'anishgiri']
    },
    'Balanced': {
        'lichess': ['DanielNaroditsky', 'Giri', 'chessexplained', 'Jospem', 'ArtemTrushko'],
        'chesscom': ['jefferyx', 'danielnaroditsky', 'gmwso']
    }
}


def collect_enhanced_dataset(
    games_per_player: int = 500,
    platforms: list = ['lichess', 'chesscom']
):
    """
    Collect comprehensive training dataset from multiple platforms.
    
    Args:
        games_per_player: Number of games to download per player
        platforms: List of platforms to collect from
    """
    output_dir = "data/training_data"
    fetcher = GameFetcher(output_dir=output_dir)
    
    dataset_summary = {
        'total_players': 0,
        'total_games_collected': 0,
        'by_style': {}
    }
    
    logger.info("\n" + "="*70)
    logger.info("ENHANCED TRAINING DATA COLLECTION")
    logger.info("="*70)
    logger.info(f"Games per player: {games_per_player}")
    logger.info(f"Platforms: {', '.join(platforms)}")
    logger.info("="*70 + "\n")
    
    for style, players_by_platform in ENHANCED_PLAYER_DATABASE.items():
        logger.info(f"\n{'='*70}")
        logger.info(f"Collecting {style} Players")
        logger.info(f"{'='*70}")
        
        style_count = 0
        
        # Lichess players
        if 'lichess' in platforms and 'lichess' in players_by_platform:
            for username in players_by_platform['lichess']:
                logger.info(f"\n📥 Lichess: {username}")
                try:
                    pgn_path = fetcher.fetch_lichess_games(
                        username,
                        max_games=games_per_player,
                        rated=True,
                        perf_type='blitz'
                    )
                    if pgn_path:
                        style_count += 1
                        dataset_summary['total_players'] += 1
                        dataset_summary['total_games_collected'] += games_per_player
                        logger.info(f"✅ Success: {username}")
                    else:
                        logger.warning(f"⚠️  Failed: {username}")
                except Exception as e:
                    logger.error(f"❌ Error for {username}: {e}")
                
                time.sleep(2)  # Rate limiting
        
        # Chess.com players
        if 'chesscom' in platforms and 'chesscom' in players_by_platform:
            for username in players_by_platform['chesscom']:
                logger.info(f"\n📥 Chess.com: {username}")
                try:
                    pgn_path = fetcher.fetch_chesscom_all_games(
                        username,
                        max_games=games_per_player
                    )
                    if pgn_path:
                        style_count += 1
                        dataset_summary['total_players'] += 1
                        dataset_summary['total_games_collected'] += games_per_player
                        logger.info(f"✅ Success: {username}")
                    else:
                        logger.warning(f"⚠️  Failed: {username}")
                except Exception as e:
                    logger.error(f"❌ Error for {username}: {e}")
                
                time.sleep(2)  # Rate limiting
        
        dataset_summary['by_style'][style] = style_count
        logger.info(f"\n{style} complete: {style_count} players")
    
    # Save summary
    summary_path = Path(output_dir).parent / "training_dataset_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(dataset_summary, f, indent=2)
    
    # Final report
    logger.info("\n" + "="*70)
    logger.info("COLLECTION COMPLETE!")
    logger.info("="*70)
    logger.info(f"Total players collected: {dataset_summary['total_players']}")
    logger.info(f"Estimated total games: ~{dataset_summary['total_games_collected']}")
    logger.info("\nBreakdown by style:")
    for style, count in dataset_summary['by_style'].items():
        logger.info(f"  {style}: {count} players")
    logger.info(f"\nData saved to: {output_dir}")
    logger.info(f"Summary saved to: {summary_path}")
    logger.info("\n" + "="*70)
    logger.info("Next steps:")
    logger.info("1. python src/models/train_style_classifier.py")
    logger.info("2. python src/models/train_position_clusters.py")
    logger.info("="*70 + "\n")
    
    return dataset_summary


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Collect enhanced training data for ChessType'
    )
    parser.add_argument(
        '--games',
        type=int,
        default=500,
        help='Games per player (default: 500)'
    )
    parser.add_argument(
        '--lichess-only',
        action='store_true',
        help='Only collect from Lichess'
    )
    parser.add_argument(
        '--chesscom-only',
        action='store_true',
        help='Only collect from Chess.com'
    )
    
    args = parser.parse_args()
    
    # Determine platforms
    if args.lichess_only:
        platforms = ['lichess']
    elif args.chesscom_only:
        platforms = ['chesscom']
    else:
        platforms = ['lichess', 'chesscom']
    
    # Run collection
    summary = collect_enhanced_dataset(
        games_per_player=args.games,
        platforms=platforms
    )
    
    logger.info("\n✅ Training data collection complete!")
    logger.info(f"Collected {summary['total_players']} players")
    logger.info(f"Estimated {summary['total_games_collected']} games")


if __name__ == "__main__":
    main()