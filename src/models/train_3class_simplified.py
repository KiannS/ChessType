"""
SIMPLIFIED 3-CLASS TRAINER
Combines overlapping styles into clearer categories: Aggressive, Positional, Balanced
"""
import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data_collection.parse_pgn import PGNParser
from src.feature_engineering.enhanced_game_features import EnhancedGameFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Simplified3ClassTrainer:
    """3-class simplified trainer for better accuracy."""
    
    FEATURE_NAMES = [
        'avg_piece_activity',
        'pawn_moves_ratio',
        'avg_piece_trades',
        'avg_center_control',
        'captures_per_game',
        'checks_per_game',
        'castles_early_pct',
        'complex_positions_pct',
        'aggressive_openings_pct',
        'king_safety_priority',
        'piece_sacrifice_rate',
        'endgame_preference',
        'positional_sacrifice_rate',
        'pawn_storm_frequency',
        'prophylaxis_moves',
        'center_pawn_moves',
        'opposite_castling_rate',
        'early_queen_development',
        'long_term_pawn_pushes',
        'weak_square_creation'
    ]
    
    # SIMPLIFIED: Tactical + Aggressive → "Aggressive"
    #             Positional + Solid → "Positional"  
    #             Balanced stays "Balanced"
    STYLE_LABELS = {
        # Aggressive (includes Tactical)
        'GMHIKARU': 'Aggressive',
        'Night-King96': 'Aggressive',
        'aladdin65': 'Aggressive',
        'nihalsarin': 'Aggressive',
        'GingerGM': 'Aggressive',  # Was Tactical
        'hikaru': 'Aggressive',
        'firouzja2003': 'Aggressive',
        'gmwso': 'Aggressive',
        'jefferyx': 'Aggressive',
        'ghandeevam2003': 'Aggressive',  # Was Tactical
        'lachesisq': 'Aggressive',  # Was Tactical
        'champ2005': 'Aggressive',  # Was Tactical
        'danielnaroditsky': 'Aggressive',  # Was Tactical
        
        # Positional (includes Solid)
        'DrNykterstein': 'Positional',
        'penguingim1': 'Positional',
        'Msb2': 'Positional',
        'RebeccaHarris': 'Positional',
        'chessbrahs': 'Positional',
        'GMWSO': 'Positional',
        'Giri': 'Positional',  # Was Solid
        'Zhigalko_Sergei': 'Positional',  # Was Solid
        'magnuscarlsen': 'Positional',
        'levon_aronian': 'Positional',
        'chessnetwork': 'Positional',
        'wesleyso': 'Positional',
        'anishgiri': 'Positional',
        'fabianocaruana': 'Positional',  # Was Solid
        'viditchess': 'Positional',  # Was Solid
        'duhless': 'Positional',  # Was Solid
        
        # Balanced
        'chessexplained': 'Balanced',
        'EricRosen': 'Balanced',
        'ChessWarrior7197': 'Balanced',
        'chessbrah': 'Balanced',
        'imrosen': 'Balanced',
    }
    
    def __init__(self, data_dir: str = "data/training_data", model_dir: str = "models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.parser = PGNParser()
        self.extractor = EnhancedGameFeatureExtractor()
        
        self.model = None
        self.scaler = None
    
    def build_dataset(self, max_games_per_player: int = 150) -> tuple:
        """Build 3-class dataset."""
        logger.info(f"Building 3-CLASS dataset (max {max_games_per_player} games/player)...")
        
        X = []
        y = []
        
        import itertools
        all_files = list(itertools.chain(
            self.data_dir.glob("*_lichess.pgn"),
            self.data_dir.glob("*_chesscom*.pgn")
        ))
        
        for idx, pgn_file in enumerate(all_files, 1):
            player_name = pgn_file.stem.replace("_lichess", "").replace("_chesscom_all_time", "")
            
            if player_name not in self.STYLE_LABELS:
                continue
            
            style = self.STYLE_LABELS[player_name]
            logger.info(f"[{idx}/{len(all_files)}] {player_name} → {style}")
            
            games = self.parser.parse_pgn_file(str(pgn_file))
            if not games:
                continue
            
            games = games[:max_games_per_player]
            game_features = self.extractor.extract_features_from_games(games, player_name)
            
            for features in game_features:
                try:
                    feature_vector = [features[name] for name in self.FEATURE_NAMES]
                    X.append(feature_vector)
                    y.append(style)
                except:
                    continue
            
            logger.info(f"  ✓ {len(game_features)} games")
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"3-CLASS DATASET: {len(X)} games")
        logger.info(f"Distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        logger.info(f"{'='*70}\n")
        
        return X, y
    
    def train(self, X, y):
        logger.info("=== Training 3-Class Model ===\n")
        
        # Balance classes
        from sklearn.utils import resample
        unique, counts = np.unique(y, return_counts=True)
        min_count = counts.min()
        
        logger.info(f"Balancing to {min_count} per class...")
        
        X_balanced = []
        y_balanced = []
        
        for label in unique:
            mask = y == label
            X_resampled, y_resampled = resample(
                X[mask], y[mask],
                n_samples=min_count,
                random_state=42,
                replace=False
            )
            X_balanced.append(X_resampled)
            y_balanced.append(y_resampled)
        
        X = np.vstack(X_balanced)
        y = np.hstack(y_balanced)
        
        logger.info(f"Balanced: {len(X)} games\n")
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train SIMPLER model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,  # Reduced
            min_samples_split=20,  # Increased
            min_samples_leaf=10,  # Increased
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        logger.info(f"Training: {train_score:.3f} ({train_score*100:.1f}%)")
        logger.info(f"Test: {test_score:.3f} ({test_score*100:.1f}%)")
        logger.info(f"Overfitting gap: {(train_score - test_score)*100:.1f}%\n")
        
        y_pred = self.model.predict(X_test_scaled)
        
        print(classification_report(y_test, y_pred, digits=3))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Feature importance
        logger.info("\nTop Features:")
        importances = self.model.feature_importances_
        for name, imp in sorted(zip(self.FEATURE_NAMES, importances), 
                                key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"  {name}: {imp:.3f}")
    
    def save_model(self):
        model_path = self.model_dir / "style_classifier.pkl"
        scaler_path = self.model_dir / "feature_scaler.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        logger.info(f"\n✓ Saved to {model_path}")


def main():
    trainer = Simplified3ClassTrainer()
    X, y = trainer.build_dataset(max_games_per_player=150)
    
    if len(X) > 0:
        trainer.train(X, y)
        trainer.save_model()
        logger.info("\n✅ 3-CLASS MODEL READY!")
    else:
        logger.error("No data")


if __name__ == "__main__":
    main()