"""
Train the playing style classifier.
"""
import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data_collection.parse_pgn import PGNParser
from src.feature_engineering.game_features import GameFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StyleClassifierTrainer:
    """Train the playing style classification model."""
    
    FEATURE_NAMES = [
        'avg_piece_activity',
        'pawn_moves_ratio',
        'avg_piece_trades',
        'avg_center_control',
        'captures_per_game',
        'checks_per_game',
        'castles_early_pct',
        'complex_positions_pct',
        'aggressive_openings_pct'
    ]
    
    STYLE_LABELS = {
        'DrNykterstein': 'Positional',
        'LyonBeast': 'Positional',
        'penguingim1': 'Positional',
        'Hikaru': 'Aggressive',
        'GMHIKARU': 'Aggressive',
        'GothamChess': 'Tactical',
        'GingerGM': 'Tactical',
        'shield-wall': 'Tactical',
        'FabianoCaruana': 'Solid',
        'RapidShooter': 'Solid',
        'Night-King96': 'Solid',
        'aladdin65': 'Solid',
        'DanielNaroditsky': 'Balanced',
        'Giri': 'Balanced',
        'chessexplained': 'Balanced'
    }
    
    def __init__(self, data_dir: str = "data/training_data", model_dir: str = "models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.parser = PGNParser()
        self.extractor = GameFeatureExtractor()
        
        self.model = None
        self.scaler = None
    
    def build_training_dataset(self) -> tuple:
        """
        Build training dataset from PGN files.
        
        Returns:
            X (features), y (labels), player_names
        """
        logger.info("Building training dataset...")
        
        X = []
        y = []
        player_names = []
        
        # Iterate through PGN files
        for pgn_file in self.data_dir.glob("*_lichess.pgn"):
            # Extract player name from filename
            player_name = pgn_file.stem.replace("_lichess", "")
            
            if player_name not in self.STYLE_LABELS:
                logger.warning(f"Unknown player: {player_name}, skipping...")
                continue
            
            style = self.STYLE_LABELS[player_name]
            logger.info(f"Processing {player_name} ({style})...")
            
            # Parse games
            games = self.parser.parse_pgn_file(str(pgn_file))
            
            if not games:
                logger.warning(f"No games found for {player_name}")
                continue
            
            # Extract features from games
            game_features = self.extractor.extract_features_from_games(games, player_name)
            
            if not game_features:
                logger.warning(f"No features extracted for {player_name}")
                continue
            
            # Aggregate features (average across all games)
            aggregated = self.extractor.aggregate_features(game_features)
            
            # Create feature vector in correct order
            feature_vector = [aggregated[name] for name in self.FEATURE_NAMES]
            
            X.append(feature_vector)
            y.append(style)
            player_names.append(player_name)
            
            logger.info(f"  Extracted {len(game_features)} games")
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"\nDataset built: {len(X)} players")
        logger.info(f"Feature shape: {X.shape}")
        logger.info(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        # Save dataset
        dataset = {
            'X': X.tolist(),
            'y': y.tolist(),
            'player_names': player_names,
            'feature_names': self.FEATURE_NAMES
        }
        
        dataset_path = self.data_dir.parent / "training_dataset.json"
        with open(dataset_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        logger.info(f"Dataset saved to {dataset_path}")
        
        return X, y, player_names
    
    def train(self, X, y):
        """
        Train the Random Forest classifier.
        
        Args:
            X: Feature matrix
            y: Labels
        """
        logger.info("\n=== Training Style Classifier ===")
        
        # Need at least 2 classes for stratified split
        if len(np.unique(y)) < 2:
            logger.error(f"Need at least 2 classes to train. Found: {np.unique(y)}")
            logger.info("Please download more training data from different style categories")
            return
        
        # Split data (if enough samples)
        if len(X) >= 4:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if len(X) >= 10 else None
            )
        else:
            logger.warning("Not enough data for train/test split. Using all data for training.")
            X_train, X_test, y_train, y_test = X, X, y, y
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,  # Reduced for small datasets
            random_state=42,
            n_jobs=-1
        )
        
        logger.info("Training Random Forest...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        logger.info(f"\nTraining accuracy: {train_score:.3f}")
        logger.info(f"Test accuracy: {test_score:.3f}")
        
        # Cross-validation (if enough samples)
        if len(X_train) >= 5:
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=min(3, len(X_train)))
            logger.info(f"Cross-validation scores: {cv_scores}")
            logger.info(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        # Detailed evaluation
        y_pred = self.model.predict(X_test_scaled)
        
        logger.info("\n=== Classification Report ===")
        print(classification_report(y_test, y_pred))
        
        logger.info("\n=== Confusion Matrix ===")
        print(confusion_matrix(y_test, y_pred))
        
        # Feature importance
        logger.info("\n=== Feature Importance ===")
        importances = self.model.feature_importances_
        for name, importance in sorted(zip(self.FEATURE_NAMES, importances), 
                                       key=lambda x: x[1], reverse=True):
            logger.info(f"{name}: {importance:.3f}")
    
    def save_model(self):
        """Save trained model and scaler."""
        if self.model is None or self.scaler is None:
            logger.error("No model to save. Train model first.")
            return
        
        model_path = self.model_dir / "style_classifier.pkl"
        scaler_path = self.model_dir / "feature_scaler.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        logger.info(f"\nModel saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")


def main():
    """Main training pipeline."""
    trainer = StyleClassifierTrainer()
    
    # Build dataset
    X, y, player_names = trainer.build_training_dataset()
    
    # Train model
    if len(X) > 0:
        trainer.train(X, y)
        trainer.save_model()
    else:
        logger.error("No training data available. Run fetch_games.py first.")


if __name__ == "__main__":
    main()