"""
Train position clustering model (K-Means).
"""
import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data_collection.parse_pgn import PGNParser
from src.feature_engineering.position_features import PositionFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PositionClusterTrainer:
    """Train position clustering model."""
    
    def __init__(self, data_dir: str = "data/training_data", model_dir: str = "models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.parser = PGNParser()
        self.extractor = PositionFeatureExtractor()
        
        self.model = None
        self.scaler = None
    
    def extract_positions_from_games(self, max_positions: int = 5000) -> np.ndarray:
        """
        Extract position features from training games.
        
        Args:
            max_positions: Maximum number of positions to extract
            
        Returns:
            Feature matrix
        """
        logger.info(f"Extracting up to {max_positions} positions...")
        
        all_features = []
        positions_per_game = 10  # Sample 10 positions per game
        
        pgn_files = list(self.data_dir.glob("*_lichess.pgn"))
        
        if not pgn_files:
            logger.error("No PGN files found. Run fetch_games.py first.")
            return np.array([])
        
        for pgn_file in pgn_files:
            if len(all_features) >= max_positions:
                break
            
            logger.info(f"Processing {pgn_file.name}...")
            games = self.parser.parse_pgn_file(str(pgn_file))
            
            for game in games[:20]:  # First 20 games per file
                if len(all_features) >= max_positions:
                    break
                
                positions = self.parser.get_positions(game)
                
                # Sample positions evenly throughout the game
                if len(positions) > positions_per_game:
                    indices = np.linspace(0, len(positions) - 1, positions_per_game, dtype=int)
                    sampled_positions = [positions[i] for i in indices]
                else:
                    sampled_positions = positions
                
                for board in sampled_positions:
                    if len(all_features) >= max_positions:
                        break
                    
                    try:
                        features = self.extractor.extract_features(board)
                        feature_vector = list(features.values())
                        all_features.append(feature_vector)
                    except Exception as e:
                        logger.warning(f"Error extracting features: {e}")
                        continue
        
        X = np.array(all_features)
        logger.info(f"Extracted {len(X)} positions with {X.shape[1] if len(X) > 0 else 0} features each")
        
        # Save position dataset
        if len(X) > 0:
            dataset_path = self.data_dir.parent / "positions_dataset.json"
            
            # Get feature names from a sample position
            sample_board = self.parser.get_positions(self.parser.parse_pgn_file(str(pgn_files[0]))[0])[0]
            feature_names = list(self.extractor.extract_features(sample_board).keys())
            
            with open(dataset_path, 'w') as f:
                json.dump({
                    'num_positions': len(X),
                    'num_features': X.shape[1],
                    'feature_names': feature_names
                }, f, indent=2)
        
        return X
    
    def train(self, X, n_clusters: int = 6):
        """
        Train K-Means clustering.
        
        Args:
            X: Feature matrix
            n_clusters: Number of clusters
        """
        logger.info(f"\n=== Training Position Clustering (k={n_clusters}) ===")
        
        if len(X) == 0:
            logger.error("No data to train on!")
            return
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train K-Means
        self.model = KMeans(
            n_clusters=n_clusters,
            init='k-means++',
            n_init=20,
            max_iter=500,
            random_state=42,
        )
        
        logger.info("Training K-Means...")
        self.model.fit(X_scaled)
        
        # Evaluate
        labels = self.model.labels_
        silhouette = silhouette_score(X_scaled, labels)
        inertia = self.model.inertia_
        
        logger.info(f"\nSilhouette Score: {silhouette:.3f}")
        logger.info(f"Inertia: {inertia:.2f}")
        logger.info(f"Cluster sizes: {np.bincount(labels)}")
        
        # Analyze and name clusters
        logger.info("\n=== Analyzing Clusters ===")
        cluster_names = self._analyze_and_name_clusters(X_scaled, labels)
        
        # Save cluster names
        cluster_names_path = self.data_dir.parent / "cluster_names.json"
        with open(cluster_names_path, 'w') as f:
            json.dump(cluster_names, f, indent=2)
        
        logger.info(f"\nCluster names saved to {cluster_names_path}")
    
    def _analyze_and_name_clusters(self, X_scaled, labels):
        """Analyze clusters and assign names based on characteristics."""
        cluster_names = {}
        
        for cluster_id in range(self.model.n_clusters):
            cluster_mask = labels == cluster_id
            cluster_data = X_scaled[cluster_mask]
            
            # Calculate mean features for this cluster
            mean_features = cluster_data.mean(axis=0)
            
            logger.info(f"\nCluster {cluster_id}:")
            logger.info(f"  Size: {cluster_mask.sum()}")
            
            # Simple heuristic naming
            name = self._infer_cluster_name(mean_features)
            cluster_names[str(cluster_id)] = name
            logger.info(f"  Assigned name: {name}")
        
        return cluster_names
    
    def _infer_cluster_name(self, mean_features):
        """Infer cluster name from feature values (simplified heuristic)."""
        # Feature indices (based on PositionFeatureExtractor order)
        # 0: piece_count, 7: hanging_pieces, 18: position_stability
        
        piece_count = mean_features[0] if len(mean_features) > 0 else 0
        hanging = mean_features[7] if len(mean_features) > 7 else 0
        stability = mean_features[18] if len(mean_features) > 18 else 0
        
        # Simple classification logic
        if piece_count < -0.5:  # Low piece count (normalized)
            return "Endgame"
        elif hanging > 0.5:  # High hanging pieces
            return "Chaotic"
        elif stability > 0.5:  # High stability
            return "Quiet"
        elif stability < -0.5:  # Low stability
            return "Sharp"
        elif hanging > 0:
            return "Tactical"
        else:
            return "Balanced"
    
    def save_model(self):
        """Save trained model and scaler."""
        if self.model is None or self.scaler is None:
            logger.error("No model to save. Train model first.")
            return
        
        model_path = self.model_dir / "position_clusters.pkl"
        scaler_path = self.model_dir / "position_scaler.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        logger.info(f"\nModel saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")


def main():
    """Main training pipeline."""
    trainer = PositionClusterTrainer()
    
    # Extract positions
    X = trainer.extract_positions_from_games(max_positions=5000)
    
    # Train clustering
    if len(X) > 0:
        trainer.train(X, n_clusters=6)
        trainer.save_model()
    else:
        logger.error("No position data available.")


if __name__ == "__main__":
    main()