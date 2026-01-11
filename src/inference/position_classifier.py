"""
Load and use trained position clustering/classification model.
"""
import pickle
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import chess
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PositionClassifier:
    """Classify positions using trained clustering model."""
    
    def __init__(self, model_dir: str = "models", data_dir: str = "data"):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.model = None
        self.scaler = None
        self.cluster_names = {}
        
        self.load_model()
    
    def load_model(self):
        """Load trained model, scaler, and cluster names."""
        model_path = self.model_dir / "position_clusters.pkl"
        scaler_path = self.model_dir / "position_scaler.pkl"
        names_path = self.data_dir / "cluster_names.json"
        
        if not model_path.exists() or not scaler_path.exists():
            raise FileNotFoundError(
                f"Model files not found in {self.model_dir}. "
                "Please train the model first using train_position_clusters.py"
            )
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        if names_path.exists():
            with open(names_path, 'r') as f:
                self.cluster_names = json.load(f)
        else:
            # Default names if file doesn't exist
            self.cluster_names = {
                str(i): f"Type_{i}" for i in range(self.model.n_clusters)
            }
        
        logger.info("Position classifier loaded successfully")
    
    def classify_position(self, position_features: Dict) -> Dict:
        """
        Classify a position.
        
        Args:
            position_features: Dictionary of position features
            
        Returns:
            Dictionary with classification results
        """
        # Create feature vector
        feature_vector = np.array([list(position_features.values())])
        
        # Scale features
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Predict cluster
        cluster_id = self.model.predict(feature_vector_scaled)[0]
        cluster_name = self.cluster_names.get(str(cluster_id), f"Type_{cluster_id}")
        
        # Calculate distance to cluster center (confidence proxy)
        distances = self.model.transform(feature_vector_scaled)[0]
        confidence = 1.0 / (1.0 + distances[cluster_id])
        
        return {
            'position_type': cluster_name,
            'cluster_id': int(cluster_id),
            'confidence': float(confidence)
        }
    
    def analyze_position_distribution(
        self,
        position_classifications: List[Dict]
    ) -> Dict:
        """
        Analyze distribution of position types.
        
        Args:
            position_classifications: List of classification results
            
        Returns:
            Dictionary with distribution statistics
        """
        if not position_classifications:
            return {}
        
        # Count position types
        type_counts = {}
        for classification in position_classifications:
            pos_type = classification['position_type']
            type_counts[pos_type] = type_counts.get(pos_type, 0) + 1
        
        total = len(position_classifications)
        
        # Calculate percentages
        distribution = {
            pos_type: (count / total) * 100
            for pos_type, count in type_counts.items()
        }
        
        return distribution


def main():
    """Test position classification."""
    from src.feature_engineering.position_features import PositionFeatureExtractor
    
    classifier = PositionClassifier()
    extractor = PositionFeatureExtractor()
    
    # Test positions
    test_positions = [
        chess.Board(),  # Starting position
        chess.Board("r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),  # Open
        chess.Board("8/5k2/3p4/1p1Pp3/pP2Pp2/P4P2/8/6K1 w - - 0 1")  # Endgame
    ]
    
    for i, board in enumerate(test_positions):
        print(f"\n=== Position {i + 1} ===")
        print(board)
        
        features = extractor.extract_features(board)
        result = classifier.classify_position(features)
        
        print(f"\nClassification: {result['position_type']}")
        print(f"Confidence: {result['confidence']:.2%}")


if __name__ == "__main__":
    main()
