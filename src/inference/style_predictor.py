"""
Load and use trained style classifier for predictions.
"""
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StylePredictor:
    """Predict playing style using trained model."""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        
        self.load_model()
    
    def load_model(self):
        """Load trained model and scaler."""
        model_path = self.model_dir / "style_classifier.pkl"
        scaler_path = self.model_dir / "feature_scaler.pkl"
        
        if not model_path.exists() or not scaler_path.exists():
            raise FileNotFoundError(
                f"Model files not found in {self.model_dir}. "
                "Please train the model first using train_style_classifier.py"
            )
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        logger.info("Style classifier loaded successfully")
    
    def predict(self, features: Dict) -> Dict:
        """
        Predict playing style from game features.
        
        Args:
            features: Dictionary of game features
            
        Returns:
            Dictionary with prediction results
        """
        # Feature names in correct order
        feature_names = [
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
        
        # Create feature vector
        feature_vector = np.array([[features[name] for name in feature_names]])
        
        # Scale features
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Predict
        prediction = self.model.predict(feature_vector_scaled)[0]
        probabilities = self.model.predict_proba(feature_vector_scaled)[0]
        
        # Get class labels
        classes = self.model.classes_
        
        # Create probability dictionary
        style_probabilities = {
            cls: float(prob) for cls, prob in zip(classes, probabilities)
        }
        
        # Sort by probability
        sorted_styles = sorted(style_probabilities.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'predicted_style': prediction,
            'confidence': float(probabilities.max()),
            'style_probabilities': style_probabilities,
            'top_3_styles': [style for style, _ in sorted_styles[:3]]
        }
    
    def get_style_description(self, style: str) -> Dict:
        """Get description and recommendations for a playing style."""
        descriptions = {
            'Positional': {
                'description': 'You excel at long-term planning and strategic maneuvering. '
                              'Your strength lies in building superior positions through '
                              'careful piece placement and pawn structure.',
                'strengths': [
                    'Endgame technique',
                    'Prophylactic thinking',
                    'Long-term planning',
                    'Pawn structure understanding'
                ],
                'weaknesses': [
                    'May miss tactical opportunities',
                    'Can be slow to adapt to sharp positions'
                ],
                'famous_players': ['Magnus Carlsen', 'Anatoly Karpov', 'Tigran Petrosian']
            },
            'Aggressive': {
                'description': 'You play with energy and initiative, constantly creating threats '
                              'and putting pressure on your opponents. Your style emphasizes '
                              'active piece play and calculated risks.',
                'strengths': [
                    'Creating attacking chances',
                    'Tactical alertness',
                    'Time pressure handling',
                    'Initiative and momentum'
                ],
                'weaknesses': [
                    'May overextend positions',
                    'Can struggle in slow, positional games'
                ],
                'famous_players': ['Hikaru Nakamura', 'Garry Kasparov', 'Bobby Fischer']
            },
            'Tactical': {
                'description': 'Your game revolves around concrete calculation and tactical '
                              'combinations. You excel at finding forcing moves and creating '
                              'complex tactical situations.',
                'strengths': [
                    'Pattern recognition',
                    'Tactical calculation',
                    'Spotting combinations',
                    'Sharp positions'
                ],
                'weaknesses': [
                    'May neglect positional play',
                    'Can overlook quiet moves'
                ],
                'famous_players': ['Mikhail Tal', 'Alexander Alekhine', 'Judit Polgar']
            },
            'Solid': {
                'description': 'You play reliable, principled chess with minimal mistakes. '
                              'Your strength is in maintaining balanced positions and '
                              'converting small advantages.',
                'strengths': [
                    'Consistency',
                    'Defensive technique',
                    'Risk management',
                    'Patience'
                ],
                'weaknesses': [
                    'May lack winning attempts',
                    'Can be too passive'
                ],
                'famous_players': ['Fabiano Caruana', 'Vladimir Kramnik', 'Viswanathan Anand']
            },
            'Balanced': {
                'description': 'You adapt your play to the position, combining positional '
                              'understanding with tactical awareness. Your versatility makes '
                              'you unpredictable.',
                'strengths': [
                    'Versatility',
                    'Adaptability',
                    'Well-rounded skills',
                    'Objective evaluation'
                ],
                'weaknesses': [
                    'May lack a distinct identity',
                    'Could specialize further'
                ],
                'famous_players': ['Levon Aronian', 'Wesley So', 'Maxime Vachier-Lagrave']
            }
        }
        
        return descriptions.get(style, {
            'description': 'Style analysis in progress.',
            'strengths': [],
            'weaknesses': [],
            'famous_players': []
        })


def main():
    """Test style prediction."""
    predictor = StylePredictor()
    
    # Example features (from a positional player)
    sample_features = {
        'avg_piece_activity': 26.5,
        'pawn_moves_ratio': 0.35,
        'avg_piece_trades': 1.8,
        'avg_center_control': 4.2,
        'captures_per_game': 6.5,
        'checks_per_game': 2.1,
        'castles_early_pct': 0.85,
        'complex_positions_pct': 0.45,
        'aggressive_openings_pct': 0.15
    }
    
    result = predictor.predict(sample_features)
    description = predictor.get_style_description(result['predicted_style'])
    
    print("\n=== Style Prediction ===")
    print(f"Predicted Style: {result['predicted_style']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nStyle Probabilities:")
    for style, prob in result['style_probabilities'].items():
        print(f"  {style}: {prob:.2%}")
    print(f"\nDescription: {description['description']}")
    print(f"\nStrengths: {', '.join(description['strengths'])}")
    print(f"\nFamous Players: {', '.join(description['famous_players'])}")


if __name__ == "__main__":
    main()