"""
Generate personalized training recommendations.
"""
from typing import Dict, List


class RecommendationEngine:
    """Generate personalized improvement recommendations."""
    
    OPENING_RECOMMENDATIONS = {
        'Positional': [
            "Queen's Gambit Declined",
            "Caro-Kann Defense",
            "Nimzo-Indian Defense",
            "English Opening",
            "Catalan Opening"
        ],
        'Aggressive': [
            "Sicilian Najdorf",
            "King's Indian Defense",
            "Dragon Variation",
            "King's Gambit",
            "Smith-Morra Gambit"
        ],
        'Tactical': [
            "Sicilian Dragon",
            "King's Indian Attack",
            "Alekhine Defense",
            "Benko Gambit",
            "Grünfeld Defense"
        ],
        'Solid': [
            "Caro-Kann Defense",
            "Queen's Gambit Declined",
            "Petroff Defense",
            "Berlin Defense",
            "French Defense"
        ],
        'Balanced': [
            "Ruy Lopez",
            "Italian Game",
            "Queen's Indian Defense",
            "Nimzo-Indian Defense",
            "Semi-Slav Defense"
        ]
    }
    
    def generate_recommendations(self, style: str, position_analysis: Dict) -> Dict:
        """Generate comprehensive recommendations."""
        openings = self.OPENING_RECOMMENDATIONS.get(style, [])
        
        improvement_tips = self._generate_improvement_tips(style, position_analysis)
        training_focus = self._generate_training_focus(style, position_analysis)
        
        return {
            'openings': openings,
            'improvement_tips': improvement_tips,
            'training_focus': training_focus
        }
    
    def _generate_improvement_tips(self, style: str, position_analysis: Dict) -> List[str]:
        """Generate improvement tips based on weak areas."""
        tips = []
        
        # General tips by style
        style_tips = {
            'Positional': "Continue leveraging your positional understanding in endgames.",
            'Aggressive': "Practice converting advantages without overextending.",
            'Tactical': "Work on positional play to complement your tactical skills.",
            'Solid': "Look for more winning attempts in favorable positions.",
            'Balanced': "Continue developing all aspects of your game evenly."
        }
        
        tips.append(style_tips.get(style, "Keep improving all aspects of your game."))
        
        # Position-specific tips
        worst = position_analysis.get('worst_position_type')
        if worst:
            worst_type = worst['type']
            tips.append(f"Practice puzzles and games in {worst_type} positions to improve your weakest area.")
        
        best = position_analysis.get('best_position_type')
        if best:
            best_type = best['type']
            tips.append(f"Leverage your strength in {best_type} positions during games.")
        
        return tips
    
    def _generate_training_focus(self, style: str, position_analysis: Dict) -> List[str]:
        """Generate specific training focus areas."""
        focus = []
        
        # Style-specific training
        style_focus = {
            'Positional': ["Study endgame masterclasses", "Analyze Carlsen's games"],
            'Aggressive': ["Practice attacking patterns", "Study Kasparov's combinations"],
            'Tactical': ["Solve tactical puzzles daily", "Study Tal's sacrifices"],
            'Solid': ["Work on prophylaxis", "Study Petrosian's games"],
            'Balanced': ["Maintain well-rounded training", "Study modern grandmaster games"]
        }
        
        focus.extend(style_focus.get(style, []))
        
        # Weak position type training
        worst = position_analysis.get('worst_position_type')
        if worst:
            worst_type = worst['type']
            focus.append(f"Dedicate 30% of training time to {worst_type} positions")
        
        return focus