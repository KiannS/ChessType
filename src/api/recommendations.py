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
    
    PAWN_STRUCTURE_RECOMMENDATIONS = {
        'Positional': {
            'study': [
                "Isolated Queen's Pawn (IQP) - Learn to exploit the space advantage",
                "Carlsbad Structure - Master the minority attack",
                "Stonewall Formation - Control key central squares",
                "Maroczy Bind - Restrict opponent's counterplay"
            ],
            'avoid': [
                "Shattered pawns - Maintain solid pawn chains",
                "Weakened king safety - Don't compromise your fortress"
            ]
        },
        'Aggressive': {
            'study': [
                "Dragon Structure - Launch kingside attacks",
                "King's Indian pawn storm - Master the g4-g5-g6 attack",
                "Benoni pawn chain - Create dynamic imbalances",
                "Open center with space - Maximize piece activity"
            ],
            'avoid': [
                "Symmetrical structures - Seek imbalances",
                "Locked positions - You thrive in open play"
            ]
        },
        'Tactical': {
            'study': [
                "Open diagonals - Maximize bishop power",
                "Weak squares around enemy king - Create tactical targets",
                "Pawn breaks (d4-d5, e4-e5) - Open lines for pieces",
                "Hanging pawns - Learn to attack or defend them"
            ],
            'avoid': [
                "Simplified structures - Keep tension",
                "Blocked positions - You need tactical opportunities"
            ]
        },
        'Solid': {
            'study': [
                "Berlin Wall - Endgame-oriented structure",
                "Symmetrical structures - Minimize weaknesses",
                "Hedgehog formation - Flexible and resilient",
                "French pawn chain - Solid but dynamic"
            ],
            'avoid': [
                "Overextended pawns - Don't create weaknesses",
                "Premature pawn breaks - Be patient"
            ]
        },
        'Balanced': {
            'study': [
                "Classical pawn center - Understand d4-e4 structures",
                "Flexible pawn formations - Adapt to positions",
                "Spanish structures - Rich middlegame plans",
                "English Opening setups - Hypermodern flexibility"
            ],
            'avoid': [
                "One-dimensional structures - Maintain flexibility",
                "Committal pawn moves too early - Stay adaptable"
            ]
        }
    }
    
    ENDGAME_RECOMMENDATIONS = {
        'Positional': {
            'strong': [
                "Rook endgames - Your technique shines here",
                "Queen endgames - Excellent maneuvering skills",
                "Minor piece endgames - Superior understanding of key squares",
                "Pawn endgames - Strong calculation and technique"
            ],
            'improve': [
                "Tactical endgames - Practice concrete variations",
                "Time pressure endgames - Improve speed of play"
            ]
        },
        'Aggressive': {
            'strong': [
                "Active piece endgames - Your aggressive style translates well",
                "Endgames with passed pawns - Push for initiative",
                "Opposite-colored bishop endgames - Create threats"
            ],
            'improve': [
                "Passive defense endgames - Learn to hold worse positions",
                "Fortress positions - Practice defensive techniques",
                "Quiet rook endgames - Improve patience and technique"
            ]
        },
        'Tactical': {
            'strong': [
                "Complex rook endgames - Your calculation helps here",
                "Endgames with tactical shots - Natural strength",
                "Queen endgames - Exploit tactical opportunities"
            ],
            'improve': [
                "Simple theoretical endgames - Study basic positions",
                "Quiet maneuvering endgames - Build patience",
                "Technical conversion - Practice winning positions methodically"
            ]
        },
        'Solid': {
            'strong': [
                "Defensive endgames - Excellent holding technique",
                "Simplified endgames - Reliable and accurate",
                "Theoretical endgames - Good preparation shows",
                "Drawn endgames - Know when to hold"
            ],
            'improve': [
                "Winning attempts in equal endgames - Be more ambitious",
                "Complex endgames - Take more risks when appropriate",
                "Time pressure situations - Speed up decisions"
            ]
        },
        'Balanced': {
            'strong': [
                "Various endgame types - Well-rounded skills",
                "Adaptive play - Adjust to the position's demands",
                "Practical endgames - Good judgment"
            ],
            'improve': [
                "Specialize in specific endgame types - Deepen knowledge",
                "Master rare endgame patterns - Study uncommon positions",
                "Advanced rook endgames - Focus on this critical type"
            ]
        }
    }
    
    def generate_recommendations(self, style: str, position_analysis: Dict) -> Dict:
        """Generate comprehensive recommendations."""
        openings = self.OPENING_RECOMMENDATIONS.get(style, [])
        pawn_structures = self.PAWN_STRUCTURE_RECOMMENDATIONS.get(style, {})
        endgames = self.ENDGAME_RECOMMENDATIONS.get(style, {})
        
        improvement_tips = self._generate_improvement_tips(style, position_analysis)
        training_focus = self._generate_training_focus(style, position_analysis)
        
        # Add endgame recommendations based on position analysis
        endgame_performance = self._analyze_endgame_performance(position_analysis)
        
        return {
            'openings': openings,
            'pawn_structures': pawn_structures,
            'endgames': endgames,
            'endgame_performance': endgame_performance,
            'improvement_tips': improvement_tips,
            'training_focus': training_focus
        }
    
    def _analyze_endgame_performance(self, position_analysis: Dict) -> Dict:
        """Analyze endgame performance from position analysis."""
        performance = position_analysis.get('performance_by_type', {})
        endgame_stats = performance.get('Endgame', {})
        
        if not endgame_stats:
            return {
                'assessment': 'Not enough endgame data',
                'recommendation': 'Play more games to get endgame analysis'
            }
        
        win_rate = endgame_stats.get('win_rate', 0)
        rating = endgame_stats.get('rating', 'Unknown')
        
        if rating == 'Excellent' or win_rate > 0.65:
            assessment = 'Strong endgame player - Continue leveraging this advantage'
            focus = 'Study advanced endgame techniques to maintain your edge'
        elif rating == 'Strong' or win_rate > 0.55:
            assessment = 'Good endgame technique - Room for improvement'
            focus = 'Study classic endgame books like Dvoretsky\'s Endgame Manual'
        elif rating == 'Average' or win_rate > 0.45:
            assessment = 'Average endgame play - This is a key area to improve'
            focus = 'Focus on fundamental endgame positions and principles'
        else:
            assessment = 'Endgames need work - Major improvement opportunity'
            focus = 'Start with basic endgame theory: K+P, R+P vs R, basic checkmates'
        
        return {
            'win_rate': win_rate,
            'rating': rating,
            'assessment': assessment,
            'focus': focus
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
        
        # Endgame-specific tip
        performance = position_analysis.get('performance_by_type', {})
        endgame_stats = performance.get('Endgame', {})
        if endgame_stats:
            endgame_rating = endgame_stats.get('rating', 'Unknown')
            if endgame_rating in ['Weak', 'Average']:
                tips.append("Dedicate time to endgame study - this will significantly boost your results.")
        
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
        
        # Endgame training based on performance
        performance = position_analysis.get('performance_by_type', {})
        endgame_stats = performance.get('Endgame', {})
        if endgame_stats:
            endgame_rating = endgame_stats.get('rating', 'Unknown')
            if endgame_rating in ['Weak', 'Average']:
                focus.append("Complete Silman's Endgame Course or similar structured endgame training")
                focus.append("Practice 10 endgame positions per day")
        
        return focus