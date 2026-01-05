# ChessType

A machine learning-powered chess personality analyzer that identifies playing styles and provides personalized improvement recommendations.

## Features
- Playing style classification (Positional, Aggressive, Tactical, Solid, Balanced)
- Position type analysis (Quiet, Chaotic, Tactical, Sharp, Endgame, Balanced)
- Performance metrics by position type
- Personalized training recommendations
- Integration with Chess.com and Lichess

## Quick Start

### Installation
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Training Models
```bash
# 1. Collect training data
python src/data_collection/fetch_games.py

# 2. Train style classifier
python src/models/train_style_classifier.py

# 3. Train position classifier
python src/models/train_position_clusters.py
```

### Running the API
```bash
python src/api/app.py
```

### API Usage
```bash
# Analyze a Lichess player
curl -X POST http://localhost:5000/analyze-lichess \
  -H "Content-Type: application/json" \
  -d '{"username": "DrNykterstein", "max_games": 50}'
```

## Project Status
- [x] Project setup
- [ ] Data collection
- [ ] Feature engineering
- [ ] Model training
- [ ] API development
- [ ] Deployment

## Tech Stack
- Python 3.10+
- Flask
- scikit-learn
- python-chess
- TensorFlow (optional)
EOF
