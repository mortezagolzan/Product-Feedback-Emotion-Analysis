# Product Feedback Emotion Analysis API

This project provides a containerized solution for analyzing product feedback using emotion classification. The implementation uses FastAPI with uvicorn workers for efficient async processing.

## Model Choice

For this demonstration, we're using the `bhadresh-savani/distilbert-base-uncased-emotion` model, which is a DistilBERT model fine-tuned for emotion classification. This model was chosen because:

1. It can detect 6 different emotions:
   - Sadness 😭 - Disappointment or unhappiness
   - Joy 😀 - Happiness and delight
   - Love ❤️ - Affection and strong positive feelings
   - Anger 🤬 - Strong negative emotions
   - Fear 😨 - Anxiety or concern
   - Surprise 😲 - Unexpected or astonished reactions

2. It's well-suited for analyzing customer feedback
3. It provides probability scores for each emotion
4. It has good performance while maintaining reasonable resource requirements

## Setup

1. Build the Docker container:
```bash
docker build -t emotion-analysis .
```

2. Run the container:
```bash
docker run -p 8000:8000 emotion-analysis
```

The API will be available at `http://localhost:8000`

## API Endpoints

### POST /analyze
Accepts a list of feedback texts and returns emotion analysis results.

Request body:
```json
{
    "texts": [
        "I'm absolutely delighted with this purchase!",
        "This product is absolutely disgusting.",
        "I'm worried about the safety of this product."
    ]
}
```

Response:
```json
{
    "predictions": [
        {
            "text": "I'm absolutely delighted with this purchase!",
            "emotions": {
                "anger": 0.01,
                "disgust": 0.01,
                "fear": 0.01,
                "joy": 0.85,
                "neutral": 0.05,
                "sadness": 0.02,
                "surprise": 0.05
            },
            "dominant_emotion": "joy"
        },
        {
            "text": "This product is absolutely disgusting.",
            "emotions": {
                "anger": 0.15,
                "disgust": 0.75,
                "fear": 0.02,
                "joy": 0.01,
                "neutral": 0.02,
                "sadness": 0.03,
                "surprise": 0.02
            },
            "dominant_emotion": "disgust"
        },
        {
            "text": "I'm worried about the safety of this product.",
            "emotions": {
                "anger": 0.05,
                "disgust": 0.02,
                "fear": 0.80,
                "joy": 0.01,
                "neutral": 0.05,
                "sadness": 0.05,
                "surprise": 0.02
            },
            "dominant_emotion": "fear"
        }
    ]
}
```

### GET /health
Health check endpoint that returns the status of the service.

## Demo Script

The included `demo.py` script demonstrates how to make parallel requests to the API. To run it:

1. Make sure the Docker container is running
2. Run the script:
```bash
python demo.py
```

The script shows:
- How to make parallel requests using ThreadPoolExecutor
- Performance metrics for parallel processing
- Detailed emotion analysis results for each feedback
- Examples of all 7 emotions in product feedback

## Architecture

- FastAPI: Modern, fast web framework for building APIs
- Uvicorn: ASGI server with multiple worker support
- Transformers: HuggingFace's library for state-of-the-art NLP
- Docker: Containerization for easy deployment

The container is configured to run 4 uvicorn workers by default, which can be adjusted based on your CPU cores and requirements.
