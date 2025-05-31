from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import uvicorn
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Product Feedback Emotion Analysis API")

MODEL_NAME = "bhadresh-savani/distilbert-base-uncased-emotion"
MAX_LENGTH = 128

EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval() 
    logger.info(f"Successfully loaded model: {MODEL_NAME}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

class FeedbackRequest(BaseModel):
    texts: List[str]

class EmotionPrediction(BaseModel):
    text: str
    emotions: Dict[str, float]
    dominant_emotion: str

class PredictionResponse(BaseModel):
    predictions: List[EmotionPrediction]

@app.post("/analyze", response_model=PredictionResponse)
async def analyze_feedback(request: FeedbackRequest):
    try:
        encoded_input = tokenizer(
            request.texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**encoded_input)
            predictions = torch.softmax(outputs.logits, dim=1)
            
        results = []
        for text, pred in zip(request.texts, predictions):
            emotion_scores = {
                label: float(score) 
                for label, score in zip(EMOTION_LABELS, pred)
            }
            
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
            
            results.append(EmotionPrediction(
                text=text,
                emotions=emotion_scores,
                dominant_emotion=dominant_emotion
            ))
            
        return PredictionResponse(predictions=results)
    
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, workers=4) 