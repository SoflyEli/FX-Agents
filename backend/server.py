from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from sentiment_analyzer import ForexSentimentAnalyzer
import asyncio
from contextlib import asynccontextmanager

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Global sentiment analyzer
sentiment_analyzer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    global sentiment_analyzer
    
    # Startup
    logger.info("Starting up sentiment analyzer...")
    sentiment_analyzer = ForexSentimentAnalyzer()
    
    # Try to load pre-trained models
    try:
        sentiment_analyzer.load_models()
        logger.info("Pre-trained models loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load pre-trained models: {e}")
        logger.info("Training new models...")
        
        # Train models if not available
        try:
            from sentiment_analyzer import train_models
            sentiment_analyzer = train_models()
            logger.info("Models trained successfully")
        except Exception as e:
            logger.error(f"Failed to train models: {e}")
            # Initialize with basic analyzer for demo
            sentiment_analyzer = ForexSentimentAnalyzer()
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app with lifespan
app = FastAPI(lifespan=lifespan)

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define Models
class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StatusCheckCreate(BaseModel):
    client_name: str

class SentimentRequest(BaseModel):
    headline: str

class SentimentResponse(BaseModel):
    headline: str
    finbert: Optional[Dict[str, Any]] = None
    nb: Optional[Dict[str, Any]] = None
    primary: Optional[str] = None

class PredictionHistory(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    headline: str
    finbert_result: Optional[Dict[str, Any]] = None
    nb_result: Optional[Dict[str, Any]] = None
    primary_model: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Existing routes
@api_router.get("/")
async def root():
    return {"message": "Forex Sentiment Analyzer API", "version": "1.0", "models_loaded": sentiment_analyzer is not None}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.dict()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.dict())
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]

# Sentiment Analysis Routes
@api_router.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: SentimentRequest):
    """
    Predict sentiment for a forex news headline
    
    Returns predictions from both FinBERT and Naive Bayes models,
    with FinBERT as primary when confidence >= 0.80
    """
    global sentiment_analyzer
    
    if not sentiment_analyzer:
        raise HTTPException(status_code=503, detail="Sentiment analyzer not available")
    
    if not request.headline or not request.headline.strip():
        raise HTTPException(status_code=400, detail="Headline cannot be empty")
    
    try:
        # Get predictions
        results = sentiment_analyzer.predict(request.headline.strip())
        
        # Store prediction in database
        prediction_record = PredictionHistory(
            headline=results['headline'],
            finbert_result=results['finbert'],
            nb_result=results['nb'],
            primary_model=results['primary']
        )
        
        await db.predictions.insert_one(prediction_record.dict())
        
        return SentimentResponse(**results)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@api_router.get("/predictions", response_model=List[PredictionHistory])
async def get_prediction_history(limit: int = 100):
    """Get recent prediction history"""
    predictions = await db.predictions.find().sort("timestamp", -1).limit(limit).to_list(limit)
    return [PredictionHistory(**pred) for pred in predictions]

@api_router.get("/model-info")
async def get_model_info():
    """Get information about loaded models"""
    global sentiment_analyzer
    
    if not sentiment_analyzer:
        return {"status": "Models not loaded"}
    
    info = {
        "traditional_ml": bool(sentiment_analyzer.tfidf_pipeline),
        "finbert": bool(sentiment_analyzer.finbert_pipeline),
        "models_trained": hasattr(sentiment_analyzer, 'evaluation_results') and sentiment_analyzer.evaluation_results is not None
    }
    
    # Add evaluation summary if available
    if hasattr(sentiment_analyzer, 'evaluation_results') and sentiment_analyzer.evaluation_results:
        info["evaluation_summary"] = {}
        for model_name, results in sentiment_analyzer.evaluation_results.items():
            if 'accuracy' in results:
                info["evaluation_summary"][model_name] = {
                    "accuracy": float(results['accuracy'])
                }
    
    return info

@api_router.post("/train")
async def trigger_training():
    """Trigger model training (for development purposes)"""
    global sentiment_analyzer
    
    try:
        logger.info("Starting model training...")
        from sentiment_analyzer import train_models
        
        # Run training in background
        loop = asyncio.get_event_loop()
        sentiment_analyzer = await loop.run_in_executor(None, train_models)
        
        return {"message": "Model training completed successfully"}
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@api_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": sentiment_analyzer is not None,
        "traditional_ml": bool(sentiment_analyzer.tfidf_pipeline) if sentiment_analyzer else False,
        "finbert": bool(sentiment_analyzer.finbert_pipeline) if sentiment_analyzer else False,
        "timestamp": datetime.utcnow()
    }

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()