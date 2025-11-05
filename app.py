"""
TimesFM Forecasting API
Lightweight FastAPI service for Google TimesFM predictions
Optimized for Railway.app deployment
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TimesFM Forecasting API",
    description="Google TimesFM time series forecasting service",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance (lazy loaded)
timesfm_model = None
MODEL_LOADED = False

class ForecastRequest(BaseModel):
    """Request schema for forecasting"""
    data: List[float] = Field(..., description="Historical time series data", min_items=10)
    horizon: int = Field(default=12, description="Number of steps to forecast", ge=1, le=100)
    frequency: Optional[str] = Field(default="5min", description="Data frequency (5min, 15min, 1h, 1d)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "data": [100.5, 101.2, 99.8, 102.1, 103.5],
                "horizon": 12,
                "frequency": "5min"
            }
        }

class ForecastResponse(BaseModel):
    """Response schema for forecasting"""
    forecast: List[float] = Field(..., description="Predicted values")
    confidence_intervals: Optional[dict] = Field(None, description="Upper and lower confidence bounds")
    metadata: dict = Field(..., description="Prediction metadata")

def load_timesfm_model():
    """
    Lazy load TimesFM model
    Uses timesfm-1.0-200m checkpoint (optimized for speed)
    """
    global timesfm_model, MODEL_LOADED
    
    if MODEL_LOADED:
        return timesfm_model
    
    try:
        logger.info("🔄 Loading TimesFM model...")
        start_time = datetime.now()
        
        # Import timesfm (heavy import, do it only once)
        import timesfm
        
        # Initialize model with 200M checkpoint (faster, less memory)
        # For production, use 1.0-200m (512MB) instead of 1.0-1.3b (2GB)
        timesfm_model = timesfm.TimesFm(
            context_len=512,  # Context window
            horizon_len=128,  # Max forecast horizon
            input_patch_len=32,
            output_patch_len=128,
            num_layers=20,
            model_dims=1280,
            backend='cpu',  # Use CPU for Railway (no GPU)
        )
        
        # Load checkpoint
        checkpoint_path = os.getenv('TIMESFM_CHECKPOINT_PATH', 'google/timesfm-1.0-200m')
        timesfm_model.load_from_checkpoint(repo_id=checkpoint_path)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"✅ TimesFM model loaded in {elapsed:.2f}s")
        
        MODEL_LOADED = True
        return timesfm_model
        
    except Exception as e:
        logger.error(f"❌ Failed to load TimesFM model: {e}")
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Preload model on startup (optional, can use lazy loading)"""
    logger.info("🚀 TimesFM API starting up...")
    # Uncomment to preload model (increases startup time but faster first request)
    # load_timesfm_model()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "TimesFM Forecasting API",
        "status": "running",
        "model_loaded": MODEL_LOADED,
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": MODEL_LOADED,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest):
    """
    Generate time series forecast using TimesFM
    
    Args:
        request: ForecastRequest with historical data and parameters
        
    Returns:
        ForecastResponse with predictions and metadata
    """
    try:
        logger.info(f"📊 Forecast request: {len(request.data)} datapoints, horizon={request.horizon}")
        
        # Load model (lazy loading)
        model = load_timesfm_model()
        
        # Prepare input data
        input_data = np.array(request.data).reshape(1, -1)  # Shape: (1, seq_len)
        
        # Generate forecast
        start_time = datetime.now()
        forecast_result = model.forecast(
            inputs=input_data,
            freq=[0],  # Frequency index (0 for custom)
            horizon=request.horizon
        )
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # Extract predictions
        predictions = forecast_result.forecast[0].tolist()  # Shape: (horizon,)
        
        logger.info(f"✅ Forecast generated in {elapsed:.2f}s")
        
        return ForecastResponse(
            forecast=predictions,
            confidence_intervals=None,  # TimesFM doesn't provide CI by default
            metadata={
                "input_length": len(request.data),
                "horizon": request.horizon,
                "frequency": request.frequency,
                "inference_time_ms": int(elapsed * 1000),
                "model": "timesfm-1.0-200m",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Forecast error: {e}")
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")

@app.post("/batch-forecast")
async def batch_forecast(requests: List[ForecastRequest]):
    """
    Batch forecasting for multiple time series
    More efficient than individual requests
    """
    try:
        logger.info(f"📦 Batch forecast: {len(requests)} series")
        
        model = load_timesfm_model()
        results = []
        
        for req in requests:
            input_data = np.array(req.data).reshape(1, -1)
            forecast_result = model.forecast(
                inputs=input_data,
                freq=[0],
                horizon=req.horizon
            )
            predictions = forecast_result.forecast[0].tolist()
            results.append({
                "forecast": predictions,
                "metadata": {
                    "input_length": len(req.data),
                    "horizon": req.horizon
                }
            })
        
        logger.info(f"✅ Batch forecast complete: {len(results)} results")
        return {"results": results}
        
    except Exception as e:
        logger.error(f"❌ Batch forecast error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch forecast failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
