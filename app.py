"""
TimesFM 2.5 Forecasting API
FastAPI service for Google TimesFM time series forecasting
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
from datetime import datetime
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TimesFM 2.5 API",
    description="Time series forecasting using Google TimesFM 2.5",
    version="2.5.0"
)

# Global model instance (lazy loaded)
model = None

def load_model():
    """Load TimesFM model (lazy loading)"""
    global model
    if model is None:
        try:
            logger.info("Loading TimesFM 2.5 model...")
            import timesfm
            import torch
            
            torch.set_float32_matmul_precision("high")
            
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                "google/timesfm-2.5-200m-pytorch"
            )
            
            # Compile with default config
            model.compile(
                timesfm.ForecastConfig(
                    max_context=2048,
                    max_horizon=512,
                    normalize_inputs=True,
                    use_continuous_quantile_head=True,
                    force_flip_invariance=True,
                    infer_is_positive=True,
                    fix_quantile_crossing=True,
                )
            )
            
            logger.info("✅ TimesFM 2.5 model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    return model

class ForecastRequest(BaseModel):
    data: List[float] = Field(..., description="Input time series data (context)", min_length=1)
    horizon: int = Field(12, description="Number of steps to forecast", ge=1, le=512)
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "data": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
                "horizon": 12
            }
        }
    }

class ForecastResponse(BaseModel):
    point_forecast: List[float] = Field(..., description="Point forecast values")
    quantile_forecast: Optional[List[List[float]]] = Field(None, description="Quantile forecasts (10th-90th percentiles)")
    context_length: int = Field(..., description="Length of input context")
    horizon: int = Field(..., description="Forecast horizon")
    timestamp: str = Field(..., description="Forecast generation timestamp")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "TimesFM 2.5 Forecasting API",
        "version": "2.5.0",
        "model": "google/timesfm-2.5-200m-pytorch",
        "endpoints": {
            "/health": "Health check",
            "/forecast": "Generate forecast (POST)",
            "/docs": "API documentation"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest):
    """
    Generate time series forecast using TimesFM 2.5
    
    Args:
        request: ForecastRequest with input data and horizon
        
    Returns:
        ForecastResponse with point and quantile forecasts
    """
    try:
        # Load model if not loaded
        tfm = load_model()
        
        # Convert input to numpy array
        inputs = np.array(request.data, dtype=np.float32)
        
        # Validate input
        if len(inputs) < 2:
            raise HTTPException(
                status_code=400,
                detail="Input data must contain at least 2 points"
            )
        
        # Generate forecast
        logger.info(f"Forecasting {request.horizon} steps from {len(inputs)} context points")
        
        point_forecast, quantile_forecast = tfm.forecast(
            horizon=request.horizon,
            inputs=[inputs]  # TimesFM expects list of arrays
        )
        
        # Extract results (batch size = 1)
        point_values = point_forecast[0].tolist()
        quantile_values = quantile_forecast[0].tolist() if quantile_forecast is not None else None
        
        return ForecastResponse(
            point_forecast=point_values,
            quantile_forecast=quantile_values,
            context_length=len(inputs),
            horizon=request.horizon,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Forecast failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Forecast failed: {e}"
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
