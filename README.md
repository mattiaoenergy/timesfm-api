# TimesFM Forecasting API

Google TimesFM time series forecasting service optimized for Railway.app deployment.

## 🚀 Features

- **Fast API**: FastAPI-based REST API
- **Lightweight**: Optimized for 512MB RAM (Railway hobby plan)
- **CPU-only**: No GPU required
- **Lazy Loading**: Model loads on first request
- **Batch Support**: Process multiple time series efficiently

## 📋 API Endpoints

### `GET /`
Health check endpoint

### `GET /health`
Detailed health status

### `POST /forecast`
Generate forecast for single time series

**Request:**
```json
{
  "data": [100.5, 101.2, 99.8, 102.1, 103.5],
  "horizon": 12,
  "frequency": "5min"
}
```

**Response:**
```json
{
  "forecast": [103.8, 104.2, 104.5, ...],
  "confidence_intervals": null,
  "metadata": {
    "input_length": 5,
    "horizon": 12,
    "inference_time_ms": 250,
    "model": "timesfm-1.0-200m"
  }
}
```

### `POST /batch-forecast`
Process multiple time series in one request

## 🔧 Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python app.py

# Test endpoint
curl http://localhost:8000/health
```

## 🚂 Railway Deployment

### Step 1: Create Railway Account
1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. Verify email

### Step 2: Create GitHub Repository
```bash
cd timesfm-api
git init
git add .
git commit -m "Initial commit: TimesFM API"
git remote add origin https://github.com/YOUR_USERNAME/timesfm-api.git
git push -u origin main
```

### Step 3: Deploy to Railway
1. Go to Railway dashboard
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose `timesfm-api` repository
5. Railway auto-detects Dockerfile and deploys

### Step 4: Get API URL
1. Go to project settings
2. Click "Generate Domain"
3. Copy URL (e.g., `https://timesfm-api-production.up.railway.app`)

### Step 5: Configure Environment Variables (Optional)
- `PORT`: Auto-set by Railway
- `TIMESFM_CHECKPOINT_PATH`: Default `google/timesfm-1.0-200m`

## 💰 Cost Estimate

**Railway Hobby Plan: $5/month**
- 512MB RAM (sufficient for TimesFM-200M)
- Always-on
- Unlimited requests
- SSL + custom domain included

**Expected Performance:**
- Cold start: ~10-15s (first request)
- Warm inference: ~200-500ms per forecast
- Memory usage: ~400-500MB

## 🔗 Integration with Trading Bot

Add to your trading bot's `.env`:
```bash
TIMESFM_API_URL=https://timesfm-api-production.up.railway.app
```

Use the TimesFMService client (see integration docs).

## 📊 Model Details

- **Model**: TimesFM 1.0-200M (Google Research)
- **Context Length**: 512 timesteps
- **Max Horizon**: 128 timesteps
- **Backend**: CPU (PyTorch)
- **Memory**: ~400MB loaded

## 🐛 Troubleshooting

### Model loading timeout
Increase Railway timeout in `railway.toml`:
```toml
[deploy]
startCommand = "python app.py"
healthcheckPath = "/health"
healthcheckTimeout = 120
```

### Out of memory
Use smaller checkpoint or upgrade Railway plan to 1GB RAM ($10/month).

### Slow inference
- Reduce `context_len` in model initialization
- Use batch forecasting for multiple series
- Consider caching predictions

## 📝 License

MIT License - Free to use and modify
