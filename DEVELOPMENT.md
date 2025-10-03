# Development Guide

## Quick Start

### First Time Setup
```bash
# 1. Clone repository
git clone https://github.com/micoverde/cultivate-uw-ml-mvp.git
cd cultivate-uw-ml-mvp

# 2. Run setup (creates venv, installs dependencies, creates logs dir)
npm run setup

# 3. Start development environment
npm start
```

That's it! The unified startup script will:
- ✅ Check for Python virtual environment
- ✅ Check for ML models
- ✅ Verify ports are available
- ✅ Start API server with health checks
- ✅ Start web server with health checks
- ✅ Monitor processes and auto-restart if needed

### Development Servers

After running `npm start`, you'll have:

- **API Server**: http://localhost:5001
  - Health: http://localhost:5001/health
  - Docs: http://localhost:5001/api/docs
  - Classic ML: `POST /api/v1/classify`
  - Ensemble ML: `POST /api/v2/classify/ensemble`

- **Web Server**: http://localhost:6061
  - Demo 1: http://localhost:6061/demo1/
  - Demo 2: http://localhost:6061/demo2/

### Stopping Servers

Press `Ctrl+C` in the terminal running `npm start`. The script will automatically:
- Kill both API and web servers
- Clean up any processes on ports 5001 and 6061
- Display cleanup confirmation

## NPM Scripts Reference

### Core Commands
```bash
npm start           # Start API + web server with health validation
npm run dev         # Alias for npm start
npm test            # Run all tests (API + scenarios)
npm run setup       # First-time setup (venv + deps + logs)
```

### Individual Services (Advanced)
```bash
npm run dev:backend   # Run API only
npm run dev:frontend  # Run web server only
```

### Testing
```bash
npm test              # Run all tests
npm run test:api      # Test API with ground truth (49 examples)
npm run test:scenarios # Test demo2 system with scenarios
```

### Code Quality
```bash
npm run lint          # Check Python code style
npm run format        # Auto-format Python code
```

### Cleanup
```bash
npm run clean         # Remove cache and logs
npm run clean:all     # Remove everything including venv
```

## Logs

All logs are stored in `logs/` directory:

```bash
# Watch API logs in real-time
tail -f logs/api.log

# Watch web server logs
tail -f logs/web.log
```

## Troubleshooting

### Port Already in Use

If you see "Port 5001 already in use":
```bash
# Kill process on port 5001
lsof -ti:5001 | xargs kill -9

# Or port 6061
lsof -ti:6061 | xargs kill -9
```

### Virtual Environment Not Found

```bash
npm run setup
```

### API Not Starting

Check the logs:
```bash
tail -20 logs/api.log
```

Common issues:
- Missing dependencies: `venv/bin/pip install -r requirements.txt`
- Missing models: Models will use heuristic fallback (warning, not error)
- Port conflict: Kill process on port 5001

### Models Not Loading

The API will still work with a heuristic fallback. To train models:
```bash
venv/bin/python train_7_model_ensemble.py
```

## Architecture

### Unified Startup Flow

```
npm start
    │
    ├─> Check Python venv exists
    ├─> Check ports 5001, 6061 available
    ├─> Start API on 5001
    │   └─> Wait for health check (30s timeout)
    ├─> Start web on 6061
    │   └─> Wait for health check (30s timeout)
    └─> Monitor both processes
        └─> Auto-exit if either dies
```

### API Configuration

- **Port**: 5001
- **Rate Limiting**: Localhost exempt (dev), 200 req/15min (production)
- **CORS**: All localhost ports allowed
- **Models**: Ensemble + Classic ML
- **PYTHONPATH**: Auto-configured

### Web Server Configuration

- **Port**: 6061
- **Directory**: `unified-demos/`
- **Static Files**: No build step required

## Git Workflow

```bash
# Create feature branch
git checkout -b fix-00XXX-description

# Make changes, test with npm start
npm start

# Run tests
npm test

# Commit (no need to specify all the patterns manually)
git add .
git commit -m "fix: description for #XXX"

# Push
git push origin fix-00XXX-description

# Do NOT merge to main yet (per Warren's instructions)
```

## Common Development Tasks

### Adding a New Endpoint

1. Edit `src/api/main.py` or create router in `src/api/endpoints/`
2. API auto-reloads (uvicorn `--reload` flag)
3. Test with: `curl http://localhost:5001/your/endpoint`

### Updating ML Models

1. Train new models: `venv/bin/python train_7_model_ensemble.py`
2. Models save to `models/ensemble_latest.pkl` and `models/classic_latest.pkl`
3. Restart API (Ctrl+C, then `npm start`)

### Testing Demo2 System

```bash
# Start services
npm start

# In another terminal, run tests
npm run test:scenarios
```

### Debugging Rate Limits

Localhost is exempt from rate limits. If you need to test rate limiting:

1. Edit `src/api/security/middleware.py`
2. Comment out the localhost exemption
3. API will auto-reload

## Performance Notes

### API Response Times

- Classic ML: ~100ms per request
- Ensemble ML: ~200ms per request (5 models voting)

### Demo2 Re-classification

- 95 questions re-classified on model switch
- No rate limiting for localhost
- Completes in ~10-20 seconds (parallel requests)

## Security

### Development vs Production

**Development (localhost)**:
- Rate limiting: Disabled
- CORS: All localhost ports
- API keys: Auto-generated (logged on startup)

**Production**:
- Rate limiting: 200 req/15min
- CORS: Specific Azure domains
- API keys: Must set `CULTIVATE_*_API_KEY` env vars

### API Keys

Development keys are auto-generated and logged:
```
Development Admin API Key: sk-admin-...
Development Monitor API Key: sk-monitor-...
```

For production, set:
```bash
export CULTIVATE_ADMIN_API_KEY="your-key"
export CULTIVATE_MONITOR_API_KEY="your-key"
```
