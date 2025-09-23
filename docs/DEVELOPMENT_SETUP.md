# 🚀 Cultivate Learning ML MVP - Development Setup Guide

## 📋 Prerequisites

Before you begin, ensure you have the following installed on your system:

### Required Software
- **Node.js** >= 18.0.0 ([Download](https://nodejs.org/))
- **Python** >= 3.9.0 ([Download](https://python.org/))
- **Git** ([Download](https://git-scm.com/))
- **npm** (comes with Node.js)
- **pip** (comes with Python)

### Recommended Tools
- **VS Code** with Python and React extensions
- **Docker** (optional, for containerized development)
- **PostgreSQL** (for production database)

## ⚡ Quick Start (< 5 minutes)

```bash
# 1. Clone the repository
git clone https://github.com/micoverde/cultivate-uw-ml-mvp.git
cd cultivate-uw-ml-mvp

# 2. Install all dependencies
npm run install:all

# 3. Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# 4. Start development servers
npm run dev
```

🎉 **You're ready!** The frontend will be available at http://localhost:3000 and the API at http://localhost:8000

## 📁 Project Structure

```
cultivate-uw-ml-mvp/
├── src/                    # Python ML backend
│   ├── api/               # FastAPI endpoints
│   ├── ml_models/         # Machine learning models
│   ├── data_pipeline/     # Data processing
│   └── evaluation/        # Model evaluation
├── demo/                  # React frontend demo
├── tests/                 # Test suite
├── data/                  # Research data
├── models/                # Trained ML models
├── docs/                  # Documentation
└── config/                # Configuration files
```

## 🐍 Python Environment Setup

### Method 1: Virtual Environment (Recommended)
```bash
# Create and activate virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

### Method 2: Using npm script
```bash
npm run setup:env
```

### Verify Python Setup
```bash
# Check Python version
python --version  # Should be >= 3.9.0

# Test ML imports
python -c "import torch, tensorflow, numpy; print('✅ ML dependencies OK')"

# Start API server
npm run dev:backend
# Visit http://localhost:8000/docs for API documentation
```

## ⚛️ Frontend Environment Setup

```bash
# Navigate to demo directory
cd demo

# Install Node.js dependencies
npm install

# Start development server
npm run dev

# Or from project root
npm run dev:frontend
```

### Verify Frontend Setup
- Frontend dev server: http://localhost:3000
- Hot reload should work when you edit files
- No console errors in browser dev tools

## 🔧 Development Scripts

### Full Stack Development
```bash
npm run dev              # Start both frontend and backend
npm run build            # Build both frontend and backend
npm run test             # Run all tests
npm run lint             # Lint both Python and JavaScript
npm run format           # Format code with black and prettier
```

### Backend Only
```bash
npm run dev:backend      # Start FastAPI server (port 8000)
npm run test:backend     # Run Python tests
npm run lint:python      # Lint Python code
npm run format:python    # Format Python code with black
```

### Frontend Only
```bash
npm run dev:frontend     # Start React dev server (port 3000)
npm run build:frontend   # Build React app
npm run serve:frontend   # Serve built frontend
```

### Utilities
```bash
npm run clean            # Clean all build artifacts
npm run install:all      # Install all dependencies
```

## 🧪 Testing

### Run All Tests
```bash
npm run test
```

### Python Tests
```bash
# Run with pytest
cd src && python -m pytest ../tests/

# Run specific test file
python -m pytest tests/test_ml_models.py

# Run with coverage
python -m pytest --cov=src tests/
```

### Frontend Tests
```bash
cd demo && npm test
```

## 🎨 Code Quality

### Automatic Formatting
```bash
# Format all code
npm run format

# Format Python only
npm run format:python

# Format JavaScript only
npm run format:frontend
```

### Linting
```bash
# Lint all code
npm run lint

# Python linting (flake8)
npm run lint:python

# JavaScript linting (ESLint)
npm run lint:frontend
```

### Pre-commit Hooks (Recommended)
```bash
# Install pre-commit
pip install pre-commit

# Set up hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

## 🌍 Environment Variables

### Required Configuration
Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

### Key Variables
- `API_HOST` - Backend host (default: localhost)
- `API_PORT` - Backend port (default: 8000)
- `FRONTEND_PORT` - Frontend port (default: 3000)
- `ML_MODEL_PATH` - Path to ML models
- `SECRET_KEY` - Security secret (change in production!)

### Development vs Production
- **Development**: Use `.env` for local overrides
- **Production**: Set environment variables directly

## 🐳 Docker Development (Optional)

### Build and Run
```bash
# Build Docker image
npm run docker:build

# Run container
npm run docker:run

# Access services
# Frontend: http://localhost:3000
# API: http://localhost:8000
```

### Docker Compose (Coming Soon)
```bash
docker-compose up -d
```

## 📊 API Documentation

### Interactive API Docs
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### API Endpoints
- `GET /health` - Health check
- `POST /analyze/audio` - Audio analysis
- `POST /analyze/video` - Video analysis
- `GET /models` - List available models

## 🔍 Troubleshooting

### Common Issues

#### "Module not found" errors
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

#### Port already in use
```bash
# Kill process using port 8000
lsof -ti:8000 | xargs kill -9

# Kill process using port 3000
lsof -ti:3000 | xargs kill -9
```

#### Import errors in Python
```bash
# Check if you're in the right directory
pwd  # Should end with cultivate-uw-ml-mvp

# Check Python path
python -c "import sys; print(sys.path)"
```

#### Node.js version issues
```bash
# Check Node version
node --version  # Should be >= 18.0.0

# Use nvm to manage versions (Mac/Linux)
nvm install 18
nvm use 18
```

### Getting Help
1. Check this documentation first
2. Look at existing GitHub issues
3. Create a new issue with:
   - Your operating system
   - Node.js and Python versions
   - Full error message
   - Steps to reproduce

## 🚀 Next Steps

After setup is complete:

1. **Explore the codebase**: Start with `src/api/main.py` and `demo/src/App.jsx`
2. **Run tests**: `npm run test` to ensure everything works
3. **Check the docs**: Visit http://localhost:8000/docs for API exploration
4. **Review the Git workflow**: See `docs/GIT_WORKFLOW.md`
5. **Start developing**: Create a feature branch and begin coding!

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Vite Documentation](https://vitejs.dev/)

---

**Need help?** Open an issue or check our [Git Workflow Guide](./GIT_WORKFLOW.md) for contribution guidelines.