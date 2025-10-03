#!/bin/bash
# Unified Development Startup Script
# Starts API and web server with health validation

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🚀 Starting Cultivate ML Development Environment"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
API_PORT=5001
WEB_PORT=6061
API_HEALTH_URL="http://localhost:${API_PORT}/health"
WEB_HEALTH_URL="http://localhost:${WEB_PORT}"

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Shutting down services...${NC}"
    if [ ! -z "$API_PID" ]; then
        kill $API_PID 2>/dev/null || true
    fi
    if [ ! -z "$WEB_PID" ]; then
        kill $WEB_PID 2>/dev/null || true
    fi
    # Kill any remaining processes on our ports
    lsof -ti:$API_PORT | xargs -r kill -9 2>/dev/null || true
    lsof -ti:$WEB_PORT | xargs -r kill -9 2>/dev/null || true
    echo -e "${GREEN}✅ Cleanup complete${NC}"
}

# Register cleanup on exit
trap cleanup EXIT INT TERM

# Check for port conflicts
check_port() {
    local port=$1
    local service=$2
    if lsof -ti:$port > /dev/null 2>&1; then
        echo -e "${RED}❌ Port $port already in use (needed for $service)${NC}"
        echo "   Kill the process with: lsof -ti:$port | xargs kill -9"
        return 1
    fi
    return 0
}

# Wait for service to be healthy
wait_for_health() {
    local url=$1
    local service=$2
    local max_attempts=30
    local attempt=0

    echo -e "${YELLOW}⏳ Waiting for $service to be healthy...${NC}"

    while [ $attempt -lt $max_attempts ]; do
        if curl -sf "$url" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ $service is healthy${NC}"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 1
    done

    echo -e "${RED}❌ $service failed to start after ${max_attempts}s${NC}"
    return 1
}

# Check Python virtual environment
echo "🔍 Checking Python environment..."
if [ ! -d "venv" ]; then
    echo -e "${RED}❌ Virtual environment not found${NC}"
    echo "   Create it with: python3 -m venv venv && venv/bin/pip install -r requirements.txt"
    exit 1
fi
echo -e "${GREEN}✅ Virtual environment found${NC}"

# Check for required models
echo "🔍 Checking ML models..."
if [ ! -f "models/ensemble_latest.pkl" ] || [ ! -f "models/classic_latest.pkl" ]; then
    echo -e "${YELLOW}⚠️  ML models not found - API will use heuristic fallback${NC}"
else
    echo -e "${GREEN}✅ ML models found${NC}"
fi

# Check ports
echo "🔍 Checking ports..."
check_port $API_PORT "API" || exit 1
check_port $WEB_PORT "Web Server" || exit 1
echo -e "${GREEN}✅ Ports available${NC}"

# Start API server
echo ""
echo "🚀 Starting API server on port $API_PORT..."
export PYTHONPATH="$PROJECT_ROOT"
venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port $API_PORT --reload > logs/api.log 2>&1 &
API_PID=$!

# Wait for API to be healthy
if ! wait_for_health "$API_HEALTH_URL" "API"; then
    echo -e "${RED}API logs:${NC}"
    tail -20 logs/api.log
    exit 1
fi

# Start web server
echo ""
echo "🚀 Starting web server on port $WEB_PORT..."
cd build
python3 -m http.server $WEB_PORT > ../logs/web.log 2>&1 &
WEB_PID=$!
cd ..

# Wait for web server
if ! wait_for_health "$WEB_HEALTH_URL" "Web Server"; then
    echo -e "${RED}Web server logs:${NC}"
    tail -20 logs/web.log
    exit 1
fi

# Print success message
echo ""
echo "================================================"
echo -e "${GREEN}✅ Development environment ready!${NC}"
echo "================================================"
echo ""
echo "📊 Services:"
echo "   API:        http://localhost:$API_PORT"
echo "   API Docs:   http://localhost:$API_PORT/api/docs"
echo "   Web Server: http://localhost:$WEB_PORT"
echo "   Demo 1:     http://localhost:$WEB_PORT/demo1/"
echo "   Demo 2:     http://localhost:$WEB_PORT/demo2/"
echo ""
echo "📝 Logs:"
echo "   API:  tail -f logs/api.log"
echo "   Web:  tail -f logs/web.log"
echo ""
echo "🛑 Press Ctrl+C to stop all services"
echo ""

# Keep script running and monitor processes
while true; do
    # Check if API is still running
    if ! kill -0 $API_PID 2>/dev/null; then
        echo -e "${RED}❌ API process died unexpectedly${NC}"
        exit 1
    fi

    # Check if web server is still running
    if ! kill -0 $WEB_PID 2>/dev/null; then
        echo -e "${RED}❌ Web server process died unexpectedly${NC}"
        exit 1
    fi

    sleep 2
done
