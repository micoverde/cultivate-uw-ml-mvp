#!/bin/bash

# Cultivate Learning ML MVP - Demo Test Runner
#
# Runs comprehensive end-to-end tests for both demo scenarios
# Can be run in headless mode for CI/CD or headed mode for debugging
#
# Usage:
#   ./run_demo_tests.sh                    # Run with browser visible
#   ./run_demo_tests.sh --headless        # Run headless for CI/CD
#   ./run_demo_tests.sh --quick           # Run quick smoke tests only

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Cultivate Learning ML MVP - Demo Test Runner${NC}"
echo "================================================================="

# Parse command line arguments
HEADLESS=false
QUICK=false

for arg in "$@"; do
    case $arg in
        --headless)
            HEADLESS=true
            echo -e "${YELLOW}Running in headless mode${NC}"
            ;;
        --quick)
            QUICK=true
            echo -e "${YELLOW}Running quick smoke tests only${NC}"
            ;;
    esac
done

# Check if servers are running
echo -e "${BLUE}Checking server status...${NC}"

# Check frontend
if curl -s http://localhost:3002 > /dev/null; then
    echo -e "${GREEN}✅ Frontend server running (localhost:3002)${NC}"
else
    echo -e "${RED}❌ Frontend server not running. Please start with: cd demo && npm run dev${NC}"
    exit 1
fi

# Check backend
if curl -s http://localhost:8000/docs > /dev/null; then
    echo -e "${GREEN}✅ Backend server running (localhost:8000)${NC}"
else
    echo -e "${RED}❌ Backend server not running. Please start with: python run_api.py${NC}"
    exit 1
fi

# Setup Python virtual environment if not exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating Python virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install testing requirements
echo -e "${BLUE}Installing test dependencies...${NC}"
pip install -q -r requirements-testing.txt

# Install Chrome WebDriver if not exists
if ! command -v chromedriver &> /dev/null; then
    echo -e "${YELLOW}Installing Chrome WebDriver...${NC}"
    # Check if Chrome is installed
    if command -v google-chrome &> /dev/null; then
        echo -e "${GREEN}Chrome browser found${NC}"
    else
        echo -e "${RED}❌ Chrome browser not found. Please install Google Chrome.${NC}"
        exit 1
    fi
fi

# Set environment variables
export HEADLESS=$HEADLESS
export FRONTEND_URL="http://localhost:3002"
export BACKEND_URL="http://localhost:8000"

# Run tests
echo -e "${BLUE}Running demo tests...${NC}"
echo "================================================================="

if [ "$QUICK" = true ]; then
    echo -e "${YELLOW}Quick smoke test (Maya scenario only)${NC}"
    # Run a minimal test for quick validation
    python3 -c "
from test_demo_scenarios import DemoTestSuite, DemoTestConfig
import sys

config = DemoTestConfig()
suite = DemoTestSuite(config)

try:
    suite.setup_driver()
    result = suite.test_maya_scenario_complete_flow()
    suite.teardown_driver()

    if result['status'] == 'passed':
        print('✅ Quick smoke test PASSED')
        sys.exit(0)
    else:
        print('❌ Quick smoke test FAILED')
        sys.exit(1)
except Exception as e:
    print(f'❌ Test execution failed: {e}')
    sys.exit(1)
"
else
    # Run full test suite
    python3 test_demo_scenarios.py
fi

TEST_EXIT_CODE=$?

# Generate summary
echo "================================================================="
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}🎉 All demo tests PASSED!${NC}"
    echo -e "${GREEN}The Cultivate Learning ML MVP is working correctly.${NC}"
else
    echo -e "${RED}❌ Some tests FAILED!${NC}"
    echo -e "${RED}Please check the test output above for details.${NC}"
fi

# Show test artifacts
if [ -d "test_screenshots" ]; then
    SCREENSHOT_COUNT=$(ls test_screenshots/*.png 2>/dev/null | wc -l)
    if [ $SCREENSHOT_COUNT -gt 0 ]; then
        echo -e "${BLUE}📸 $SCREENSHOT_COUNT screenshots saved in test_screenshots/${NC}"
    fi
fi

if ls test_results_*.json 1> /dev/null 2>&1; then
    LATEST_RESULT=$(ls -t test_results_*.json | head -n1)
    echo -e "${BLUE}📄 Detailed results saved in $LATEST_RESULT${NC}"
fi

echo "================================================================="
exit $TEST_EXIT_CODE