#!/bin/bash

# Cultivate Learning ML MVP - Comprehensive Demo Test Runner
# Executes complete end-to-end testing suite with reporting
# Supports headed/headless modes and comprehensive validation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${BLUE}🧪 Cultivate Learning ML MVP - Comprehensive Demo Test Suite${NC}"
echo "================================================================="

# Configuration
FRONTEND_URL="http://localhost:3002"
BACKEND_URL="http://localhost:8000"
TEST_MODE="headless"
QUICK_MODE=""
OUTPUT_DIR="test_results/$(date +%Y%m%d_%H%M%S)"
PYTHON_ENV="venv/bin/activate"

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Run comprehensive end-to-end tests for Cultivate Learning ML MVP"
    echo ""
    echo "OPTIONS:"
    echo "  --headed        Run tests with visible browser (default: headless)"
    echo "  --headless      Run tests in headless mode"
    echo "  --quick         Run quick validation tests only"
    echo "  --full          Run complete test suite (default)"
    echo "  --frontend URL  Override frontend URL (default: $FRONTEND_URL)"
    echo "  --backend URL   Override backend URL (default: $BACKEND_URL)"
    echo "  --output DIR    Override output directory"
    echo "  --help, -h      Show this help message"
    echo ""
    echo "EXAMPLES:"
    echo "  $0                    # Run full headless test suite"
    echo "  $0 --headed           # Run with visible browser"
    echo "  $0 --quick --headed   # Quick validation with visible browser"
    echo "  $0 --output my_tests  # Custom output directory"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --headed)
            TEST_MODE="headed"
            shift
            ;;
        --headless)
            TEST_MODE="headless"
            shift
            ;;
        --quick)
            QUICK_MODE="--quick"
            shift
            ;;
        --full)
            QUICK_MODE=""
            shift
            ;;
        --frontend)
            FRONTEND_URL="$2"
            shift 2
            ;;
        --backend)
            BACKEND_URL="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

echo -e "${BLUE}Test Configuration:${NC}"
echo -e "  Frontend URL: $FRONTEND_URL"
echo -e "  Backend URL: $BACKEND_URL"
echo -e "  Test Mode: $TEST_MODE"
echo -e "  Quick Mode: ${QUICK_MODE:-"disabled"}"
echo -e "  Output Directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/screenshots"
mkdir -p "$OUTPUT_DIR/logs"

# Function to check server health
check_server_health() {
    local service_name=$1
    local url=$2
    local timeout=30

    echo -e "${BLUE}Checking $service_name health...${NC}"

    for i in $(seq 1 $timeout); do
        if curl -s -f "$url/health" >/dev/null 2>&1 || curl -s -f "$url/api/health" >/dev/null 2>&1 || curl -s -f "$url" >/dev/null 2>&1; then
            echo -e "${GREEN}✅ $service_name is healthy${NC}"
            return 0
        fi
        echo -n "."
        sleep 1
    done

    echo -e "${RED}❌ $service_name is not responding after ${timeout}s${NC}"
    return 1
}

# Function to check Python environment
check_python_environment() {
    echo -e "${BLUE}Checking Python environment...${NC}"

    if [ ! -f "$PYTHON_ENV" ]; then
        echo -e "${RED}❌ Python virtual environment not found at $PYTHON_ENV${NC}"
        echo -e "${YELLOW}Please run: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt${NC}"
        return 1
    fi

    source "$PYTHON_ENV"

    # Check required packages
    local required_packages=("selenium" "requests" "pytest" "beautifulsoup4")
    for package in "${required_packages[@]}"; do
        if ! python -c "import $package" 2>/dev/null; then
            echo -e "${RED}❌ Required package '$package' not installed${NC}"
            echo -e "${YELLOW}Installing missing packages...${NC}"
            pip install selenium requests pytest beautifulsoup4 webdriver-manager
            break
        fi
    done

    echo -e "${GREEN}✅ Python environment is ready${NC}"
    return 0
}

# Function to run Maya scenario tests
run_maya_tests() {
    echo -e "${BLUE}🧩 Running Maya Scenario Tests...${NC}"

    source "$PYTHON_ENV"

    # Set environment variables for the test
    export FRONTEND_URL="$FRONTEND_URL"
    export BACKEND_URL="$BACKEND_URL"
    export TEST_MODE="$TEST_MODE"
    export OUTPUT_DIR="$OUTPUT_DIR"
    export HEADLESS=$([ "$TEST_MODE" = "headless" ] && echo "true" || echo "false")

    # Run the comprehensive Maya scenario test
    if python test_maya_scenario_comprehensive.py $QUICK_MODE; then
        echo -e "${GREEN}✅ Maya scenario tests completed successfully${NC}"
        return 0
    else
        echo -e "${RED}❌ Maya scenario tests failed${NC}"
        return 1
    fi
}

# Function to generate test report
generate_report() {
    local report_file="$OUTPUT_DIR/test_report.html"
    local start_time=$1
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    echo -e "${BLUE}📊 Generating test report...${NC}"

    cat > "$report_file" << EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cultivate Learning ML MVP - Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
        .header { background: #2196F3; color: white; padding: 20px; border-radius: 8px; }
        .summary { background: #f5f5f5; padding: 15px; border-radius: 8px; margin: 20px 0; }
        .success { color: #4CAF50; font-weight: bold; }
        .error { color: #f44336; font-weight: bold; }
        .screenshot { max-width: 800px; border: 1px solid #ddd; margin: 10px 0; }
        .log-section { background: #1e1e1e; color: #fff; padding: 15px; border-radius: 4px; overflow-x: auto; }
        pre { margin: 0; white-space: pre-wrap; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧪 Cultivate Learning ML MVP</h1>
        <h2>Comprehensive Demo Test Report</h2>
        <p>Generated: $(date)</p>
    </div>

    <div class="summary">
        <h3>Test Execution Summary</h3>
        <ul>
            <li><strong>Test Mode:</strong> $TEST_MODE</li>
            <li><strong>Frontend URL:</strong> $FRONTEND_URL</li>
            <li><strong>Backend URL:</strong> $BACKEND_URL</li>
            <li><strong>Duration:</strong> ${duration}s</li>
            <li><strong>Output Directory:</strong> $OUTPUT_DIR</li>
        </ul>
    </div>

    <div class="test-results">
        <h3>Test Results</h3>
        <!-- Results will be populated by Python test script -->
    </div>

    <div class="screenshots">
        <h3>Screenshots</h3>
        <p>Test screenshots are available in the screenshots/ directory:</p>
        <ul>
EOF

    # Add screenshot links if they exist
    if ls "$OUTPUT_DIR/screenshots"/*.png 1> /dev/null 2>&1; then
        for screenshot in "$OUTPUT_DIR/screenshots"/*.png; do
            local filename=$(basename "$screenshot")
            echo "            <li><a href=\"screenshots/$filename\">$filename</a></li>" >> "$report_file"
        done
    else
        echo "            <li>No screenshots generated</li>" >> "$report_file"
    fi

    cat >> "$report_file" << EOF
        </ul>
    </div>

    <div class="logs">
        <h3>Test Logs</h3>
        <div class="log-section">
            <pre id="test-logs">
<!-- Test logs will be inserted here -->
            </pre>
        </div>
    </div>
</body>
</html>
EOF

    echo -e "${GREEN}✅ Test report generated: $report_file${NC}"
    echo -e "${BLUE}📂 View report: file://$PWD/$report_file${NC}"
}

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}🧹 Cleaning up...${NC}"
    # Kill any remaining processes if needed
    # (Browser processes are handled by the Python script)
}

trap cleanup EXIT

# Main execution
main() {
    local start_time=$(date +%s)

    echo -e "${PURPLE}Phase 1: Environment Validation${NC}"
    echo "================================="

    # Check Python environment
    if ! check_python_environment; then
        echo -e "${RED}❌ Environment check failed${NC}"
        exit 1
    fi

    # Check server health
    echo -e "\n${PURPLE}Phase 2: Server Health Checks${NC}"
    echo "================================="

    if ! check_server_health "Frontend" "$FRONTEND_URL"; then
        echo -e "${RED}❌ Frontend server check failed${NC}"
        echo -e "${YELLOW}💡 Start frontend: cd demo && npm run dev${NC}"
        exit 1
    fi

    if ! check_server_health "Backend" "$BACKEND_URL"; then
        echo -e "${RED}❌ Backend server check failed${NC}"
        echo -e "${YELLOW}💡 Start backend: source venv/bin/activate && python run_api.py${NC}"
        exit 1
    fi

    # Run tests
    echo -e "\n${PURPLE}Phase 3: Test Execution${NC}"
    echo "================================="

    if ! run_maya_tests; then
        echo -e "${RED}❌ Test execution failed${NC}"
        generate_report $start_time
        exit 1
    fi

    # Generate report
    echo -e "\n${PURPLE}Phase 4: Report Generation${NC}"
    echo "================================="

    generate_report $start_time

    echo ""
    echo "================================================================="
    echo -e "${GREEN}🎉 Test execution completed successfully!${NC}"
    echo -e "${GREEN}⏱️  Total execution time: $(($(date +%s) - start_time))s${NC}"
    echo ""
    echo -e "${BLUE}📁 Results Location:${NC}"
    echo -e "  Report: $OUTPUT_DIR/test_report.html"
    echo -e "  Screenshots: $OUTPUT_DIR/screenshots/"
    echo -e "  Logs: $OUTPUT_DIR/logs/"
    echo ""
    echo -e "${BLUE}🌐 View Report:${NC}"
    echo -e "  file://$PWD/$OUTPUT_DIR/test_report.html"
    echo "================================================================="
}

# Execute main function
main "$@"