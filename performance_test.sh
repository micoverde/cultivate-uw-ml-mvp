#!/bin/bash

# Cultivate Learning ML MVP - Performance Testing & Monitoring
#
# Continuous performance benchmarking for demo scenarios
# Tracks response times, analysis performance, and user experience metrics

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}📊 Cultivate Learning ML MVP - Performance Testing${NC}"
echo "================================================================="

# Create performance logs directory
mkdir -p performance_logs
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
PERF_LOG="performance_logs/perf_${TIMESTAMP}.log"

# Function to log performance metrics
log_metric() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$PERF_LOG"
}

# Test frontend response time
echo -e "${BLUE}Testing frontend response time...${NC}"
FRONTEND_START=$(date +%s.%N)
curl -s -o /dev/null http://localhost:3002
FRONTEND_END=$(date +%s.%N)
FRONTEND_TIME=$(echo "$FRONTEND_END - $FRONTEND_START" | bc)
log_metric "Frontend Response Time: ${FRONTEND_TIME}s"

if (( $(echo "$FRONTEND_TIME < 2.0" | bc -l) )); then
    echo -e "${GREEN}✅ Frontend response time: ${FRONTEND_TIME}s (Good)${NC}"
else
    echo -e "${YELLOW}⚠️ Frontend response time: ${FRONTEND_TIME}s (Slow)${NC}"
fi

# Test backend API response time
echo -e "${BLUE}Testing backend API response time...${NC}"
BACKEND_START=$(date +%s.%N)
curl -s -o /dev/null http://localhost:8000/docs
BACKEND_END=$(date +%s.%N)
BACKEND_TIME=$(echo "$BACKEND_END - $BACKEND_START" | bc)
log_metric "Backend Response Time: ${BACKEND_TIME}s"

if (( $(echo "$BACKEND_TIME < 1.0" | bc -l) )); then
    echo -e "${GREEN}✅ Backend response time: ${BACKEND_TIME}s (Excellent)${NC}"
else
    echo -e "${YELLOW}⚠️ Backend response time: ${BACKEND_TIME}s (Slow)${NC}"
fi

# Test educator response analysis performance
echo -e "${BLUE}Testing educator response analysis performance...${NC}"

ANALYSIS_START=$(date +%s.%N)

# Submit analysis request
ANALYSIS_RESPONSE=$(curl -s -X POST http://localhost:8000/api/analyze/educator-response \
  -H "Content-Type: application/json" \
  -d '{
    "scenario_id": "maya-puzzle-frustration",
    "scenario_context": "During free play time, 4-year-old Maya is working on a 12-piece puzzle.",
    "audio_transcript": "Maya: This is stupid! I cant do it!",
    "educator_response": "Maya, I can see you are feeling frustrated with that puzzle. That is okay - puzzles can be tricky! I noticed you got three pieces to fit perfectly. Would you like to try one more piece or take a break?",
    "analysis_categories": [
      {"id": "emotional_support", "name": "Emotional Support & Validation"}
    ],
    "evidence_metrics": [
      {
        "strategy": "emotion_labeling_validation",
        "name": "Emotion labeling and validation",
        "effectiveness": 89,
        "description": "Explicitly acknowledging and naming child emotions"
      }
    ]
  }')

ANALYSIS_ID=$(echo "$ANALYSIS_RESPONSE" | grep -o '"analysis_id":"[^"]*"' | cut -d'"' -f4)

if [ -n "$ANALYSIS_ID" ]; then
    echo "Analysis ID: $ANALYSIS_ID"

    # Poll for completion
    POLL_COUNT=0
    while [ $POLL_COUNT -lt 30 ]; do
        sleep 2
        STATUS_RESPONSE=$(curl -s "http://localhost:8000/api/analyze/educator-response/status/$ANALYSIS_ID")
        STATUS=$(echo "$STATUS_RESPONSE" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)

        if [ "$STATUS" = "complete" ]; then
            break
        elif [ "$STATUS" = "error" ]; then
            echo -e "${RED}❌ Analysis failed${NC}"
            exit 1
        fi

        POLL_COUNT=$((POLL_COUNT + 1))
    done

    ANALYSIS_END=$(date +%s.%N)
    ANALYSIS_TIME=$(echo "$ANALYSIS_END - $ANALYSIS_START" | bc)
    log_metric "Educator Response Analysis Time: ${ANALYSIS_TIME}s"

    if (( $(echo "$ANALYSIS_TIME < 5.0" | bc -l) )); then
        echo -e "${GREEN}✅ Analysis time: ${ANALYSIS_TIME}s (Excellent)${NC}"
    elif (( $(echo "$ANALYSIS_TIME < 10.0" | bc -l) )); then
        echo -e "${YELLOW}⚠️ Analysis time: ${ANALYSIS_TIME}s (Acceptable)${NC}"
    else
        echo -e "${RED}❌ Analysis time: ${ANALYSIS_TIME}s (Too slow)${NC}"
    fi
else
    echo -e "${RED}❌ Failed to submit analysis request${NC}"
fi

# Memory usage check
echo -e "${BLUE}Checking system resources...${NC}"
MEMORY_USAGE=$(free | grep MemAvailable | awk '{printf "%.1f", ($2-$7)/$2 * 100.0}')
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')

log_metric "Memory Usage: ${MEMORY_USAGE}%"
log_metric "CPU Usage: ${CPU_USAGE}%"

echo -e "${BLUE}Memory Usage: ${MEMORY_USAGE}%${NC}"
echo -e "${BLUE}CPU Usage: ${CPU_USAGE}%${NC}"

# Generate performance summary
echo "================================================================="
echo -e "${BLUE}📈 Performance Summary${NC}"
echo -e "Frontend Response: ${FRONTEND_TIME}s"
echo -e "Backend Response: ${BACKEND_TIME}s"
if [ -n "$ANALYSIS_TIME" ]; then
    echo -e "Analysis Time: ${ANALYSIS_TIME}s"
fi
echo -e "Memory Usage: ${MEMORY_USAGE}%"
echo -e "CPU Usage: ${CPU_USAGE}%"
echo ""
echo -e "${BLUE}📄 Detailed log: $PERF_LOG${NC}"
echo "================================================================="

# Performance thresholds check
OVERALL_GRADE="EXCELLENT"
if (( $(echo "$FRONTEND_TIME > 2.0" | bc -l) )) || (( $(echo "$BACKEND_TIME > 1.0" | bc -l) )); then
    OVERALL_GRADE="NEEDS_IMPROVEMENT"
fi

if [ -n "$ANALYSIS_TIME" ] && (( $(echo "$ANALYSIS_TIME > 10.0" | bc -l) )); then
    OVERALL_GRADE="NEEDS_IMPROVEMENT"
fi

log_metric "Overall Performance Grade: $OVERALL_GRADE"

if [ "$OVERALL_GRADE" = "EXCELLENT" ]; then
    echo -e "${GREEN}🎉 Overall Performance: EXCELLENT${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️ Overall Performance: NEEDS IMPROVEMENT${NC}"
    exit 1
fi