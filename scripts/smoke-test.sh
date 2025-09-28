#!/bin/bash

# Smoke Test Script for Cultivate ML MVP
# Tests basic connectivity and ML pipeline functionality

set -e

BACKEND_URL="http://cultivate-ml-api-pag.westus2.azurecontainer.io:8000"
TEST_TRANSCRIPT='{"transcript":"Teacher: What do you think will happen if we mix red and blue?\n\nChild: Maybe purple!\n\nTeacher: Great thinking! Let us try it and see what happens.\n\nChild: Wow, it really did make purple!\n\nTeacher: Excellent observation! What other colors do you think we could mix?"}'

echo "🧪 Starting Cultivate ML MVP Smoke Tests..."
echo "Backend URL: $BACKEND_URL"
echo

# Test 1: Health Check
echo "1️⃣ Testing backend health endpoint..."
HEALTH_RESPONSE=$(curl -s "$BACKEND_URL/api/health")
if [[ $HEALTH_RESPONSE == *"healthy"* ]]; then
    echo "✅ Backend health check passed"
else
    echo "❌ Backend health check failed"
    exit 1
fi

# Test 2: ML Analysis Pipeline
echo
echo "2️⃣ Testing ML analysis pipeline..."
echo "Submitting test transcript..."

ANALYSIS_RESPONSE=$(curl -s -X POST "$BACKEND_URL/api/v1/analyze/transcript" \
    -H "Content-Type: application/json" \
    -d "$TEST_TRANSCRIPT")

if [[ $ANALYSIS_RESPONSE == *"analysis_id"* ]]; then
    ANALYSIS_ID=$(echo "$ANALYSIS_RESPONSE" | grep -o '"analysis_id":"[^"]*"' | cut -d'"' -f4)
    echo "✅ Analysis submitted successfully (ID: $ANALYSIS_ID)"

    # Wait for processing
    echo "⏳ Waiting for analysis to complete..."

    for i in {1..30}; do
        sleep 2
        STATUS_RESPONSE=$(curl -s "$BACKEND_URL/api/v1/analyze/status/$ANALYSIS_ID")
        if [[ $STATUS_RESPONSE == *'"status":"complete"'* ]]; then
            echo "✅ Analysis completed successfully"

            # Get results
            RESULTS_RESPONSE=$(curl -s "$BACKEND_URL/api/v1/analyze/results/$ANALYSIS_ID")
            if [[ $RESULTS_RESPONSE == *'"class_scores"'* ]] && [[ $RESULTS_RESPONSE == *'"recommendations"'* ]]; then
                echo "✅ Results retrieved successfully with CLASS scores and recommendations"
                break
            else
                echo "❌ Results incomplete or malformed"
                exit 1
            fi
        elif [[ $STATUS_RESPONSE == *'"status":"error"'* ]]; then
            echo "❌ Analysis failed with error"
            exit 1
        fi

        if [ $i -eq 30 ]; then
            echo "❌ Analysis timed out after 60 seconds"
            exit 1
        fi
    done
else
    echo "❌ Analysis submission failed"
    echo "Response: $ANALYSIS_RESPONSE"
    exit 1
fi

# Test 3: Response Time Check
echo
echo "3️⃣ Testing response time..."
START_TIME=$(date +%s%N)
curl -s "$BACKEND_URL/api/health" > /dev/null
END_TIME=$(date +%s%N)
RESPONSE_TIME=$(( ($END_TIME - $START_TIME) / 1000000 ))

if [ $RESPONSE_TIME -lt 5000 ]; then
    echo "✅ Response time: ${RESPONSE_TIME}ms (under 5s limit)"
else
    echo "⚠️  Response time: ${RESPONSE_TIME}ms (over 5s, but acceptable for MVP)"
fi

echo
echo "🎉 All smoke tests passed!"
echo "Backend is ready for frontend integration testing."