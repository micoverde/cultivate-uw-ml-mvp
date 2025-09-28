#!/bin/bash

# Cultivate Learning ML MVP - Docker Build Script
# Issue #119: Production Docker Deployment Architecture
#
# Demonstrates building all Docker stages with optimization flags
# Supports local development, testing, and production deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${BLUE}🐳 Cultivate Learning ML MVP - Docker Build Suite${NC}"
echo "================================================================="

# Parse command line arguments
BUILD_STAGE=""
PUSH_TO_REGISTRY=""
REGISTRY_URL=""
NO_CACHE=""

show_help() {
    echo "Usage: $0 [OPTIONS] [STAGE]"
    echo ""
    echo "Build Docker images for Cultivate Learning ML MVP"
    echo ""
    echo "STAGES:"
    echo "  production   Build lightweight production image (default)"
    echo "  testing      Build testing image with Selenium support"
    echo "  fullml       Build complete ML stack image"
    echo "  all          Build all stages"
    echo ""
    echo "OPTIONS:"
    echo "  --push REGISTRY_URL    Push built images to registry"
    echo "  --no-cache            Build without Docker cache"
    echo "  --help, -h            Show this help message"
    echo ""
    echo "EXAMPLES:"
    echo "  $0                     # Build production image"
    echo "  $0 testing             # Build testing image"
    echo "  $0 all --no-cache      # Build all images without cache"
    echo "  $0 production --push cultivate.azurecr.io"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --push)
            PUSH_TO_REGISTRY="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        production|testing|fullml|all)
            BUILD_STAGE="$1"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Default to production if no stage specified
if [ -z "$BUILD_STAGE" ]; then
    BUILD_STAGE="production"
fi

# Build configuration
IMAGE_NAME="cultivate-ml-mvp"
TAG_SUFFIX=$(date +"%Y%m%d-%H%M%S")

# Function to build a specific stage
build_stage() {
    local stage=$1
    local stage_name=$2
    local description=$3

    echo -e "${BLUE}Building $stage_name image...${NC}"
    echo -e "${PURPLE}Stage: $stage${NC}"
    echo -e "${PURPLE}Description: $description${NC}"

    # Build with stage-specific optimizations
    BUILD_ARGS=""
    case $stage in
        "production")
            BUILD_ARGS="--build-arg BUILD_ENV=production"
            ;;
        "testing")
            BUILD_ARGS="--build-arg BUILD_ENV=testing"
            ;;
        "fullml")
            BUILD_ARGS="--build-arg BUILD_ENV=fullml"
            ;;
    esac

    BUILD_START=$(date +%s)

    if docker build $NO_CACHE \
        --target $stage \
        $BUILD_ARGS \
        --tag ${IMAGE_NAME}:${stage} \
        --tag ${IMAGE_NAME}:${stage}-${TAG_SUFFIX} \
        --label "com.cultivate.stage=${stage}" \
        --label "com.cultivate.build-date=$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --label "com.cultivate.version=${TAG_SUFFIX}" \
        .; then

        BUILD_END=$(date +%s)
        BUILD_TIME=$((BUILD_END - BUILD_START))

        echo -e "${GREEN}✅ $stage_name build completed in ${BUILD_TIME}s${NC}"

        # Show image size
        IMAGE_SIZE=$(docker images ${IMAGE_NAME}:${stage} --format "{{.Size}}")
        echo -e "${GREEN}📦 Image size: $IMAGE_SIZE${NC}"

        # Push to registry if requested
        if [ -n "$PUSH_TO_REGISTRY" ]; then
            echo -e "${BLUE}📤 Pushing to registry: $PUSH_TO_REGISTRY${NC}"
            docker tag ${IMAGE_NAME}:${stage} ${PUSH_TO_REGISTRY}/${IMAGE_NAME}:${stage}
            docker tag ${IMAGE_NAME}:${stage} ${PUSH_TO_REGISTRY}/${IMAGE_NAME}:${stage}-${TAG_SUFFIX}

            if docker push ${PUSH_TO_REGISTRY}/${IMAGE_NAME}:${stage} && \
               docker push ${PUSH_TO_REGISTRY}/${IMAGE_NAME}:${stage}-${TAG_SUFFIX}; then
                echo -e "${GREEN}✅ Push completed${NC}"
            else
                echo -e "${RED}❌ Push failed${NC}"
                return 1
            fi
        fi

    else
        echo -e "${RED}❌ $stage_name build failed${NC}"
        return 1
    fi

    echo ""
}

# Function to analyze and display image information
analyze_image() {
    local stage=$1
    local image_tag="${IMAGE_NAME}:${stage}"

    if docker image inspect $image_tag >/dev/null 2>&1; then
        echo -e "${BLUE}📊 Image Analysis: $stage${NC}"

        # Get image details
        IMAGE_ID=$(docker images $image_tag --format "{{.ID}}")
        IMAGE_SIZE=$(docker images $image_tag --format "{{.Size}}")
        CREATED=$(docker images $image_tag --format "{{.CreatedSince}}")

        echo -e "  Image ID: $IMAGE_ID"
        echo -e "  Size: $IMAGE_SIZE"
        echo -e "  Created: $CREATED"

        # Layer count
        LAYER_COUNT=$(docker history $image_tag --quiet | wc -l)
        echo -e "  Layers: $LAYER_COUNT"

        # Security scan if available
        if command -v docker scan &> /dev/null; then
            echo -e "${YELLOW}🔍 Security scanning available (run 'docker scan $image_tag')${NC}"
        fi

        echo ""
    fi
}

# Main build execution
echo -e "${BLUE}Build Configuration:${NC}"
echo -e "  Stage: $BUILD_STAGE"
echo -e "  Registry: ${PUSH_TO_REGISTRY:-"(local only)"}"
echo -e "  No Cache: ${NO_CACHE:-"false"}"
echo ""

TOTAL_START=$(date +%s)

case $BUILD_STAGE in
    "production")
        build_stage "production" "Production (Lightweight)" "Optimized for demo reliability and cost-effectiveness"
        analyze_image "production"
        ;;

    "testing")
        build_stage "testing" "Testing (E2E Selenium)" "Includes comprehensive testing capabilities"
        analyze_image "testing"
        ;;

    "fullml")
        build_stage "fullml" "Full ML Stack" "Complete ML capabilities with torch, tensorflow, transformers"
        analyze_image "fullml"
        ;;

    "all")
        echo -e "${PURPLE}Building all stages...${NC}"
        build_stage "production" "Production (Lightweight)" "Optimized for demo reliability and cost-effectiveness"
        build_stage "testing" "Testing (E2E Selenium)" "Includes comprehensive testing capabilities"
        build_stage "fullml" "Full ML Stack" "Complete ML capabilities with torch, tensorflow, transformers"

        echo -e "${BLUE}📊 All Images Analysis${NC}"
        analyze_image "production"
        analyze_image "testing"
        analyze_image "fullml"
        ;;

    *)
        echo -e "${RED}❌ Invalid stage: $BUILD_STAGE${NC}"
        show_help
        exit 1
        ;;
esac

TOTAL_END=$(date +%s)
TOTAL_TIME=$((TOTAL_END - TOTAL_START))

echo "================================================================="
echo -e "${GREEN}🎉 Docker build completed successfully!${NC}"
echo -e "${GREEN}⏱️  Total build time: ${TOTAL_TIME}s${NC}"
echo ""

# Show available images
echo -e "${BLUE}📦 Built Images:${NC}"
docker images ${IMAGE_NAME} --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}"

echo ""
echo -e "${BLUE}🚀 Usage Examples:${NC}"
echo -e "  # Run production container:"
echo -e "  docker run -p 8000:8000 ${IMAGE_NAME}:production"
echo ""
echo -e "  # Run with docker-compose:"
echo -e "  docker-compose up"
echo ""
echo -e "  # Run testing container:"
echo -e "  docker run ${IMAGE_NAME}:testing"

if [ -n "$PUSH_TO_REGISTRY" ]; then
    echo ""
    echo -e "${BLUE}🌐 Registry Images:${NC}"
    echo -e "  ${PUSH_TO_REGISTRY}/${IMAGE_NAME}:${BUILD_STAGE}"
    echo -e "  ${PUSH_TO_REGISTRY}/${IMAGE_NAME}:${BUILD_STAGE}-${TAG_SUFFIX}"
fi

echo "================================================================="