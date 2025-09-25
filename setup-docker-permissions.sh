#!/bin/bash

# Setup Docker Permissions for Cultivate Learning ML MVP
# Run this script with sudo: sudo bash setup-docker-permissions.sh

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🐳 Setting up Docker permissions for Cultivate Learning ML MVP${NC}"
echo "================================================================="

# Get the original user (not root)
ORIGINAL_USER=${SUDO_USER:-$USER}
echo -e "${BLUE}Setting up Docker permissions for user: $ORIGINAL_USER${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Installing Docker...${NC}"
    # Install Docker (Ubuntu/Debian)
    apt-get update
    apt-get install -y docker.io
    systemctl enable docker
    systemctl start docker
    echo -e "${GREEN}✅ Docker installed${NC}"
else
    echo -e "${GREEN}✅ Docker is already installed${NC}"
fi

# Check if Docker is running
if ! systemctl is-active --quiet docker; then
    echo -e "${YELLOW}🔄 Starting Docker daemon...${NC}"
    systemctl start docker
    systemctl enable docker
    sleep 3
fi

if systemctl is-active --quiet docker; then
    echo -e "${GREEN}✅ Docker daemon is running${NC}"
else
    echo -e "${RED}❌ Failed to start Docker daemon${NC}"
    exit 1
fi

# Add user to docker group
echo -e "${BLUE}Adding $ORIGINAL_USER to docker group...${NC}"
usermod -aG docker $ORIGINAL_USER

# Set proper ownership of docker socket
echo -e "${BLUE}Setting Docker socket permissions...${NC}"
chown root:docker /var/run/docker.sock
chmod 664 /var/run/docker.sock

# Test Docker access as the original user
echo -e "${BLUE}Testing Docker access...${NC}"
if sudo -u $ORIGINAL_USER docker --version; then
    echo -e "${GREEN}✅ Docker access configured successfully${NC}"
else
    echo -e "${RED}❌ Docker access test failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}🎉 Docker permissions setup complete!${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "1. Log out and back in (or run: newgrp docker)"
echo -e "2. Test: ${BLUE}docker ps${NC}"
echo -e "3. Build: ${BLUE}./docker-build.sh production${NC}"
echo ""
echo "================================================================="