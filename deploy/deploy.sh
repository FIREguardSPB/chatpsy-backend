#!/bin/bash
set -e

echo "🚀 ChatPsy Deployment Script"
echo "=============================="

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOMAIN="chatpsy.online"
VPS_IP="77.222.60.126"
BACKEND_REPO="https://github.com/FIREguardSPB/chatpsy-backend.git"
FRONTEND_REPO="https://github.com/FIREguardSPB/chatpsy-front.git"
DEPLOY_DIR="/var/www/chatpsy"

echo -e "${BLUE}[1/8] Updating system...${NC}"
sudo apt update && sudo apt upgrade -y

echo -e "${BLUE}[2/8] Installing dependencies...${NC}"
sudo apt install -y nginx python3 python3-pip python3-venv git certbot python3-certbot-nginx curl

echo -e "${BLUE}[3/8] Installing Node.js (for frontend build)...${NC}"
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

echo -e "${BLUE}[4/8] Creating deployment directory...${NC}"
sudo mkdir -p $DEPLOY_DIR/backend
sudo mkdir -p $DEPLOY_DIR/frontend

echo -e "${BLUE}[5/8] Cloning backend repository...${NC}"
cd $DEPLOY_DIR/backend
sudo git clone $BACKEND_REPO .

echo -e "${BLUE}[6/8] Setting up Python virtual environment...${NC}"
sudo python3 -m venv venv
sudo $DEPLOY_DIR/backend/venv/bin/pip install --upgrade pip
sudo $DEPLOY_DIR/backend/venv/bin/pip install -r requirements.txt

echo -e "${BLUE}[7/8] Cloning and building frontend...${NC}"
cd /tmp
git clone $FRONTEND_REPO frontend-build
cd frontend-build
npm install
npm run build
sudo cp -r dist/* $DEPLOY_DIR/frontend/

echo -e "${BLUE}[8/8] Setting up services...${NC}"
# Copy systemd service
sudo cp $DEPLOY_DIR/backend/deploy/chatpsy-backend.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable chatpsy-backend
sudo systemctl start chatpsy-backend

# Copy nginx config
sudo cp $DEPLOY_DIR/backend/deploy/nginx.conf /etc/nginx/sites-available/chatpsy
sudo ln -sf /etc/nginx/sites-available/chatpsy /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl restart nginx

echo -e "${GREEN}✅ Deployment complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Create .env file: sudo nano $DEPLOY_DIR/backend/.env"
echo "2. Add your API keys and configuration"
echo "3. Restart backend: sudo systemctl restart chatpsy-backend"
echo "4. Install SSL: sudo certbot --nginx -d $DOMAIN -d www.$DOMAIN"
echo ""
echo "Check status:"
echo "  Backend: sudo systemctl status chatpsy-backend"
echo "  Nginx: sudo systemctl status nginx"
echo "  Logs: sudo journalctl -u chatpsy-backend -f"
