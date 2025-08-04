#!/bin/bash

# VS Code SSH Configuration Update Script
# Updates the eva-dev host configuration with new IP address

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

NEW_IP="$1"

if [[ -z "$NEW_IP" ]]; then
    echo "Usage: $0 <new_ip_address>"
    echo "Example: $0 34.168.123.45"
    exit 1
fi

echo -e "${BLUE}🔧 Updating VS Code SSH Configuration${NC}"
echo -e "${BLUE}===================================${NC}"
echo ""
echo -e "${YELLOW}📍 New IP Address: ${NEW_IP}${NC}"
echo ""

# Function to print status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# SSH config file location
SSH_CONFIG="$HOME/.ssh/config"

# Backup existing SSH config
if [[ -f "$SSH_CONFIG" ]]; then
    cp "$SSH_CONFIG" "$SSH_CONFIG.backup.$(date +%Y%m%d-%H%M%S)"
    print_status "SSH config backed up"
fi

# Check if eva-dev host exists in SSH config
if grep -q "Host eva-dev" "$SSH_CONFIG" 2>/dev/null; then
    print_status "Found existing eva-dev host configuration"
    
    # Update the HostName
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "/Host eva-dev/,/^Host / { s/HostName .*/HostName $NEW_IP/; }" "$SSH_CONFIG"
    else
        # Linux
        sed -i "/Host eva-dev/,/^Host / { s/HostName .*/HostName $NEW_IP/; }" "$SSH_CONFIG"
    fi
    
    print_status "Updated eva-dev HostName to $NEW_IP"
else
    print_warning "eva-dev host not found. Adding new configuration..."
    
    # Add new eva-dev host configuration
    cat >> "$SSH_CONFIG" << EOF

# Eva DeepSeek-V3 Development Environment
Host eva-dev
    HostName $NEW_IP
    User eva
    IdentityFile ~/.ssh/google_compute_engine
    ServerAliveInterval 60
    ServerAliveCountMax 3
    LocalForward 8888 localhost:8888
    LocalForward 6006 localhost:6006
    LocalForward 8080 localhost:8080
EOF
    
    print_status "Added new eva-dev host configuration"
fi

# Test SSH connection
echo -e "${BLUE}🔍 Testing SSH connection...${NC}"
if ssh -o ConnectTimeout=10 -o BatchMode=yes eva-dev "echo 'SSH connection successful'" 2>/dev/null; then
    print_status "SSH connection test successful"
else
    print_warning "SSH connection test failed. This might be normal if the instance is still starting up."
    echo "You can test manually with: ssh eva-dev"
fi

# Display current configuration
echo ""
echo -e "${BLUE}📋 Current eva-dev SSH Configuration:${NC}"
echo -e "${BLUE}====================================${NC}"
grep -A 10 "Host eva-dev" "$SSH_CONFIG" || print_error "Configuration not found"

echo ""
echo -e "${GREEN}🎉 VS Code SSH Configuration Updated!${NC}"
echo -e "${GREEN}====================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Open VS Code"
echo "  2. Press Ctrl+Shift+P (or Cmd+Shift+P on Mac)"
echo "  3. Type 'Remote-SSH: Connect to Host'"
echo "  4. Select 'eva-dev' from the list"
echo "  5. Open folder: /home/eva/workspace/eva"
echo ""
echo "Connection details:"
echo "  • Host: eva-dev"
echo "  • IP: $NEW_IP"
echo "  • User: eva"
echo "  • Jupyter Lab: http://$NEW_IP:8888 (password: eva2025)"
