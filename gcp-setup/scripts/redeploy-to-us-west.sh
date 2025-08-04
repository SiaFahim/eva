#!/bin/bash

# Eva DeepSeek-V3 Environment Recreation Script
# Redeploys infrastructure to us-west region and recreates development environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID="eva-deepseek-dev"
NEW_REGION="us-west1"
NEW_ZONE="us-west1-a"
ORIGINAL_REGION="us-central1"
ORIGINAL_ZONE="us-central1-a"

echo -e "${BLUE}🚀 Eva DeepSeek-V3 Environment Recreation${NC}"
echo -e "${BLUE}=======================================${NC}"
echo ""
echo -e "${YELLOW}📍 Deploying to new region: ${NEW_REGION}${NC}"
echo -e "${YELLOW}📍 Original region: ${ORIGINAL_REGION}${NC}"
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

# Check if gcloud is available
if ! command -v gcloud &> /dev/null; then
    print_error "gcloud CLI not found. Please install Google Cloud SDK."
    exit 1
fi

# Set up gcloud path if needed
if [[ -f "$HOME/google-cloud-sdk/bin/gcloud" ]]; then
    export PATH="$HOME/google-cloud-sdk/bin:$PATH"
fi

# Authenticate if needed
echo -e "${BLUE}🔐 Checking authentication...${NC}"
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    print_warning "Authentication required"
    gcloud auth login
fi

# Set project
gcloud config set project $PROJECT_ID
print_status "Project set to $PROJECT_ID"

# Check if original instance exists and try to create disk snapshot
echo -e "${BLUE}💾 Attempting to backup original instance data...${NC}"
ORIGINAL_INSTANCE="eva-dev-dev"

# Try to create snapshot of original instance disks
if gcloud compute instances describe $ORIGINAL_INSTANCE --zone=$ORIGINAL_ZONE &>/dev/null; then
    print_status "Original instance found. Creating disk snapshots..."
    
    # Create snapshot of boot disk
    BOOT_DISK=$(gcloud compute instances describe $ORIGINAL_INSTANCE --zone=$ORIGINAL_ZONE --format="value(disks[0].source)" | sed 's|.*/||')
    SNAPSHOT_NAME="eva-backup-$(date +%Y%m%d-%H%M%S)"
    
    echo "Creating snapshot of boot disk: $BOOT_DISK"
    gcloud compute disks snapshot $BOOT_DISK --zone=$ORIGINAL_ZONE --snapshot-names="${SNAPSHOT_NAME}-boot" --async
    
    # Create snapshot of data disk if it exists
    DATA_DISK=$(gcloud compute instances describe $ORIGINAL_INSTANCE --zone=$ORIGINAL_ZONE --format="value(disks[1].source)" 2>/dev/null | sed 's|.*/||' || echo "")
    if [[ -n "$DATA_DISK" ]]; then
        echo "Creating snapshot of data disk: $DATA_DISK"
        gcloud compute disks snapshot $DATA_DISK --zone=$ORIGINAL_ZONE --snapshot-names="${SNAPSHOT_NAME}-data" --async
    fi
    
    print_status "Snapshots initiated (running in background)"
    echo "Boot disk snapshot: ${SNAPSHOT_NAME}-boot"
    if [[ -n "$DATA_DISK" ]]; then
        echo "Data disk snapshot: ${SNAPSHOT_NAME}-data"
    fi
else
    print_warning "Original instance not found or not accessible. Proceeding with fresh deployment."
fi

# Navigate to terraform directory
cd "$(dirname "$0")/../terraform"

# Initialize Terraform
echo -e "${BLUE}🏗️  Initializing Terraform...${NC}"
terraform init
print_status "Terraform initialized"

# Plan deployment
echo -e "${BLUE}📋 Planning deployment...${NC}"
terraform plan -var="region=$NEW_REGION" -var="zone=$NEW_ZONE"

# Ask for confirmation
echo ""
read -p "Do you want to proceed with the deployment? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_warning "Deployment cancelled"
    exit 0
fi

# Apply Terraform configuration
echo -e "${BLUE}🚀 Deploying infrastructure...${NC}"
terraform apply -var="region=$NEW_REGION" -var="zone=$NEW_ZONE" -auto-approve

# Get new instance details
NEW_INSTANCE_NAME=$(terraform output -raw instance_name)
NEW_EXTERNAL_IP=$(terraform output -raw instance_external_ip)

print_status "Infrastructure deployed successfully!"
echo "Instance name: $NEW_INSTANCE_NAME"
echo "External IP: $NEW_EXTERNAL_IP"

# Wait for instance to be ready
echo -e "${BLUE}⏳ Waiting for instance to be ready...${NC}"
sleep 30

# Wait for startup script to complete
echo -e "${BLUE}⏳ Waiting for startup script to complete...${NC}"
for i in {1..20}; do
    if gcloud compute ssh eva@$NEW_INSTANCE_NAME --zone=$NEW_ZONE --command="test -f /var/log/eva-startup-complete" 2>/dev/null; then
        print_status "Startup script completed"
        break
    fi
    echo "Waiting... ($i/20)"
    sleep 30
done

print_status "New Eva DeepSeek-V3 environment deployed successfully!"
echo ""
echo -e "${GREEN}🎉 Deployment Complete!${NC}"
echo -e "${GREEN}=====================${NC}"
echo ""
echo "New instance details:"
echo "  • Name: $NEW_INSTANCE_NAME"
echo "  • External IP: $NEW_EXTERNAL_IP"
echo "  • Region: $NEW_REGION"
echo "  • Zone: $NEW_ZONE"
echo ""
echo "Next steps:"
echo "  1. Update VS Code SSH configuration with new IP"
echo "  2. Connect and restore your development work"
echo "  3. Access Jupyter Lab at: http://$NEW_EXTERNAL_IP:8888"
echo ""
echo "SSH command:"
echo "  gcloud compute ssh eva@$NEW_INSTANCE_NAME --zone=$NEW_ZONE"
