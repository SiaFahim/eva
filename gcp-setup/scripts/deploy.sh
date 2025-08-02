#!/bin/bash
# Eva DeepSeek-V3 GCP Deployment Script
# This script automates the deployment of the development environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TERRAFORM_DIR="$PROJECT_ROOT/terraform"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check if terraform is installed
    if ! command -v terraform &> /dev/null; then
        log_error "Terraform is not installed. Please install it first."
        exit 1
    fi
    
    # Check if authenticated with gcloud
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        log_error "Not authenticated with gcloud. Please run 'gcloud auth login'"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

setup_gcp_project() {
    log_info "Setting up GCP project..."
    
    # Get current project
    PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
    
    if [ -z "$PROJECT_ID" ]; then
        log_error "No GCP project set. Please run 'gcloud config set project YOUR_PROJECT_ID'"
        exit 1
    fi
    
    log_info "Using GCP project: $PROJECT_ID"
    
    # Enable required APIs
    log_info "Enabling required GCP APIs..."
    gcloud services enable compute.googleapis.com
    gcloud services enable storage.googleapis.com
    gcloud services enable iam.googleapis.com
    gcloud services enable cloudresourcemanager.googleapis.com
    
    log_success "GCP project setup completed"
}

setup_ssh_keys() {
    log_info "Setting up SSH keys..."
    
    SSH_KEY_PATH="$HOME/.ssh/id_rsa"
    
    if [ ! -f "$SSH_KEY_PATH" ]; then
        log_info "Generating SSH key pair..."
        ssh-keygen -t rsa -b 4096 -f "$SSH_KEY_PATH" -N "" -C "eva-dev-$(whoami)"
        log_success "SSH key pair generated"
    else
        log_info "SSH key already exists"
    fi
    
    # Add SSH key to gcloud
    if ! gcloud compute os-login ssh-keys list --format="value(key)" | grep -q "$(cat $SSH_KEY_PATH.pub | cut -d' ' -f2)"; then
        log_info "Adding SSH key to GCP..."
        gcloud compute os-login ssh-keys add --key-file="$SSH_KEY_PATH.pub"
        log_success "SSH key added to GCP"
    else
        log_info "SSH key already added to GCP"
    fi
}

get_public_ip() {
    log_info "Getting your public IP address..."
    
    PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || curl -s ipinfo.io/ip 2>/dev/null || echo "")
    
    if [ -n "$PUBLIC_IP" ]; then
        log_info "Your public IP: $PUBLIC_IP"
        echo "$PUBLIC_IP/32"
    else
        log_warning "Could not determine public IP. Using 0.0.0.0/0 (less secure)"
        echo "0.0.0.0/0"
    fi
}

setup_terraform() {
    log_info "Setting up Terraform configuration..."
    
    cd "$TERRAFORM_DIR"
    
    # Create terraform.tfvars if it doesn't exist
    if [ ! -f "terraform.tfvars" ]; then
        log_info "Creating terraform.tfvars from example..."
        cp terraform.tfvars.example terraform.tfvars
        
        # Get project ID and public IP
        PROJECT_ID=$(gcloud config get-value project)
        PUBLIC_IP=$(get_public_ip)
        
        # Update terraform.tfvars
        sed -i.bak "s/your-gcp-project-id/$PROJECT_ID/g" terraform.tfvars
        sed -i.bak "s/YOUR.IP.ADDRESS.HERE\/32/$PUBLIC_IP/g" terraform.tfvars
        rm terraform.tfvars.bak
        
        log_warning "Please review and customize terraform.tfvars before proceeding"
        log_info "Key settings to review:"
        log_info "- instance_type: Choose based on your budget and GPU needs"
        log_info "- preemptible: Set to false for persistent instances"
        log_info "- allowed_ips: Ensure your IP is correctly set"
        
        read -p "Press Enter to continue after reviewing terraform.tfvars..."
    fi
    
    # Initialize Terraform
    log_info "Initializing Terraform..."
    terraform init
    
    log_success "Terraform setup completed"
}

deploy_infrastructure() {
    log_info "Deploying infrastructure with Terraform..."
    
    cd "$TERRAFORM_DIR"
    
    # Plan deployment
    log_info "Creating deployment plan..."
    terraform plan -out=tfplan
    
    # Ask for confirmation
    echo ""
    log_warning "Review the deployment plan above."
    read -p "Do you want to proceed with deployment? (y/N): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Apply deployment
        log_info "Applying deployment..."
        terraform apply tfplan
        
        log_success "Infrastructure deployment completed!"
        
        # Show connection information
        echo ""
        log_info "Connection Information:"
        terraform output -raw ssh_command
        echo ""
        log_info "External IP: $(terraform output -raw instance_external_ip)"
        log_info "Jupyter Lab: http://$(terraform output -raw instance_external_ip):8888"
        log_info "Storage Bucket: $(terraform output -raw storage_bucket)"
        
    else
        log_info "Deployment cancelled"
        rm -f tfplan
        exit 0
    fi
}

wait_for_instance() {
    log_info "Waiting for instance to be ready..."
    
    cd "$TERRAFORM_DIR"
    INSTANCE_NAME=$(terraform output -raw instance_name)
    ZONE=$(terraform output -raw zone 2>/dev/null || echo "us-central1-a")
    
    # Wait for instance to be running
    while true; do
        STATUS=$(gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" --format="value(status)")
        if [ "$STATUS" = "RUNNING" ]; then
            break
        fi
        log_info "Instance status: $STATUS. Waiting..."
        sleep 10
    done
    
    # Wait for startup script to complete
    log_info "Waiting for startup script to complete (this may take 10-15 minutes)..."
    
    while true; do
        if gcloud compute ssh eva@"$INSTANCE_NAME" --zone="$ZONE" --command="test -f /var/log/eva-startup.log && tail -1 /var/log/eva-startup.log | grep -q 'Setup completed'" 2>/dev/null; then
            break
        fi
        log_info "Startup script still running..."
        sleep 30
    done
    
    log_success "Instance is ready!"
}

show_usage_instructions() {
    cd "$TERRAFORM_DIR"
    
    INSTANCE_NAME=$(terraform output -raw instance_name)
    EXTERNAL_IP=$(terraform output -raw instance_external_ip)
    SSH_COMMAND=$(terraform output -raw ssh_command)
    STORAGE_BUCKET=$(terraform output -raw storage_bucket)
    
    echo ""
    log_success "🚀 Eva DeepSeek-V3 Development Environment is Ready!"
    echo "=============================================="
    echo ""
    echo "📡 Connection Information:"
    echo "   SSH: $SSH_COMMAND"
    echo "   External IP: $EXTERNAL_IP"
    echo "   Jupyter Lab: http://$EXTERNAL_IP:8888"
    echo "   Storage Bucket: $STORAGE_BUCKET"
    echo ""
    echo "🔧 Quick Start:"
    echo "   1. Connect via SSH: $SSH_COMMAND"
    echo "   2. Activate environment: source ~/activate_env.sh"
    echo "   3. Start coding in ~/workspace/"
    echo ""
    echo "📊 VS Code Remote Setup:"
    echo "   1. Install 'Remote - SSH' extension"
    echo "   2. Add SSH config: Host eva-dev"
    echo "                      HostName $EXTERNAL_IP"
    echo "                      User eva"
    echo "   3. Connect to eva-dev in VS Code"
    echo ""
    echo "💾 File Sync:"
    echo "   Upload: gsutil -m cp -r local_folder/ gs://$STORAGE_BUCKET/"
    echo "   Download: gsutil -m cp -r gs://$STORAGE_BUCKET/folder/ ."
    echo ""
    echo "⚠️  Important Notes:"
    echo "   - Instance may be preemptible (can be terminated)"
    echo "   - Auto-shutdown after 30 min of inactivity"
    echo "   - GPU costs ~$3-5/hour, monitor usage!"
    echo ""
    echo "🛠️  Management Commands:"
    echo "   Start: gcloud compute instances start $INSTANCE_NAME --zone=$ZONE"
    echo "   Stop: gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE"
    echo "   Destroy: cd $TERRAFORM_DIR && terraform destroy"
    echo ""
}

# Main execution
main() {
    echo "🧠 Eva DeepSeek-V3 GCP Deployment Script"
    echo "========================================"
    echo ""
    
    check_prerequisites
    setup_gcp_project
    setup_ssh_keys
    setup_terraform
    deploy_infrastructure
    wait_for_instance
    show_usage_instructions
    
    log_success "Deployment completed successfully! 🎉"
}

# Handle script arguments
case "${1:-}" in
    "plan")
        cd "$TERRAFORM_DIR"
        terraform plan
        ;;
    "destroy")
        cd "$TERRAFORM_DIR"
        log_warning "This will destroy all resources!"
        read -p "Are you sure? (y/N): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            terraform destroy
        fi
        ;;
    "status")
        cd "$TERRAFORM_DIR"
        if [ -f "terraform.tfstate" ]; then
            terraform show
        else
            log_info "No infrastructure deployed"
        fi
        ;;
    *)
        main
        ;;
esac
