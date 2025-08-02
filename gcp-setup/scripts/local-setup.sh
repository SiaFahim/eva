#!/bin/bash
# Local Environment Setup for Eva DeepSeek-V3 Development
# This script sets up your local machine for remote development

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

install_gcloud_cli() {
    local os=$(detect_os)
    
    if command -v gcloud &> /dev/null; then
        log_info "Google Cloud CLI already installed"
        return 0
    fi
    
    log_info "Installing Google Cloud CLI..."
    
    case $os in
        "macos")
            if command -v brew &> /dev/null; then
                brew install --cask google-cloud-sdk
            else
                log_info "Installing via curl (Homebrew not found)..."
                curl https://sdk.cloud.google.com | bash
                exec -l $SHELL
            fi
            ;;
        "linux")
            # Add Google Cloud SDK repository
            echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
            curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
            sudo apt-get update && sudo apt-get install google-cloud-cli
            ;;
        "windows")
            log_warning "Please install Google Cloud CLI manually from: https://cloud.google.com/sdk/docs/install"
            log_warning "Then run this script again."
            exit 1
            ;;
        *)
            log_error "Unsupported operating system"
            exit 1
            ;;
    esac
    
    log_success "Google Cloud CLI installed"
}

install_terraform() {
    local os=$(detect_os)
    
    if command -v terraform &> /dev/null; then
        log_info "Terraform already installed"
        return 0
    fi
    
    log_info "Installing Terraform..."
    
    case $os in
        "macos")
            if command -v brew &> /dev/null; then
                brew tap hashicorp/tap
                brew install hashicorp/tap/terraform
            else
                log_warning "Please install Homebrew first or install Terraform manually"
                exit 1
            fi
            ;;
        "linux")
            wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor | sudo tee /usr/share/keyrings/hashicorp-archive-keyring.gpg
            echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
            sudo apt update && sudo apt install terraform
            ;;
        "windows")
            log_warning "Please install Terraform manually from: https://www.terraform.io/downloads"
            exit 1
            ;;
        *)
            log_error "Unsupported operating system"
            exit 1
            ;;
    esac
    
    log_success "Terraform installed"
}

setup_conda_environment() {
    log_info "Setting up local conda environment..."
    
    # Check if conda is installed
    if ! command -v conda &> /dev/null; then
        log_warning "Conda not found. Please install Miniconda or Anaconda first."
        log_info "Download from: https://docs.conda.io/en/latest/miniconda.html"
        return 1
    fi
    
    # Create eva environment if it doesn't exist
    if ! conda env list | grep -q "^eva "; then
        log_info "Creating conda environment 'eva'..."
        conda create -n eva python=3.10 -y
    else
        log_info "Conda environment 'eva' already exists"
    fi
    
    # Activate and install packages
    log_info "Installing local development packages..."
    eval "$(conda shell.bash hook)"
    conda activate eva
    
    # Install essential packages for local development
    pip install --upgrade pip
    
    # Development tools
    pip install jupyter jupyterlab
    pip install black isort flake8 mypy
    pip install pre-commit
    
    # Basic ML libraries (for local testing)
    pip install numpy pandas matplotlib seaborn
    pip install scikit-learn
    
    # Google Cloud libraries
    pip install google-cloud-storage
    pip install google-cloud-compute
    
    # Remote development tools
    pip install fabric  # For remote command execution
    pip install paramiko  # SSH client
    
    log_success "Local conda environment setup completed"
}

setup_vscode_config() {
    log_info "Setting up VS Code configuration..."
    
    # Check if VS Code is installed
    if ! command -v code &> /dev/null; then
        log_warning "VS Code not found. Please install VS Code first."
        log_info "Download from: https://code.visualstudio.com/"
        return 1
    fi
    
    # Install essential extensions
    log_info "Installing VS Code extensions..."
    
    extensions=(
        "ms-vscode-remote.remote-ssh"
        "ms-vscode-remote.remote-ssh-edit"
        "ms-python.python"
        "ms-toolsai.jupyter"
        "ms-python.black-formatter"
        "ms-python.isort"
        "ms-python.flake8"
        "ms-python.mypy-type-checker"
        "ms-vscode.vscode-json"
        "redhat.vscode-yaml"
        "hashicorp.terraform"
        "ms-vscode.vscode-docker"
    )
    
    for extension in "${extensions[@]}"; do
        if ! code --list-extensions | grep -q "$extension"; then
            log_info "Installing extension: $extension"
            code --install-extension "$extension"
        else
            log_info "Extension already installed: $extension"
        fi
    done
    
    # Create VS Code settings for remote development
    VSCODE_SETTINGS_DIR="$HOME/.vscode"
    mkdir -p "$VSCODE_SETTINGS_DIR"
    
    cat > "$VSCODE_SETTINGS_DIR/eva-remote-settings.json" << 'VSCODE_SETTINGS'
{
    "python.defaultInterpreterPath": "/opt/conda/envs/eva/bin/python",
    "python.terminal.activateEnvironment": true,
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "jupyter.jupyterServerType": "remote",
    "files.watcherExclude": {
        "**/.git/objects/**": true,
        "**/.git/subtree-cache/**": true,
        "**/node_modules/*/**": true,
        "**/__pycache__/**": true,
        "**/.*cache/**": true
    },
    "terminal.integrated.defaultProfile.linux": "bash",
    "terminal.integrated.profiles.linux": {
        "bash": {
            "path": "/bin/bash",
            "args": ["-c", "source /home/eva/activate_env.sh && exec bash"]
        }
    }
}
VSCODE_SETTINGS
    
    log_success "VS Code configuration completed"
    log_info "Remote settings saved to: $VSCODE_SETTINGS_DIR/eva-remote-settings.json"
}

create_sync_scripts() {
    log_info "Creating file synchronization scripts..."
    
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    # Upload script
    cat > "$SCRIPT_DIR/sync-to-remote.sh" << 'SYNC_UP'
#!/bin/bash
# Sync local files to remote GCP instance

set -e

# Configuration
LOCAL_DIR="${1:-.}"
REMOTE_DIR="${2:-~/workspace/projects/eva}"
BUCKET_NAME="${3:-}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Get instance info from Terraform
TERRAFORM_DIR="$(dirname "$(dirname "$0")")/terraform"
if [ -f "$TERRAFORM_DIR/terraform.tfstate" ]; then
    cd "$TERRAFORM_DIR"
    INSTANCE_NAME=$(terraform output -raw instance_name 2>/dev/null || echo "")
    ZONE=$(terraform output -raw zone 2>/dev/null || echo "us-central1-a")
    BUCKET_NAME=$(terraform output -raw storage_bucket 2>/dev/null || echo "$BUCKET_NAME")
fi

if [ -z "$INSTANCE_NAME" ]; then
    echo "Error: Could not determine instance name. Is infrastructure deployed?"
    exit 1
fi

log_info "Syncing $LOCAL_DIR to remote instance..."

# Sync via rsync over SSH
gcloud compute scp --recurse "$LOCAL_DIR" eva@"$INSTANCE_NAME":"$REMOTE_DIR" --zone="$ZONE"

log_success "Files synced to remote instance"

# Also sync to bucket if specified
if [ -n "$BUCKET_NAME" ]; then
    log_info "Syncing to Cloud Storage bucket: $BUCKET_NAME"
    gsutil -m rsync -r -d "$LOCAL_DIR" "gs://$BUCKET_NAME/projects/eva/"
    log_success "Files synced to Cloud Storage"
fi
SYNC_UP
    
    # Download script
    cat > "$SCRIPT_DIR/sync-from-remote.sh" << 'SYNC_DOWN'
#!/bin/bash
# Sync files from remote GCP instance to local

set -e

# Configuration
REMOTE_DIR="${1:-~/workspace/projects/eva}"
LOCAL_DIR="${2:-.}"
BUCKET_NAME="${3:-}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Get instance info from Terraform
TERRAFORM_DIR="$(dirname "$(dirname "$0")")/terraform"
if [ -f "$TERRAFORM_DIR/terraform.tfstate" ]; then
    cd "$TERRAFORM_DIR"
    INSTANCE_NAME=$(terraform output -raw instance_name 2>/dev/null || echo "")
    ZONE=$(terraform output -raw zone 2>/dev/null || echo "us-central1-a")
    BUCKET_NAME=$(terraform output -raw storage_bucket 2>/dev/null || echo "$BUCKET_NAME")
fi

if [ -z "$INSTANCE_NAME" ]; then
    echo "Error: Could not determine instance name. Is infrastructure deployed?"
    exit 1
fi

log_info "Syncing from remote instance to $LOCAL_DIR..."

# Sync via rsync over SSH
gcloud compute scp --recurse eva@"$INSTANCE_NAME":"$REMOTE_DIR" "$LOCAL_DIR" --zone="$ZONE"

log_success "Files synced from remote instance"

# Also sync from bucket if specified
if [ -n "$BUCKET_NAME" ]; then
    log_info "Syncing from Cloud Storage bucket: $BUCKET_NAME"
    gsutil -m rsync -r -d "gs://$BUCKET_NAME/projects/eva/" "$LOCAL_DIR"
    log_success "Files synced from Cloud Storage"
fi
SYNC_DOWN
    
    chmod +x "$SCRIPT_DIR/sync-to-remote.sh"
    chmod +x "$SCRIPT_DIR/sync-from-remote.sh"
    
    log_success "Sync scripts created"
    log_info "Upload files: $SCRIPT_DIR/sync-to-remote.sh [local_dir] [remote_dir]"
    log_info "Download files: $SCRIPT_DIR/sync-from-remote.sh [remote_dir] [local_dir]"
}

setup_ssh_config() {
    log_info "Setting up SSH configuration..."
    
    SSH_CONFIG="$HOME/.ssh/config"
    
    # Backup existing config
    if [ -f "$SSH_CONFIG" ]; then
        cp "$SSH_CONFIG" "$SSH_CONFIG.backup.$(date +%Y%m%d_%H%M%S)"
    fi
    
    # Add Eva development host configuration
    if ! grep -q "Host eva-dev" "$SSH_CONFIG" 2>/dev/null; then
        cat >> "$SSH_CONFIG" << 'SSH_CONFIG'

# Eva DeepSeek-V3 Development Environment
Host eva-dev
    HostName [YOUR_INSTANCE_IP]
    User eva
    IdentityFile ~/.ssh/google_compute_engine
    ServerAliveInterval 60
    ServerAliveCountMax 3
    LocalForward 8888 localhost:8888
    LocalForward 6006 localhost:6006
    LocalForward 8080 localhost:8080
SSH_CONFIG
        
        log_success "SSH config added for eva-dev"
        log_warning "Update the HostName in ~/.ssh/config with your instance IP after deployment"
    else
        log_info "SSH config for eva-dev already exists"
    fi
}

show_next_steps() {
    echo ""
    log_success "🚀 Local Environment Setup Complete!"
    echo "====================================="
    echo ""
    echo "Next Steps:"
    echo "1. Deploy GCP infrastructure:"
    echo "   cd gcp-setup/scripts && ./deploy.sh"
    echo ""
    echo "2. Update SSH config with instance IP:"
    echo "   Edit ~/.ssh/config and replace INSTANCE_IP_PLACEHOLDER"
    echo ""
    echo "3. Connect with VS Code:"
    echo "   - Open VS Code"
    echo "   - Press Ctrl+Shift+P (Cmd+Shift+P on Mac)"
    echo "   - Type 'Remote-SSH: Connect to Host'"
    echo "   - Select 'eva-dev'"
    echo ""
    echo "4. Sync your code:"
    echo "   ./sync-to-remote.sh /path/to/your/code"
    echo ""
    echo "Happy coding! 🧠✨"
}

# Main execution
main() {
    echo "🧠 Eva DeepSeek-V3 Local Setup"
    echo "============================="
    echo ""
    
    install_gcloud_cli
    install_terraform
    setup_conda_environment
    setup_vscode_config
    create_sync_scripts
    setup_ssh_config
    show_next_steps
    
    log_success "Local setup completed successfully! 🎉"
}

# Run main function
main "$@"
