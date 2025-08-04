#!/bin/bash

# Eva DeepSeek-V3 Development Environment Recreation Script
# Recreates the exact development environment on the new instance

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
INSTANCE_NAME="$1"
ZONE="$2"

if [[ -z "$INSTANCE_NAME" || -z "$ZONE" ]]; then
    echo "Usage: $0 <instance_name> <zone>"
    echo "Example: $0 eva-dev-dev us-west1-a"
    exit 1
fi

echo -e "${BLUE}🔧 Recreating Eva DeepSeek-V3 Development Environment${NC}"
echo -e "${BLUE}=================================================${NC}"
echo ""
echo -e "${YELLOW}📍 Instance: ${INSTANCE_NAME}${NC}"
echo -e "${YELLOW}📍 Zone: ${ZONE}${NC}"
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

# Function to run commands on remote instance
run_remote() {
    gcloud compute ssh eva@$INSTANCE_NAME --zone=$ZONE --command="$1"
}

# Wait for instance to be accessible
echo -e "${BLUE}⏳ Waiting for instance to be accessible...${NC}"
for i in {1..10}; do
    if run_remote "echo 'Instance accessible'" &>/dev/null; then
        print_status "Instance is accessible"
        break
    fi
    echo "Waiting... ($i/10)"
    sleep 10
done

# Check if conda is installed
echo -e "${BLUE}🐍 Setting up Conda environment...${NC}"
run_remote "export PATH=/opt/conda/bin:\$PATH && conda --version" || {
    print_error "Conda not found. Please ensure the startup script completed successfully."
    exit 1
}

# Accept conda terms of service
echo -e "${BLUE}📋 Accepting Conda terms of service...${NC}"
run_remote "export PATH=/opt/conda/bin:\$PATH && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r" || true

# Create eva conda environment
echo -e "${BLUE}🔧 Creating eva conda environment...${NC}"
run_remote "export PATH=/opt/conda/bin:\$PATH && conda create -n eva python=3.10 -y"
print_status "Eva conda environment created"

# Install PyTorch and ML packages
echo -e "${BLUE}🤖 Installing PyTorch and ML packages...${NC}"
run_remote "export PATH=/opt/conda/bin:\$PATH && source activate eva && pip install --upgrade pip"
run_remote "export PATH=/opt/conda/bin:\$PATH && source activate eva && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
print_status "PyTorch installed"

run_remote "export PATH=/opt/conda/bin:\$PATH && source activate eva && pip install transformers datasets tokenizers accelerate jupyterlab notebook pandas matplotlib seaborn scikit-learn"
print_status "ML packages installed"

# Configure Jupyter Lab
echo -e "${BLUE}📓 Configuring Jupyter Lab...${NC}"
run_remote "export PATH=/opt/conda/bin:\$PATH && source activate eva && jupyter lab --generate-config"

# Set up Jupyter Lab configuration with password eva2025
run_remote "cat > ~/.jupyter/jupyter_lab_config.py << 'EOF'
# Jupyter Lab Configuration for Eva DeepSeek-V3 Development
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_remote_access = True
c.ServerApp.password = 'argon2:\$argon2id\$v=19\$m=10240,t=10,p=8\$eGDm1XeLvKiJ7+lk2LPRHg\$CbrK1LidDVZA5agC9NMhBoJ9i9iDI7cVKYLh8DpfVaA'
c.ServerApp.root_dir = '/home/eva/workspace'
c.ServerApp.notebook_dir = '/home/eva/workspace'
c.ServerApp.allow_origin = '*'
c.ServerApp.disable_check_xsrf = True
EOF"
print_status "Jupyter Lab configured with password: eva2025"

# Start Jupyter Lab
echo -e "${BLUE}🚀 Starting Jupyter Lab...${NC}"
run_remote "export PATH=/opt/conda/bin:\$PATH && source activate eva && nohup jupyter lab > /tmp/jupyter.log 2>&1 &"
sleep 5
print_status "Jupyter Lab started"

# Clone Eva repository
echo -e "${BLUE}📂 Cloning Eva DeepSeek-V3 repository...${NC}"
run_remote "cd workspace && git clone https://github.com/SiaFahim/eva.git"
print_status "Repository cloned to /home/eva/workspace/eva"

# Create environment activation script
echo -e "${BLUE}⚙️  Creating environment activation script...${NC}"
run_remote "cat > ~/activate_env.sh << 'EOF'
#!/bin/bash
# Eva DeepSeek-V3 Environment Activation Script
export PATH=/opt/conda/bin:\$PATH
source activate eva
echo \"🧠 Eva DeepSeek-V3 Development Environment Activated\"
echo \"📍 Python: \$(python --version)\"
echo \"📍 PyTorch: \$(python -c 'import torch; print(torch.__version__)')\"
echo \"📍 Working Directory: \$(pwd)\"
echo \"📍 Jupyter Lab: http://\$(curl -s ifconfig.me):8888 (password: eva2025)\"
EOF"
run_remote "chmod +x ~/activate_env.sh"
print_status "Environment activation script created"

# Test the environment
echo -e "${BLUE}🧪 Testing development environment...${NC}"
run_remote "export PATH=/opt/conda/bin:\$PATH && source activate eva && python -c \"
import torch
import transformers
import pandas as pd
import numpy as np
print('🎉 Environment Test Results:')
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ Transformers: {transformers.__version__}')
print(f'✅ Pandas: {pd.__version__}')
print(f'✅ NumPy: {np.__version__}')
print(f'✅ CUDA Available: {torch.cuda.is_available()}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'✅ Device: {device}')
print('🚀 Eva DeepSeek-V3 Development Environment Ready!')
\""

print_status "Environment test completed successfully!"

# Get instance IP
EXTERNAL_IP=$(gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --format="value(networkInterfaces[0].accessConfigs[0].natIP)")

echo ""
echo -e "${GREEN}🎉 Development Environment Recreation Complete!${NC}"
echo -e "${GREEN}=============================================${NC}"
echo ""
echo "Environment details:"
echo "  • Instance: $INSTANCE_NAME"
echo "  • External IP: $EXTERNAL_IP"
echo "  • Jupyter Lab: http://$EXTERNAL_IP:8888 (password: eva2025)"
echo "  • Repository: /home/eva/workspace/eva"
echo "  • Environment: conda activate eva"
echo ""
echo "Next steps:"
echo "  1. Update your VS Code SSH configuration:"
echo "     Host eva-dev"
echo "       HostName $EXTERNAL_IP"
echo "  2. Connect via VS Code Remote-SSH"
echo "  3. Open folder: /home/eva/workspace/eva"
echo "  4. Start developing!"
