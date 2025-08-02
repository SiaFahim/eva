#!/bin/bash
# Enhanced startup script for Eva DeepSeek-V3 development instance
# This script sets up a complete ML development environment

set -e

# Configuration
EVA_USER="eva"
CONDA_VERSION="latest"
PYTHON_VERSION="3.10"
CUDA_VERSION="12.4"

# Logging
exec > >(tee -a /var/log/eva-startup.log)
exec 2>&1
echo "$(date): Starting Eva development environment setup..."

# Update system
echo "$(date): Updating system packages..."
apt-get update -y
apt-get upgrade -y

# Install essential packages
echo "$(date): Installing essential packages..."
apt-get install -y \
    git curl wget htop tree vim nano \
    build-essential software-properties-common \
    apt-transport-https ca-certificates gnupg lsb-release \
    unzip zip p7zip-full \
    tmux screen \
    rsync \
    jq

# Install NVIDIA drivers and CUDA
echo "$(date): Installing NVIDIA drivers and CUDA..."
apt-get install -y ubuntu-drivers-common
ubuntu-drivers autoinstall

# Add NVIDIA package repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb
apt-get update

# Install CUDA toolkit
apt-get install -y cuda-toolkit-12-4 nvidia-cuda-toolkit

# Install Docker
echo "$(date): Installing Docker..."
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
rm get-docker.sh

# Install NVIDIA Docker runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update
apt-get install -y nvidia-container-toolkit
systemctl restart docker

# Install Google Cloud SDK
echo "$(date): Installing Google Cloud SDK..."
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
apt-get update
apt-get install -y google-cloud-cli

# Install Miniconda
echo "$(date): Installing Miniconda..."
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p /opt/conda
rm miniconda.sh
echo 'export PATH="/opt/conda/bin:$PATH"' >> /etc/profile
export PATH="/opt/conda/bin:$PATH"

# Create eva user if not exists
if ! id "$EVA_USER" &>/dev/null; then
    echo "$(date): Creating eva user..."
    useradd -m -s /bin/bash $EVA_USER
    usermod -aG sudo,docker $EVA_USER
    echo "$EVA_USER ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
    
    # Setup SSH directory
    mkdir -p /home/$EVA_USER/.ssh
    chmod 700 /home/$EVA_USER/.ssh
    chown $EVA_USER:$EVA_USER /home/$EVA_USER/.ssh
fi

# Mount additional disk
echo "$(date): Setting up additional storage..."
DEVICE="/dev/sdb"
MOUNT_POINT="/home/$EVA_USER/workspace"

if [ -b "$DEVICE" ]; then
    # Format disk if not already formatted
    if ! blkid $DEVICE; then
        mkfs.ext4 $DEVICE
    fi
    
    # Create mount point and mount
    mkdir -p $MOUNT_POINT
    mount $DEVICE $MOUNT_POINT
    
    # Add to fstab for persistent mounting
    echo "$DEVICE $MOUNT_POINT ext4 defaults 0 2" >> /etc/fstab
    
    # Set ownership
    chown -R $EVA_USER:$EVA_USER $MOUNT_POINT
fi

# Setup conda environment for eva user
echo "$(date): Setting up conda environment..."
sudo -u $EVA_USER bash << 'CONDA_SETUP'
export PATH="/opt/conda/bin:$PATH"

# Create eva environment
conda create -n eva python=3.10 -y
source activate eva

# Install core ML packages
pip install --upgrade pip

# PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# TensorFlow with GPU support
pip install tensorflow[and-cuda]==2.15.0

# Hugging Face ecosystem
pip install transformers datasets tokenizers accelerate

# Development tools
pip install jupyterlab==4.0.7 notebook
pip install ipywidgets jupyter_contrib_nbextensions

# Scientific computing
pip install numpy pandas matplotlib seaborn plotly
pip install scikit-learn scipy

# ML utilities
pip install wandb tensorboard
pip install mlflow optuna

# Development and testing
pip install pytest black isort flake8 mypy
pip install pre-commit

# DeepSpeed for distributed training
pip install deepspeed

# Additional ML libraries
pip install xformers  # Memory efficient transformers
pip install flash-attn  # Flash attention implementation

# Create activation script
cat > /home/eva/activate_env.sh << 'ACTIVATE_EOF'
#!/bin/bash
export PATH="/opt/conda/bin:$PATH"
conda activate eva

# CUDA environment variables
export CUDA_VISIBLE_DEVICES=0
export TF_CPP_MIN_LOG_LEVEL=1
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Hugging Face cache
export HF_HOME="/home/eva/workspace/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/eva/workspace/.cache/huggingface/transformers"

echo "🧠 Eva development environment activated"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
fi
ACTIVATE_EOF

chmod +x /home/eva/activate_env.sh
CONDA_SETUP

# Setup Jupyter Lab
echo "$(date): Configuring Jupyter Lab..."
sudo -u $EVA_USER bash << 'JUPYTER_SETUP'
export PATH="/opt/conda/bin:$PATH"
source activate eva

# Generate Jupyter config
jupyter lab --generate-config

# Create Jupyter config
cat > /home/eva/.jupyter/jupyter_lab_config.py << 'JUPYTER_CONFIG'
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_root = False
c.ServerApp.token = ''
c.ServerApp.password = ''
c.ServerApp.allow_remote_access = True
c.ServerApp.notebook_dir = '/home/eva/workspace'
JUPYTER_CONFIG

# Install Jupyter extensions
jupyter labextension install @jupyter-widgets/jupyterlab-manager
JUPYTER_SETUP

# Setup auto-shutdown script
echo "$(date): Setting up auto-shutdown..."
cat > /home/$EVA_USER/auto-shutdown.sh << 'SHUTDOWN_EOF'
#!/bin/bash
# Auto-shutdown if no active sessions for 30 minutes

IDLE_TIME=30  # minutes
LOG_FILE="/var/log/auto-shutdown.log"

# Check for active SSH sessions
ACTIVE_SESSIONS=$(who | wc -l)

# Check for running Jupyter processes
JUPYTER_PROCESSES=$(pgrep -f jupyter | wc -l)

# Check for GPU utilization
GPU_UTIL=0
if command -v nvidia-smi &> /dev/null; then
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
fi

echo "$(date): Sessions: $ACTIVE_SESSIONS, Jupyter: $JUPYTER_PROCESSES, GPU: $GPU_UTIL%" >> $LOG_FILE

# Shutdown if no activity
if [ $ACTIVE_SESSIONS -eq 0 ] && [ $JUPYTER_PROCESSES -eq 0 ] && [ $GPU_UTIL -lt 5 ]; then
    echo "$(date): No activity detected. Scheduling shutdown in $IDLE_TIME minutes." >> $LOG_FILE
    shutdown -h +$IDLE_TIME "Auto-shutdown: No active development sessions detected"
else
    echo "$(date): Activity detected. Cancelling any pending shutdowns." >> $LOG_FILE
    shutdown -c 2>/dev/null || true
fi
SHUTDOWN_EOF

chmod +x /home/$EVA_USER/auto-shutdown.sh

# Add to crontab for eva user
sudo -u $EVA_USER bash -c 'echo "*/15 * * * * /home/eva/auto-shutdown.sh" | crontab -'

# Setup systemd service for Jupyter
cat > /etc/systemd/system/eva-jupyter.service << 'SERVICE_EOF'
[Unit]
Description=Eva Jupyter Lab
After=network.target

[Service]
Type=simple
User=eva
WorkingDirectory=/home/eva/workspace
Environment=PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=/bin/bash -c 'source /opt/conda/bin/activate eva && jupyter lab'
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
SERVICE_EOF

systemctl daemon-reload
systemctl enable eva-jupyter.service

# Create workspace directories
echo "$(date): Setting up workspace directories..."
sudo -u $EVA_USER mkdir -p /home/$EVA_USER/workspace/{projects,datasets,models,notebooks,scripts}

# Set proper ownership
chown -R $EVA_USER:$EVA_USER /home/$EVA_USER/

# Create welcome script
cat > /home/$EVA_USER/welcome.sh << 'WELCOME_EOF'
#!/bin/bash
echo "🚀 Welcome to Eva DeepSeek-V3 Development Environment!"
echo "=================================================="
echo ""
echo "Quick Start:"
echo "1. Activate environment: source ~/activate_env.sh"
echo "2. Start Jupyter Lab: jupyter lab (or systemctl start eva-jupyter)"
echo "3. Access Jupyter at: http://$(curl -s ifconfig.me):8888"
echo ""
echo "Useful commands:"
echo "- Check GPU: nvidia-smi"
echo "- Monitor resources: htop"
echo "- Sync files: gsutil rsync -r gs://your-bucket/ ~/workspace/"
echo ""
echo "Directories:"
echo "- ~/workspace/projects/ - Your code projects"
echo "- ~/workspace/datasets/ - Training datasets"
echo "- ~/workspace/models/ - Model checkpoints"
echo "- ~/workspace/notebooks/ - Jupyter notebooks"
echo ""
echo "Happy coding! 🧠✨"
WELCOME_EOF

chmod +x /home/$EVA_USER/welcome.sh

# Add welcome to bashrc
echo "source /home/$EVA_USER/welcome.sh" >> /home/$EVA_USER/.bashrc

echo "$(date): Eva development environment setup completed successfully!"
echo "$(date): Instance is ready for DeepSeek-V3 development."

# Start Jupyter service
systemctl start eva-jupyter.service

# Final system info
echo "$(date): System Information:"
echo "- OS: $(lsb_release -d | cut -f2)"
echo "- Kernel: $(uname -r)"
echo "- CUDA: $(nvcc --version | grep release | cut -d' ' -f5-6)"
echo "- Docker: $(docker --version)"
echo "- Python: $(python3 --version)"

if command -v nvidia-smi &> /dev/null; then
    echo "- GPU Info:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

echo "$(date): Setup completed. Rebooting to ensure all drivers are loaded..."
reboot
