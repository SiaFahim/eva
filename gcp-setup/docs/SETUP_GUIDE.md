# Eva DeepSeek-V3 GCP Development Environment Setup Guide

This guide provides step-by-step instructions for setting up a complete Google Cloud Platform (GCP) development environment optimized for DeepSeek-V3 model development.

## 🎯 Overview

The setup includes:
- **High-performance GPU instances** (A100/T4) for ML development
- **VS Code Remote Development** with SSH integration
- **Automated environment setup** with CUDA, PyTorch, TensorFlow
- **Cost optimization** with preemptible instances and auto-shutdown
- **File synchronization** between local and remote environments
- **Jupyter Lab** for interactive development

## 📋 Prerequisites

### Local Machine Requirements
- **Operating System**: macOS, Linux, or Windows with WSL2
- **Tools**: Git, SSH client
- **Optional**: VS Code, Conda/Miniconda

### GCP Requirements
- **GCP Account** with billing enabled
- **Project** with Compute Engine API enabled
- **Quotas**: GPU quotas in your chosen region
- **Permissions**: Compute Admin, Storage Admin, IAM Admin

## 🚀 Quick Start

### 1. Local Environment Setup

Run the local setup script to install required tools:

```bash
cd gcp-setup/scripts
chmod +x local-setup.sh
./local-setup.sh
```

This script will:
- Install Google Cloud CLI
- Install Terraform
- Set up conda environment
- Configure VS Code extensions
- Create file sync scripts

### 2. GCP Authentication

Authenticate with Google Cloud:

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### 3. Deploy Infrastructure

Deploy the development environment:

```bash
cd gcp-setup/scripts
chmod +x deploy.sh
./deploy.sh
```

The deployment script will:
- Enable required GCP APIs
- Set up SSH keys
- Create Terraform configuration
- Deploy infrastructure
- Wait for instance to be ready

### 4. Connect and Start Developing

Once deployed, you can connect via:

**SSH:**
```bash
gcloud compute ssh eva@eva-dev-dev --zone=us-central1-a
```

**VS Code Remote:**
1. Open VS Code
2. Install "Remote - SSH" extension
3. Press `Ctrl+Shift+P` → "Remote-SSH: Connect to Host"
4. Select `eva-dev`

**Jupyter Lab:**
Access at `http://INSTANCE_IP:8888`

## 🏗️ Architecture Details

### Instance Configuration

**High-Performance Option (Recommended):**
- **Instance**: `a2-highgpu-1g`
- **GPU**: 1x NVIDIA A100 (40GB VRAM)
- **CPU**: 12 vCPUs
- **Memory**: 85GB RAM
- **Cost**: ~$3.67/hour (preemptible: ~$1.10/hour)

**Budget Option:**
- **Instance**: `n1-standard-8`
- **GPU**: 1x NVIDIA Tesla T4 (16GB VRAM)
- **CPU**: 8 vCPUs
- **Memory**: 30GB RAM
- **Cost**: ~$0.95/hour (preemptible: ~$0.29/hour)

### Software Stack

**Base System:**
- Ubuntu 22.04 LTS
- NVIDIA drivers + CUDA 12.4
- Docker with NVIDIA runtime

**Python Environment:**
- Miniconda with Python 3.10
- PyTorch 2.x with CUDA support
- TensorFlow 2.15 with GPU support
- Hugging Face Transformers
- Jupyter Lab + extensions

**Development Tools:**
- Git, tmux, htop
- Black, isort, flake8, mypy
- Pre-commit hooks
- DeepSpeed for distributed training

## 💰 Cost Management

### Preemptible Instances
- **Savings**: 60-91% cost reduction
- **Trade-off**: Can be terminated with 30-second notice
- **Best for**: Development, experimentation, fault-tolerant workloads

### Auto-Shutdown
- Automatically shuts down after 30 minutes of inactivity
- Checks for active SSH sessions, Jupyter processes, and GPU utilization
- Prevents accidental charges from idle instances

### Manual Management
```bash
# Start instance
gcloud compute instances start eva-dev-dev --zone=us-central1-a

# Stop instance
gcloud compute instances stop eva-dev-dev --zone=us-central1-a

# Check status
gcloud compute instances list
```

## 📁 File Synchronization

### Method 1: Git Workflow (Recommended)
```bash
# On local machine
git add . && git commit -m "Update" && git push

# On remote instance
git pull
```

### Method 2: Direct Sync Scripts
```bash
# Upload files
./sync-to-remote.sh /local/path /remote/path

# Download files
./sync-from-remote.sh /remote/path /local/path
```

### Method 3: Cloud Storage
```bash
# Upload to bucket
gsutil -m cp -r local_folder/ gs://your-bucket/

# Download from bucket
gsutil -m cp -r gs://your-bucket/folder/ .
```

## 🔧 VS Code Remote Development

### Initial Setup
1. Install "Remote - SSH" extension
2. Update SSH config with instance IP:
   ```
   Host eva-dev
       HostName YOUR_INSTANCE_IP
       User eva
       IdentityFile ~/.ssh/id_rsa
       LocalForward 8888 localhost:8888
       LocalForward 6006 localhost:6006
   ```

### Recommended Extensions
- Python
- Jupyter
- Black Formatter
- isort
- Flake8
- MyPy Type Checker
- Docker
- Terraform

### Settings
Copy `configs/vscode-remote-settings.json` to your VS Code settings for optimal remote development experience.

## 🛠️ Troubleshooting

### Common Issues

**1. GPU Not Available**
```bash
# Check GPU status
nvidia-smi

# Verify CUDA installation
nvcc --version

# Test PyTorch GPU
python -c "import torch; print(torch.cuda.is_available())"
```

**2. Jupyter Lab Not Accessible**
```bash
# Check service status
systemctl status eva-jupyter

# Restart service
sudo systemctl restart eva-jupyter

# Check firewall
gcloud compute firewall-rules list --filter="name:eva"
```

**3. SSH Connection Issues**
```bash
# Check instance status
gcloud compute instances list

# Reset SSH keys
gcloud compute os-login ssh-keys add --key-file=~/.ssh/id_rsa.pub

# Connect with verbose output
ssh -v eva@INSTANCE_IP
```

**4. Out of Disk Space**
```bash
# Check disk usage
df -h

# Clean up conda cache
conda clean --all

# Clean up pip cache
pip cache purge

# Clean up Docker
docker system prune -a
```

### Performance Optimization

**1. Monitor Resource Usage**
```bash
# GPU utilization
nvidia-smi -l 1

# CPU and memory
htop

# Disk I/O
iotop
```

**2. Optimize Data Loading**
- Use SSD persistent disks for datasets
- Implement efficient data pipelines
- Use multiple workers for data loading
- Cache preprocessed data

## 🔒 Security Best Practices

### Network Security
- Restrict SSH access to your IP address
- Use VPC with private subnets
- Enable OS Login for centralized SSH key management

### Access Control
- Use service accounts with minimal permissions
- Regularly rotate SSH keys
- Monitor access logs

### Data Protection
- Enable disk encryption
- Use Cloud Storage with versioning
- Regular backups of important data

## 📊 Monitoring and Logging

### Built-in Monitoring
- Cloud Monitoring for instance metrics
- Stackdriver Logging for application logs
- Custom metrics for ML training

### Local Monitoring
```bash
# Check startup logs
tail -f /var/log/eva-startup.log

# Monitor auto-shutdown
tail -f /var/log/auto-shutdown.log

# System logs
journalctl -u eva-jupyter -f
```

## 🔄 Maintenance

### Regular Updates
```bash
# Update system packages
sudo apt update && sudo apt upgrade

# Update conda environment
conda update --all

# Update pip packages
pip list --outdated | cut -d' ' -f1 | xargs pip install --upgrade
```

### Backup Strategy
1. **Code**: Use Git repositories
2. **Data**: Sync to Cloud Storage buckets
3. **Models**: Save checkpoints to persistent storage
4. **Environment**: Document dependencies in requirements.txt

## 🆘 Support and Resources

### Documentation
- [Google Cloud Compute Engine](https://cloud.google.com/compute/docs)
- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [VS Code Remote Development](https://code.visualstudio.com/docs/remote/remote-overview)

### Community
- [DeepSeek GitHub](https://github.com/deepseek-ai)
- [Google Cloud Community](https://cloud.google.com/community)
- [PyTorch Forums](https://discuss.pytorch.org/)

### Getting Help
1. Check the troubleshooting section above
2. Review logs for error messages
3. Search community forums
4. Create GitHub issues for project-specific problems

---

**Happy coding with Eva DeepSeek-V3! 🧠✨**
