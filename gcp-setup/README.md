# Eva DeepSeek-V3 GCP Development Environment

A complete, production-ready Google Cloud Platform development environment optimized for DeepSeek-V3 model development and large-scale machine learning workloads.

## 🚀 Quick Start

```bash
# 1. Setup local environment
cd gcp-setup/scripts
./local-setup.sh

# 2. Deploy to GCP
./deploy.sh

# 3. Connect and start coding!
gcloud compute ssh eva@[INSTANCE_NAME] --zone=[ZONE]
```

## ✨ Features

### 🏗️ Infrastructure
- **High-performance GPU instances** (A100, T4) with CUDA 12.4
- **Automated provisioning** with Terraform
- **Cost optimization** with preemptible instances (60-91% savings)
- **Auto-shutdown** after inactivity to prevent charges
- **Secure networking** with VPC and firewall rules

### 💻 Development Environment
- **Complete ML stack**: PyTorch, TensorFlow, Hugging Face Transformers
- **Jupyter Lab** with GPU support and extensions
- **VS Code Remote Development** with SSH integration
- **Pre-configured conda environment** with all dependencies
- **Development tools**: Black, isort, flake8, mypy, pre-commit

### 📁 File Management
- **Multiple sync strategies**: Git, rsync, Cloud Storage
- **Automated backup** to Cloud Storage buckets
- **Large dataset handling** with persistent SSD storage
- **Efficient data pipelines** for ML workloads

### 🔒 Security & Monitoring
- **SSH key management** with OS Login
- **IP-restricted access** for enhanced security
- **Comprehensive logging** and monitoring
- **Service accounts** with minimal permissions

## 📊 Instance Options

| Configuration | GPU | vCPUs | RAM | Storage | Cost/hour | Use Case |
|---------------|-----|-------|-----|---------|-----------|----------|
| **High-Performance** | A100 (40GB) | 12 | 85GB | 500GB SSD | $3.67 ($1.10*) | Large models, training |
| **Ultra-Performance** | A100 (80GB) | 12 | 85GB | 500GB SSD | $4.89 ($1.47*) | Massive models |
| **Budget** | T4 (16GB) | 8 | 30GB | 500GB SSD | $0.95 ($0.29*) | Development, testing |

*Preemptible pricing (60-91% savings)

## 🛠️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Local Dev     │    │   GCP Instance   │    │ Cloud Storage   │
│                 │    │                  │    │                 │
│ • VS Code       │◄──►│ • GPU Instance   │◄──►│ • Datasets      │
│ • Git           │    │ • Jupyter Lab    │    │ • Models        │
│ • Sync Scripts  │    │ • CUDA/PyTorch   │    │ • Backups       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

- **GCP Account** with billing enabled
- **Project** with Compute Engine API access
- **GPU Quotas** in your chosen region
- **Local Tools**: Git, SSH client (installed by setup script)

## 📖 Documentation

- **[Setup Guide](docs/SETUP_GUIDE.md)** - Comprehensive setup instructions
- **[Cost Optimization](docs/SETUP_GUIDE.md#cost-management)** - Strategies to minimize costs
- **[Troubleshooting](docs/SETUP_GUIDE.md#troubleshooting)** - Common issues and solutions
- **[Security](docs/SETUP_GUIDE.md#security-best-practices)** - Best practices for secure development

## 🗂️ Project Structure

```
gcp-setup/
├── terraform/                 # Infrastructure as Code
│   ├── main.tf               # Main Terraform configuration
│   └── terraform.tfvars.example
├── scripts/                   # Automation scripts
│   ├── deploy.sh             # Main deployment script
│   ├── local-setup.sh        # Local environment setup
│   ├── enhanced-startup-script.sh  # Instance initialization
│   ├── sync-to-remote.sh     # Upload files to remote
│   └── sync-from-remote.sh   # Download files from remote
├── configs/                   # Configuration files
│   └── vscode-remote-settings.json
└── docs/                     # Documentation
    └── SETUP_GUIDE.md
```

## 🎯 Use Cases

### DeepSeek-V3 Development
- **Model Architecture**: Implement MoE, MLA, MTP components
- **Training Pipeline**: Pre-training, SFT, RLHF with GRPO
- **Inference Optimization**: 128K context handling
- **Distributed Training**: Multi-GPU setups with DeepSpeed

### General ML Development
- **Large Language Models**: Training and fine-tuning
- **Computer Vision**: Image classification, object detection
- **Research**: Experiment tracking with Weights & Biases
- **Production**: Model serving and deployment

## 💡 Best Practices

### Cost Management
1. **Use preemptible instances** for development (60-91% savings)
2. **Enable auto-shutdown** to prevent idle charges
3. **Monitor usage** with Cloud Monitoring
4. **Stop instances** when not in use

### Development Workflow
1. **Code locally** with VS Code Remote-SSH
2. **Version control** with Git
3. **Sync large files** via Cloud Storage
4. **Run experiments** on GPU instances
5. **Save checkpoints** to persistent storage

### Security
1. **Restrict SSH access** to your IP
2. **Use service accounts** with minimal permissions
3. **Enable OS Login** for centralized key management
4. **Regular security updates**

## 🔧 Management Commands

```bash
# Deploy infrastructure
./scripts/deploy.sh

# Check deployment status
./scripts/deploy.sh status

# Plan changes
./scripts/deploy.sh plan

# Destroy infrastructure
./scripts/deploy.sh destroy

# Start/stop instances
gcloud compute instances start [INSTANCE_NAME] --zone=[ZONE]
gcloud compute instances stop [INSTANCE_NAME] --zone=[ZONE]

# Sync files
./scripts/sync-to-remote.sh /local/path
./scripts/sync-from-remote.sh /remote/path
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## 🆘 Support

- **Documentation**: Check the [Setup Guide](docs/SETUP_GUIDE.md)
- **Issues**: Create GitHub issues for bugs or feature requests
- **Community**: Join discussions in the main repository

## 🙏 Acknowledgments

- **DeepSeek Team** for the original model architecture
- **Google Cloud** for the infrastructure platform
- **Open Source Community** for the tools and libraries

---

**Ready to build the next generation of AI models? Let's get started! 🧠✨**
