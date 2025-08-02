# Eva DeepSeek-V3

A comprehensive full replication of **DeepSeek-V3**, the state-of-the-art 671B parameter Mixture-of-Experts (MoE) model with 37B activated parameters per token. This project implements all core architectural innovations including Multi-head Latent Attention (MLA), advanced MoE with 256 experts, and Multi-Token Prediction (MTP) capabilities.

## 🎯 Project Overview

Eva DeepSeek-V3 is an ambitious reverse-engineering project that aims to fully replicate DeepSeek-V3's capabilities from the ground up. Our implementation focuses on:

- **Complete architectural replication** of the 671B parameter MoE model
- **Production-ready training pipeline** supporting 14.8T token pre-training
- **Advanced serving infrastructure** with 128K context window support
- **Comprehensive alignment methodologies** using Group Relative Policy Optimization (GRPO)

## 🏗️ Key Technical Components

### Multi-head Latent Attention (MLA)
- Reduces KV cache memory usage by **93.3%** compared to vanilla Transformers
- Enables efficient handling of 128K context windows
- Optimized for both training and inference workloads

### DeepSeekMoE Architecture
- **256 expert networks** with auxiliary-loss-free load balancing
- Fine-grained expert routing with shared expert mechanisms
- Prevents routing collapse while maintaining training stability

### Multi-Token Prediction (MTP)
- Accelerates inference through parallel token generation
- Maintains model quality while improving throughput
- Integrated with MLA for optimal memory efficiency

## 📋 Development Phases

The project is organized into 4 main development phases, each with comprehensive documentation and implementation guides:

### Phase 1: MLA Implementation
- Core Multi-head Latent Attention mechanisms
- KV cache optimization and memory management
- Component-level testing and validation

### Phase 2: Advanced MoE Architecture
- DeepSeekMoE layer implementation with 256 experts
- Load balancing and routing optimization
- Expert specialization and training dynamics

### Phase 3: Distributed Training
- Multi-node training infrastructure
- FP8 mixed precision implementation
- DualPipe parallelism for compute/communication overlap

### Phase 4: Training Pipeline
- Complete pre-training pipeline (14.8T tokens)
- Supervised Fine-Tuning (SFT) implementation
- GRPO-based reinforcement learning alignment

## 📁 Project Structure

```
eva/
├── docs/                           # Comprehensive technical documentation
│   ├── phase0/                     # Development environment setup
│   ├── phase1/                     # Core components (MLA, basic MoE)
│   ├── phase2/                     # Advanced MoE architecture
│   ├── phase3/                     # Distributed training
│   ├── phase4/                     # Training pipeline
│   ├── phase5/                     # Alignment and GRPO
│   ├── phase6/                     # Serving and optimization
│   ├── PROJECT_OBJECTIVE.md        # Detailed project goals and scope
│   ├── DeepSeek-V3-Technical-Reference.md
│   └── Engineering-Documentation-Development-Plan.md
├── gcp-setup/                      # Google Cloud Platform development environment
│   ├── terraform/                  # Infrastructure as Code
│   ├── scripts/                    # Deployment and setup scripts
│   └── configs/                    # Environment configurations
└── README.md                       # This file
```

## 🚀 Development Environment

The project uses a Google Cloud Platform-based development environment optimized for large-scale ML workloads:

### GCP Instance Configuration
- **Instance Type**: n1-standard-4 (preemptible for cost optimization)
- **Development Access**: SSH via VS Code Remote-SSH
- **Jupyter Lab**: Available at configured instance IP on port 8888
- **Auto-shutdown**: 30-minute inactivity timeout for cost control

### Environment Setup
```bash
# Connect to development instance
ssh eva-dev

# Activate conda environment
source activate eva
# or
source ~/activate_env.sh

# Verify environment
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

## 🛠️ Getting Started

### Prerequisites
- Google Cloud Platform account with appropriate quotas
- SSH access configured for the development instance
- VS Code with Remote-SSH extension (recommended)

### Quick Start
1. **Connect to Development Environment**:
   ```bash
   ssh eva-dev
   cd /home/eva/workspace/eva
   ```

2. **Activate Environment**:
   ```bash
   source activate eva
   ```

3. **Explore Documentation**:
   ```bash
   # Review project objectives
   cat docs/PROJECT_OBJECTIVE.md

   # Check development plan
   cat docs/Engineering-Documentation-Development-Plan.md
   ```

4. **Start Development**:
   - Use VS Code Remote-SSH for primary development
   - Access Jupyter Lab via the configured instance IP on port 8888
   - Follow phase-specific documentation in `docs/phase*/`

### Development Workflow
1. **Phase-based Development**: Follow the structured 4-phase approach
2. **Test-as-you-Develop**: Use synthetic data validation to avoid expensive pre-training
3. **Documentation-Driven**: Each component has comprehensive implementation guides
4. **Incremental Validation**: Benchmark components against published DeepSeek-V3 metrics

## 📊 Technical Specifications

| Component | Specification |
|-----------|---------------|
| **Model Size** | 671B total parameters, 37B activated per token |
| **Context Window** | 128K tokens |
| **Training Data** | 14.8T tokens (87% code, 13% natural language) |
| **MoE Experts** | 256 experts with auxiliary-loss-free load balancing |
| **Attention Mechanism** | Multi-head Latent Attention (93.3% KV cache reduction) |
| **Precision** | FP8 mixed precision training |
| **RL Algorithm** | Group Relative Policy Optimization (GRPO) |

## 🎯 Performance Targets

Our implementation aims to match the original DeepSeek-V3 performance:
- **MMLU**: 88.5
- **HumanEval**: 65.2
- **GSM8K**: 89.3
- **Inference Latency**: <100ms/token (128K context)

## 📚 Documentation

The `docs/` directory contains comprehensive technical documentation organized by development phases:

- **Phase 0**: Development environment and infrastructure setup
- **Phase 1-6**: Step-by-step implementation guides for each architectural component
- **Cross-Phase Documentation**: Quality assurance and testing methodologies
- **Technical Reference**: Detailed architectural specifications and design decisions

## 🤝 Contributing

This project follows a systematic, documentation-driven development approach. Contributors should:

1. Review the relevant phase documentation before making changes
2. Follow the test-as-you-develop methodology
3. Validate components using synthetic data before integration
4. Update documentation to reflect implementation changes

## 📄 License

This project is developed for research and educational purposes, implementing the architectural innovations described in the DeepSeek-V3 research paper.

---

**Note**: This is an active research project. The development environment is optimized for experimentation and may require adjustments based on available compute resources and quotas.
