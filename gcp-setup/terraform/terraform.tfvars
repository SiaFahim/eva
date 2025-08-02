# Eva DeepSeek-V3 Development Environment Configuration
# Copy this file to terraform.tfvars and customize for your setup

# GCP Project Configuration
project_id = "eva-deepseek-dev"
region     = "us-central1"  # Choose region with GPU availability
zone       = "us-central1-a"

# Environment
environment = "dev"

# Instance Configuration
# Options:
# - a2-highgpu-1g: 1x A100 (40GB), 12 vCPUs, 85GB RAM - $3.67/hour
# - a2-ultragpu-1g: 1x A100 (80GB), 12 vCPUs, 85GB RAM - $4.89/hour
# - n1-standard-8 + nvidia-tesla-t4: 1x T4 (16GB), 8 vCPUs, 30GB RAM - $0.95/hour
# - n1-standard-4: CPU-only, 4 vCPUs, 15GB RAM - $0.19/hour (for testing)
instance_type = "n1-standard-4"

# GPU Configuration (disabled due to quota limits)
gpu_type  = "nvidia-tesla-t4"
gpu_count = 0

# Storage
disk_size = 100  # GB for development workspace (reduced due to quota)

# Cost Optimization
preemptible = true  # 60-91% cost savings, but instances can be terminated

# Security - IMPORTANT: Replace with your actual IP address
allowed_ips = [
  "187.190.190.250/32"  # Your current public IP
  # "203.0.113.0/24"    # Example: office network
]

# Alternative Budget Configuration (uncomment to use)
# instance_type = "n1-standard-8"
# gpu_type      = "nvidia-tesla-t4"
# gpu_count     = 1
