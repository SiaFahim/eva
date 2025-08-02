# Eva DeepSeek-V3 Terraform Infrastructure

This directory contains Terraform configuration for deploying the Eva DeepSeek-V3 development environment on Google Cloud Platform.

## 🚀 Quick Start

### Prerequisites
- [Terraform](https://www.terraform.io/downloads.html) >= 1.0
- [Google Cloud CLI](https://cloud.google.com/sdk/docs/install)
- GCP project with billing enabled

### Setup Steps

1. **Authenticate with GCP:**
   ```bash
   gcloud auth login
   gcloud auth application-default login
   gcloud config set project YOUR_PROJECT_ID
   ```

2. **Configure Terraform:**
   ```bash
   # Copy and customize the configuration
   cp terraform.tfvars.example terraform.tfvars
   
   # Edit terraform.tfvars with your settings:
   # - project_id: Your GCP project ID
   # - allowed_ips: Your public IP address
   # - instance_type: Choose based on your needs and quotas
   ```

3. **Deploy Infrastructure:**
   ```bash
   terraform init
   terraform plan
   terraform apply
   ```

4. **Connect to Instance:**
   ```bash
   # Use the SSH command from terraform output
   gcloud compute ssh eva@INSTANCE_NAME --zone=ZONE
   ```

## 📁 Files

- `main.tf` - Main Terraform configuration
- `terraform.tfvars.example` - Configuration template
- `terraform.tfvars` - Your actual configuration (not in git)
- `.terraform/` - Terraform working directory (not in git)
- `*.tfstate` - Terraform state files (not in git)

## 🔒 Security Notes

- **Never commit** `terraform.tfvars` - contains sensitive data
- **Never commit** `.terraform/` directory - contains provider binaries
- **Never commit** `*.tfstate` files - may contain sensitive data
- Always use least-privilege service accounts
- Restrict SSH access to your IP address only

## 💰 Cost Management

- Use preemptible instances for development (60-91% savings)
- Monitor usage with `gcloud billing budgets list`
- Stop instances when not in use: `gcloud compute instances stop INSTANCE_NAME`

## 🛠️ Troubleshooting

### Common Issues

1. **Quota Exceeded:**
   - Request quota increases in GCP Console
   - Use smaller instance types or fewer GPUs

2. **Authentication Errors:**
   ```bash
   gcloud auth application-default login
   ```

3. **Provider Download Issues:**
   ```bash
   terraform init -upgrade
   ```

### Cleanup

To destroy all resources:
```bash
terraform destroy
```

**⚠️ Warning:** This will permanently delete all resources!

## 📚 Documentation

- [Terraform GCP Provider](https://registry.terraform.io/providers/hashicorp/google/latest/docs)
- [GCP Compute Engine](https://cloud.google.com/compute/docs)
- [Main Setup Guide](../docs/SETUP_GUIDE.md)
