# Security Guidelines for Eva DeepSeek-V3 Project

## 🔒 Security Overview

This document outlines security best practices and guidelines for the Eva DeepSeek-V3 project to prevent accidental exposure of sensitive information.

## ⚠️ Sensitive Information Types

### Never Commit These Items:
- **IP Addresses**: Public IP addresses of development instances
- **Passwords**: Any passwords, including Jupyter Lab passwords
- **API Keys**: GCP service account keys, API tokens
- **SSH Keys**: Private SSH keys (`.pem`, `.key` files)
- **Terraform State**: `.tfstate` files containing infrastructure details
- **Configuration Files**: `terraform.tfvars` with actual values
- **Credentials**: Any authentication credentials or secrets

### GCP-Specific Sensitive Information:
- **Project IDs**: Actual GCP project identifiers
- **Service Account Emails**: `*@*.iam.gserviceaccount.com`
- **Organization IDs**: GCP organization identifiers
- **Billing Account IDs**: GCP billing account numbers
- **Bucket Names**: Cloud Storage bucket names with project identifiers
- **Instance Names**: Specific compute instance names
- **Zone/Region**: When tied to specific account configurations
- **Resource Quotas**: Account-specific quota information

### Examples of What NOT to Commit:
```bash
# ❌ BAD - Never commit these
password = "eva2025"
ip_address = "34.9.120.163"
api_key = "AIzaSyD..."
ssh_key = "-----BEGIN PRIVATE KEY-----"
project_id = "my-actual-project-123"
service_account = "eva-compute@my-project.iam.gserviceaccount.com"
billing_account = "01234A-567890-BCDEF1"
bucket_name = "my-project-eva-dev-data"
instance_name = "eva-dev-dev"
```

## ✅ Security Best Practices

### 1. Use Placeholders in Documentation
Instead of actual values, use placeholders:
```bash
# ✅ GOOD - Use placeholders
password = "[YOUR_PASSWORD]"
ip_address = "[YOUR_INSTANCE_IP]"
api_key = "[YOUR_API_KEY]"
project_id = "[YOUR_PROJECT_ID]"
service_account = "[SERVICE_ACCOUNT_EMAIL]"
billing_account = "[YOUR_BILLING_ACCOUNT]"
bucket_name = "[PROJECT_ID]-eva-[ENVIRONMENT]-data"
instance_name = "[INSTANCE_NAME]"
zone = "[ZONE]"
```

### 2. Environment Variables
Store sensitive values in environment variables:
```bash
export JUPYTER_PASSWORD="your_secure_password"
export GCP_PROJECT_ID="your-project-id"
```

### 3. Local Configuration Files
Keep sensitive configurations in local files that are gitignored:
- `terraform.tfvars` (actual values)
- `.env` files
- `secrets/` directory
- `credentials/` directory

### 4. Git Hooks (Recommended)
Set up pre-commit hooks to scan for secrets:
```bash
# Install pre-commit
pip install pre-commit

# Add to .pre-commit-config.yaml
repos:
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
```

## 🔐 GCP-Specific Security Guidelines

### Project ID Protection
- Never commit actual project IDs in documentation
- Use `[YOUR_PROJECT_ID]` or `[PROJECT_ID]` placeholders
- Store actual project ID in `terraform.tfvars` (gitignored)

### Service Account Security
- Never commit service account JSON keys
- Use IAM roles with least privilege principle
- Rotate service account keys regularly
- Use workload identity when possible

### Billing Information Protection
- Never commit billing account IDs
- Keep cost estimates generic (avoid account-specific quotas)
- Use placeholder values in documentation: `[YOUR_BILLING_ACCOUNT]`

### Infrastructure Naming
- Use parameterized naming in Terraform: `${var.project_id}-eva-${var.environment}`
- Avoid hardcoded instance names in documentation
- Use placeholders: `[INSTANCE_NAME]`, `[ZONE]`, `[REGION]`

### Cloud Storage Security
- Bucket names should use variables: `${var.project_id}-eva-${var.environment}-data`
- Never commit `gs://` URLs with actual bucket names
- Use IAM for bucket access control, not ACLs

## 🛡️ Current Security Measures

### .gitignore Protection
The following patterns are already protected in `.gitignore`:
- `*.tfvars` - Terraform variable files
- `*.tfstate*` - Terraform state files
- `*.json` - Service account keys (with exceptions for configs)
- `*.pem`, `*.key` - Private keys
- `.env*` - Environment files
- `**/secrets/` - Secrets directories
- `**/*secret*`, `**/*password*`, `**/*token*` - Files with sensitive names

### Template Files
Use `.example` files for configuration templates:
- `terraform.tfvars.example` - Template with placeholder values
- `.env.example` - Environment variable template

## 🚨 If Secrets Are Accidentally Committed

### Immediate Actions:
1. **Change the exposed credentials immediately**
2. **Remove from git history** (if recently committed):
   ```bash
   git filter-branch --force --index-filter \
   'git rm --cached --ignore-unmatch path/to/sensitive/file' \
   --prune-empty --tag-name-filter cat -- --all
   ```
3. **Force push** (⚠️ only if safe to do so):
   ```bash
   git push origin --force --all
   ```
4. **Notify team members** to re-clone the repository

### For Public Repositories:
- Consider the exposed information **permanently compromised**
- Rotate all exposed credentials immediately
- Review access logs for unauthorized usage

## 📋 Security Checklist

Before committing, verify:
- [ ] No IP addresses in files
- [ ] No passwords or secrets in files
- [ ] No API keys or tokens in files
- [ ] Configuration files use placeholders
- [ ] Sensitive files are in `.gitignore`
- [ ] Only `.example` template files are committed

### GCP-Specific Checklist:
- [ ] No actual project IDs (use `[YOUR_PROJECT_ID]`)
- [ ] No service account emails (use `[SERVICE_ACCOUNT_EMAIL]`)
- [ ] No billing account IDs (use `[YOUR_BILLING_ACCOUNT]`)
- [ ] No specific instance names (use `[INSTANCE_NAME]`)
- [ ] No actual bucket names (use `[PROJECT_ID]-eva-[ENVIRONMENT]-data`)
- [ ] No organization IDs (use `[YOUR_ORG_ID]`)
- [ ] Terraform uses variables, not hardcoded values
- [ ] All gcloud commands use placeholders in documentation

## 🔍 Regular Security Audits

### Monthly Review:
1. Scan repository for accidentally committed secrets
2. Review `.gitignore` effectiveness
3. Update security guidelines as needed
4. Rotate development credentials

### Tools for Scanning:
```bash
# Scan for secrets
git secrets --scan

# Check for IP addresses
grep -r "([0-9]{1,3}\.){3}[0-9]{1,3}" . --exclude-dir=.git

# Check for common secret patterns
grep -r "password\|secret\|token\|api_key" . --exclude-dir=.git

# GCP-specific scans
grep -r "project.*id.*=" . --exclude-dir=.git
grep -r "@.*\.iam\.gserviceaccount\.com" . --exclude-dir=.git
grep -r "gs://.*-.*-.*" . --exclude-dir=.git
grep -r "billing.*account" . --exclude-dir=.git
grep -r "organization.*id" . --exclude-dir=.git
```

## 📞 Reporting Security Issues

If you discover a security vulnerability or accidentally committed sensitive information:

1. **Do not create a public issue**
2. **Contact the project maintainer directly**
3. **Provide details about the exposure**
4. **Follow the remediation steps above**

## 📚 Additional Resources

- [GitHub Security Best Practices](https://docs.github.com/en/code-security)
- [Git Secrets Tool](https://github.com/awslabs/git-secrets)
- [Pre-commit Hooks](https://pre-commit.com/)
- [Terraform Security Best Practices](https://learn.hashicorp.com/tutorials/terraform/security)

---

**Remember**: Security is everyone's responsibility. When in doubt, ask before committing!
