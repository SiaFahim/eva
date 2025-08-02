# Eva DeepSeek-V3 Development Environment - GCP Infrastructure
# Terraform configuration for ML development environment

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# Variables
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP Zone"
  type        = string
  default     = "us-central1-a"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "instance_type" {
  description = "Instance type for development VM"
  type        = string
  default     = "a2-highgpu-1g"  # 1x A100, 12 vCPUs, 85GB RAM
}

variable "gpu_type" {
  description = "GPU type"
  type        = string
  default     = "nvidia-tesla-a100"
}

variable "gpu_count" {
  description = "Number of GPUs"
  type        = number
  default     = 1
}

variable "disk_size" {
  description = "Boot disk size in GB"
  type        = number
  default     = 500
}

variable "preemptible" {
  description = "Use preemptible instances for cost savings"
  type        = bool
  default     = true
}

variable "allowed_ips" {
  description = "List of IP addresses allowed to SSH"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # Replace with your IP
}

# Provider configuration
provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# Data sources
data "google_compute_image" "ubuntu" {
  family  = "ubuntu-2204-lts"
  project = "ubuntu-os-cloud"
}

# VPC Network
resource "google_compute_network" "eva_network" {
  name                    = "eva-${var.environment}-network"
  auto_create_subnetworks = false
  description             = "Eva DeepSeek-V3 development network"
}

# Subnet
resource "google_compute_subnetwork" "eva_subnet" {
  name          = "eva-${var.environment}-subnet"
  ip_cidr_range = "10.0.0.0/24"
  region        = var.region
  network       = google_compute_network.eva_network.id
  description   = "Eva development subnet"
}

# Firewall rules
resource "google_compute_firewall" "eva_ssh" {
  name    = "eva-${var.environment}-allow-ssh"
  network = google_compute_network.eva_network.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = var.allowed_ips
  target_tags   = ["eva-dev"]
  description   = "Allow SSH access to Eva development instances"
}

resource "google_compute_firewall" "eva_internal" {
  name    = "eva-${var.environment}-allow-internal"
  network = google_compute_network.eva_network.name

  allow {
    protocol = "tcp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "udp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "icmp"
  }

  source_ranges = ["10.0.0.0/24"]
  target_tags   = ["eva-dev"]
  description   = "Allow internal communication"
}

resource "google_compute_firewall" "eva_jupyter" {
  name    = "eva-${var.environment}-allow-jupyter"
  network = google_compute_network.eva_network.name

  allow {
    protocol = "tcp"
    ports    = ["8888", "6006", "8080"]  # Jupyter, TensorBoard, custom apps
  }

  source_ranges = var.allowed_ips
  target_tags   = ["eva-dev"]
  description   = "Allow access to development services"
}

# Service Account
resource "google_service_account" "eva_compute" {
  account_id   = "eva-${var.environment}-compute"
  display_name = "Eva Development Compute Service Account"
  description  = "Service account for Eva development instances"
}

# IAM bindings for service account
resource "google_project_iam_member" "eva_compute_storage" {
  project = var.project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:${google_service_account.eva_compute.email}"
}

resource "google_project_iam_member" "eva_compute_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.eva_compute.email}"
}

resource "google_project_iam_member" "eva_compute_monitoring" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.eva_compute.email}"
}

# Cloud Storage bucket for datasets and models
resource "google_storage_bucket" "eva_data" {
  name          = "${var.project_id}-eva-${var.environment}-data"
  location      = var.region
  force_destroy = false

  uniform_bucket_level_access = true
  
  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }

  labels = {
    environment = var.environment
    project     = "eva"
    purpose     = "ml-data"
  }
}

# Persistent disk for development
resource "google_compute_disk" "eva_dev_disk" {
  name  = "eva-${var.environment}-dev-disk"
  type  = "pd-ssd"
  zone  = var.zone
  size  = var.disk_size
  
  labels = {
    environment = var.environment
    project     = "eva"
  }
}

# Development instance
resource "google_compute_instance" "eva_dev" {
  name         = "eva-${var.environment}-dev"
  machine_type = var.instance_type
  zone         = var.zone

  tags = ["eva-dev"]

  boot_disk {
    initialize_params {
      image = data.google_compute_image.ubuntu.self_link
      size  = 100
      type  = "pd-ssd"
    }
  }

  attached_disk {
    source      = google_compute_disk.eva_dev_disk.id
    device_name = "eva-dev-disk"
  }

  dynamic "guest_accelerator" {
    for_each = var.gpu_count > 0 ? [1] : []
    content {
      type  = var.gpu_type
      count = var.gpu_count
    }
  }

  network_interface {
    network    = google_compute_network.eva_network.id
    subnetwork = google_compute_subnetwork.eva_subnet.id
    
    access_config {
      # Ephemeral public IP
    }
  }

  service_account {
    email  = google_service_account.eva_compute.email
    scopes = ["cloud-platform"]
  }

  scheduling {
    preemptible       = var.preemptible
    automatic_restart = !var.preemptible
    on_host_maintenance = (var.preemptible || var.gpu_count > 0) ? "TERMINATE" : "MIGRATE"
  }

  metadata = {
    ssh-keys = "eva:${file("~/.ssh/id_rsa.pub")}"
    startup-script = file("${path.module}/../scripts/enhanced-startup-script.sh")
  }

  metadata_startup_script = file("${path.module}/../scripts/enhanced-startup-script.sh")

  labels = {
    environment = var.environment
    project     = "eva"
    purpose     = "ml-development"
  }
}

# Outputs
output "instance_name" {
  description = "Name of the created instance"
  value       = google_compute_instance.eva_dev.name
}

output "instance_external_ip" {
  description = "External IP address of the instance"
  value       = google_compute_instance.eva_dev.network_interface[0].access_config[0].nat_ip
}

output "instance_internal_ip" {
  description = "Internal IP address of the instance"
  value       = google_compute_instance.eva_dev.network_interface[0].network_ip
}

output "ssh_command" {
  description = "SSH command to connect to the instance"
  value       = "gcloud compute ssh eva@${google_compute_instance.eva_dev.name} --zone=${var.zone}"
}

output "storage_bucket" {
  description = "Name of the created storage bucket"
  value       = google_storage_bucket.eva_data.name
}
