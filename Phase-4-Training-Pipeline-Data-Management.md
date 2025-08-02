# Phase 4: Training Pipeline & Data Management
## DeepSeek-V3 TensorFlow Implementation

### Overview

This document provides comprehensive engineering guidance for implementing the complete training pipeline for DeepSeek-V3, including pre-training data pipeline for 14.8T tokens, training orchestration and monitoring systems, distributed checkpointing for 671B parameters, and progressive context extension from 4K→32K→128K tokens.

---

## 1. Pre-training Data Pipeline (14.8T Tokens)

### 1.1 Data Composition and Sources

**DeepSeek-V3 Data Distribution:**
- **Code:** 87% (12.876T tokens) - Programming languages and repositories
- **Natural Language:** 13% (1.924T tokens) - Multilingual text data
- **Quality Control:** Extensive filtering, deduplication, and validation

```python
# data/data_pipeline.py
import tensorflow as tf
from typing import Dict, List, Optional, Tuple, Iterator
import json
import glob
import hashlib

class DeepSeekDataPipeline:
    """
    Comprehensive data pipeline for DeepSeek-V3 pre-training
    Handles 14.8T tokens with 87% code and 13% natural language
    """
    
    def __init__(self,
                 data_root: str,
                 tokenizer_path: str,
                 sequence_length: int = 4096,
                 batch_size: int = 32,
                 num_parallel_calls: int = tf.data.AUTOTUNE):
        self.data_root = data_root
        self.tokenizer_path = tokenizer_path
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_parallel_calls = num_parallel_calls
        
        # Load tokenizer
        self.tokenizer = self._load_tokenizer()
        
        # Data composition ratios
        self.data_composition = {
            'code': 0.87,
            'natural_language': 0.13
        }
        
        # Quality filters
        self.quality_filters = {
            'min_length': 50,
            'max_length': 100000,
            'min_unique_tokens': 10,
            'max_repetition_ratio': 0.3
        }
        
        # Deduplication cache
        self.seen_hashes = set()
    
    def _load_tokenizer(self):
        """Load tokenizer for text processing"""
        # Placeholder for actual tokenizer loading
        # In practice, would load DeepSeek's custom tokenizer
        return None
    
    def create_training_dataset(self, 
                               data_sources: Dict[str, List[str]],
                               shuffle_buffer_size: int = 10000,
                               prefetch_buffer_size: int = tf.data.AUTOTUNE) -> tf.data.Dataset:
        """
        Create training dataset from multiple data sources
        
        Args:
            data_sources: Dictionary mapping data types to file paths
            shuffle_buffer_size: Buffer size for shuffling
            prefetch_buffer_size: Buffer size for prefetching
            
        Returns:
            dataset: Training dataset ready for consumption
        """
        # Create datasets for each data type
        datasets = []
        
        for data_type, file_paths in data_sources.items():
            if data_type in self.data_composition:
                # Create dataset for this data type
                type_dataset = self._create_dataset_for_type(data_type, file_paths)
                
                # Sample according to composition ratio
                sample_ratio = self.data_composition[data_type]
                sampled_dataset = type_dataset.filter(
                    lambda x: tf.random.uniform([]) < sample_ratio
                )
                
                datasets.append(sampled_dataset)
        
        # Combine datasets
        combined_dataset = tf.data.Dataset.from_tensor_slices([])
        for dataset in datasets:
            combined_dataset = combined_dataset.concatenate(dataset)
        
        # Apply processing pipeline
        processed_dataset = (combined_dataset
                           .shuffle(shuffle_buffer_size)
                           .map(self._tokenize_and_format, 
                               num_parallel_calls=self.num_parallel_calls)
                           .filter(self._quality_filter)
                           .map(self._create_training_example)
                           .batch(self.batch_size, drop_remainder=True)
                           .prefetch(prefetch_buffer_size))
        
        return processed_dataset
    
    def _create_dataset_for_type(self, data_type: str, file_paths: List[str]) -> tf.data.Dataset:
        """Create dataset for specific data type"""
        if data_type == 'code':
            return self._create_code_dataset(file_paths)
        elif data_type == 'natural_language':
            return self._create_natural_language_dataset(file_paths)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    def _create_code_dataset(self, file_paths: List[str]) -> tf.data.Dataset:
        """Create dataset from code files"""
        # Create dataset from code files
        dataset = tf.data.Dataset.from_tensor_slices(file_paths)
        
        # Read and process code files
        dataset = dataset.map(
            lambda path: tf.py_function(
                self._read_code_file, [path], tf.string
            ),
            num_parallel_calls=self.num_parallel_calls
        )
        
        # Filter out empty files
        dataset = dataset.filter(lambda x: tf.strings.length(x) > 0)
        
        return dataset
    
    def _create_natural_language_dataset(self, file_paths: List[str]) -> tf.data.Dataset:
        """Create dataset from natural language files"""
        # Create dataset from text files
        dataset = tf.data.Dataset.from_tensor_slices(file_paths)
        
        # Read and process text files
        dataset = dataset.map(
            lambda path: tf.py_function(
                self._read_text_file, [path], tf.string
            ),
            num_parallel_calls=self.num_parallel_calls
        )
        
        # Filter out empty files
        dataset = dataset.filter(lambda x: tf.strings.length(x) > 0)
        
        return dataset
    
    def _read_code_file(self, file_path: tf.Tensor) -> tf.Tensor:
        """Read and preprocess code file"""
        file_path_str = file_path.numpy().decode('utf-8')
        
        try:
            with open(file_path_str, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Apply code-specific preprocessing
            content = self._preprocess_code(content)
            
            return tf.constant(content, dtype=tf.string)
        except Exception as e:
            return tf.constant('', dtype=tf.string)
    
    def _read_text_file(self, file_path: tf.Tensor) -> tf.Tensor:
        """Read and preprocess text file"""
        file_path_str = file_path.numpy().decode('utf-8')
        
        try:
            with open(file_path_str, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Apply text-specific preprocessing
            content = self._preprocess_text(content)
            
            return tf.constant(content, dtype=tf.string)
        except Exception as e:
            return tf.constant('', dtype=tf.string)
    
    def _preprocess_code(self, code_content: str) -> str:
        """Preprocess code content"""
        # Remove excessive whitespace
        lines = code_content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove trailing whitespace
            line = line.rstrip()
            
            # Skip empty lines at the beginning/end
            if line or cleaned_lines:
                cleaned_lines.append(line)
        
        # Remove trailing empty lines
        while cleaned_lines and not cleaned_lines[-1]:
            cleaned_lines.pop()
        
        return '\n'.join(cleaned_lines)
    
    def _preprocess_text(self, text_content: str) -> str:
        """Preprocess natural language text"""
        # Basic text cleaning
        # Remove excessive whitespace
        text_content = ' '.join(text_content.split())
        
        # Remove control characters
        text_content = ''.join(char for char in text_content if ord(char) >= 32 or char in '\n\t')
        
        return text_content
    
    def _tokenize_and_format(self, text: tf.Tensor) -> tf.Tensor:
        """Tokenize text and format for training"""
        # Placeholder for actual tokenization
        # In practice, would use DeepSeek's tokenizer
        
        # For now, simulate tokenization by converting to bytes
        text_bytes = tf.strings.bytes_split(text)
        
        # Convert to token IDs (simplified)
        token_ids = tf.cast(tf.strings.to_number(text_bytes, out_type=tf.int32), tf.int32)
        
        # Pad or truncate to sequence length
        token_ids = token_ids[:self.sequence_length]
        padded_tokens = tf.pad(
            token_ids,
            [[0, self.sequence_length - tf.shape(token_ids)[0]]],
            constant_values=0
        )
        
        return padded_tokens
    
    def _quality_filter(self, tokens: tf.Tensor) -> tf.bool:
        """Apply quality filters to tokenized content"""
        # Check minimum length
        non_zero_tokens = tf.reduce_sum(tf.cast(tokens > 0, tf.int32))
        if non_zero_tokens < self.quality_filters['min_length']:
            return False
        
        # Check for excessive repetition
        unique_tokens = tf.size(tf.unique(tokens)[0])
        repetition_ratio = 1.0 - tf.cast(unique_tokens, tf.float32) / tf.cast(non_zero_tokens, tf.float32)
        
        if repetition_ratio > self.quality_filters['max_repetition_ratio']:
            return False
        
        return True
    
    def _create_training_example(self, tokens: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Create training example with input and target tokens"""
        return {
            'input_ids': tokens[:-1],  # All tokens except last
            'labels': tokens[1:]       # All tokens except first (shifted)
        }
    
    def get_dataset_statistics(self, dataset: tf.data.Dataset, num_samples: int = 1000) -> Dict:
        """Get statistics about the dataset"""
        token_lengths = []
        unique_tokens_counts = []
        
        for i, batch in enumerate(dataset.take(num_samples // self.batch_size)):
            for example in batch['input_ids']:
                non_zero_tokens = tf.reduce_sum(tf.cast(example > 0, tf.int32))
                unique_tokens = tf.size(tf.unique(example)[0])
                
                token_lengths.append(int(non_zero_tokens))
                unique_tokens_counts.append(int(unique_tokens))
        
        return {
            'num_samples_analyzed': len(token_lengths),
            'avg_token_length': sum(token_lengths) / len(token_lengths),
            'avg_unique_tokens': sum(unique_tokens_counts) / len(unique_tokens_counts),
            'min_token_length': min(token_lengths),
            'max_token_length': max(token_lengths)
        }
```

### 1.2 Data Quality Control and Filtering

```python
class DataQualityController:
    """
    Advanced data quality control for large-scale training
    """
    
    def __init__(self):
        self.deduplication_cache = {}
        self.quality_metrics = {
            'total_documents': 0,
            'filtered_documents': 0,
            'duplicate_documents': 0,
            'quality_passed': 0
        }
    
    def apply_deduplication(self, text: str) -> bool:
        """Apply deduplication using content hashing"""
        # Create hash of content
        content_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        if content_hash in self.deduplication_cache:
            self.quality_metrics['duplicate_documents'] += 1
            return False
        
        self.deduplication_cache[content_hash] = True
        return True
    
    def apply_content_filters(self, text: str, data_type: str) -> bool:
        """Apply content-specific quality filters"""
        if data_type == 'code':
            return self._filter_code_content(text)
        elif data_type == 'natural_language':
            return self._filter_text_content(text)
        return True
    
    def _filter_code_content(self, code: str) -> bool:
        """Filter code content for quality"""
        lines = code.split('\n')
        
        # Check for minimum meaningful content
        non_empty_lines = [line for line in lines if line.strip()]
        if len(non_empty_lines) < 5:
            return False
        
        # Check for excessive comments (might indicate low-quality code)
        comment_lines = sum(1 for line in lines if line.strip().startswith(('#', '//', '/*', '*')))
        if comment_lines / len(non_empty_lines) > 0.7:
            return False
        
        # Check for reasonable line length distribution
        avg_line_length = sum(len(line) for line in non_empty_lines) / len(non_empty_lines)
        if avg_line_length < 10 or avg_line_length > 200:
            return False
        
        return True
    
    def _filter_text_content(self, text: str) -> bool:
        """Filter natural language content for quality"""
        words = text.split()
        
        # Check minimum word count
        if len(words) < 20:
            return False
        
        # Check for reasonable word length distribution
        avg_word_length = sum(len(word) for word in words) / len(words)
        if avg_word_length < 3 or avg_word_length > 15:
            return False
        
        # Check for excessive repetition
        unique_words = set(words)
        if len(unique_words) / len(words) < 0.3:
            return False
        
        return True
    
    def get_quality_report(self) -> Dict:
        """Get data quality report"""
        total = self.quality_metrics['total_documents']
        if total == 0:
            return {'message': 'No documents processed'}
        
        return {
            'total_documents': total,
            'quality_pass_rate': self.quality_metrics['quality_passed'] / total,
            'deduplication_rate': self.quality_metrics['duplicate_documents'] / total,
            'filter_rate': self.quality_metrics['filtered_documents'] / total
        }
```

---

## 2. Training Orchestration & Monitoring

### 2.1 Training Orchestrator

```python
# training/orchestrator.py
import tensorflow as tf
from typing import Dict, List, Optional, Callable
import time
import json
import logging

class DeepSeekTrainingOrchestrator:
    """
    Comprehensive training orchestrator for DeepSeek-V3
    Manages training lifecycle, monitoring, and coordination
    """
    
    def __init__(self,
                 model: tf.keras.Model,
                 optimizer: tf.keras.optimizers.Optimizer,
                 strategy: tf.distribute.Strategy,
                 config: Dict):
        self.model = model
        self.optimizer = optimizer
        self.strategy = strategy
        self.config = config
        
        # Training state
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.epoch = tf.Variable(0, trainable=False, name='epoch')
        self.best_loss = tf.Variable(float('inf'), trainable=False, name='best_loss')
        
        # Monitoring
        self.metrics_history = {
            'train_loss': [],
            'learning_rate': [],
            'expert_utilization': [],
            'memory_usage': [],
            'throughput': []
        }
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Callbacks
        self.callbacks = []
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger('deepseek_training')
        logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler('training.log')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def train(self,
              train_dataset: tf.data.Dataset,
              num_epochs: int,
              steps_per_epoch: int,
              validation_dataset: Optional[tf.data.Dataset] = None,
              validation_steps: Optional[int] = None):
        """
        Main training loop with comprehensive monitoring
        
        Args:
            train_dataset: Training dataset
            num_epochs: Number of training epochs
            steps_per_epoch: Steps per epoch
            validation_dataset: Optional validation dataset
            validation_steps: Steps for validation
        """
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Steps per epoch: {steps_per_epoch}")
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Training phase
            train_metrics = self._train_epoch(train_dataset, steps_per_epoch)
            
            # Validation phase
            if validation_dataset is not None:
                val_metrics = self._validate_epoch(validation_dataset, validation_steps)
                self.logger.info(f"Validation Loss: {val_metrics['loss']:.4f}")
            
            # Update epoch counter
            self.epoch.assign_add(1)
            
            # Log epoch summary
            epoch_time = time.time() - epoch_start_time
            self.logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
            self.logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
            
            # Save checkpoint if best model
            if train_metrics['loss'] < self.best_loss:
                self.best_loss.assign(train_metrics['loss'])
                self._save_checkpoint('best_model')
            
            # Regular checkpoint
            if (epoch + 1) % self.config.get('checkpoint_every_n_epochs', 5) == 0:
                self._save_checkpoint(f'epoch_{epoch + 1}')
            
            # Execute callbacks
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, train_metrics)
    
    def _train_epoch(self, dataset: tf.data.Dataset, steps_per_epoch: int) -> Dict:
        """Train for one epoch"""
        epoch_metrics = {
            'loss': tf.keras.metrics.Mean(),
            'throughput': tf.keras.metrics.Mean(),
            'expert_utilization_variance': tf.keras.metrics.Mean()
        }
        
        step_start_time = time.time()
        
        for step, batch in enumerate(dataset.take(steps_per_epoch)):
            # Training step
            step_metrics = self._train_step(batch)
            
            # Update metrics
            epoch_metrics['loss'].update_state(step_metrics['loss'])
            
            # Calculate throughput
            step_time = time.time() - step_start_time
            tokens_per_second = (self.config['batch_size'] * self.config['sequence_length']) / step_time
            epoch_metrics['throughput'].update_state(tokens_per_second)
            
            # Update expert utilization metrics
            if 'expert_utilization_variance' in step_metrics:
                epoch_metrics['expert_utilization_variance'].update_state(
                    step_metrics['expert_utilization_variance']
                )
            
            # Log progress
            if step % self.config.get('log_every_n_steps', 100) == 0:
                self.logger.info(
                    f"Step {step}: Loss = {epoch_metrics['loss'].result():.4f}, "
                    f"Throughput = {epoch_metrics['throughput'].result():.0f} tokens/sec"
                )
            
            # Update global step
            self.global_step.assign_add(1)
            step_start_time = time.time()
        
        # Return epoch metrics
        return {key: metric.result().numpy() for key, metric in epoch_metrics.items()}
    
    @tf.function
    def _train_step(self, batch: Dict[str, tf.Tensor]) -> Dict:
        """Single training step"""
        with tf.GradientTape() as tape:
            # Forward pass
            outputs = self.model(batch['input_ids'], training=True)
            
            # Compute loss
            loss = tf.keras.losses.sparse_categorical_crossentropy(
                batch['labels'],
                outputs,
                from_logits=True
            )
            loss = tf.reduce_mean(loss)
            
            # Scale loss for mixed precision
            if self.config.get('use_mixed_precision', False):
                scaled_loss = self.optimizer.get_scaled_loss(loss)
            else:
                scaled_loss = loss
        
        # Compute gradients
        if self.config.get('use_mixed_precision', False):
            scaled_gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Collect metrics
        metrics = {
            'loss': loss,
            'learning_rate': self.optimizer.learning_rate,
            'gradient_norm': tf.linalg.global_norm(gradients)
        }
        
        # Add expert utilization metrics if available
        if hasattr(self.model, 'get_expert_utilization_stats'):
            expert_stats = self.model.get_expert_utilization_stats()
            metrics['expert_utilization_variance'] = expert_stats.get('utilization_variance', 0.0)
        
        return metrics
    
    def _validate_epoch(self, dataset: tf.data.Dataset, validation_steps: int) -> Dict:
        """Validate for one epoch"""
        val_loss = tf.keras.metrics.Mean()
        
        for step, batch in enumerate(dataset.take(validation_steps)):
            # Validation step
            outputs = self.model(batch['input_ids'], training=False)
            loss = tf.keras.losses.sparse_categorical_crossentropy(
                batch['labels'],
                outputs,
                from_logits=True
            )
            val_loss.update_state(tf.reduce_mean(loss))
        
        return {'loss': val_loss.result().numpy()}
    
    def _save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint"""
        checkpoint_path = f"checkpoints/{checkpoint_name}"
        
        # Save model weights
        self.model.save_weights(f"{checkpoint_path}/model_weights")
        
        # Save optimizer state
        with open(f"{checkpoint_path}/optimizer_state.json", 'w') as f:
            json.dump({
                'learning_rate': float(self.optimizer.learning_rate),
                'global_step': int(self.global_step),
                'epoch': int(self.epoch)
            }, f)
        
        # Save training metrics
        with open(f"{checkpoint_path}/metrics.json", 'w') as f:
            json.dump(self.metrics_history, f)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def add_callback(self, callback):
        """Add training callback"""
        self.callbacks.append(callback)
```

### 2.2 Advanced Monitoring System

```python
class DeepSeekMonitor:
    """
    Advanced monitoring system for DeepSeek-V3 training
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.metrics_buffer = {}
        self.alert_thresholds = {
            'loss_spike': 2.0,  # Alert if loss increases by 2x
            'memory_usage': 0.95,  # Alert if memory usage > 95%
            'expert_imbalance': 0.3,  # Alert if expert utilization CV > 0.3
            'throughput_drop': 0.5  # Alert if throughput drops by 50%
        }
        
        # Initialize monitoring components
        self.system_monitor = SystemResourceMonitor()
        self.model_monitor = ModelPerformanceMonitor()
        self.expert_monitor = ExpertUtilizationMonitor()
    
    def collect_metrics(self, step: int, model: tf.keras.Model, metrics: Dict):
        """Collect comprehensive metrics"""
        timestamp = time.time()
        
        # System metrics
        system_metrics = self.system_monitor.get_metrics()
        
        # Model metrics
        model_metrics = self.model_monitor.get_metrics(model)
        
        # Expert metrics
        expert_metrics = self.expert_monitor.get_metrics(model)
        
        # Combine all metrics
        combined_metrics = {
            'timestamp': timestamp,
            'step': step,
            'training': metrics,
            'system': system_metrics,
            'model': model_metrics,
            'experts': expert_metrics
        }
        
        # Store metrics
        self.metrics_buffer[step] = combined_metrics
        
        # Check for alerts
        self._check_alerts(combined_metrics)
        
        return combined_metrics
    
    def _check_alerts(self, metrics: Dict):
        """Check for alert conditions"""
        # Loss spike detection
        if len(self.metrics_buffer) > 10:
            recent_losses = [m['training']['loss'] for m in list(self.metrics_buffer.values())[-10:]]
            if recent_losses[-1] > min(recent_losses[:-1]) * self.alert_thresholds['loss_spike']:
                self._send_alert('LOSS_SPIKE', f"Loss spiked to {recent_losses[-1]:.4f}")
        
        # Memory usage alert
        if metrics['system']['memory_usage'] > self.alert_thresholds['memory_usage']:
            self._send_alert('HIGH_MEMORY', f"Memory usage: {metrics['system']['memory_usage']:.1%}")
        
        # Expert imbalance alert
        if metrics['experts']['utilization_cv'] > self.alert_thresholds['expert_imbalance']:
            self._send_alert('EXPERT_IMBALANCE', f"Expert CV: {metrics['experts']['utilization_cv']:.3f}")
    
    def _send_alert(self, alert_type: str, message: str):
        """Send alert notification"""
        alert_message = f"[{alert_type}] {message}"
        print(f"ALERT: {alert_message}")
        
        # In production, would send to monitoring system
        # (e.g., Slack, email, PagerDuty, etc.)

class SystemResourceMonitor:
    """Monitor system resources"""
    
    def get_metrics(self) -> Dict:
        """Get system resource metrics"""
        try:
            import psutil
            import GPUtil
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # GPU metrics
            gpus = GPUtil.getGPUs()
            gpu_metrics = []
            for gpu in gpus:
                gpu_metrics.append({
                    'id': gpu.id,
                    'utilization': gpu.load,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'temperature': gpu.temperature
                })
            
            return {
                'cpu_percent': cpu_percent,
                'memory_usage': memory.percent / 100.0,
                'memory_available_gb': memory.available / (1024**3),
                'gpu_metrics': gpu_metrics
            }
        except ImportError:
            return {'error': 'System monitoring libraries not available'}

class ModelPerformanceMonitor:
    """Monitor model performance metrics"""
    
    def get_metrics(self, model: tf.keras.Model) -> Dict:
        """Get model performance metrics"""
        metrics = {}
        
        # Model size metrics
        total_params = sum([tf.size(var) for var in model.trainable_variables])
        metrics['total_parameters'] = int(total_params)
        
        # Memory usage (approximate)
        param_memory = total_params * 4  # Assume 4 bytes per parameter
        metrics['parameter_memory_gb'] = float(param_memory) / (1024**3)
        
        return metrics

class ExpertUtilizationMonitor:
    """Monitor expert utilization in MoE layers"""
    
    def get_metrics(self, model: tf.keras.Model) -> Dict:
        """Get expert utilization metrics"""
        if not hasattr(model, 'get_expert_utilization_stats'):
            return {'message': 'No expert utilization available'}
        
        expert_stats = model.get_expert_utilization_stats()
        
        # Calculate coefficient of variation
        utilization = expert_stats.get('utilization', [])
        if len(utilization) > 0:
            mean_util = sum(utilization) / len(utilization)
            variance = sum((u - mean_util) ** 2 for u in utilization) / len(utilization)
            cv = (variance ** 0.5) / mean_util if mean_util > 0 else 0
        else:
            cv = 0
        
        return {
            'utilization_mean': mean_util if len(utilization) > 0 else 0,
            'utilization_cv': cv,
            'utilization_min': min(utilization) if utilization else 0,
            'utilization_max': max(utilization) if utilization else 0,
            'num_experts': len(utilization)
        }
```

---

## 3. Distributed Checkpointing for 671B Parameters

### 3.1 Distributed Checkpoint Manager

```python
# checkpointing/distributed_checkpoint.py
import tensorflow as tf
from typing import Dict, List, Optional
import os
import json
import time
import threading

class DistributedCheckpointManager:
    """
    Distributed checkpoint manager for 671B parameter models
    Handles sharded checkpointing across multiple nodes
    """
    
    def __init__(self,
                 checkpoint_dir: str,
                 max_to_keep: int = 5,
                 save_interval_steps: int = 1000,
                 async_save: bool = True):
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep
        self.save_interval_steps = save_interval_steps
        self.async_save = async_save
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Checkpoint metadata
        self.checkpoint_metadata = {}
        self.checkpoint_counter = 0
        
        # Async saving
        if async_save:
            self.save_queue = []
            self.save_thread = threading.Thread(target=self._async_save_worker, daemon=True)
            self.save_thread.start()
    
    def save_checkpoint(self,
                       model: tf.keras.Model,
                       optimizer: tf.keras.optimizers.Optimizer,
                       step: int,
                       metrics: Dict,
                       blocking: bool = False) -> str:
        """
        Save distributed checkpoint
        
        Args:
            model: Model to checkpoint
            optimizer: Optimizer to checkpoint
            step: Current training step
            metrics: Training metrics
            blocking: Whether to block until save completes
            
        Returns:
            checkpoint_path: Path to saved checkpoint
        """
        checkpoint_name = f"checkpoint_{step}"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        # Create checkpoint data
        checkpoint_data = {
            'model': model,
            'optimizer': optimizer,
            'step': step,
            'metrics': metrics,
            'timestamp': time.time(),
            'path': checkpoint_path
        }
        
        if self.async_save and not blocking:
            # Add to async save queue
            self.save_queue.append(checkpoint_data)
        else:
            # Save synchronously
            self._save_checkpoint_data(checkpoint_data)
        
        return checkpoint_path
    
    def _save_checkpoint_data(self, checkpoint_data: Dict):
        """Save checkpoint data to disk"""
        checkpoint_path = checkpoint_data['path']
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save model weights in shards
        self._save_model_shards(checkpoint_data['model'], checkpoint_path)
        
        # Save optimizer state
        self._save_optimizer_state(checkpoint_data['optimizer'], checkpoint_path)
        
        # Save metadata
        metadata = {
            'step': checkpoint_data['step'],
            'timestamp': checkpoint_data['timestamp'],
            'metrics': checkpoint_data['metrics']
        }
        
        with open(os.path.join(checkpoint_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update checkpoint registry
        self._update_checkpoint_registry(checkpoint_path, metadata)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_model_shards(self, model: tf.keras.Model, checkpoint_path: str):
        """Save model weights in shards for distributed loading"""
        weights_dir = os.path.join(checkpoint_path, 'model_weights')
        os.makedirs(weights_dir, exist_ok=True)
        
        # Group variables by layer/component
        variable_groups = self._group_variables_by_component(model.trainable_variables)
        
        shard_info = {}
        
        for group_name, variables in variable_groups.items():
            shard_path = os.path.join(weights_dir, f'{group_name}.weights')
            
            # Save variables in this group
            checkpoint = tf.train.Checkpoint(**{f'var_{i}': var for i, var in enumerate(variables)})
            checkpoint.write(shard_path)
            
            # Record shard information
            shard_info[group_name] = {
                'path': shard_path,
                'variables': [var.name for var in variables],
                'shapes': [var.shape.as_list() for var in variables],
                'dtypes': [var.dtype.name for var in variables]
            }
        
        # Save shard index
        with open(os.path.join(weights_dir, 'shard_index.json'), 'w') as f:
            json.dump(shard_info, f, indent=2)
    
    def _group_variables_by_component(self, variables: List[tf.Variable]) -> Dict[str, List[tf.Variable]]:
        """Group variables by model component for efficient sharding"""
        groups = {}
        
        for var in variables:
            # Extract component name from variable name
            var_name = var.name
            
            if 'attention' in var_name:
                component = 'attention'
            elif 'moe' in var_name or 'expert' in var_name:
                component = 'moe'
            elif 'embedding' in var_name:
                component = 'embedding'
            elif 'layer_norm' in var_name:
                component = 'layer_norm'
            else:
                component = 'other'
            
            if component not in groups:
                groups[component] = []
            groups[component].append(var)
        
        return groups
    
    def _save_optimizer_state(self, optimizer: tf.keras.optimizers.Optimizer, checkpoint_path: str):
        """Save optimizer state"""
        optimizer_dir = os.path.join(checkpoint_path, 'optimizer')
        os.makedirs(optimizer_dir, exist_ok=True)
        
        # Save optimizer checkpoint
        optimizer_checkpoint = tf.train.Checkpoint(optimizer=optimizer)
        optimizer_checkpoint.write(os.path.join(optimizer_dir, 'optimizer_state'))
    
    def _update_checkpoint_registry(self, checkpoint_path: str, metadata: Dict):
        """Update checkpoint registry and cleanup old checkpoints"""
        self.checkpoint_metadata[checkpoint_path] = metadata
        self.checkpoint_counter += 1
        
        # Cleanup old checkpoints
        if len(self.checkpoint_metadata) > self.max_to_keep:
            # Sort by timestamp and remove oldest
            sorted_checkpoints = sorted(
                self.checkpoint_metadata.items(),
                key=lambda x: x[1]['timestamp']
            )
            
            for old_path, _ in sorted_checkpoints[:-self.max_to_keep]:
                self._remove_checkpoint(old_path)
                del self.checkpoint_metadata[old_path]
    
    def _remove_checkpoint(self, checkpoint_path: str):
        """Remove old checkpoint"""
        import shutil
        if os.path.exists(checkpoint_path):
            shutil.rmtree(checkpoint_path)
            print(f"Removed old checkpoint: {checkpoint_path}")
    
    def _async_save_worker(self):
        """Async save worker thread"""
        while True:
            if self.save_queue:
                checkpoint_data = self.save_queue.pop(0)
                self._save_checkpoint_data(checkpoint_data)
            else:
                time.sleep(1)  # Wait for new checkpoints
    
    def load_checkpoint(self, checkpoint_path: str, model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer) -> Dict:
        """
        Load distributed checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint
            model: Model to load weights into
            optimizer: Optimizer to load state into
            
        Returns:
            metadata: Checkpoint metadata
        """
        # Load metadata
        with open(os.path.join(checkpoint_path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        # Load model weights
        self._load_model_shards(model, checkpoint_path)
        
        # Load optimizer state
        self._load_optimizer_state(optimizer, checkpoint_path)
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        return metadata
    
    def _load_model_shards(self, model: tf.keras.Model, checkpoint_path: str):
        """Load model weights from shards"""
        weights_dir = os.path.join(checkpoint_path, 'model_weights')
        
        # Load shard index
        with open(os.path.join(weights_dir, 'shard_index.json'), 'r') as f:
            shard_info = json.load(f)
        
        # Load each shard
        for group_name, info in shard_info.items():
            shard_path = info['path']
            
            # Create temporary checkpoint to load variables
            temp_vars = [tf.Variable(tf.zeros(shape), dtype=dtype) 
                        for shape, dtype in zip(info['shapes'], info['dtypes'])]
            
            temp_checkpoint = tf.train.Checkpoint(**{f'var_{i}': var for i, var in enumerate(temp_vars)})
            temp_checkpoint.read(shard_path)
            
            # Assign loaded values to model variables
            model_vars = [var for var in model.trainable_variables if any(name in var.name for name in info['variables'])]
            
            for model_var, temp_var in zip(model_vars, temp_vars):
                model_var.assign(temp_var)
    
    def _load_optimizer_state(self, optimizer: tf.keras.optimizers.Optimizer, checkpoint_path: str):
        """Load optimizer state"""
        optimizer_dir = os.path.join(checkpoint_path, 'optimizer')
        
        optimizer_checkpoint = tf.train.Checkpoint(optimizer=optimizer)
        optimizer_checkpoint.read(os.path.join(optimizer_dir, 'optimizer_state'))
    
    def list_checkpoints(self) -> List[Dict]:
        """List available checkpoints"""
        return [
            {'path': path, **metadata} 
            for path, metadata in sorted(
                self.checkpoint_metadata.items(),
                key=lambda x: x[1]['timestamp'],
                reverse=True
            )
        ]
```

---

## 4. Progressive Context Extension (4K → 32K → 128K)

### 4.1 YaRN Technique Implementation

```python
# training/context_extension.py
import tensorflow as tf
import numpy as np
from typing import Dict, Tuple

class YaRNContextExtension:
    """
    Yet another RoPE extensioN (YaRN) for progressive context extension
    Extends context from 4K → 32K → 128K tokens
    """
    
    def __init__(self,
                 base_context_length: int = 4096,
                 extension_stages: List[int] = [32768, 131072],
                 alpha: float = 1.0,
                 beta: float = 32.0):
        self.base_context_length = base_context_length
        self.extension_stages = extension_stages
        self.alpha = alpha
        self.beta = beta
        
        # Current context length
        self.current_context_length = base_context_length
        self.current_stage = 0
    
    def extend_context_length(self, new_length: int, model: tf.keras.Model):
        """
        Extend context length using YaRN technique
        
        Args:
            new_length: New context length to extend to
            model: Model to update
        """
        if new_length <= self.current_context_length:
            return
        
        print(f"Extending context from {self.current_context_length} to {new_length}")
        
        # Update RoPE frequencies in all attention layers
        self._update_rope_frequencies(model, new_length)
        
        # Update position embeddings if present
        self._update_position_embeddings(model, new_length)
        
        self.current_context_length = new_length
        
        print(f"Context extension complete: {new_length} tokens")
    
    def _update_rope_frequencies(self, model: tf.keras.Model, new_length: int):
        """Update RoPE frequencies for context extension"""
        scale_factor = new_length / self.base_context_length
        
        # Find all attention layers
        attention_layers = self._find_attention_layers(model)
        
        for layer in attention_layers:
            if hasattr(layer, 'rope_freqs'):
                # Apply YaRN scaling
                original_freqs = layer.rope_freqs
                scaled_freqs = self._apply_yarn_scaling(original_freqs, scale_factor)
                layer.rope_freqs.assign(scaled_freqs)
    
    def _apply_yarn_scaling(self, freqs: tf.Tensor, scale_factor: float) -> tf.Tensor:
        """
        Apply YaRN scaling to RoPE frequencies
        
        Args:
            freqs: Original RoPE frequencies
            scale_factor: Context length scale factor
            
        Returns:
            scaled_freqs: YaRN-scaled frequencies
        """
        # YaRN scaling formula
        # For frequencies below threshold: scale by 1/scale_factor
        # For frequencies above threshold: apply interpolation
        
        dim = tf.shape(freqs)[0]
        
        # Calculate critical dimension
        critical_dim = dim * np.log(self.alpha) / (2 * np.log(scale_factor))
        critical_dim = tf.cast(critical_dim, tf.int32)
        
        # Create scaling factors
        indices = tf.range(dim, dtype=tf.float32)
        
        # Low frequency scaling (direct scaling)
        low_freq_scale = 1.0 / scale_factor
        
        # High frequency scaling (interpolation)
        high_freq_scale = tf.where(
            indices < critical_dim,
            low_freq_scale,
            (1.0 - self.beta / scale_factor) * tf.cos(
                np.pi * (indices - critical_dim) / (2 * (dim - critical_dim))
            ) + self.beta / scale_factor
        )
        
        # Apply scaling
        scaled_freqs = freqs * high_freq_scale
        
        return scaled_freqs
    
    def _find_attention_layers(self, model: tf.keras.Model) -> List:
        """Find all attention layers in the model"""
        attention_layers = []
        
        def find_layers(layer):
            if hasattr(layer, 'rope_freqs'):
                attention_layers.append(layer)
            
            if hasattr(layer, 'layers'):
                for sublayer in layer.layers:
                    find_layers(sublayer)
        
        find_layers(model)
        return attention_layers
    
    def _update_position_embeddings(self, model: tf.keras.Model, new_length: int):
        """Update position embeddings for extended context"""
        # Find position embedding layers
        for layer in model.layers:
            if hasattr(layer, 'position_embeddings'):
                current_length = layer.position_embeddings.input_dim
                
                if current_length < new_length:
                    # Extend position embeddings
                    self._extend_position_embeddings(layer, new_length)
    
    def _extend_position_embeddings(self, layer, new_length: int):
        """Extend position embeddings to new length"""
        current_embeddings = layer.position_embeddings.embeddings
        current_length, embed_dim = current_embeddings.shape
        
        # Create extended embeddings
        extended_embeddings = tf.Variable(
            tf.zeros([new_length, embed_dim]),
            trainable=True
        )
        
        # Copy existing embeddings
        extended_embeddings[:current_length].assign(current_embeddings)
        
        # Initialize new positions with interpolation
        for pos in range(current_length, new_length):
            # Simple interpolation from existing positions
            interp_pos = (pos * current_length) // new_length
            extended_embeddings[pos].assign(current_embeddings[interp_pos])
        
        # Update layer
        layer.position_embeddings.embeddings = extended_embeddings
        layer.position_embeddings.input_dim = new_length

class ProgressiveContextTrainer:
    """
    Progressive context extension trainer
    """
    
    def __init__(self, yarn_extender: YaRNContextExtension):
        self.yarn_extender = yarn_extender
        self.extension_schedule = [
            {'context_length': 4096, 'epochs': 10},
            {'context_length': 32768, 'epochs': 5},
            {'context_length': 131072, 'epochs': 3}
        ]
    
    def train_with_progressive_extension(self,
                                       model: tf.keras.Model,
                                       optimizer: tf.keras.optimizers.Optimizer,
                                       dataset_fn: Callable,
                                       orchestrator):
        """
        Train with progressive context extension
        
        Args:
            model: Model to train
            optimizer: Optimizer
            dataset_fn: Function to create dataset with given context length
            orchestrator: Training orchestrator
        """
        for stage in self.extension_schedule:
            context_length = stage['context_length']
            epochs = stage['epochs']
            
            print(f"Training stage: {context_length} context length for {epochs} epochs")
            
            # Extend context if needed
            if context_length > self.yarn_extender.current_context_length:
                self.yarn_extender.extend_context_length(context_length, model)
            
            # Create dataset with new context length
            train_dataset = dataset_fn(context_length)
            
            # Train for specified epochs
            orchestrator.train(
                train_dataset=train_dataset,
                num_epochs=epochs,
                steps_per_epoch=1000  # Adjust as needed
            )
            
            print(f"Completed training stage: {context_length} tokens")
```

---

## 5. Success Criteria and Validation Targets

### 5.1 Data Pipeline Requirements
- [ ] Process 14.8T tokens with 87% code and 13% natural language
- [ ] Data quality filters removing < 10% of content
- [ ] Deduplication rate < 5% (indicating diverse content)
- [ ] Throughput > 10,000 tokens/second/worker during data loading
- [ ] Memory-efficient streaming without full dataset loading

### 5.2 Training Orchestration Requirements
- [ ] Training stability > 99.9% uptime over multi-day runs
- [ ] Comprehensive monitoring with < 1 minute alert latency
- [ ] Automatic recovery from transient failures
- [ ] Expert utilization coefficient of variation < 0.2
- [ ] Memory usage monitoring with predictive alerts

### 5.3 Checkpointing Requirements
- [ ] Distributed checkpoint save time < 10 minutes for 671B parameters
- [ ] Checkpoint compression ratio > 50% without quality loss
- [ ] Recovery time < 30 minutes from checkpoint
- [ ] Checkpoint integrity validation with 100% success rate
- [ ] Automatic cleanup maintaining only N most recent checkpoints

### 5.4 Context Extension Requirements
- [ ] Successful extension from 4K → 32K → 128K tokens
- [ ] Performance degradation < 5% after each extension stage
- [ ] Memory scaling sub-linear with context length
- [ ] YaRN technique maintaining attention quality
- [ ] Progressive training completing within timeline

This comprehensive training pipeline provides the infrastructure needed to successfully train DeepSeek-V3's 671B parameter model with proper data management, monitoring, checkpointing, and context extension capabilities.
