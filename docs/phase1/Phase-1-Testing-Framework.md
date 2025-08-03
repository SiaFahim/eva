# Phase 1: Testing Framework and Validation Strategy
## Comprehensive Testing for DeepSeek-V3 Components

**Version:** 1.0  
**Date:** 2025-08-02  
**Testing Philosophy:** Test-as-you-develop with synthetic data validation

---

## 1. Testing Strategy Overview

### 1.1 Testing Pyramid for LLM Components

```
                    Integration Tests
                   /                 \
              Component Tests     Performance Tests
             /                                     \
        Unit Tests                            Synthetic Data Tests
       /          \                          /                    \
  Functional   Mathematical            Convergence           Benchmark
   Tests        Validation              Tests                 Tests
```

### 1.2 Testing Principles

**Test-Driven Development:** Write tests before implementation to define expected behavior
**Synthetic Data Focus:** Use synthetic datasets to avoid expensive pre-training during development
**Performance Validation:** Ensure components meet performance targets before integration
**Mathematical Correctness:** Validate mathematical operations and numerical stability
**Production Readiness:** Test components under realistic conditions and edge cases

---

## 2. Unit Testing Framework

### 2.1 MLA Component Tests

```python
# tests/unit/test_mla.py
import tensorflow as tf
import numpy as np
import pytest
from components.attention.mla import MultiHeadLatentAttention

class TestMultiHeadLatentAttention:
    """Comprehensive MLA unit tests"""
    
    @pytest.fixture
    def mla_config(self):
        return {
            'd_model': 512,
            'num_heads': 8,
            'd_latent': 128,
            'rope_dim': 32,
            'batch_size': 2,
            'seq_len': 64
        }
    
    @pytest.fixture
    def mla_layer(self, mla_config):
        return MultiHeadLatentAttention(
            d_model=mla_config['d_model'],
            num_heads=mla_config['num_heads'],
            d_latent=mla_config['d_latent'],
            rope_dim=mla_config['rope_dim']
        )
    
    def test_forward_pass_shapes(self, mla_layer, mla_config):
        """Test forward pass produces correct output shapes"""
        inputs = tf.random.normal([
            mla_config['batch_size'], 
            mla_config['seq_len'], 
            mla_config['d_model']
        ])
        
        output, cache = mla_layer(inputs, use_cache=True)
        
        # Validate output shape
        assert output.shape == inputs.shape
        
        # Validate cache shapes
        assert cache is not None
        k_cache, v_cache = cache
        expected_cache_shape = [
            mla_config['batch_size'],
            mla_config['num_heads'],
            mla_config['seq_len'],
            mla_config['d_model'] // mla_config['num_heads']
        ]
        assert list(k_cache.shape) == expected_cache_shape
        assert list(v_cache.shape) == expected_cache_shape
    
    def test_compression_decompression_consistency(self, mla_layer, mla_config):
        """Test compression-decompression maintains information"""
        inputs = tf.random.normal([1, mla_config['seq_len'], mla_config['d_model']])
        
        # Get compressed representation
        compressed = mla_layer.compression(inputs)
        assert compressed.shape[-1] == mla_config['d_latent']
        
        # Test decompression dimensions
        c_qk = compressed[:, :, :mla_config['d_latent']//2]
        c_v = compressed[:, :, mla_config['d_latent']//2:]
        
        q_decompressed = mla_layer.q_decompression(c_qk)
        k_decompressed = mla_layer.k_decompression(c_qk)
        v_decompressed = mla_layer.v_decompression(c_v)
        
        expected_qk_dim = mla_config['num_heads'] * (
            mla_config['d_model'] // mla_config['num_heads'] - mla_config['rope_dim']
        )
        expected_v_dim = mla_config['num_heads'] * (mla_config['d_model'] // mla_config['num_heads'])
        
        assert q_decompressed.shape[-1] == expected_qk_dim
        assert k_decompressed.shape[-1] == expected_qk_dim
        assert v_decompressed.shape[-1] == expected_v_dim
    
    def test_kv_cache_functionality(self, mla_layer, mla_config):
        """Test KV cache accumulation and usage"""
        seq_len_1, seq_len_2 = 32, 16
        
        # First forward pass
        inputs_1 = tf.random.normal([1, seq_len_1, mla_config['d_model']])
        output_1, cache_1 = mla_layer(inputs_1, use_cache=True)
        
        # Second forward pass with cache
        inputs_2 = tf.random.normal([1, seq_len_2, mla_config['d_model']])
        output_2, cache_2 = mla_layer(inputs_2, past_key_value=cache_1, use_cache=True)
        
        # Validate cache growth
        k_cache_1, v_cache_1 = cache_1
        k_cache_2, v_cache_2 = cache_2
        
        assert k_cache_2.shape[2] == seq_len_1 + seq_len_2
        assert v_cache_2.shape[2] == seq_len_1 + seq_len_2
    
    def test_attention_mask_application(self, mla_layer, mla_config):
        """Test attention mask correctly applied"""
        inputs = tf.random.normal([1, mla_config['seq_len'], mla_config['d_model']])
        
        # Create causal mask
        mask = tf.linalg.band_part(
            tf.ones([mla_config['seq_len'], mla_config['seq_len']]), -1, 0
        )
        mask = (1.0 - mask) * -1e9
        mask = mask[None, :, :]
        
        output_masked, _ = mla_layer(inputs, attention_mask=mask)
        output_unmasked, _ = mla_layer(inputs)
        
        # Outputs should differ when mask applied
        assert not tf.reduce_all(tf.abs(output_masked - output_unmasked) < 1e-6)
    
    def test_rope_positional_encoding(self, mla_layer, mla_config):
        """Test RoPE positional encoding functionality"""
        inputs = tf.random.normal([1, mla_config['seq_len'], mla_config['d_model']])
        
        # Test with different position IDs
        pos_ids_1 = tf.range(mla_config['seq_len'], dtype=tf.int32)[None, :]
        pos_ids_2 = tf.range(10, mla_config['seq_len'] + 10, dtype=tf.int32)[None, :]
        
        output_1, _ = mla_layer(inputs, position_ids=pos_ids_1)
        output_2, _ = mla_layer(inputs, position_ids=pos_ids_2)
        
        # Different positions should produce different outputs
        assert not tf.reduce_all(tf.abs(output_1 - output_2) < 1e-6)
    
    def test_gradient_flow(self, mla_layer, mla_config):
        """Test gradient flow through MLA"""
        inputs = tf.random.normal([1, mla_config['seq_len'], mla_config['d_model']])
        
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            output, _ = mla_layer(inputs)
            loss = tf.reduce_mean(tf.square(output))
        
        gradients = tape.gradient(loss, inputs)
        
        assert gradients is not None
        assert not tf.reduce_any(tf.math.is_nan(gradients))
        assert tf.reduce_any(tf.abs(gradients) > 1e-8)
    
    def test_memory_efficiency(self, mla_layer, mla_config):
        """Test memory efficiency vs standard attention"""
        # This test requires GPU memory monitoring
        if not tf.config.list_physical_devices('GPU'):
            pytest.skip("GPU not available for memory testing")
        
        long_seq_len = 1024
        inputs = tf.random.normal([1, long_seq_len, mla_config['d_model']])
        
        # Monitor memory usage
        initial_memory = tf.config.experimental.get_memory_info('GPU:0')['current']
        
        output, cache = mla_layer(inputs, use_cache=True)
        
        peak_memory = tf.config.experimental.get_memory_info('GPU:0')['peak']
        memory_used = peak_memory - initial_memory
        
        # Memory usage should be reasonable for long sequences
        assert memory_used > 0
        print(f"Memory used for seq_len={long_seq_len}: {memory_used / (1024**2):.2f} MB")
```

### 2.2 MoE Component Tests

```python
# tests/unit/test_moe.py
import tensorflow as tf
import numpy as np
import pytest
from components.moe.basic_moe import BasicMoELayer

class TestBasicMoELayer:
    """Comprehensive MoE unit tests"""
    
    @pytest.fixture
    def moe_config(self):
        return {
            'd_model': 256,
            'd_ff': 1024,
            'num_experts': 8,
            'top_k': 2,
            'batch_size': 4,
            'seq_len': 32
        }
    
    @pytest.fixture
    def moe_layer(self, moe_config):
        return BasicMoELayer(
            d_model=moe_config['d_model'],
            d_ff=moe_config['d_ff'],
            num_experts=moe_config['num_experts'],
            top_k=moe_config['top_k']
        )
    
    def test_forward_pass_shapes(self, moe_layer, moe_config):
        """Test MoE forward pass shape consistency"""
        inputs = tf.random.normal([
            moe_config['batch_size'],
            moe_config['seq_len'],
            moe_config['d_model']
        ])
        
        output = moe_layer(inputs, training=True)
        
        assert output.shape == inputs.shape
        assert not tf.reduce_any(tf.math.is_nan(output))
    
    def test_expert_routing(self, moe_layer, moe_config):
        """Test expert routing mechanism"""
        inputs = tf.random.normal([1, 1, moe_config['d_model']])
        
        # Get router logits
        inputs_flat = tf.reshape(inputs, [-1, moe_config['d_model']])
        router_logits = moe_layer.router(inputs_flat)
        
        # Test top-k selection
        top_k_logits, top_k_indices = tf.nn.top_k(router_logits, k=moe_config['top_k'])
        
        assert top_k_indices.shape == [1, moe_config['top_k']]
        assert tf.reduce_all(top_k_indices >= 0)
        assert tf.reduce_all(top_k_indices < moe_config['num_experts'])
    
    def test_expert_utilization_tracking(self, moe_layer, moe_config):
        """Test expert utilization tracking"""
        inputs = tf.random.normal([
            moe_config['batch_size'],
            moe_config['seq_len'],
            moe_config['d_model']
        ])
        
        # Reset counters
        moe_layer.reset_expert_counts()
        
        # Run multiple forward passes
        for _ in range(10):
            _ = moe_layer(inputs, training=True)
        
        # Check utilization
        utilization = moe_layer.get_expert_utilization()
        
        # All experts should have some utilization
        assert np.all(utilization['expert_counts'] > 0)
        assert utilization['variance'] < 0.5  # Reasonable load balance
    
    def test_load_balancing(self, moe_layer, moe_config):
        """Test load balancing effectiveness"""
        # Generate diverse inputs to encourage different expert usage
        inputs_list = []
        for i in range(moe_config['num_experts']):
            # Create inputs with different patterns
            pattern = tf.random.normal([4, 16, moe_config['d_model']]) * (i + 1) * 0.1
            inputs_list.append(pattern)
        
        moe_layer.reset_expert_counts()
        
        # Process different input patterns
        for inputs in inputs_list:
            _ = moe_layer(inputs, training=True)
        
        utilization = moe_layer.get_expert_utilization()
        
        # Check that load balancing is working
        assert utilization['variance'] < 0.3
        assert utilization['max_utilization'] / utilization['min_utilization'] < 5.0
    
    def test_gradient_flow_to_all_experts(self, moe_layer, moe_config):
        """Test gradients flow to all experts"""
        inputs = tf.random.normal([
            moe_config['batch_size'],
            moe_config['seq_len'],
            moe_config['d_model']
        ])
        
        with tf.GradientTape() as tape:
            output = moe_layer(inputs, training=True)
            loss = tf.reduce_mean(tf.square(output))
        
        # Get gradients for all expert parameters
        expert_gradients = []
        for expert in moe_layer.experts:
            expert_grads = tape.gradient(loss, expert.trainable_variables)
            expert_gradients.extend(expert_grads)
        
        # Check that most experts receive gradients
        non_none_grads = [g for g in expert_gradients if g is not None]
        assert len(non_none_grads) > len(expert_gradients) * 0.5  # At least 50% of experts
        
        # Check gradients are finite
        for grad in non_none_grads:
            assert tf.reduce_all(tf.math.is_finite(grad))
```

---

## 3. Integration Testing Framework

### 3.1 Component Integration Tests

```python
# tests/integration/test_component_integration.py
import tensorflow as tf
import pytest
from components.attention.mla import MultiHeadLatentAttention
from components.moe.basic_moe import BasicMoELayer
from components.integration.transformer_block import TransformerBlockWithMLA

class TestComponentIntegration:
    """Test integration between MLA, MoE, and other components"""
    
    @pytest.fixture
    def integration_config(self):
        return {
            'd_model': 512,
            'num_heads': 8,
            'd_ff': 2048,
            'num_experts': 8,
            'top_k': 2,
            'd_latent': 128,
            'batch_size': 2,
            'seq_len': 64
        }
    
    def test_transformer_block_integration(self, integration_config):
        """Test MLA + MoE transformer block"""
        transformer_block = TransformerBlockWithMLA(
            d_model=integration_config['d_model'],
            num_heads=integration_config['num_heads'],
            d_ff=integration_config['d_ff'],
            num_experts=integration_config['num_experts'],
            top_k=integration_config['top_k'],
            d_latent=integration_config['d_latent']
        )
        
        inputs = tf.random.normal([
            integration_config['batch_size'],
            integration_config['seq_len'],
            integration_config['d_model']
        ])
        
        output = transformer_block(inputs, training=True)
        
        assert output.shape == inputs.shape
        assert not tf.reduce_any(tf.math.is_nan(output))
    
    def test_multi_layer_model(self, integration_config):
        """Test multi-layer model with integrated components"""
        from components.integration.model_utils import create_mini_model
        
        model = create_mini_model(
            num_layers=4,
            d_model=integration_config['d_model'],
            num_heads=integration_config['num_heads'],
            d_ff=integration_config['d_ff'],
            num_experts=integration_config['num_experts']
        )
        
        inputs = tf.random.normal([
            integration_config['batch_size'],
            integration_config['seq_len'],
            integration_config['d_model']
        ])
        
        output = model(inputs, training=True)
        
        assert output.shape == inputs.shape
        assert not tf.reduce_any(tf.math.is_nan(output))
    
    def test_end_to_end_training(self, integration_config):
        """Test end-to-end training with synthetic data"""
        from tests.synthetic_data.data_generators import create_synthetic_dataset
        
        # Create model and synthetic data
        model = create_mini_model(num_layers=2, **integration_config)
        train_data = create_synthetic_dataset(
            batch_size=integration_config['batch_size'],
            seq_len=integration_config['seq_len'],
            d_model=integration_config['d_model'],
            num_batches=10
        )
        
        # Simple training loop
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        losses = []
        
        for batch in train_data:
            with tf.GradientTape() as tape:
                predictions = model(batch['inputs'], training=True)
                loss = tf.reduce_mean(tf.square(predictions - batch['targets']))
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            losses.append(loss.numpy())
        
        # Check training progress
        assert len(losses) == 10
        assert all(np.isfinite(loss) for loss in losses)
        # Loss should generally decrease or remain stable
        assert losses[-1] <= losses[0] * 2.0  # Allow some variance
```

---

## 4. Performance Testing Framework

### 4.1 Benchmark Tests

```python
# tests/benchmarks/performance_benchmarks.py
import tensorflow as tf
import time
import numpy as np
from components.attention.mla import MultiHeadLatentAttention
from components.moe.basic_moe import BasicMoELayer

class PerformanceBenchmarks:
    """Comprehensive performance benchmarking suite"""
    
    def benchmark_mla_memory_efficiency(self):
        """Benchmark MLA memory efficiency vs standard attention"""
        configs = [
            {'seq_len': 128, 'd_model': 512, 'num_heads': 8},
            {'seq_len': 512, 'd_model': 512, 'num_heads': 8},
            {'seq_len': 1024, 'd_model': 512, 'num_heads': 8},
            {'seq_len': 2048, 'd_model': 512, 'num_heads': 8},
        ]
        
        results = []
        
        for config in configs:
            mla = MultiHeadLatentAttention(
                d_model=config['d_model'],
                num_heads=config['num_heads'],
                d_latent=128
            )
            
            inputs = tf.random.normal([1, config['seq_len'], config['d_model']])
            
            # Measure memory usage
            initial_memory = tf.config.experimental.get_memory_info('GPU:0')['current']
            output, cache = mla(inputs, use_cache=True)
            peak_memory = tf.config.experimental.get_memory_info('GPU:0')['peak']
            
            memory_used = peak_memory - initial_memory
            
            result = {
                'seq_len': config['seq_len'],
                'memory_mb': memory_used / (1024**2),
                'memory_per_token': memory_used / config['seq_len']
            }
            results.append(result)
        
        return results
    
    def benchmark_moe_scaling(self):
        """Benchmark MoE scaling with expert count"""
        expert_counts = [4, 8, 16, 32]
        base_config = {'d_model': 512, 'd_ff': 2048, 'top_k': 2}
        
        results = []
        
        for num_experts in expert_counts:
            moe = BasicMoELayer(
                num_experts=num_experts,
                **base_config
            )
            
            inputs = tf.random.normal([4, 128, base_config['d_model']])
            
            # Warmup
            for _ in range(5):
                _ = moe(inputs)
            
            # Benchmark
            times = []
            for _ in range(20):
                start = time.time()
                _ = moe(inputs)
                times.append(time.time() - start)
            
            avg_time = np.mean(times)
            throughput = (4 * 128) / avg_time  # tokens per second
            
            result = {
                'num_experts': num_experts,
                'avg_time': avg_time,
                'throughput': throughput,
                'time_per_expert': avg_time / num_experts
            }
            results.append(result)
        
        return results
    
    def benchmark_fp8_performance(self):
        """Benchmark FP8 vs FP32 performance"""
        from components.precision.mixed_precision import FP8MixedPrecisionTrainer
        
        # Create test model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256)
        ])
        
        inputs = tf.random.normal([32, 256])
        targets = tf.random.normal([32, 256])
        
        # FP32 benchmark
        fp32_times = []
        for _ in range(10):
            start = time.time()
            with tf.GradientTape() as tape:
                predictions = model(inputs, training=True)
                loss = tf.reduce_mean(tf.square(predictions - targets))
            gradients = tape.gradient(loss, model.trainable_variables)
            fp32_times.append(time.time() - start)
        
        # FP8 benchmark
        fp8_trainer = FP8MixedPrecisionTrainer()
        fp8_times = []
        for _ in range(10):
            start = time.time()
            loss, info = fp8_trainer.compute_loss_with_fp8(
                model, inputs, targets, tf.keras.losses.mse
            )
            fp8_times.append(time.time() - start)
        
        return {
            'fp32_avg_time': np.mean(fp32_times),
            'fp8_avg_time': np.mean(fp8_times),
            'speedup': np.mean(fp32_times) / np.mean(fp8_times)
        }
```

---

## 5. Synthetic Data Testing

### 5.1 Synthetic Data Generators

```python
# tests/synthetic_data/data_generators.py
import tensorflow as tf
import numpy as np

def create_synthetic_dataset(batch_size: int, seq_len: int, d_model: int, num_batches: int):
    """Create synthetic dataset for component testing"""
    
    def generate_batch():
        # Create inputs with structured patterns
        inputs = tf.random.normal([batch_size, seq_len, d_model])
        
        # Create targets with some relationship to inputs
        # This allows testing of learning capability
        targets = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        targets = tf.tile(targets, [1, 1, d_model])
        targets += tf.random.normal([batch_size, seq_len, d_model]) * 0.1
        
        return {'inputs': inputs, 'targets': targets}
    
    dataset = tf.data.Dataset.from_generator(
        generate_batch,
        output_signature={
            'inputs': tf.TensorSpec([batch_size, seq_len, d_model], tf.float32),
            'targets': tf.TensorSpec([batch_size, seq_len, d_model], tf.float32)
        }
    )
    
    return dataset.take(num_batches)

def create_convergence_test_data(d_model: int, seq_len: int):
    """Create data for testing training convergence"""
    # Simple pattern that should be learnable
    x = tf.random.normal([100, seq_len, d_model])
    # Target is a simple transformation
    y = tf.nn.tanh(tf.reduce_mean(x, axis=-1, keepdims=True))
    y = tf.tile(y, [1, 1, d_model])
    
    return tf.data.Dataset.from_tensor_slices({'inputs': x, 'targets': y}).batch(10)
```

This comprehensive testing framework ensures that all Phase 1 components are thoroughly validated before integration and provides the foundation for reliable development of advanced DeepSeek-V3 features.
