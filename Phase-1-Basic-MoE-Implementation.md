# Phase 1: Basic MoE Layer Implementation
## DeepSeek-V3 TensorFlow Implementation

### Overview

This document provides engineering guidance for implementing basic Mixture-of-Experts (MoE) layers in TensorFlow as foundation for DeepSeek-V3's advanced MoE architecture. This phase focuses on core routing mechanisms and expert computation before scaling to 256 experts.

---

## 1. Basic MoE Architecture

### 1.1 Core Components

```python
# Basic MoE consists of:
# 1. Expert Networks (Feed-forward layers)
# 2. Router/Gating Network (Expert selection)
# 3. Load Balancing Mechanism
# 4. Expert Combination Logic

class BasicMoELayer(tf.keras.layers.Layer):
    """
    Basic Mixture of Experts layer for DeepSeek-V3
    Implements core routing and expert computation
    """
    
    def __init__(self, 
                 d_model: int,
                 d_ff: int,
                 num_experts: int = 8,  # Start small, scale later
                 top_k: int = 2,        # Number of experts per token
                 activation: str = 'swish',
                 use_bias: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k
        self.activation = activation
        self.use_bias = use_bias
        
        # Router network
        self.router = tf.keras.layers.Dense(
            num_experts,
            use_bias=use_bias,
            name='router'
        )
        
        # Expert networks
        self.experts = []
        for i in range(num_experts):
            expert = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    d_ff, 
                    activation=activation,
                    use_bias=use_bias,
                    name=f'expert_{i}_up'
                ),
                tf.keras.layers.Dense(
                    d_model,
                    use_bias=use_bias,
                    name=f'expert_{i}_down'
                )
            ], name=f'expert_{i}')
            self.experts.append(expert)
        
        # Load balancing tracking
        self.expert_counts = tf.Variable(
            tf.zeros(num_experts),
            trainable=False,
            name='expert_counts'
        )
    
    def call(self, inputs, training=None):
        """
        Forward pass through MoE layer
        
        Args:
            inputs: [batch_size, seq_len, d_model]
            training: Training mode flag
            
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = tf.shape(inputs)
        
        # Flatten for routing: [batch_size * seq_len, d_model]
        inputs_flat = tf.reshape(inputs, [-1, d_model])
        
        # Router computation
        router_logits = self.router(inputs_flat)  # [batch_size * seq_len, num_experts]
        
        # Top-k expert selection
        top_k_logits, top_k_indices = tf.nn.top_k(router_logits, k=self.top_k)
        top_k_probs = tf.nn.softmax(top_k_logits, axis=-1)
        
        # Initialize output
        output = tf.zeros_like(inputs_flat)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens assigned to this expert
            expert_mask = tf.reduce_any(tf.equal(top_k_indices, expert_idx), axis=-1)
            expert_tokens = tf.boolean_mask(inputs_flat, expert_mask)
            
            if tf.shape(expert_tokens)[0] > 0:  # If expert has tokens
                # Process through expert
                expert_output = self.experts[expert_idx](expert_tokens)
                
                # Get routing weights for this expert
                expert_positions = tf.where(expert_mask)[:, 0]
                expert_weights = tf.gather_nd(
                    top_k_probs,
                    tf.stack([
                        expert_positions,
                        tf.where(tf.equal(
                            tf.gather(top_k_indices, expert_positions),
                            expert_idx
                        ))[:, 1]
                    ], axis=1)
                )
                
                # Weight expert output
                weighted_output = expert_output * expert_weights[:, None]
                
                # Scatter back to full output
                output = tf.tensor_scatter_nd_add(
                    output,
                    expert_positions[:, None],
                    weighted_output
                )
                
                # Update expert counts for load balancing
                if training:
                    self.expert_counts[expert_idx].assign_add(
                        tf.cast(tf.shape(expert_tokens)[0], tf.float32)
                    )
        
        # Reshape back to original shape
        output = tf.reshape(output, [batch_size, seq_len, d_model])
        
        return output
    
    def get_expert_utilization(self):
        """Get current expert utilization statistics"""
        total_tokens = tf.reduce_sum(self.expert_counts)
        utilization = self.expert_counts / (total_tokens + 1e-8)
        
        return {
            'expert_counts': self.expert_counts.numpy(),
            'utilization': utilization.numpy(),
            'variance': tf.math.reduce_variance(utilization).numpy(),
            'max_utilization': tf.reduce_max(utilization).numpy(),
            'min_utilization': tf.reduce_min(utilization).numpy()
        }
    
    def reset_expert_counts(self):
        """Reset expert utilization counters"""
        self.expert_counts.assign(tf.zeros_like(self.expert_counts))
```

---

## 2. Testing Framework

### 2.1 Basic MoE Tests

```python
# tests/test_basic_moe.py
import tensorflow as tf
import numpy as np
import pytest
from components.basic_moe import BasicMoELayer

class TestBasicMoE:
    
    def setup_method(self):
        self.config = {
            'd_model': 256,
            'd_ff': 1024,
            'num_experts': 8,
            'top_k': 2,
            'batch_size': 4,
            'seq_len': 32
        }
        
        self.moe = BasicMoELayer(
            d_model=self.config['d_model'],
            d_ff=self.config['d_ff'],
            num_experts=self.config['num_experts'],
            top_k=self.config['top_k']
        )
    
    def test_forward_pass(self):
        """Test basic forward pass functionality"""
        inputs = tf.random.normal([
            self.config['batch_size'],
            self.config['seq_len'],
            self.config['d_model']
        ])
        
        output = self.moe(inputs, training=True)
        
        # Check output shape
        assert output.shape == inputs.shape
        assert not tf.reduce_any(tf.math.is_nan(output))
    
    def test_expert_utilization(self):
        """Test expert utilization tracking"""
        inputs = tf.random.normal([
            self.config['batch_size'],
            self.config['seq_len'],
            self.config['d_model']
        ])
        
        # Reset counters
        self.moe.reset_expert_counts()
        
        # Run multiple forward passes
        for _ in range(10):
            _ = self.moe(inputs, training=True)
        
        # Check utilization
        utilization = self.moe.get_expert_utilization()
        
        # All experts should have some utilization
        assert np.all(utilization['expert_counts'] > 0)
        assert utilization['variance'] < 0.1  # Reasonable load balance
    
    def test_top_k_routing(self):
        """Test top-k expert routing"""
        inputs = tf.random.normal([1, 1, self.config['d_model']])
        
        # Get router logits
        inputs_flat = tf.reshape(inputs, [-1, self.config['d_model']])
        router_logits = self.moe.router(inputs_flat)
        
        # Check top-k selection
        top_k_logits, top_k_indices = tf.nn.top_k(router_logits, k=self.config['top_k'])
        
        assert top_k_indices.shape == [1, self.config['top_k']]
        assert tf.reduce_all(top_k_indices >= 0)
        assert tf.reduce_all(top_k_indices < self.config['num_experts'])
    
    def test_gradient_flow(self):
        """Test gradient flow through MoE"""
        inputs = tf.random.normal([
            self.config['batch_size'],
            self.config['seq_len'],
            self.config['d_model']
        ])
        
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            output = self.moe(inputs, training=True)
            loss = tf.reduce_mean(tf.square(output))
        
        gradients = tape.gradient(loss, inputs)
        
        assert gradients is not None
        assert not tf.reduce_any(tf.math.is_nan(gradients))
        assert tf.reduce_any(tf.abs(gradients) > 1e-8)
```

---

## 3. Performance Benchmarking

### 3.1 MoE Scaling Benchmark

```python
# benchmarks/basic_moe_benchmark.py
import tensorflow as tf
import time
import numpy as np
from components.basic_moe import BasicMoELayer

class BasicMoEBenchmark:
    
    def benchmark_expert_scaling(self):
        """Benchmark performance with different expert counts"""
        expert_counts = [4, 8, 16, 32]
        top_k_values = [1, 2, 4]
        
        results = []
        
        for num_experts in expert_counts:
            for top_k in top_k_values:
                if top_k <= num_experts:
                    config = {
                        'd_model': 512,
                        'd_ff': 2048,
                        'num_experts': num_experts,
                        'top_k': top_k
                    }
                    
                    moe = BasicMoELayer(**config)
                    inputs = tf.random.normal([4, 128, config['d_model']])
                    
                    # Benchmark forward pass
                    times = []
                    for _ in range(10):  # Warmup
                        _ = moe(inputs)
                    
                    for _ in range(100):  # Benchmark
                        start = time.time()
                        _ = moe(inputs)
                        times.append(time.time() - start)
                    
                    avg_time = np.mean(times)
                    
                    results.append({
                        'num_experts': num_experts,
                        'top_k': top_k,
                        'avg_time': avg_time,
                        'efficiency': top_k / num_experts
                    })
        
        return results
    
    def benchmark_sequence_scaling(self):
        """Benchmark performance with different sequence lengths"""
        seq_lengths = [64, 128, 256, 512, 1024]
        
        moe = BasicMoELayer(
            d_model=512,
            d_ff=2048,
            num_experts=8,
            top_k=2
        )
        
        results = []
        
        for seq_len in seq_lengths:
            inputs = tf.random.normal([2, seq_len, 512])
            
            # Benchmark
            times = []
            for _ in range(50):
                start = time.time()
                _ = moe(inputs)
                times.append(time.time() - start)
            
            avg_time = np.mean(times)
            throughput = (2 * seq_len) / avg_time  # tokens per second
            
            results.append({
                'seq_len': seq_len,
                'avg_time': avg_time,
                'throughput': throughput
            })
        
        return results

if __name__ == "__main__":
    benchmark = BasicMoEBenchmark()
    
    print("=== Expert Scaling Benchmark ===")
    expert_results = benchmark.benchmark_expert_scaling()
    for result in expert_results:
        print(f"Experts: {result['num_experts']}, Top-K: {result['top_k']}, "
              f"Time: {result['avg_time']:.4f}s, Efficiency: {result['efficiency']:.2f}")
    
    print("\n=== Sequence Scaling Benchmark ===")
    seq_results = benchmark.benchmark_sequence_scaling()
    for result in seq_results:
        print(f"Seq Len: {result['seq_len']}, Time: {result['avg_time']:.4f}s, "
              f"Throughput: {result['throughput']:.0f} tokens/sec")
```

---

## 4. Integration and Validation

### 4.1 Transformer Block Integration

```python
# components/transformer_with_moe.py
import tensorflow as tf
from components.mla import MultiHeadLatentAttention
from components.basic_moe import BasicMoELayer

class TransformerBlockWithMoE(tf.keras.layers.Layer):
    """Transformer block with MLA attention and MoE feed-forward"""
    
    def __init__(self, 
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 num_experts: int = 8,
                 top_k: int = 2,
                 d_latent: int = 512,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.attention = MultiHeadLatentAttention(
            d_model=d_model,
            num_heads=num_heads,
            d_latent=d_latent
        )
        
        self.moe = BasicMoELayer(
            d_model=d_model,
            d_ff=d_ff,
            num_experts=num_experts,
            top_k=top_k
        )
        
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, inputs, attention_mask=None, training=None):
        # Self-attention with residual connection
        attention_output, _ = self.attention(
            inputs,
            attention_mask=attention_mask,
            training=training
        )
        attention_output = self.layer_norm_1(inputs + attention_output)
        
        # MoE feed-forward with residual connection
        moe_output = self.moe(attention_output, training=training)
        output = self.layer_norm_2(attention_output + moe_output)
        
        return output
    
    def get_expert_utilization(self):
        """Get expert utilization from MoE layer"""
        return self.moe.get_expert_utilization()
```

---

## 5. Success Criteria and Next Steps

### 5.1 Validation Targets

**Functional Requirements:**
- [ ] Basic MoE routing functional with top-k selection
- [ ] Expert utilization tracking working
- [ ] Load balancing within reasonable variance (<20%)
- [ ] Gradient flow stable through all experts
- [ ] Integration with attention mechanisms

**Performance Requirements:**
- [ ] Linear scaling with number of experts (up to 32)
- [ ] Throughput > 500 tokens/sec/GPU for basic config
- [ ] Memory usage scales appropriately with expert count
- [ ] Expert utilization variance < 0.1

### 5.2 Development Workflow

**Phase 1A: Core Implementation**
1. Implement basic MoE layer with simple routing
2. Add expert utilization tracking
3. Test with small expert counts (4-8 experts)
4. Validate gradient flow and training stability

**Phase 1B: Scaling Validation**
1. Test scaling to 16-32 experts
2. Benchmark performance vs dense layers
3. Validate load balancing mechanisms
4. Test integration with MLA attention

**Phase 1C: Optimization**
1. Optimize routing computation
2. Implement efficient expert batching
3. Add memory optimization techniques
4. Prepare for advanced MoE features

### 5.3 Common Issues and Solutions

**Load Balancing:**
- Issue: Some experts unused, others overloaded
- Solution: Monitor utilization, adjust routing temperature

**Memory Usage:**
- Issue: Memory grows with expert count
- Solution: Implement expert batching, gradient checkpointing

**Training Instability:**
- Issue: Gradient variance across experts
- Solution: Proper initialization, gradient clipping

This basic MoE implementation provides the foundation for DeepSeek-V3's advanced MoE architecture with auxiliary-loss-free load balancing.
