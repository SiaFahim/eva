"""
DeepSeek-V3 Advanced MoE Architecture Implementation

This module implements the advanced Mixture-of-Experts architecture from DeepSeek-V3,
featuring 256 routed experts + 1 shared expert, auxiliary-loss-free load balancing,
and affinity-based routing with expert centroids.

Key Innovations:
- Fine-grained expert segmentation (256 routed experts)
- Shared expert always activated for stable learning
- Auxiliary-loss-free load balancing via bias adjustment
- Affinity-based routing using expert centroids
- Non-differentiable bias updates for load balancing
- Expert parallelism support for distributed training

Mathematical Foundation:
DeepSeek MoE: Y = SharedExpert(X) + Σ(i=1 to k) w_i * RoutedExpert_i(X)
where w_i = sigmoid(affinity(X, centroid_i) + bias_i)

Author: Eva DeepSeek-V3 Project
Date: 2025-08-05
"""

import tensorflow as tf
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import math


class DeepSeekMoELayer(tf.keras.layers.Layer):
    """
    Advanced MoE layer with 256 routed + 1 shared expert
    Implements auxiliary-loss-free load balancing with bias adjustment
    
    This is the core innovation of DeepSeek-V3's MoE architecture, providing
    fine-grained expert specialization while maintaining training stability
    through shared experts and bias-based load balancing.
    
    Args:
        d_model: Model dimension (input/output size)
        d_ff: Feed-forward dimension (expert hidden size)
        num_routed_experts: Number of routed experts (default: 256)
        num_shared_experts: Number of shared experts (default: 1)
        top_k: Number of experts to route each token to (default: 8)
        expert_capacity_factor: Capacity factor for expert load balancing
        use_bias: Whether to use bias in expert networks
        activation: Activation function for experts ('swish', 'relu', 'gelu')
        bias_update_rate: Rate for bias updates in load balancing
    """
    
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 num_routed_experts: int = 256,
                 num_shared_experts: int = 1,
                 top_k: int = 8,
                 expert_capacity_factor: float = 1.25,
                 use_bias: bool = False,
                 activation: str = 'swish',
                 bias_update_rate: float = 1e-3,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_routed_experts = num_routed_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k
        self.expert_capacity_factor = expert_capacity_factor
        self.use_bias = use_bias
        self.activation = activation
        self.bias_update_rate = bias_update_rate
        
        # Validation
        if top_k > num_routed_experts:
            raise ValueError(f"top_k ({top_k}) cannot be larger than num_routed_experts ({num_routed_experts})")
        
        print(f"DeepSeek MoE Configuration:")
        print(f"  d_model: {d_model}, d_ff: {d_ff}")
        print(f"  routed_experts: {num_routed_experts}, shared_experts: {num_shared_experts}")
        print(f"  top_k: {top_k}, activation: {activation}")
        print(f"  bias_update_rate: {bias_update_rate}")
    
    def build(self, input_shape):
        """Build the DeepSeekMoE layer components"""
        super().build(input_shape)
        
        # Shared experts (always activated)
        self.shared_experts = []
        for i in range(self.num_shared_experts):
            expert = self._create_expert(name=f'shared_expert_{i}')
            self.shared_experts.append(expert)
        
        # Routed experts (selectively activated)
        self.routed_experts = []
        for i in range(self.num_routed_experts):
            expert = self._create_expert(name=f'routed_expert_{i}')
            self.routed_experts.append(expert)
        
        # Expert centroids for affinity-based routing
        self.expert_centroids = self.add_weight(
            name='expert_centroids',
            shape=(self.num_routed_experts, self.d_model),
            initializer='random_normal',
            trainable=True
        )
        
        # Bias for auxiliary-loss-free load balancing (non-trainable)
        self.expert_biases = self.add_weight(
            name='expert_biases',
            shape=(self.num_routed_experts,),
            initializer='zeros',
            trainable=False
        )
        
        # Load balancing tracking (non-trainable)
        self.expert_counts = self.add_weight(
            name='expert_counts',
            shape=(self.num_routed_experts,),
            initializer='zeros',
            trainable=False
        )
        
        self.total_tokens = self.add_weight(
            name='total_tokens',
            shape=(),
            initializer='zeros',
            trainable=False
        )
        
        print(f"DeepSeek MoE built with {self.num_shared_experts} shared + {self.num_routed_experts} routed experts")
        print(f"Total parameters: {self._count_parameters():,}")
    
    def _create_expert(self, name: str) -> tf.keras.Sequential:
        """Create individual expert network"""
        return tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.d_ff,
                activation=self.activation,
                use_bias=self.use_bias,
                name=f'{name}_up'
            ),
            tf.keras.layers.Dense(
                self.d_model,
                use_bias=self.use_bias,
                name=f'{name}_down'
            )
        ], name=name)
    
    def _count_parameters(self) -> int:
        """Count total parameters in the DeepSeekMoE layer"""
        # Shared expert parameters
        shared_params = self.num_shared_experts * (
            self.d_model * self.d_ff +  # Up projection
            self.d_ff * self.d_model    # Down projection
        )
        if self.use_bias:
            shared_params += self.num_shared_experts * (self.d_ff + self.d_model)
        
        # Routed expert parameters
        routed_params = self.num_routed_experts * (
            self.d_model * self.d_ff +  # Up projection
            self.d_ff * self.d_model    # Down projection
        )
        if self.use_bias:
            routed_params += self.num_routed_experts * (self.d_ff + self.d_model)
        
        # Expert centroids
        centroid_params = self.num_routed_experts * self.d_model
        
        return shared_params + routed_params + centroid_params
    
    def _compute_routing_weights(self, inputs: tf.Tensor, training: bool = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute routing weights using affinity-based routing with bias adjustment
        
        Args:
            inputs: [batch_size, seq_len, d_model]
            training: Training mode flag
            
        Returns:
            top_k_indices: [batch_size, seq_len, top_k]
            routing_weights: [batch_size, seq_len, top_k]
        """
        batch_size, seq_len, _ = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]
        
        # Flatten for routing computation
        inputs_flat = tf.reshape(inputs, [-1, self.d_model])  # [batch*seq, d_model]
        
        # Compute affinity scores with expert centroids
        affinity_scores = tf.matmul(inputs_flat, self.expert_centroids, transpose_b=True)
        # Shape: [batch*seq, num_routed_experts]
        
        # Add bias for load balancing (only affects top-k selection, not final weights)
        biased_scores = affinity_scores + self.expert_biases
        
        # Select top-k experts using biased scores
        _, top_k_indices = tf.nn.top_k(biased_scores, k=self.top_k)
        # Shape: [batch*seq, top_k]
        
        # Compute routing weights using original affinity scores (no bias)
        selected_scores = tf.gather(affinity_scores, top_k_indices, batch_dims=1)
        # Shape: [batch*seq, top_k]
        
        # Use sigmoid activation instead of softmax for better load balancing
        routing_weights = tf.nn.sigmoid(selected_scores)
        
        # Normalize weights to sum to 1
        routing_weights = routing_weights / (tf.reduce_sum(routing_weights, axis=-1, keepdims=True) + 1e-8)
        
        # Reshape back to original batch structure
        top_k_indices = tf.reshape(top_k_indices, [batch_size, seq_len, self.top_k])
        routing_weights = tf.reshape(routing_weights, [batch_size, seq_len, self.top_k])
        
        return top_k_indices, routing_weights
    
    def _update_expert_biases(self, expert_loads: tf.Tensor, target_load: float):
        """
        Update expert biases for load balancing without gradients
        
        Args:
            expert_loads: Current load for each expert [num_routed_experts]
            target_load: Target load per expert (total_tokens / num_experts)
        """
        # Calculate load errors
        load_errors = expert_loads - target_load
        
        # Update bias based on error sign (non-differentiable)
        bias_updates = -self.bias_update_rate * tf.sign(load_errors)
        
        # Apply bias updates
        self.expert_biases.assign_add(bias_updates)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass through DeepSeekMoE layer

        Args:
            inputs: [batch_size, seq_len, d_model]
            training: Training mode flag

        Returns:
            output: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]
        total_tokens = batch_size * seq_len

        # Step 1: Shared expert computation (always active)
        shared_output = tf.zeros_like(inputs)
        for shared_expert in self.shared_experts:
            shared_output += shared_expert(inputs)

        # Step 2: Routed expert computation
        top_k_indices, routing_weights = self._compute_routing_weights(inputs, training)

        # Initialize routed output
        routed_output = tf.zeros_like(inputs)

        # Track expert loads for bias updates
        if training:
            expert_loads = tf.zeros(self.num_routed_experts)

        # Step 3: Process each position in top-k
        for k in range(self.top_k):
            # Get expert indices and weights for this k position
            expert_indices = top_k_indices[:, :, k]  # [batch_size, seq_len]
            weights = routing_weights[:, :, k:k+1]    # [batch_size, seq_len, 1]

            # Process each expert
            for expert_id in range(self.num_routed_experts):
                # Create mask for tokens assigned to this expert
                expert_mask = tf.equal(expert_indices, expert_id)

                if tf.reduce_any(expert_mask):
                    # Extract tokens for this expert
                    expert_tokens = tf.boolean_mask(inputs, expert_mask)
                    expert_weights = tf.boolean_mask(weights, expert_mask)

                    if tf.shape(expert_tokens)[0] > 0:
                        # Process through expert
                        expert_output = self.routed_experts[expert_id](expert_tokens)

                        # Apply routing weights
                        weighted_output = expert_output * expert_weights

                        # Scatter back to full tensor
                        indices = tf.where(expert_mask)
                        routed_output = tf.tensor_scatter_nd_add(
                            routed_output,
                            indices,
                            weighted_output
                        )

                        # Update expert load tracking
                        if training:
                            expert_loads = tf.tensor_scatter_nd_add(
                                expert_loads,
                                [[expert_id]],
                                [tf.cast(tf.shape(expert_tokens)[0], tf.float32)]
                            )

        # Step 4: Update expert biases for load balancing
        if training:
            target_load = tf.cast(total_tokens, tf.float32) / tf.cast(self.num_routed_experts, tf.float32)
            self._update_expert_biases(expert_loads, target_load)

            # Update expert counts for monitoring
            self.expert_counts.assign_add(expert_loads)
            self.total_tokens.assign_add(tf.cast(total_tokens, tf.float32))

        # Step 5: Combine shared and routed outputs
        final_output = shared_output + routed_output

        return final_output

    def get_expert_utilization_stats(self) -> Dict[str, Any]:
        """Get expert utilization statistics for monitoring"""
        if self.total_tokens == 0:
            return {
                'expert_counts': np.zeros(self.num_routed_experts),
                'utilization': np.zeros(self.num_routed_experts),
                'utilization_variance': 0.0,
                'max_utilization': 0.0,
                'min_utilization': 0.0,
                'expert_biases': np.zeros(self.num_routed_experts),
                'total_tokens': 0
            }

        total_count = tf.reduce_sum(self.expert_counts)
        utilization = self.expert_counts / (total_count + 1e-8)

        return {
            'expert_counts': self.expert_counts.numpy(),
            'utilization': utilization.numpy(),
            'utilization_variance': tf.math.reduce_variance(utilization).numpy(),
            'max_utilization': tf.reduce_max(utilization).numpy(),
            'min_utilization': tf.reduce_min(utilization).numpy(),
            'expert_biases': self.expert_biases.numpy(),
            'total_tokens': self.total_tokens.numpy()
        }

    def reset_expert_stats(self):
        """Reset expert utilization statistics"""
        self.expert_counts.assign(tf.zeros_like(self.expert_counts))
        self.total_tokens.assign(0.0)
        self.expert_biases.assign(tf.zeros_like(self.expert_biases))

    def get_load_balance_metrics(self) -> Dict[str, float]:
        """Get comprehensive load balancing metrics"""
        stats = self.get_expert_utilization_stats()

        if stats['total_tokens'] == 0:
            return {'load_balance_score': 1.0, 'routing_entropy': 0.0}

        # Load balance score (1.0 = perfect, 0.0 = worst)
        ideal_utilization = 1.0 / self.num_routed_experts
        variance = stats['utilization_variance']
        max_variance = ideal_utilization * (1 - ideal_utilization)
        load_balance_score = max(0.0, 1.0 - (variance / max_variance))

        # Routing entropy (higher = more diverse)
        utilization = stats['utilization']
        utilization = utilization + 1e-8  # Avoid log(0)
        entropy = -np.sum(utilization * np.log(utilization))
        max_entropy = np.log(self.num_routed_experts)
        normalized_entropy = entropy / max_entropy

        return {
            'load_balance_score': float(load_balance_score),
            'routing_entropy': float(normalized_entropy),
            'utilization_cv': float(np.std(utilization) / (np.mean(utilization) + 1e-8))
        }

    def get_config(self):
        """Return layer configuration for serialization"""
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            'num_routed_experts': self.num_routed_experts,
            'num_shared_experts': self.num_shared_experts,
            'top_k': self.top_k,
            'expert_capacity_factor': self.expert_capacity_factor,
            'use_bias': self.use_bias,
            'activation': self.activation,
            'bias_update_rate': self.bias_update_rate
        })
        return config


# Testing and Validation
if __name__ == "__main__":
    print("🚀 Testing DeepSeek MoE Implementation...")

    # Test configuration (scaled down for testing)
    config = {
        'd_model': 512,
        'd_ff': 2048,
        'num_routed_experts': 32,  # Scaled down from 256 for testing
        'num_shared_experts': 1,
        'top_k': 4,  # Scaled down from 8 for testing
        'activation': 'swish',
        'bias_update_rate': 1e-3
    }

    # Create DeepSeek MoE layer
    moe = DeepSeekMoELayer(**config)

    # Test data
    batch_size, seq_len = 4, 64
    inputs = tf.random.normal([batch_size, seq_len, config['d_model']])

    # Build the layer
    moe.build(inputs.shape)

    print(f"\n📊 DeepSeek MoE Statistics:")
    print(f"  Total parameters: {moe._count_parameters():,}")
    print(f"  Shared experts: {config['num_shared_experts']}")
    print(f"  Routed experts: {config['num_routed_experts']}")
    print(f"  Top-k routing: {config['top_k']}")

    print("\n🔄 Testing Forward Pass...")
    # Reset counters for clean test
    moe.reset_expert_stats()

    # Test forward pass
    output = moe(inputs, training=True)
    print(f"  Input shape: {inputs.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output is finite: {tf.reduce_all(tf.math.is_finite(output))}")

    print("\n📈 Testing Load Balancing...")
    # Run multiple forward passes to test load balancing
    for i in range(20):
        batch = tf.random.normal([batch_size, seq_len, config['d_model']])
        _ = moe(batch, training=True)

    stats = moe.get_expert_utilization_stats()
    metrics = moe.get_load_balance_metrics()

    print(f"  Total tokens processed: {stats['total_tokens']:,.0f}")
    print(f"  Utilization variance: {stats['utilization_variance']:.6f}")
    print(f"  Load balance score: {metrics['load_balance_score']:.3f}")
    print(f"  Routing entropy: {metrics['routing_entropy']:.3f}")
    print(f"  Max utilization: {stats['max_utilization']:.4f}")
    print(f"  Min utilization: {stats['min_utilization']:.4f}")

    print("\n🎯 Testing Bias Updates...")
    initial_biases = moe.expert_biases.numpy().copy()

    # Run more training to see bias updates
    for _ in range(10):
        batch = tf.random.normal([batch_size, seq_len, config['d_model']])
        _ = moe(batch, training=True)

    final_biases = moe.expert_biases.numpy()
    bias_change = np.mean(np.abs(final_biases - initial_biases))
    print(f"  Average bias change: {bias_change:.6f}")
    print(f"  Bias range: [{np.min(final_biases):.4f}, {np.max(final_biases):.4f}]")

    # Success criteria
    success_criteria = {
        'output_shape_correct': output.shape == inputs.shape,
        'output_finite': tf.reduce_all(tf.math.is_finite(output)),
        'load_balance_reasonable': stats['utilization_variance'] < 0.01,
        'all_experts_used': stats['min_utilization'] > 0,
        'biases_updating': bias_change > 1e-6,
        'routing_diverse': metrics['routing_entropy'] > 0.5
    }

    print(f"\n✅ Success Criteria:")
    for criterion, passed in success_criteria.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {criterion}: {passed}")

    all_passed = all(success_criteria.values())
    if all_passed:
        print(f"\n🎉 All DeepSeek MoE tests passed successfully!")
        print(f"🎯 Load balance score: {metrics['load_balance_score']:.3f}")
        print(f"🧠 Ready for 256-expert scaling!")
    else:
        print(f"\n⚠️  Some tests failed - check implementation")

    print(f"💡 DeepSeek MoE provides expert specialization with auxiliary-loss-free load balancing!")
