"""
Basic Mixture-of-Experts (MoE) Implementation for DeepSeek-V3

This module implements a basic MoE layer that serves as the foundation for
DeepSeek-V3's advanced MoE architecture. It includes expert routing, load
balancing, and utilization tracking.

Key Features:
- Configurable number of experts (scales from 4 to 256+)
- Top-k expert selection with learnable routing
- Load balancing mechanisms to prevent expert collapse
- Expert utilization tracking for monitoring
- Efficient batched expert computation

Mathematical Foundation:
Traditional FFN: Y = FFN(X) for all tokens
MoE: Y = Σ(i=1 to k) w_i * Expert_i(X) where w_i = Router(X)
This allows specialization while maintaining computational efficiency.

Author: Eva DeepSeek-V3 Project
Date: 2025-08-03
"""

import tensorflow as tf
import numpy as np
from typing import Optional, Dict, List, Any
import math


class BasicMoELayer(tf.keras.layers.Layer):
    """
    Basic Mixture-of-Experts layer for DeepSeek-V3
    
    Implements core MoE functionality with expert routing and load balancing.
    Each expert is a feed-forward network that can specialize on different
    types of input patterns.
    
    Args:
        d_model: Model dimension (input/output size)
        d_ff: Feed-forward dimension (expert hidden size)
        num_experts: Number of expert networks (e.g., 4, 8, 16, 32)
        top_k: Number of experts to route each token to (typically 1 or 2)
        activation: Activation function for experts ('swish', 'relu', 'gelu')
        use_bias: Whether to use bias in expert networks
        expert_dropout: Dropout rate within expert networks
        routing_dropout: Dropout rate for routing probabilities
    """
    
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 num_experts: int = 8,
                 top_k: int = 2,
                 activation: str = 'swish',
                 use_bias: bool = False,
                 expert_dropout: float = 0.0,
                 routing_dropout: float = 0.0,
                 **kwargs):
        super().__init__(**kwargs)
        
        # Store configuration
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k
        self.activation = activation
        self.use_bias = use_bias
        self.expert_dropout = expert_dropout
        self.routing_dropout = routing_dropout
        
        # Validate configuration
        if top_k > num_experts:
            raise ValueError(f"top_k ({top_k}) cannot be larger than num_experts ({num_experts})")
        
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")
        
        print(f"MoE Configuration:")
        print(f"  d_model: {d_model}, d_ff: {d_ff}")
        print(f"  num_experts: {num_experts}, top_k: {top_k}")
        print(f"  activation: {activation}, expert_dropout: {expert_dropout}")
        print(f"  Theoretical speedup: {num_experts / top_k:.1f}x vs dense layer")
    
    def build(self, input_shape):
        """Build the MoE layer components"""
        super().build(input_shape)
        
        # Router network: maps input to expert selection probabilities
        # This is the "gating" mechanism that decides which experts to use
        self.router = self.add_weight(
            name='router_weight',
            shape=(self.d_model, self.num_experts),
            initializer='glorot_uniform',
            trainable=True
        )
        
        if self.use_bias:
            self.router_bias = self.add_weight(
                name='router_bias',
                shape=(self.num_experts,),
                initializer='zeros',
                trainable=True
            )
        
        # Expert networks: each expert is a 2-layer FFN
        # We create separate weights for each expert for maximum flexibility
        self.expert_weights_1 = []  # First layer weights (up projection)
        self.expert_weights_2 = []  # Second layer weights (down projection)
        self.expert_biases_1 = []   # First layer biases
        self.expert_biases_2 = []   # Second layer biases
        
        for i in range(self.num_experts):
            # First layer: d_model -> d_ff (up projection)
            w1 = self.add_weight(
                name=f'expert_{i}_weight_1',
                shape=(self.d_model, self.d_ff),
                initializer='glorot_uniform',
                trainable=True
            )
            self.expert_weights_1.append(w1)
            
            # Second layer: d_ff -> d_model (down projection)
            w2 = self.add_weight(
                name=f'expert_{i}_weight_2',
                shape=(self.d_ff, self.d_model),
                initializer='glorot_uniform',
                trainable=True
            )
            self.expert_weights_2.append(w2)
            
            if self.use_bias:
                b1 = self.add_weight(
                    name=f'expert_{i}_bias_1',
                    shape=(self.d_ff,),
                    initializer='zeros',
                    trainable=True
                )
                self.expert_biases_1.append(b1)
                
                b2 = self.add_weight(
                    name=f'expert_{i}_bias_2',
                    shape=(self.d_model,),
                    initializer='zeros',
                    trainable=True
                )
                self.expert_biases_2.append(b2)
        
        # Expert utilization tracking (non-trainable)
        # This helps us monitor load balancing and expert specialization
        self.expert_counts = self.add_weight(
            name='expert_counts',
            shape=(self.num_experts,),
            initializer='zeros',
            trainable=False
        )
        
        # Total token count for utilization calculation
        self.total_tokens = self.add_weight(
            name='total_tokens',
            shape=(),
            initializer='zeros',
            trainable=False
        )
        
        # Dropout layers
        if self.expert_dropout > 0:
            self.expert_dropout_layer = tf.keras.layers.Dropout(self.expert_dropout)
        else:
            self.expert_dropout_layer = None
            
        if self.routing_dropout > 0:
            self.routing_dropout_layer = tf.keras.layers.Dropout(self.routing_dropout)
        else:
            self.routing_dropout_layer = None
        
        print(f"MoE built with {len(self.expert_weights_1)} experts")
        print(f"Total parameters: {self._count_parameters():,}")
    
    def _count_parameters(self) -> int:
        """Count total parameters in the MoE layer"""
        # Router parameters
        router_params = self.d_model * self.num_experts
        if self.use_bias:
            router_params += self.num_experts
        
        # Expert parameters
        expert_params = self.num_experts * (
            self.d_model * self.d_ff +  # First layer
            self.d_ff * self.d_model    # Second layer
        )
        if self.use_bias:
            expert_params += self.num_experts * (self.d_ff + self.d_model)
        
        return router_params + expert_params
    
    def _apply_activation(self, x: tf.Tensor) -> tf.Tensor:
        """Apply the specified activation function"""
        if self.activation == 'swish':
            return tf.nn.swish(x)
        elif self.activation == 'relu':
            return tf.nn.relu(x)
        elif self.activation == 'gelu':
            return tf.nn.gelu(x)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
    
    def _compute_expert_output(self, inputs: tf.Tensor, expert_idx: int, training: Optional[bool] = None) -> tf.Tensor:
        """
        Compute output for a specific expert
        
        Args:
            inputs: Input tensor [num_tokens, d_model]
            expert_idx: Index of the expert to use
            training: Training mode flag
            
        Returns:
            output: Expert output [num_tokens, d_model]
        """
        # First layer: d_model -> d_ff
        hidden = tf.matmul(inputs, self.expert_weights_1[expert_idx])
        if self.use_bias:
            hidden = tf.nn.bias_add(hidden, self.expert_biases_1[expert_idx])
        
        # Apply activation
        hidden = self._apply_activation(hidden)
        
        # Apply dropout if in training mode
        if self.expert_dropout_layer is not None and training:
            hidden = self.expert_dropout_layer(hidden, training=training)
        
        # Second layer: d_ff -> d_model
        output = tf.matmul(hidden, self.expert_weights_2[expert_idx])
        if self.use_bias:
            output = tf.nn.bias_add(output, self.expert_biases_2[expert_idx])
        
        return output

    def _compute_routing_weights(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tuple:
        """
        Compute routing weights and expert assignments

        This is the core of the MoE mechanism - deciding which experts
        should process which tokens. We use a learned router network
        followed by top-k selection.

        Args:
            inputs: Input tensor [batch_size, seq_len, d_model]
            training: Training mode flag

        Returns:
            top_k_weights: Routing weights [batch_size * seq_len, top_k]
            top_k_indices: Expert indices [batch_size * seq_len, top_k]
            router_logits: Raw router outputs [batch_size * seq_len, num_experts]
        """
        batch_size, seq_len = tf.shape(inputs)[0], tf.shape(inputs)[1]

        # Flatten inputs for routing: [batch_size * seq_len, d_model]
        inputs_flat = tf.reshape(inputs, [-1, self.d_model])

        # Compute router logits: input -> expert selection scores
        router_logits = tf.matmul(inputs_flat, self.router)
        if self.use_bias:
            router_logits = tf.nn.bias_add(router_logits, self.router_bias)

        # Apply routing dropout if in training mode
        if self.routing_dropout_layer is not None and training:
            router_logits = self.routing_dropout_layer(router_logits, training=training)

        # Top-k expert selection
        # This selects the k most suitable experts for each token
        top_k_logits, top_k_indices = tf.nn.top_k(router_logits, k=self.top_k)

        # Convert logits to probabilities (softmax over selected experts)
        top_k_weights = tf.nn.softmax(top_k_logits, axis=-1)

        return top_k_weights, top_k_indices, router_logits

    def _update_expert_utilization(self, top_k_indices: tf.Tensor, training: Optional[bool] = None):
        """
        Update expert utilization statistics

        This tracks how often each expert is used, which is crucial for
        monitoring load balancing and preventing expert collapse.

        Args:
            top_k_indices: Expert indices [batch_size * seq_len, top_k]
            training: Training mode flag
        """
        if not training:
            return  # Only track during training

        num_tokens = tf.shape(top_k_indices)[0]

        # Count how many times each expert is selected
        for expert_idx in range(self.num_experts):
            # Count occurrences of this expert in the top-k selections
            expert_mask = tf.equal(top_k_indices, expert_idx)
            expert_count = tf.reduce_sum(tf.cast(expert_mask, tf.float32))

            # Update running count using assign_add on the variable
            new_count = self.expert_counts[expert_idx] + expert_count
            self.expert_counts[expert_idx].assign(new_count)

        # Update total token count
        new_total = self.total_tokens + tf.cast(num_tokens, tf.float32)
        self.total_tokens.assign(new_total)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass through MoE layer

        This orchestrates the entire MoE process:
        1. Compute routing weights (which experts for which tokens)
        2. Route tokens to selected experts
        3. Compute expert outputs
        4. Combine outputs with routing weights
        5. Update utilization statistics

        Args:
            inputs: Input tensor [batch_size, seq_len, d_model]
            training: Training mode flag

        Returns:
            output: MoE output [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = tf.shape(inputs)[0], tf.shape(inputs)[1]

        # Step 1: Compute routing weights and expert assignments
        top_k_weights, top_k_indices, router_logits = self._compute_routing_weights(inputs, training)

        # Step 2: Update expert utilization statistics
        self._update_expert_utilization(top_k_indices, training)

        # Step 3: Flatten inputs for expert processing
        inputs_flat = tf.reshape(inputs, [-1, self.d_model])  # [batch*seq, d_model]
        num_tokens = tf.shape(inputs_flat)[0]

        # Step 4: Initialize output tensor
        output_flat = tf.zeros_like(inputs_flat)  # [batch*seq, d_model]

        # Step 5: Process tokens through selected experts
        # We iterate through each expert and process all tokens assigned to it
        for expert_idx in range(self.num_experts):
            # Find tokens assigned to this expert (across all top-k positions)
            expert_mask = tf.reduce_any(tf.equal(top_k_indices, expert_idx), axis=-1)  # [batch*seq]

            # Get tokens for this expert
            expert_tokens = tf.boolean_mask(inputs_flat, expert_mask)  # [num_expert_tokens, d_model]

            # Skip if no tokens assigned to this expert
            if tf.shape(expert_tokens)[0] == 0:
                continue

            # Compute expert output
            expert_output = self._compute_expert_output(expert_tokens, expert_idx, training)

            # Get routing weights for this expert's tokens
            expert_token_indices = tf.where(expert_mask)[:, 0]  # Indices of tokens for this expert

            # For each token assigned to this expert, find its weight
            expert_weights_list = []
            for i in range(tf.shape(expert_token_indices)[0]):
                token_idx = expert_token_indices[i]
                # Find which position in top_k this expert appears for this token
                expert_positions = tf.where(tf.equal(top_k_indices[token_idx], expert_idx))
                if tf.shape(expert_positions)[0] > 0:
                    pos = expert_positions[0, 0]
                    weight = top_k_weights[token_idx, pos]
                    expert_weights_list.append(weight)
                else:
                    expert_weights_list.append(0.0)

            expert_weights = tf.stack(expert_weights_list)  # [num_expert_tokens]

            # Weight the expert output
            weighted_expert_output = expert_output * expert_weights[:, None]  # [num_expert_tokens, d_model]

            # Scatter the weighted output back to the full output tensor
            output_flat = tf.tensor_scatter_nd_add(
                output_flat,
                expert_token_indices[:, None],
                weighted_expert_output
            )

        # Step 6: Reshape output back to original shape
        output = tf.reshape(output_flat, [batch_size, seq_len, self.d_model])

        return output

    def get_expert_utilization(self) -> Dict[str, Any]:
        """
        Get current expert utilization statistics

        Returns:
            Dictionary with utilization metrics for monitoring load balancing
        """
        if self.total_tokens == 0:
            return {
                'expert_counts': np.zeros(self.num_experts),
                'utilization': np.zeros(self.num_experts),
                'variance': 0.0,
                'max_utilization': 0.0,
                'min_utilization': 0.0,
                'load_balance_score': 1.0,
                'total_tokens': 0
            }

        # Get current counts
        expert_counts = self.expert_counts.numpy()
        total_tokens = self.total_tokens.numpy()

        # Calculate utilization percentages
        utilization = expert_counts / (total_tokens + 1e-8)

        # Calculate load balancing metrics
        variance = np.var(utilization)
        max_util = np.max(utilization)
        min_util = np.min(utilization)

        # Load balance score: 1.0 = perfect balance, 0.0 = worst case
        ideal_utilization = 1.0 / self.num_experts
        load_balance_score = 1.0 - (variance / (ideal_utilization * (1 - ideal_utilization)))
        load_balance_score = max(0.0, min(1.0, load_balance_score))

        return {
            'expert_counts': expert_counts,
            'utilization': utilization,
            'variance': float(variance),
            'max_utilization': float(max_util),
            'min_utilization': float(min_util),
            'load_balance_score': float(load_balance_score),
            'total_tokens': float(total_tokens)
        }

    def reset_expert_counts(self):
        """Reset expert utilization counters"""
        self.expert_counts.assign(tf.zeros_like(self.expert_counts))
        self.total_tokens.assign(0.0)

    def get_routing_entropy(self, inputs: tf.Tensor) -> float:
        """
        Calculate routing entropy to measure expert diversity

        Higher entropy indicates more diverse expert usage,
        lower entropy suggests routing collapse.

        Args:
            inputs: Input tensor [batch_size, seq_len, d_model]

        Returns:
            entropy: Routing entropy (higher = more diverse)
        """
        _, _, router_logits = self._compute_routing_weights(inputs, training=False)

        # Convert to probabilities
        router_probs = tf.nn.softmax(router_logits, axis=-1)

        # Calculate entropy: -Σ p_i * log(p_i)
        log_probs = tf.nn.log_softmax(router_logits, axis=-1)
        entropy = -tf.reduce_mean(tf.reduce_sum(router_probs * log_probs, axis=-1))

        return entropy.numpy()

    def get_config(self):
        """Return layer configuration for serialization"""
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            'num_experts': self.num_experts,
            'top_k': self.top_k,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'expert_dropout': self.expert_dropout,
            'routing_dropout': self.routing_dropout
        })
        return config


# Comprehensive MoE Testing
if __name__ == "__main__":
    print("🚀 Testing Complete MoE Implementation...")

    # Test configuration
    config = {
        'd_model': 256,
        'd_ff': 1024,
        'num_experts': 8,
        'top_k': 2,
        'activation': 'swish'
    }

    # Create MoE layer
    moe = BasicMoELayer(**config)

    # Test data
    batch_size, seq_len = 4, 32
    inputs = tf.random.normal([batch_size, seq_len, config['d_model']])

    # Build the layer
    moe.build(inputs.shape)

    print(f"\n📊 MoE Statistics:")
    print(f"  Total parameters: {moe._count_parameters():,}")
    print(f"  Theoretical speedup: {config['num_experts'] / config['top_k']:.1f}x vs dense")

    print("\n🔄 Testing Forward Pass...")
    # Reset counters for clean test
    moe.reset_expert_counts()

    # Test forward pass
    output = moe(inputs, training=True)
    print(f"  Input shape: {inputs.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output is finite: {tf.reduce_all(tf.math.is_finite(output))}")

    print("\n📈 Testing Expert Utilization...")
    # Run multiple forward passes to build utilization statistics
    for _ in range(10):
        batch = tf.random.normal([batch_size, seq_len, config['d_model']])
        _ = moe(batch, training=True)

    utilization = moe.get_expert_utilization()
    print(f"  Total tokens processed: {utilization['total_tokens']:,.0f}")
    print(f"  Expert utilization variance: {utilization['variance']:.4f}")
    print(f"  Load balance score: {utilization['load_balance_score']:.3f}")
    print(f"  Max utilization: {utilization['max_utilization']:.3f}")
    print(f"  Min utilization: {utilization['min_utilization']:.3f}")

    print("\n🎯 Testing Routing Diversity...")
    entropy = moe.get_routing_entropy(inputs)
    max_entropy = math.log(config['num_experts'])
    print(f"  Routing entropy: {entropy:.3f} / {max_entropy:.3f}")
    print(f"  Entropy ratio: {entropy / max_entropy:.3f}")

    print("\n🧪 Testing Load Balancing...")
    # Test with diverse input patterns
    moe.reset_expert_counts()

    for i in range(config['num_experts']):
        # Create inputs with different patterns to encourage expert specialization
        pattern = tf.random.normal([2, 16, config['d_model']]) * (i + 1) * 0.1
        _ = moe(pattern, training=True)

    final_utilization = moe.get_expert_utilization()
    print(f"  Final load balance score: {final_utilization['load_balance_score']:.3f}")
    print(f"  Utilization range: [{final_utilization['min_utilization']:.3f}, {final_utilization['max_utilization']:.3f}]")

    # Check success criteria
    success_criteria = {
        'output_shape_correct': output.shape == inputs.shape,
        'output_finite': tf.reduce_all(tf.math.is_finite(output)),
        'load_balance_reasonable': final_utilization['variance'] < 0.1,
        'all_experts_used': final_utilization['min_utilization'] > 0,
        'routing_diverse': entropy / max_entropy > 0.5
    }

    print(f"\n✅ Success Criteria:")
    for criterion, passed in success_criteria.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {criterion}: {passed}")

    all_passed = all(success_criteria.values())
    if all_passed:
        print(f"\n🎉 All MoE tests passed successfully!")
        print(f"🎯 Load balance score: {final_utilization['load_balance_score']:.3f}")
    else:
        print(f"\n⚠️  Some tests failed - check implementation")

    print(f"💡 MoE provides {config['num_experts'] / config['top_k']:.1f}x theoretical speedup over dense layers!")
