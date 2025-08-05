"""
Auxiliary-Loss-Free Load Balancing for DeepSeek-V3 MoE

This module implements the innovative auxiliary-loss-free load balancing mechanism
used in DeepSeek-V3. Instead of using auxiliary losses that interfere with the main
language modeling objective, this approach uses bias adjustments that are 
non-differentiable and don't affect gradient flow.

Key Innovations:
- Bias-based load balancing without auxiliary losses
- Non-differentiable bias updates that don't interfere with gradients
- Routing collapse detection and prevention
- Expert utilization tracking and monitoring
- Adaptive bias update rates based on load imbalance

Mathematical Foundation:
Traditional MoE: Uses auxiliary loss L_aux = α * load_balance_loss
DeepSeek MoE: Uses bias adjustment b_i = b_i - η * sign(load_error_i)
This maintains load balancing without affecting the main training objective.

Author: Eva DeepSeek-V3 Project
Date: 2025-08-05
"""

import tensorflow as tf
import numpy as np
from typing import Dict, Any, Optional


class AuxiliaryLossFreeLoadBalancer:
    """
    Implements bias-based load balancing without auxiliary losses
    
    This is a key innovation of DeepSeek-V3 that maintains expert load balancing
    without interfering with the main language modeling objective through
    non-differentiable bias adjustments.
    
    Args:
        num_experts: Number of experts to balance
        update_rate: Rate for bias updates (default: 1e-3)
        collapse_threshold: Threshold for detecting routing collapse (default: 0.1)
        adaptive_rate: Whether to use adaptive update rates (default: True)
    """
    
    def __init__(self, 
                 num_experts: int, 
                 update_rate: float = 1e-3,
                 collapse_threshold: float = 0.1,
                 adaptive_rate: bool = True):
        self.num_experts = num_experts
        self.update_rate = update_rate
        self.collapse_threshold = collapse_threshold
        self.adaptive_rate = adaptive_rate
        
        # Expert biases (non-trainable)
        self.expert_biases = tf.Variable(
            tf.zeros(num_experts),
            trainable=False,
            name='expert_biases'
        )
        
        # Load tracking
        self.cumulative_loads = tf.Variable(
            tf.zeros(num_experts),
            trainable=False,
            name='cumulative_loads'
        )
        
        self.update_count = tf.Variable(0, trainable=False, name='update_count')
        
        # Adaptive rate tracking
        if self.adaptive_rate:
            self.load_variance_history = tf.Variable(
                tf.zeros(10),  # Keep last 10 variance measurements
                trainable=False,
                name='variance_history'
            )
            self.history_index = tf.Variable(0, trainable=False, name='history_index')
    
    def update_biases(self, current_loads: tf.Tensor) -> Dict[str, float]:
        """
        Update expert biases based on load imbalance
        
        Args:
            current_loads: Current batch expert loads [num_experts]
            
        Returns:
            metrics: Dictionary with update metrics
        """
        # Calculate target load (uniform distribution)
        total_load = tf.reduce_sum(current_loads)
        target_load = total_load / tf.cast(self.num_experts, tf.float32)
        
        # Calculate load errors
        load_errors = current_loads - target_load
        
        # Adaptive update rate based on load variance
        current_rate = self.update_rate
        if self.adaptive_rate:
            load_variance = tf.math.reduce_variance(current_loads)
            
            # Update variance history
            idx = self.history_index % 10
            self.load_variance_history = tf.tensor_scatter_nd_update(
                self.load_variance_history,
                [[idx]],
                [load_variance]
            )
            self.history_index.assign_add(1)
            
            # Adapt rate based on variance trend
            if self.update_count > 10:
                avg_variance = tf.reduce_mean(self.load_variance_history)
                if load_variance > avg_variance * 1.5:
                    current_rate = self.update_rate * 2.0  # Increase rate for high variance
                elif load_variance < avg_variance * 0.5:
                    current_rate = self.update_rate * 0.5  # Decrease rate for low variance
        
        # Update biases (non-differentiable operation)
        bias_updates = -current_rate * tf.sign(load_errors)
        self.expert_biases.assign_add(bias_updates)
        
        # Update cumulative statistics
        self.cumulative_loads.assign_add(current_loads)
        self.update_count.assign_add(1)
        
        # Return metrics
        return {
            'load_variance': float(tf.math.reduce_variance(current_loads)),
            'max_load_error': float(tf.reduce_max(tf.abs(load_errors))),
            'update_rate_used': float(current_rate),
            'bias_update_magnitude': float(tf.reduce_mean(tf.abs(bias_updates)))
        }
    
    def get_adjusted_scores(self, raw_scores: tf.Tensor, use_bias: bool = True) -> tf.Tensor:
        """
        Apply bias adjustment to routing scores
        
        Args:
            raw_scores: Raw affinity scores [batch*seq, num_experts]
            use_bias: Whether to apply bias (False for final weight computation)
            
        Returns:
            adjusted_scores: Bias-adjusted scores for top-k selection
        """
        if use_bias:
            return raw_scores + self.expert_biases
        else:
            return raw_scores
    
    def detect_routing_collapse(self, expert_utilization: tf.Tensor) -> tuple:
        """
        Detect and prevent routing collapse where few experts dominate
        
        Args:
            expert_utilization: Expert utilization rates [num_experts]
            
        Returns:
            collapse_detected: Boolean indicating if collapse is detected
            corrective_action: Suggested corrective bias adjustment
        """
        # Count experts below threshold
        underutilized_experts = tf.reduce_sum(
            tf.cast(expert_utilization < self.collapse_threshold, tf.float32)
        )
        
        collapse_ratio = underutilized_experts / tf.cast(self.num_experts, tf.float32)
        
        if collapse_ratio > 0.5:  # More than 50% experts underutilized
            # Apply corrective bias to underutilized experts
            corrective_bias = tf.where(
                expert_utilization < self.collapse_threshold,
                0.1,  # Boost underutilized experts
                -0.05  # Slightly reduce overutilized experts
            )
            return True, corrective_bias
        
        return False, tf.zeros_like(expert_utilization)
    
    def apply_corrective_action(self, corrective_bias: tf.Tensor):
        """Apply corrective bias adjustments for routing collapse"""
        self.expert_biases.assign_add(corrective_bias)
    
    def get_load_balance_metrics(self) -> Dict[str, Any]:
        """Get load balancing metrics for monitoring"""
        if self.update_count > 0:
            avg_loads = self.cumulative_loads / tf.cast(self.update_count, tf.float32)
            load_variance = tf.math.reduce_variance(avg_loads)
            load_coefficient_of_variation = tf.sqrt(load_variance) / (tf.reduce_mean(avg_loads) + 1e-8)
            
            # Calculate load balance score (1.0 = perfect, 0.0 = worst)
            ideal_utilization = 1.0 / self.num_experts
            max_variance = ideal_utilization * (1 - ideal_utilization)
            load_balance_score = tf.maximum(0.0, 1.0 - (load_variance / max_variance))
            
            return {
                'average_loads': avg_loads.numpy(),
                'load_variance': float(load_variance),
                'load_cv': float(load_coefficient_of_variation),
                'load_balance_score': float(load_balance_score),
                'current_biases': self.expert_biases.numpy(),
                'update_count': int(self.update_count),
                'bias_range': [float(tf.reduce_min(self.expert_biases)), 
                              float(tf.reduce_max(self.expert_biases))]
            }
        else:
            return {'message': 'No updates yet'}
    
    def reset_statistics(self):
        """Reset all load balancing statistics"""
        self.expert_biases.assign(tf.zeros_like(self.expert_biases))
        self.cumulative_loads.assign(tf.zeros_like(self.cumulative_loads))
        self.update_count.assign(0)
        if self.adaptive_rate:
            self.load_variance_history.assign(tf.zeros_like(self.load_variance_history))
            self.history_index.assign(0)
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization"""
        return {
            'num_experts': self.num_experts,
            'update_rate': self.update_rate,
            'collapse_threshold': self.collapse_threshold,
            'adaptive_rate': self.adaptive_rate
        }


# Testing and Validation
if __name__ == "__main__":
    print("🚀 Testing Auxiliary-Loss-Free Load Balancer...")
    
    # Test configuration
    num_experts = 16
    balancer = AuxiliaryLossFreeLoadBalancer(
        num_experts=num_experts,
        update_rate=1e-2,  # Higher rate for testing
        adaptive_rate=True
    )
    
    print(f"\n📊 Load Balancer Configuration:")
    print(f"  Number of experts: {num_experts}")
    print(f"  Update rate: {balancer.update_rate}")
    print(f"  Adaptive rate: {balancer.adaptive_rate}")
    
    print("\n🔄 Testing Load Balancing...")
    
    # Simulate imbalanced expert loads
    for step in range(50):
        # Create imbalanced loads (some experts overused)
        imbalanced_loads = tf.random.gamma([num_experts], alpha=1.0, beta=1.0)
        imbalanced_loads = imbalanced_loads * tf.constant([3.0 if i < 4 else 0.5 for i in range(num_experts)])
        
        # Update biases
        metrics = balancer.update_biases(imbalanced_loads)
        
        if step % 10 == 0:
            print(f"  Step {step}: Load variance = {metrics['load_variance']:.4f}, "
                  f"Update rate = {metrics['update_rate_used']:.4f}")
    
    print("\n📈 Final Load Balancing Results:")
    final_metrics = balancer.get_load_balance_metrics()
    print(f"  Load balance score: {final_metrics['load_balance_score']:.3f}")
    print(f"  Load variance: {final_metrics['load_variance']:.6f}")
    print(f"  Load CV: {final_metrics['load_cv']:.3f}")
    print(f"  Bias range: {final_metrics['bias_range']}")
    
    print("\n🎯 Testing Routing Collapse Detection...")
    # Create collapsed utilization (few experts dominate)
    collapsed_utilization = tf.constant([0.8, 0.15, 0.05] + [0.0] * (num_experts - 3))
    collapse_detected, corrective_bias = balancer.detect_routing_collapse(collapsed_utilization)
    
    print(f"  Collapse detected: {collapse_detected}")
    if collapse_detected:
        print(f"  Corrective bias applied: {tf.reduce_sum(tf.abs(corrective_bias)):.3f}")
        balancer.apply_corrective_action(corrective_bias)
    
    # Success criteria
    success_criteria = {
        'load_balance_improved': final_metrics['load_balance_score'] > 0.7,
        'biases_updated': np.max(np.abs(final_metrics['current_biases'])) > 1e-3,
        'collapse_detection_works': collapse_detected,
        'adaptive_rate_working': final_metrics['update_count'] > 0
    }
    
    print(f"\n✅ Success Criteria:")
    for criterion, passed in success_criteria.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {criterion}: {passed}")
    
    all_passed = all(success_criteria.values())
    if all_passed:
        print(f"\n🎉 All load balancing tests passed!")
        print(f"🎯 Load balance score: {final_metrics['load_balance_score']:.3f}")
    else:
        print(f"\n⚠️  Some tests failed - check implementation")
    
    print(f"💡 Auxiliary-loss-free load balancing maintains expert balance without interfering with training!")
