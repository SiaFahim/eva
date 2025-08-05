"""
Comprehensive Testing Suite for Advanced MoE Architecture

This module provides comprehensive testing for all Phase 2 components of the
DeepSeek-V3 implementation, including:
- DeepSeekMoE layer functionality
- Auxiliary-loss-free load balancing
- Expert parallelism simulation
- Multi-Token Prediction (MTP)
- Integration testing

Author: Eva DeepSeek-V3 Project
Date: 2025-08-05
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import tensorflow as tf
import numpy as np
import pytest
import time
from typing import Dict, Any

# Import components to test
from components.moe.deepseek_moe import DeepSeekMoELayer
from components.moe.load_balancing import AuxiliaryLossFreeLoadBalancer
from components.moe.expert_parallelism import ExpertParallelismManager
from components.moe.multi_token_prediction import MultiTokenPredictionHead, MTPTrainingStrategy


class TestAdvancedMoE:
    """Comprehensive test suite for advanced MoE components"""
    
    def setup_method(self):
        """Set up test configuration"""
        self.config = {
            'd_model': 512,
            'd_ff': 2048,
            'num_routed_experts': 32,  # Scaled down for testing
            'num_shared_experts': 1,
            'top_k': 4,  # Scaled down for testing
            'batch_size': 2,
            'seq_len': 64,
            'vocab_size': 1000
        }
        
        # Create test data
        self.test_inputs = tf.random.normal([
            self.config['batch_size'],
            self.config['seq_len'],
            self.config['d_model']
        ])
        
        print(f"Test setup complete with config: {self.config}")
    
    def test_deepseek_moe_basic_functionality(self):
        """Test basic DeepSeekMoE functionality"""
        print("\n🧪 Testing DeepSeekMoE Basic Functionality...")
        
        moe = DeepSeekMoELayer(
            d_model=self.config['d_model'],
            d_ff=self.config['d_ff'],
            num_routed_experts=self.config['num_routed_experts'],
            num_shared_experts=self.config['num_shared_experts'],
            top_k=self.config['top_k']
        )
        
        # Build and test forward pass
        moe.build(self.test_inputs.shape)
        output = moe(self.test_inputs, training=True)
        
        # Assertions
        assert output.shape == self.test_inputs.shape, f"Output shape mismatch: {output.shape} vs {self.test_inputs.shape}"
        assert tf.reduce_all(tf.math.is_finite(output)), "Output contains non-finite values"
        
        print("✅ Basic functionality test passed")
    
    def test_auxiliary_loss_free_load_balancing(self):
        """Test auxiliary-loss-free load balancing mechanism"""
        print("\n🧪 Testing Auxiliary-Loss-Free Load Balancing...")
        
        moe = DeepSeekMoELayer(
            d_model=self.config['d_model'],
            d_ff=self.config['d_ff'],
            num_routed_experts=self.config['num_routed_experts'],
            num_shared_experts=self.config['num_shared_experts'],
            top_k=self.config['top_k']
        )
        moe.build(self.test_inputs.shape)
        
        # Reset expert statistics
        moe.reset_expert_stats()
        
        # Run multiple forward passes
        for _ in range(20):
            batch = tf.random.normal([
                self.config['batch_size'],
                self.config['seq_len'],
                self.config['d_model']
            ])
            _ = moe(batch, training=True)
        
        # Check load balancing
        stats = moe.get_expert_utilization_stats()
        metrics = moe.get_load_balance_metrics()
        
        # Assertions
        assert stats['utilization_variance'] < 0.05, f"High utilization variance: {stats['utilization_variance']}"
        assert stats['min_utilization'] > 0.001, f"Some experts completely unused: {stats['min_utilization']}"
        assert np.std(stats['expert_biases']) > 0, "Biases are not being updated"
        assert metrics['load_balance_score'] > 0.5, f"Poor load balance score: {metrics['load_balance_score']}"
        
        print(f"✅ Load balancing test passed (score: {metrics['load_balance_score']:.3f})")
    
    def test_expert_specialization(self):
        """Test that experts develop specialization"""
        print("\n🧪 Testing Expert Specialization...")
        
        moe = DeepSeekMoELayer(
            d_model=self.config['d_model'],
            d_ff=self.config['d_ff'],
            num_routed_experts=self.config['num_routed_experts'],
            num_shared_experts=self.config['num_shared_experts'],
            top_k=self.config['top_k']
        )
        moe.build(self.test_inputs.shape)
        
        # Create specialized inputs
        math_inputs = tf.random.normal([1, 32, self.config['d_model']]) * 0.1
        code_inputs = tf.random.normal([1, 32, self.config['d_model']]) * 0.1 + 1.0
        
        # Process different types of inputs
        math_output = moe(math_inputs, training=True)
        code_output = moe(code_inputs, training=True)
        
        # Verify outputs are different (indicating specialization)
        output_diff = tf.reduce_mean(tf.abs(math_output - code_output))
        
        assert output_diff > 0.1, f"Insufficient specialization: {output_diff}"
        
        print(f"✅ Expert specialization test passed (diff: {output_diff:.3f})")
    
    def test_routing_stability(self):
        """Test routing stability over time"""
        print("\n🧪 Testing Routing Stability...")
        
        moe = DeepSeekMoELayer(
            d_model=self.config['d_model'],
            d_ff=self.config['d_ff'],
            num_routed_experts=self.config['num_routed_experts'],
            num_shared_experts=self.config['num_shared_experts'],
            top_k=self.config['top_k']
        )
        moe.build(self.test_inputs.shape)
        
        # Get initial routing
        initial_indices, _ = moe._compute_routing_weights(self.test_inputs, training=False)
        
        # Run training for several steps
        for _ in range(10):
            _ = moe(self.test_inputs, training=True)
        
        # Get final routing
        final_indices, _ = moe._compute_routing_weights(self.test_inputs, training=False)
        
        # Check routing stability (shouldn't change drastically)
        routing_similarity = tf.reduce_mean(
            tf.cast(tf.equal(initial_indices, final_indices), tf.float32)
        )
        
        assert routing_similarity > 0.6, f"Routing too unstable: {routing_similarity}"
        
        print(f"✅ Routing stability test passed (similarity: {routing_similarity:.3f})")
    
    def test_load_balancer_standalone(self):
        """Test standalone load balancer functionality"""
        print("\n🧪 Testing Standalone Load Balancer...")
        
        balancer = AuxiliaryLossFreeLoadBalancer(
            num_experts=self.config['num_routed_experts'],
            update_rate=1e-2,
            adaptive_rate=True
        )
        
        # Simulate imbalanced loads
        for _ in range(20):
            imbalanced_loads = tf.random.gamma([self.config['num_routed_experts']], alpha=1.0, beta=1.0)
            imbalanced_loads = imbalanced_loads * tf.constant([2.0 if i < 8 else 0.5 for i in range(self.config['num_routed_experts'])])
            
            metrics = balancer.update_biases(imbalanced_loads)
        
        final_metrics = balancer.get_load_balance_metrics()
        
        # Test routing collapse detection
        collapsed_utilization = tf.constant([0.8, 0.15, 0.05] + [0.0] * (self.config['num_routed_experts'] - 3))
        collapse_detected, corrective_bias = balancer.detect_routing_collapse(collapsed_utilization)
        
        # Assertions
        assert final_metrics['load_balance_score'] > 0.3, f"Load balancer not working: {final_metrics['load_balance_score']}"
        assert collapse_detected, "Routing collapse not detected"
        assert tf.reduce_sum(tf.abs(corrective_bias)) > 0, "No corrective action generated"
        
        print(f"✅ Load balancer test passed (score: {final_metrics['load_balance_score']:.3f})")
    
    def test_expert_parallelism_simulation(self):
        """Test expert parallelism simulation"""
        print("\n🧪 Testing Expert Parallelism Simulation...")
        
        manager = ExpertParallelismManager(
            num_experts=self.config['num_routed_experts'],
            num_nodes=4,
            compression_enabled=True
        )
        
        # Create test routing data
        expert_assignments = tf.random.uniform(
            [self.config['batch_size'], self.config['seq_len'], self.config['top_k']], 
            maxval=self.config['num_routed_experts'], 
            dtype=tf.int32
        )
        routing_weights = tf.nn.softmax(tf.random.normal([
            self.config['batch_size'], 
            self.config['seq_len'], 
            self.config['top_k']
        ]), axis=-1)
        
        # Test all-to-all communication
        node_data = manager.simulate_all_to_all_routing(
            self.test_inputs, expert_assignments, routing_weights
        )
        
        # Test node processing
        node_outputs = {}
        for node_id in range(4):
            output = manager.simulate_node_processing(node_id, node_data[node_id], [])
            node_outputs[node_id] = output
        
        # Test load balance across nodes
        load_balance = manager.get_load_balance_across_nodes(expert_assignments)
        
        # Assertions
        assert len(node_data) == 4, f"Wrong number of nodes: {len(node_data)}"
        assert all(len(manager.get_node_experts(i)) > 0 for i in range(4)), "Some nodes have no experts"
        assert load_balance['load_cv'] < 3.0, f"Poor load balance across nodes: {load_balance['load_cv']}"
        
        print(f"✅ Expert parallelism test passed (load CV: {load_balance['load_cv']:.3f})")
    
    def test_mtp_functionality(self):
        """Test Multi-Token Prediction functionality"""
        print("\n🧪 Testing Multi-Token Prediction...")
        
        mtp_head = MultiTokenPredictionHead(
            vocab_size=self.config['vocab_size'],
            d_model=self.config['d_model'],
            num_predict_tokens=4
        )
        mtp_head.build(self.test_inputs.shape)
        
        # Test forward pass
        predictions = mtp_head(self.test_inputs)
        
        # Test generation
        def mock_model_forward(input_ids):
            batch_size, seq_len = tf.shape(input_ids)[0], tf.shape(input_ids)[1]
            return tf.random.normal([batch_size, seq_len, self.config['d_model']])
        
        input_ids = tf.random.uniform([1, 10], maxval=self.config['vocab_size'], dtype=tf.int32)
        generated_ids, stats = mtp_head.generate_with_mtp(
            input_ids=input_ids,
            model_forward_fn=mock_model_forward,
            max_length=30,
            acceptance_threshold=0.5
        )
        
        # Test training strategy
        training_strategy = MTPTrainingStrategy(mtp_loss_weight=0.1, warmup_steps=10)
        target_tokens = tf.random.uniform([self.config['batch_size'], self.config['seq_len'] + 4], 
                                         maxval=self.config['vocab_size'], dtype=tf.int32)
        mask = tf.ones([self.config['batch_size'], self.config['seq_len'] + 4])
        mtp_loss = training_strategy.compute_mtp_loss(predictions, target_tokens, mask)
        
        # Assertions
        expected_shape = [self.config['batch_size'], self.config['seq_len'], 4, self.config['vocab_size']]
        assert list(predictions.shape) == expected_shape, f"Wrong prediction shape: {predictions.shape}"
        assert tf.reduce_all(tf.math.is_finite(predictions)), "Predictions contain non-finite values"
        assert stats['generated_length'] > 0, "No tokens generated"
        assert stats['speedup_ratio'] >= 1.0, f"No speedup achieved: {stats['speedup_ratio']}"
        assert tf.math.is_finite(mtp_loss), "MTP loss is not finite"
        
        print(f"✅ MTP test passed (speedup: {stats['speedup_ratio']:.2f}x)")
    
    def test_integration_all_components(self):
        """Test integration of all components together"""
        print("\n🧪 Testing Integration of All Components...")
        
        # Create all components
        moe = DeepSeekMoELayer(
            d_model=self.config['d_model'],
            d_ff=self.config['d_ff'],
            num_routed_experts=16,  # Smaller for integration test
            num_shared_experts=1,
            top_k=2
        )
        
        mtp_head = MultiTokenPredictionHead(
            vocab_size=self.config['vocab_size'],
            d_model=self.config['d_model'],
            num_predict_tokens=2
        )
        
        # Build components
        moe.build(self.test_inputs.shape)
        mtp_head.build(self.test_inputs.shape)
        
        # Test integrated forward pass
        moe_output = moe(self.test_inputs, training=True)
        mtp_predictions = mtp_head(moe_output, training=True)
        
        # Assertions
        assert moe_output.shape == self.test_inputs.shape, "MoE output shape mismatch"
        assert tf.reduce_all(tf.math.is_finite(moe_output)), "MoE output not finite"
        assert tf.reduce_all(tf.math.is_finite(mtp_predictions)), "MTP predictions not finite"
        
        print("✅ Integration test passed")


def run_comprehensive_tests():
    """Run all comprehensive tests"""
    print("🚀 Running Comprehensive Advanced MoE Test Suite...")
    print("=" * 60)
    
    test_suite = TestAdvancedMoE()
    test_suite.setup_method()
    
    tests = [
        test_suite.test_deepseek_moe_basic_functionality,
        test_suite.test_auxiliary_loss_free_load_balancing,
        test_suite.test_expert_specialization,
        test_suite.test_routing_stability,
        test_suite.test_load_balancer_standalone,
        test_suite.test_expert_parallelism_simulation,
        test_suite.test_mtp_functionality,
        test_suite.test_integration_all_components
    ]
    
    passed_tests = 0
    failed_tests = 0
    
    for test in tests:
        try:
            test()
            passed_tests += 1
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {str(e)}")
            failed_tests += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results:")
    print(f"  ✅ Passed: {passed_tests}")
    print(f"  ❌ Failed: {failed_tests}")
    print(f"  📈 Success Rate: {passed_tests / (passed_tests + failed_tests) * 100:.1f}%")
    
    if failed_tests == 0:
        print("\n🎉 All tests passed! Advanced MoE implementation is ready!")
    else:
        print(f"\n⚠️  {failed_tests} tests failed. Please review implementation.")
    
    return passed_tests, failed_tests


if __name__ == "__main__":
    run_comprehensive_tests()
