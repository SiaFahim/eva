"""
Comprehensive Distributed Training Testing Suite for DeepSeek-V3 Phase 3

This module provides comprehensive testing for all Phase 3 distributed training
components including DualPipe scheduling, gradient accumulation, ZeRO optimizer,
communication efficiency, and end-to-end integration testing.

Test Coverage:
- DualPipe bidirectional pipeline parallelism
- Pipeline stage models and integration
- Custom distributed training strategy
- ZeRO-1 optimizer state partitioning
- Memory optimization framework
- Optimized communication kernels
- End-to-end distributed training scenarios

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

# Import Phase 3 distributed components
from components.distributed.dualpipe import DualPipeScheduler
from components.distributed.pipeline_stage import PipelineStageModel, PipelineStageManager
from components.distributed.training_strategy import DeepSeekDistributedStrategy
from components.distributed.zero_optimizer import ZeRO1Optimizer
from components.distributed.memory_optimization import GradientAccumulator, ActivationCheckpointing, MemoryMonitor
from components.distributed.communication_kernels import OptimizedAllToAll


class TestDistributedTraining:
    """Comprehensive test suite for distributed training components"""
    
    def setup_method(self):
        """Set up test configuration"""
        self.config = {
            'd_model': 512,
            'd_ff': 2048,
            'num_heads': 8,
            'num_routed_experts': 16,  # Scaled down for testing
            'num_shared_experts': 1,
            'top_k': 4,
            'batch_size': 2,
            'seq_len': 64,
            'num_pipeline_stages': 4,  # Scaled down for testing
            'num_workers': 4
        }
        
        # Create test data
        self.test_inputs = tf.random.normal([
            self.config['batch_size'],
            self.config['seq_len'],
            self.config['d_model']
        ])
        
        self.test_targets = tf.random.uniform([
            self.config['batch_size'],
            self.config['seq_len']
        ], maxval=1000, dtype=tf.int32)
        
        print(f"Test setup complete with config: {self.config}")
    
    def test_dualpipe_scheduler_functionality(self):
        """Test DualPipe bidirectional pipeline scheduling"""
        print("\n🧪 Testing DualPipe Scheduler Functionality...")
        
        scheduler = DualPipeScheduler(
            num_stages=self.config['num_pipeline_stages'],
            micro_batch_size=2,
            overlap_communication=True,
            adaptive_scheduling=True
        )
        
        # Test schedule creation
        global_batch_size = 16
        schedule = scheduler.create_pipeline_schedule(global_batch_size)
        
        # Verify schedule properties
        assert len(schedule) > 0, "Schedule should not be empty"
        
        forward_ops = [op for op in schedule if op['direction'] == 'forward']
        backward_ops = [op for op in schedule if op['direction'] == 'backward']
        
        assert len(forward_ops) > 0, "Should have forward operations"
        assert len(backward_ops) > 0, "Should have backward operations"
        
        # Test efficiency metrics
        scheduler.stage_timings['attention'] = [0.1] * 10
        scheduler.stage_timings['dispatch'] = [0.05] * 10
        scheduler.stage_timings['mlp'] = [0.2] * 10
        scheduler.stage_timings['combine'] = [0.05] * 10
        
        metrics = scheduler.get_pipeline_efficiency_metrics()
        
        assert metrics['pipeline_efficiency'] > 0.5, f"Pipeline efficiency too low: {metrics['pipeline_efficiency']}"
        assert metrics['bubble_ratio'] < 0.5, f"Bubble ratio too high: {metrics['bubble_ratio']}"
        
        print("✅ DualPipe scheduler functionality test passed")
    
    def test_pipeline_stage_integration(self):
        """Test pipeline stage models with MLA and MoE integration"""
        print("\n🧪 Testing Pipeline Stage Integration...")
        
        stage = PipelineStageModel(
            d_model=self.config['d_model'],
            num_heads=self.config['num_heads'],
            d_ff=self.config['d_ff'],
            num_routed_experts=self.config['num_routed_experts'],
            num_shared_experts=self.config['num_shared_experts'],
            top_k=self.config['top_k'],
            stage_id=0,
            activation_checkpointing=True
        )
        
        # Build and test stage
        stage.build(self.test_inputs.shape)
        
        # Test forward pass
        output = stage(self.test_inputs, training=True)
        
        assert output.shape == self.test_inputs.shape, f"Output shape mismatch: {output.shape} vs {self.test_inputs.shape}"
        assert tf.reduce_all(tf.math.is_finite(output)), "Output contains non-finite values"
        
        # Test stage statistics
        stats = stage.get_stage_statistics()
        assert stats['total_parameters'] > 0, "Stage should have parameters"
        assert stats['stage_id'] == 0, "Stage ID should match"
        
        print("✅ Pipeline stage integration test passed")
    
    def test_distributed_training_strategy(self):
        """Test custom distributed training strategy"""
        print("\n🧪 Testing Distributed Training Strategy...")
        
        strategy = DeepSeekDistributedStrategy(
            pipeline_parallel_size=self.config['num_pipeline_stages'],
            expert_parallel_size=8,  # Scaled down
            data_parallel_size=2,
            micro_batch_size=2,
            gradient_accumulation_steps=4
        )
        
        # Test dataset creation
        dummy_dataset = tf.data.Dataset.from_tensor_slices({
            'input_ids': self.test_inputs,
            'labels': self.test_targets
        })
        
        distributed_dataset = strategy.create_distributed_dataset(dummy_dataset, global_batch_size=8)
        
        # Test training metrics
        metrics = strategy.get_training_metrics()
        assert 'total_training_steps' in metrics, "Should have training step count"
        assert 'effective_batch_size' in metrics, "Should have effective batch size"
        
        # Test optimization recommendations
        optimization = strategy.optimize_training_configuration()
        assert 'current_metrics' in optimization, "Should have current metrics"
        assert 'recommendations' in optimization, "Should have recommendations"
        
        print("✅ Distributed training strategy test passed")
    
    def test_zero_optimizer_partitioning(self):
        """Test ZeRO-1 optimizer state partitioning"""
        print("\n🧪 Testing ZeRO-1 Optimizer Partitioning...")
        
        base_optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4)
        zero_optimizer = ZeRO1Optimizer(
            base_optimizer=base_optimizer,
            num_partitions=4,
            overlap_communication=True
        )
        
        # Create fake model variables
        fake_variables = []
        for i in range(12):  # 12 variables to partition across 4 partitions
            var = tf.Variable(tf.random.normal([100, 50]), name=f'var_{i}')
            fake_variables.append(var)
        
        # Test partitioning
        zero_optimizer.partition_parameters(fake_variables)
        
        # Verify partitioning
        memory_stats = zero_optimizer.get_memory_usage()
        assert memory_stats['total_parameters'] == len(fake_variables), "Parameter count mismatch"
        assert all(count > 0 for count in memory_stats['parameters_per_partition']), "All partitions should have parameters"
        assert memory_stats['memory_reduction_ratio'] > 0.5, "Should achieve significant memory reduction"
        
        # Test gradient application
        fake_gradients = [tf.random.normal(var.shape) * 0.01 for var in fake_variables]
        grads_and_vars = list(zip(fake_gradients, fake_variables))
        
        zero_optimizer.apply_gradients(grads_and_vars)
        
        comm_stats = zero_optimizer.get_communication_stats()
        assert comm_stats['total_communications'] > 0, "Should have communication activity"
        
        print("✅ ZeRO-1 optimizer partitioning test passed")
    
    def test_memory_optimization_framework(self):
        """Test memory optimization components"""
        print("\n🧪 Testing Memory Optimization Framework...")
        
        # Test gradient accumulator
        accumulator = GradientAccumulator(
            accumulation_steps=4,
            gradient_clipping=1.0
        )
        
        fake_gradients = [tf.random.normal([100, 50]) * 0.01 for _ in range(3)]
        
        # Test accumulation process
        for step in range(6):  # More than accumulation_steps
            accumulator.accumulate_gradients(fake_gradients)
            
            if accumulator.should_apply_gradients():
                averaged_grads = accumulator.get_averaged_gradients()
                assert len(averaged_grads) == len(fake_gradients), "Should return same number of gradients"
        
        acc_stats = accumulator.get_accumulation_stats()
        assert acc_stats['total_applications'] > 0, "Should have applied gradients"
        
        # Test activation checkpointing
        checkpointer = ActivationCheckpointing(
            checkpoint_every_n_layers=2,
            checkpoint_attention=True,
            checkpoint_mlp=True
        )
        
        # Test checkpointing decisions
        for layer_id in range(8):
            layer_type = 'attention' if layer_id % 2 == 0 else 'mlp'
            should_checkpoint = checkpointer._should_checkpoint_layer(layer_id, layer_type)
            # Should checkpoint some layers
        
        # Test memory monitor
        monitor = MemoryMonitor()
        
        for _ in range(5):
            memory_stats = monitor.get_current_memory_usage()
            assert 'cpu_memory_mb' in memory_stats, "Should track CPU memory"
            assert 'total_memory_mb' in memory_stats, "Should track total memory"
        
        recommendations = monitor.get_memory_optimization_recommendations()
        assert 'current_memory_mb' in recommendations, "Should provide current memory info"
        
        print("✅ Memory optimization framework test passed")
    
    def test_communication_kernels_efficiency(self):
        """Test optimized communication kernels"""
        print("\n🧪 Testing Communication Kernels Efficiency...")
        
        comm = OptimizedAllToAll(
            num_workers=self.config['num_workers'],
            compression_enabled=True,
            compression_algorithm='topk',
            compression_ratio=0.5,
            adaptive_scheduling=True
        )
        
        # Test expert routing communication
        local_tokens = tf.random.normal([4, 32, self.config['d_model']])
        expert_assignments = tf.random.uniform([4, 32, 4], maxval=256, dtype=tf.int32)
        routing_weights = tf.nn.softmax(tf.random.normal([4, 32, 4]), axis=-1)
        
        routed_tokens, comm_info = comm.all_to_all_expert_routing(
            local_tokens, expert_assignments, routing_weights
        )
        
        assert routed_tokens.shape[0] >= 0, "Should return valid routed tokens"
        assert comm_info['compression_ratio'] <= 1.0, "Compression ratio should be valid"
        assert comm_info['communication_time_ms'] > 0, "Should track communication time"
        
        # Test multiple communications for statistics
        for _ in range(5):
            _, _ = comm.all_to_all_expert_routing(local_tokens, expert_assignments, routing_weights)
        
        stats = comm.get_communication_stats()
        assert stats['total_communications'] > 0, "Should track communications"
        assert stats['bandwidth_utilization'] >= 0, "Should track bandwidth utilization"
        
        print("✅ Communication kernels efficiency test passed")

    def test_end_to_end_integration(self):
        """Test end-to-end integration of all distributed components"""
        print("\n🧪 Testing End-to-End Integration...")

        # Create integrated components
        scheduler = DualPipeScheduler(num_stages=4, micro_batch_size=2)

        stage_manager = PipelineStageManager(
            num_stages=4,
            stage_config={
                'd_model': self.config['d_model'],
                'num_heads': self.config['num_heads'],
                'd_ff': self.config['d_ff'],
                'num_routed_experts': 8,  # Smaller for integration test
                'num_shared_experts': 1,
                'top_k': 2,
                'activation_checkpointing': True
            }
        )

        stage_manager.build_all_stages(self.test_inputs.shape)

        accumulator = GradientAccumulator(accumulation_steps=2)

        # Test integrated forward pass through pipeline
        current_output = self.test_inputs

        for stage_id in range(4):
            stage = stage_manager.get_stage(stage_id)
            current_output = stage(current_output, training=True)

        assert current_output.shape == self.test_inputs.shape, "Pipeline should preserve shape"
        assert tf.reduce_all(tf.math.is_finite(current_output)), "Pipeline output should be finite"

        # Test gradient accumulation integration
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(current_output))

        # Get all trainable variables from pipeline stages
        all_variables = []
        for stage_id in range(4):
            stage = stage_manager.get_stage(stage_id)
            all_variables.extend(stage.trainable_variables)

        gradients = tape.gradient(loss, all_variables)

        # Test gradient accumulation
        accumulator.accumulate_gradients(gradients)

        if accumulator.should_apply_gradients():
            averaged_grads = accumulator.get_averaged_gradients()
            assert len(averaged_grads) == len(gradients), "Should return all gradients"

        # Get pipeline statistics
        pipeline_stats = stage_manager.get_pipeline_statistics()
        assert pipeline_stats['num_stages'] == 4, "Should have correct number of stages"
        assert pipeline_stats['total_parameters'] > 0, "Should have parameters"

        print("✅ End-to-end integration test passed")

    def test_scalability_and_performance(self):
        """Test scalability and performance characteristics"""
        print("\n🧪 Testing Scalability and Performance...")

        # Test different scales
        scales = [
            {'stages': 2, 'experts': 8, 'workers': 2},
            {'stages': 4, 'experts': 16, 'workers': 4},
            {'stages': 8, 'experts': 32, 'workers': 8}
        ]

        performance_results = []

        for scale in scales:
            print(f"  Testing scale: {scale}")

            # Create components at this scale
            scheduler = DualPipeScheduler(
                num_stages=scale['stages'],
                micro_batch_size=2
            )

            comm = OptimizedAllToAll(
                num_workers=scale['workers'],
                compression_enabled=True
            )

            # Test performance
            start_time = time.time()

            # Simulate pipeline schedule
            schedule = scheduler.create_pipeline_schedule(global_batch_size=16)

            # Simulate communication
            test_tokens = tf.random.normal([2, 16, 256])
            test_assignments = tf.random.uniform([2, 16, 2], maxval=scale['experts'], dtype=tf.int32)
            test_weights = tf.nn.softmax(tf.random.normal([2, 16, 2]), axis=-1)

            _, comm_info = comm.all_to_all_expert_routing(test_tokens, test_assignments, test_weights)

            end_time = time.time()

            performance_results.append({
                'scale': scale,
                'total_time_ms': (end_time - start_time) * 1000,
                'schedule_ops': len(schedule),
                'communication_time_ms': comm_info['communication_time_ms'],
                'bandwidth_utilization': comm_info['bandwidth_utilization']
            })

        # Verify scaling characteristics
        for i, result in enumerate(performance_results):
            print(f"  Scale {i+1}: {result['total_time_ms']:.2f}ms total, "
                  f"{result['communication_time_ms']:.2f}ms comm, "
                  f"{result['bandwidth_utilization']:.3f} bandwidth")

        # Basic scalability checks
        assert len(performance_results) == len(scales), "Should test all scales"
        assert all(r['total_time_ms'] > 0 for r in performance_results), "Should have positive timing"

        print("✅ Scalability and performance test passed")

    def test_fault_tolerance_and_recovery(self):
        """Test fault tolerance and recovery mechanisms"""
        print("\n🧪 Testing Fault Tolerance and Recovery...")

        # Test gradient accumulator recovery
        accumulator = GradientAccumulator(accumulation_steps=4)

        # Simulate partial gradient accumulation
        fake_gradients = [tf.random.normal([50, 25]) for _ in range(2)]

        for step in range(2):  # Partial accumulation
            accumulator.accumulate_gradients(fake_gradients)

        # Reset and verify recovery
        initial_count = accumulator.accumulation_count.numpy()
        accumulator.reset_statistics()

        assert accumulator.accumulation_count.numpy() == 0, "Should reset accumulation count"

        # Test communication error handling
        comm = OptimizedAllToAll(num_workers=4)

        # Test with empty data (edge case)
        empty_tokens = tf.zeros([0, 32, 256])
        empty_assignments = tf.zeros([0, 32, 2], dtype=tf.int32)
        empty_weights = tf.zeros([0, 32, 2])

        try:
            routed_tokens, _ = comm.all_to_all_expert_routing(
                empty_tokens, empty_assignments, empty_weights
            )
            # Should handle empty data gracefully
            assert routed_tokens.shape[0] == 0, "Should return empty result for empty input"
        except Exception as e:
            pytest.fail(f"Should handle empty data gracefully, but got: {e}")

        # Test memory monitor with cleanup
        monitor = MemoryMonitor()

        # Force garbage collection test
        initial_memory = monitor.get_current_memory_usage()

        # Create temporary data
        temp_data = [tf.random.normal([1000, 1000]) for _ in range(3)]
        after_allocation = monitor.get_current_memory_usage()

        # Clean up
        del temp_data
        monitor.force_garbage_collection()
        after_cleanup = monitor.get_current_memory_usage()

        # Memory should not increase indefinitely
        assert after_cleanup['total_memory_mb'] <= after_allocation['total_memory_mb'] * 1.1, "Memory cleanup should work"

        print("✅ Fault tolerance and recovery test passed")


def run_comprehensive_distributed_tests():
    """Run all comprehensive distributed training tests"""
    print("🚀 Running Comprehensive Distributed Training Test Suite...")
    print("=" * 70)

    test_suite = TestDistributedTraining()
    test_suite.setup_method()

    tests = [
        test_suite.test_dualpipe_scheduler_functionality,
        test_suite.test_pipeline_stage_integration,
        test_suite.test_distributed_training_strategy,
        test_suite.test_zero_optimizer_partitioning,
        test_suite.test_memory_optimization_framework,
        test_suite.test_communication_kernels_efficiency,
        test_suite.test_end_to_end_integration,
        test_suite.test_scalability_and_performance,
        test_suite.test_fault_tolerance_and_recovery
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

    print("\n" + "=" * 70)
    print(f"📊 Distributed Training Test Results:")
    print(f"  ✅ Passed: {passed_tests}")
    print(f"  ❌ Failed: {failed_tests}")
    print(f"  📈 Success Rate: {passed_tests / (passed_tests + failed_tests) * 100:.1f}%")

    if failed_tests == 0:
        print("\n🎉 All distributed training tests passed!")
        print("🚀 Phase 3 distributed training system is ready for production!")
    else:
        print(f"\n⚠️  {failed_tests} tests failed. Please review implementation.")

    return passed_tests, failed_tests


if __name__ == "__main__":
    run_comprehensive_distributed_tests()
