"""
Performance Benchmarking Suite for Advanced MoE Architecture

This module provides comprehensive performance benchmarking for all Phase 2 
components of the DeepSeek-V3 implementation, including:
- Expert scaling performance analysis
- Load balancing overhead measurement
- MTP speedup validation
- Expert parallelism efficiency testing
- Memory usage profiling

Author: Eva DeepSeek-V3 Project
Date: 2025-08-05
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import tensorflow as tf
import numpy as np
import time
import psutil
import gc
from typing import Dict, List, Any, Tuple

# Import components to benchmark
from components.moe.deepseek_moe import DeepSeekMoELayer
from components.moe.load_balancing import AuxiliaryLossFreeLoadBalancer
from components.moe.expert_parallelism import ExpertParallelismManager
from components.moe.multi_token_prediction import MultiTokenPredictionHead


class AdvancedMoEBenchmark:
    """Comprehensive benchmarking suite for advanced MoE components"""
    
    def __init__(self):
        self.results = {}
        self.warmup_iterations = 5
        self.benchmark_iterations = 20
        
        print("🚀 Advanced MoE Benchmarking Suite Initialized")
        print(f"  Warmup iterations: {self.warmup_iterations}")
        print(f"  Benchmark iterations: {self.benchmark_iterations}")
    
    def benchmark_expert_scaling(self) -> Dict[str, Any]:
        """Benchmark performance with different expert counts"""
        print("\n📊 Benchmarking Expert Scaling Performance...")
        
        expert_counts = [8, 16, 32, 64]
        top_k_values = [2, 4, 8]
        d_model = 512
        d_ff = 2048
        batch_size, seq_len = 4, 128
        
        results = []
        
        for num_experts in expert_counts:
            for top_k in top_k_values:
                if top_k <= min(num_experts, 16):  # Reasonable top_k limit
                    print(f"  Testing {num_experts} experts, top-k={top_k}...")
                    
                    # Create MoE layer
                    moe = DeepSeekMoELayer(
                        d_model=d_model,
                        d_ff=d_ff,
                        num_routed_experts=num_experts,
                        top_k=top_k
                    )
                    
                    inputs = tf.random.normal([batch_size, seq_len, d_model])
                    moe.build(inputs.shape)
                    
                    # Warmup
                    for _ in range(self.warmup_iterations):
                        _ = moe(inputs, training=True)
                    
                    # Benchmark
                    times = []
                    memory_usage = []
                    
                    for _ in range(self.benchmark_iterations):
                        # Memory before
                        mem_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                        
                        start_time = time.time()
                        _ = moe(inputs, training=True)
                        end_time = time.time()
                        
                        # Memory after
                        mem_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                        
                        times.append(end_time - start_time)
                        memory_usage.append(mem_after - mem_before)
                    
                    avg_time = np.mean(times)
                    std_time = np.std(times)
                    throughput = (batch_size * seq_len) / avg_time  # tokens per second
                    avg_memory = np.mean(memory_usage)
                    
                    result = {
                        'num_experts': num_experts,
                        'top_k': top_k,
                        'avg_time': avg_time,
                        'std_time': std_time,
                        'throughput': throughput,
                        'avg_memory_mb': avg_memory,
                        'efficiency': top_k / num_experts,
                        'theoretical_speedup': num_experts / top_k
                    }
                    results.append(result)
                    
                    print(f"    Time: {avg_time*1000:.2f}±{std_time*1000:.2f}ms, "
                          f"Throughput: {throughput:.0f} tokens/sec")
                    
                    # Cleanup
                    del moe
                    gc.collect()
        
        self.results['expert_scaling'] = results
        return results
    
    def benchmark_load_balancing_overhead(self) -> Dict[str, Any]:
        """Benchmark overhead of load balancing mechanism"""
        print("\n⚖️ Benchmarking Load Balancing Overhead...")
        
        d_model = 512
        d_ff = 2048
        num_experts = 32
        top_k = 4
        batch_size, seq_len = 4, 128
        
        # MoE with load balancing
        moe_with_balancing = DeepSeekMoELayer(
            d_model=d_model,
            d_ff=d_ff,
            num_routed_experts=num_experts,
            top_k=top_k,
            bias_update_rate=1e-3
        )
        
        # MoE without load balancing (set update rate to 0)
        moe_without_balancing = DeepSeekMoELayer(
            d_model=d_model,
            d_ff=d_ff,
            num_routed_experts=num_experts,
            top_k=top_k,
            bias_update_rate=0.0  # Disable bias updates
        )
        
        inputs = tf.random.normal([batch_size, seq_len, d_model])
        moe_with_balancing.build(inputs.shape)
        moe_without_balancing.build(inputs.shape)
        
        # Warmup both models
        for _ in range(self.warmup_iterations):
            _ = moe_with_balancing(inputs, training=True)
            _ = moe_without_balancing(inputs, training=True)
        
        # Benchmark with load balancing
        times_with = []
        for _ in range(self.benchmark_iterations):
            start_time = time.time()
            _ = moe_with_balancing(inputs, training=True)
            times_with.append(time.time() - start_time)
        
        # Benchmark without load balancing
        times_without = []
        for _ in range(self.benchmark_iterations):
            start_time = time.time()
            _ = moe_without_balancing(inputs, training=True)
            times_without.append(time.time() - start_time)
        
        avg_time_with = np.mean(times_with)
        avg_time_without = np.mean(times_without)
        overhead = (avg_time_with - avg_time_without) / avg_time_without * 100
        
        result = {
            'time_with_balancing_ms': avg_time_with * 1000,
            'time_without_balancing_ms': avg_time_without * 1000,
            'overhead_percentage': overhead,
            'overhead_absolute_ms': (avg_time_with - avg_time_without) * 1000
        }
        
        print(f"  With balancing: {avg_time_with*1000:.2f}ms")
        print(f"  Without balancing: {avg_time_without*1000:.2f}ms")
        print(f"  Overhead: {overhead:.2f}%")
        
        self.results['load_balancing_overhead'] = result
        return result
    
    def benchmark_mtp_speedup(self) -> Dict[str, Any]:
        """Benchmark Multi-Token Prediction speedup"""
        print("\n🚀 Benchmarking Multi-Token Prediction Speedup...")
        
        vocab_size = 1000
        d_model = 512
        batch_size = 1
        seq_len = 32
        max_length = 100
        
        # Test different prediction lengths
        prediction_lengths = [1, 2, 4, 8]
        results = []
        
        for num_predict in prediction_lengths:
            print(f"  Testing {num_predict} token prediction...")
            
            mtp_head = MultiTokenPredictionHead(
                vocab_size=vocab_size,
                d_model=d_model,
                num_predict_tokens=num_predict,
                temperature=1.0
            )
            
            hidden_states = tf.random.normal([batch_size, seq_len, d_model])
            mtp_head.build(hidden_states.shape)
            
            # Mock model function
            def mock_model_forward(input_ids):
                batch_size, seq_len = tf.shape(input_ids)[0], tf.shape(input_ids)[1]
                return tf.random.normal([batch_size, seq_len, d_model])
            
            # Warmup
            input_ids = tf.random.uniform([batch_size, 10], maxval=vocab_size, dtype=tf.int32)
            for _ in range(3):
                _, _ = mtp_head.generate_with_mtp(
                    input_ids=input_ids,
                    model_forward_fn=mock_model_forward,
                    max_length=20,
                    acceptance_threshold=0.7
                )
            
            # Benchmark generation
            generation_times = []
            speedup_ratios = []
            acceptance_rates = []
            
            for _ in range(10):  # Fewer iterations for generation benchmark
                start_time = time.time()
                generated_ids, stats = mtp_head.generate_with_mtp(
                    input_ids=input_ids,
                    model_forward_fn=mock_model_forward,
                    max_length=max_length,
                    acceptance_threshold=0.7
                )
                generation_time = time.time() - start_time
                
                generation_times.append(generation_time)
                speedup_ratios.append(stats['speedup_ratio'])
                acceptance_rates.append(stats['acceptance_rate'])
            
            result = {
                'num_predict_tokens': num_predict,
                'avg_generation_time': np.mean(generation_times),
                'avg_speedup_ratio': np.mean(speedup_ratios),
                'avg_acceptance_rate': np.mean(acceptance_rates),
                'theoretical_max_speedup': num_predict
            }
            results.append(result)
            
            print(f"    Speedup: {result['avg_speedup_ratio']:.2f}x "
                  f"(max: {result['theoretical_max_speedup']:.1f}x), "
                  f"Acceptance: {result['avg_acceptance_rate']:.1%}")
            
            del mtp_head
            gc.collect()
        
        self.results['mtp_speedup'] = results
        return results
    
    def benchmark_expert_parallelism_efficiency(self) -> Dict[str, Any]:
        """Benchmark expert parallelism efficiency"""
        print("\n🔗 Benchmarking Expert Parallelism Efficiency...")
        
        num_experts = 32
        node_counts = [2, 4, 8]
        batch_size, seq_len, d_model = 4, 64, 256
        top_k = 4
        
        results = []
        
        for num_nodes in node_counts:
            print(f"  Testing {num_nodes} nodes...")
            
            manager = ExpertParallelismManager(
                num_experts=num_experts,
                num_nodes=num_nodes,
                compression_enabled=True
            )
            
            # Create test data
            tokens = tf.random.normal([batch_size, seq_len, d_model])
            expert_assignments = tf.random.uniform(
                [batch_size, seq_len, top_k], 
                maxval=num_experts, 
                dtype=tf.int32
            )
            routing_weights = tf.nn.softmax(tf.random.normal([batch_size, seq_len, top_k]), axis=-1)
            
            # Warmup
            for _ in range(self.warmup_iterations):
                _ = manager.simulate_all_to_all_routing(tokens, expert_assignments, routing_weights)
            
            # Benchmark communication
            communication_times = []
            for _ in range(self.benchmark_iterations):
                start_time = time.time()
                node_data = manager.simulate_all_to_all_routing(tokens, expert_assignments, routing_weights)
                communication_time = time.time() - start_time
                communication_times.append(communication_time)
            
            # Benchmark node processing
            processing_times = []
            for _ in range(self.benchmark_iterations):
                node_data = manager.simulate_all_to_all_routing(tokens, expert_assignments, routing_weights)
                
                start_time = time.time()
                for node_id in range(num_nodes):
                    _ = manager.simulate_node_processing(node_id, node_data[node_id], [])
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
            
            # Calculate load balance
            load_balance = manager.get_load_balance_across_nodes(expert_assignments)
            
            result = {
                'num_nodes': num_nodes,
                'experts_per_node': num_experts // num_nodes,
                'avg_communication_time': np.mean(communication_times),
                'avg_processing_time': np.mean(processing_times),
                'total_time': np.mean(communication_times) + np.mean(processing_times),
                'load_balance_cv': load_balance['load_cv'],
                'parallelism_efficiency': 1.0 / num_nodes  # Theoretical efficiency
            }
            results.append(result)
            
            print(f"    Communication: {result['avg_communication_time']*1000:.2f}ms, "
                  f"Processing: {result['avg_processing_time']*1000:.2f}ms, "
                  f"Load CV: {result['load_balance_cv']:.3f}")
        
        self.results['expert_parallelism'] = results
        return results
    
    def generate_benchmark_report(self) -> str:
        """Generate comprehensive benchmark report"""
        report = []
        report.append("🚀 Advanced MoE Benchmark Report")
        report.append("=" * 50)
        
        # Expert Scaling Results
        if 'expert_scaling' in self.results:
            report.append("\n📊 Expert Scaling Performance:")
            for result in self.results['expert_scaling']:
                report.append(f"  {result['num_experts']} experts, top-k={result['top_k']}: "
                            f"{result['throughput']:.0f} tokens/sec, "
                            f"{result['theoretical_speedup']:.1f}x theoretical speedup")
        
        # Load Balancing Overhead
        if 'load_balancing_overhead' in self.results:
            result = self.results['load_balancing_overhead']
            report.append(f"\n⚖️ Load Balancing Overhead: {result['overhead_percentage']:.2f}%")
        
        # MTP Speedup Results
        if 'mtp_speedup' in self.results:
            report.append("\n🚀 Multi-Token Prediction Speedup:")
            for result in self.results['mtp_speedup']:
                report.append(f"  {result['num_predict_tokens']} tokens: "
                            f"{result['avg_speedup_ratio']:.2f}x speedup, "
                            f"{result['avg_acceptance_rate']:.1%} acceptance")
        
        # Expert Parallelism Efficiency
        if 'expert_parallelism' in self.results:
            report.append("\n🔗 Expert Parallelism Efficiency:")
            for result in self.results['expert_parallelism']:
                report.append(f"  {result['num_nodes']} nodes: "
                            f"{result['total_time']*1000:.2f}ms total, "
                            f"Load CV: {result['load_balance_cv']:.3f}")
        
        return "\n".join(report)
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmarks and return results"""
        print("🚀 Running All Advanced MoE Benchmarks...")
        print("=" * 60)
        
        try:
            self.benchmark_expert_scaling()
            self.benchmark_load_balancing_overhead()
            self.benchmark_mtp_speedup()
            self.benchmark_expert_parallelism_efficiency()
            
            print("\n" + self.generate_benchmark_report())
            print("\n✅ All benchmarks completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Benchmark failed: {str(e)}")
        
        return self.results


if __name__ == "__main__":
    benchmark = AdvancedMoEBenchmark()
    results = benchmark.run_all_benchmarks()
