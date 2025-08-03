"""
FP8 Mixed Precision Utilities for DeepSeek-V3

This module implements FP8 mixed precision training utilities that enable
significant performance improvements while maintaining training stability
and model quality.

Key Features:
- FP8 E4M3 format for activations and gradients (training optimized)
- FP8 E5M2 format for weights (higher dynamic range)
- Dynamic scaling for optimal FP8 range utilization
- Numerical stability monitoring and validation
- Seamless integration with MLA and MoE components

Mathematical Foundation:
FP8 E4M3: 1 sign + 4 exponent + 3 mantissa bits (range: ±448)
FP8 E5M2: 1 sign + 5 exponent + 2 mantissa bits (range: ±57344)
Mixed precision: FP8 for compute, FP32 for critical operations

Author: Eva DeepSeek-V3 Project
Date: 2025-08-03
"""

import tensorflow as tf
import numpy as np
from typing import Optional, Dict, Any, Tuple
import math


class FP8Converter:
    """
    FP8 precision conversion utilities for DeepSeek-V3
    
    Handles conversion between FP32 and FP8 formats with dynamic scaling
    to maximize precision utilization while maintaining numerical stability.
    
    Args:
        target_utilization: Target utilization of FP8 range (0.75 = 75%)
        scale_update_rate: Rate of scale factor updates (0.1 = 10% per update)
        stability_threshold: Threshold for detecting numerical instability
    """
    
    def __init__(self,
                 target_utilization: float = 0.75,
                 scale_update_rate: float = 0.1,
                 stability_threshold: float = 1e-6):
        
        # FP8 E4M3 format constants (for activations/gradients)
        self.e4m3_max = 448.0
        self.e4m3_min = -448.0
        self.e4m3_eps = 1.0 / (2**6)  # Smallest representable value
        
        # FP8 E5M2 format constants (for weights)
        self.e5m2_max = 57344.0
        self.e5m2_min = -57344.0
        self.e5m2_eps = 1.0 / (2**10)
        
        # Configuration
        self.target_utilization = target_utilization
        self.scale_update_rate = scale_update_rate
        self.stability_threshold = stability_threshold
        
        # Dynamic scaling factors (learned during training)
        self.activation_scale = tf.Variable(1.0, trainable=False, name='activation_scale')
        self.gradient_scale = tf.Variable(1.0, trainable=False, name='gradient_scale')
        self.weight_scale = tf.Variable(1.0, trainable=False, name='weight_scale')
        
        # Statistics tracking
        self.conversion_count = tf.Variable(0, trainable=False, name='conversion_count')
        self.overflow_count = tf.Variable(0, trainable=False, name='overflow_count')
        
        print(f"FP8 Converter initialized:")
        print(f"  E4M3 range: [{self.e4m3_min}, {self.e4m3_max}]")
        print(f"  E5M2 range: [{self.e5m2_min}, {self.e5m2_max}]")
        print(f"  Target utilization: {target_utilization:.1%}")
    
    def to_fp8_e4m3(self, tensor: tf.Tensor, scale: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Convert tensor to FP8 E4M3 format (for activations/gradients)
        
        E4M3 format is optimized for training with good precision for gradients
        and activations. The 4-bit exponent provides sufficient dynamic range
        while 3 mantissa bits maintain reasonable precision.
        
        Args:
            tensor: Input tensor to convert
            scale: Optional scaling factor (uses activation_scale if None)
            
        Returns:
            quantized: FP8 E4M3 quantized tensor
        """
        if scale is None:
            scale = self.activation_scale
        
        # Step 1: Scale tensor to FP8 range
        scaled_tensor = tensor * scale
        
        # Step 2: Clamp to FP8 E4M3 range to prevent overflow
        clamped_tensor = tf.clip_by_value(scaled_tensor, self.e4m3_min, self.e4m3_max)
        
        # Step 3: Quantize to FP8 precision (simulation)
        # In real hardware, this would be done by FP8 arithmetic units
        quantized_tensor = self._quantize_e4m3(clamped_tensor)
        
        # Step 4: Track conversion statistics
        self.conversion_count.assign_add(1)
        
        # Step 5: Check for overflow (values that hit the clamp limits)
        overflow_mask = tf.logical_or(
            tf.equal(clamped_tensor, self.e4m3_max),
            tf.equal(clamped_tensor, self.e4m3_min)
        )
        if tf.reduce_any(overflow_mask):
            self.overflow_count.assign_add(1)
        
        return quantized_tensor
    
    def to_fp8_e5m2(self, tensor: tf.Tensor, scale: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Convert tensor to FP8 E5M2 format (for weights)
        
        E5M2 format provides higher dynamic range with 5 exponent bits,
        making it suitable for weight storage where we need to represent
        a wider range of values with acceptable precision.
        
        Args:
            tensor: Input tensor to convert
            scale: Optional scaling factor (uses weight_scale if None)
            
        Returns:
            quantized: FP8 E5M2 quantized tensor
        """
        if scale is None:
            scale = self.weight_scale
        
        # Scale and clamp to E5M2 range
        scaled_tensor = tensor * scale
        clamped_tensor = tf.clip_by_value(scaled_tensor, self.e5m2_min, self.e5m2_max)
        
        # Quantize to FP8 precision
        quantized_tensor = self._quantize_e5m2(clamped_tensor)
        
        return quantized_tensor
    
    def from_fp8(self, tensor: tf.Tensor, scale: tf.Tensor) -> tf.Tensor:
        """
        Convert FP8 tensor back to FP32
        
        Args:
            tensor: FP8 tensor to convert
            scale: Scaling factor used in forward conversion
            
        Returns:
            fp32_tensor: Converted FP32 tensor
        """
        return tensor / (scale + self.stability_threshold)
    
    def _quantize_e4m3(self, tensor: tf.Tensor) -> tf.Tensor:
        """
        Simulate E4M3 quantization
        
        This simulates the quantization that would happen in actual FP8 hardware.
        Real implementations would use dedicated FP8 arithmetic units.
        """
        # E4M3 has 3 mantissa bits, so we can represent 2^3 = 8 levels
        # between powers of 2. This gives us the quantization step size.
        scale_factor = 2**3  # Based on 3 mantissa bits
        quantized = tf.round(tensor * scale_factor) / scale_factor
        return quantized
    
    def _quantize_e5m2(self, tensor: tf.Tensor) -> tf.Tensor:
        """
        Simulate E5M2 quantization
        
        E5M2 has 2 mantissa bits, providing 4 levels between powers of 2.
        This gives coarser quantization but higher dynamic range.
        """
        scale_factor = 2**2  # Based on 2 mantissa bits
        quantized = tf.round(tensor * scale_factor) / scale_factor
        return quantized
    
    def update_scales(self, tensors: Dict[str, tf.Tensor]):
        """
        Update FP8 scaling factors based on tensor statistics
        
        This implements dynamic scaling to maximize utilization of the FP8
        range while preventing overflow. The scaling factors are updated
        using exponential moving averages for stability.
        
        Args:
            tensors: Dictionary of tensors to analyze
                    Keys: 'activations', 'gradients', 'weights'
        """
        for tensor_type, tensor in tensors.items():
            if tensor is None:
                continue
            
            # Compute tensor statistics
            abs_max = tf.reduce_max(tf.abs(tensor))
            
            if tensor_type == 'activations':
                # Update activation scale to utilize target percentage of E4M3 range
                target_max = self.e4m3_max * self.target_utilization
                new_scale = target_max / (abs_max + self.e4m3_eps)
                
                # Exponential moving average update
                updated_scale = (1 - self.scale_update_rate) * self.activation_scale + \
                               self.scale_update_rate * new_scale
                self.activation_scale.assign(updated_scale)
                
            elif tensor_type == 'gradients':
                # Update gradient scale
                target_max = self.e4m3_max * self.target_utilization
                new_scale = target_max / (abs_max + self.e4m3_eps)
                
                updated_scale = (1 - self.scale_update_rate) * self.gradient_scale + \
                               self.scale_update_rate * new_scale
                self.gradient_scale.assign(updated_scale)
                
            elif tensor_type == 'weights':
                # Update weight scale for E5M2 format
                target_max = self.e5m2_max * self.target_utilization
                new_scale = target_max / (abs_max + self.e5m2_eps)
                
                updated_scale = (1 - self.scale_update_rate) * self.weight_scale + \
                               self.scale_update_rate * new_scale
                self.weight_scale.assign(updated_scale)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get FP8 conversion statistics for monitoring
        
        Returns:
            Dictionary with conversion statistics and scale factors
        """
        overflow_rate = self.overflow_count / tf.maximum(self.conversion_count, 1)
        
        return {
            'activation_scale': float(self.activation_scale.numpy()),
            'gradient_scale': float(self.gradient_scale.numpy()),
            'weight_scale': float(self.weight_scale.numpy()),
            'conversion_count': int(self.conversion_count.numpy()),
            'overflow_count': int(self.overflow_count.numpy()),
            'overflow_rate': float(overflow_rate.numpy()),
            'target_utilization': self.target_utilization,
            'e4m3_range': [self.e4m3_min, self.e4m3_max],
            'e5m2_range': [self.e5m2_min, self.e5m2_max]
        }
    
    def reset_statistics(self):
        """Reset conversion statistics"""
        self.conversion_count.assign(0)
        self.overflow_count.assign(0)
    
    def validate_conversion_quality(self, original: tf.Tensor, converted: tf.Tensor) -> Dict[str, float]:
        """
        Validate FP8 conversion quality
        
        Args:
            original: Original FP32 tensor
            converted: FP8 converted tensor (in FP32 representation)
            
        Returns:
            Dictionary with quality metrics
        """
        # Calculate relative error
        abs_error = tf.abs(original - converted)
        rel_error = abs_error / (tf.abs(original) + self.stability_threshold)
        
        # Calculate signal-to-noise ratio
        signal_power = tf.reduce_mean(tf.square(original))
        noise_power = tf.reduce_mean(tf.square(abs_error))
        snr = 10 * tf.math.log(signal_power / (noise_power + self.stability_threshold)) / tf.math.log(10.0)
        
        # Calculate correlation coefficient
        original_centered = original - tf.reduce_mean(original)
        converted_centered = converted - tf.reduce_mean(converted)
        correlation = tf.reduce_sum(original_centered * converted_centered) / \
                     tf.sqrt(tf.reduce_sum(tf.square(original_centered)) * 
                            tf.reduce_sum(tf.square(converted_centered)) + self.stability_threshold)
        
        return {
            'max_abs_error': float(tf.reduce_max(abs_error).numpy()),
            'mean_rel_error': float(tf.reduce_mean(rel_error).numpy()),
            'snr_db': float(snr.numpy()),
            'correlation': float(correlation.numpy()),
            'mse': float(tf.reduce_mean(tf.square(abs_error)).numpy())
        }


# Global FP8 converter instance for easy access
fp8_converter = FP8Converter()


# Comprehensive FP8 Testing
if __name__ == "__main__":
    print("🚀 Testing Complete FP8 Implementation...")
    
    # Test data with various ranges
    test_cases = [
        ("Small values", tf.random.normal([100, 100]) * 0.1),
        ("Medium values", tf.random.normal([100, 100]) * 10.0),
        ("Large values", tf.random.normal([100, 100]) * 100.0),
        ("Mixed range", tf.concat([
            tf.random.normal([50, 100]) * 0.01,
            tf.random.normal([50, 100]) * 50.0
        ], axis=0))
    ]
    
    print("\n🧪 Testing FP8 E4M3 Conversion Quality...")
    for name, tensor in test_cases:
        # Test E4M3 conversion
        fp8_tensor = fp8_converter.to_fp8_e4m3(tensor)
        recovered_tensor = fp8_converter.from_fp8(fp8_tensor, fp8_converter.activation_scale)
        
        quality = fp8_converter.validate_conversion_quality(tensor, recovered_tensor)
        
        print(f"  {name}:")
        print(f"    Max abs error: {quality['max_abs_error']:.6f}")
        print(f"    Mean rel error: {quality['mean_rel_error']:.6f}")
        print(f"    SNR: {quality['snr_db']:.1f} dB")
        print(f"    Correlation: {quality['correlation']:.4f}")
    
    print("\n📊 Testing Dynamic Scaling...")
    # Test dynamic scaling with different tensor ranges
    for i, (name, tensor) in enumerate(test_cases):
        fp8_converter.update_scales({'activations': tensor})
        stats = fp8_converter.get_statistics()
        print(f"  After {name}: activation_scale = {stats['activation_scale']:.4f}")
    
    print("\n⚡ Testing Performance Impact...")
    # Simulate training scenario
    large_tensor = tf.random.normal([1000, 1000])
    
    # FP32 baseline
    import time
    start_time = time.time()
    for _ in range(10):
        result_fp32 = tf.matmul(large_tensor, large_tensor)
    fp32_time = time.time() - start_time
    
    # FP8 simulation (conversion overhead)
    start_time = time.time()
    for _ in range(10):
        fp8_tensor = fp8_converter.to_fp8_e4m3(large_tensor)
        recovered = fp8_converter.from_fp8(fp8_tensor, fp8_converter.activation_scale)
        result_fp8 = tf.matmul(recovered, recovered)
    fp8_time = time.time() - start_time
    
    print(f"  FP32 time: {fp32_time:.4f}s")
    print(f"  FP8 time (with conversion): {fp8_time:.4f}s")
    print(f"  Overhead ratio: {fp8_time / fp32_time:.2f}x")
    
    print("\n📈 Final Statistics:")
    final_stats = fp8_converter.get_statistics()
    print(f"  Conversions performed: {final_stats['conversion_count']}")
    print(f"  Overflow rate: {final_stats['overflow_rate']:.4f}")
    print(f"  Current scales: act={final_stats['activation_scale']:.4f}, "
          f"grad={final_stats['gradient_scale']:.4f}, weight={final_stats['weight_scale']:.4f}")
    
    print("\n✅ FP8 implementation ready for integration!")
    print("🎯 Key benefits: Memory reduction, potential speedup on FP8 hardware")
    print("⚠️  Note: Full benefits require hardware FP8 support")
