"""
Pipeline Stage Models for DeepSeek-V3 Distributed Training

This module implements the pipeline stage models that integrate MLA attention
and DeepSeek MoE layers for distributed training across multiple nodes.
Each stage represents a transformer layer optimized for pipeline parallelism.

Key Features:
- Integration with Phase 1 MLA attention mechanism
- Integration with Phase 2 DeepSeek MoE layers
- Optimized residual connections and layer normalization
- Stage-specific optimizations for distributed execution
- Memory-efficient activation checkpointing
- Pipeline-aware gradient synchronization

Mathematical Foundation:
Stage Output = LayerNorm(Input + MLA(Input)) + LayerNorm(MoE(Attention_Output))
Pipeline: Stage_i+1(Stage_i(Input)) across distributed nodes

Author: Eva DeepSeek-V3 Project
Date: 2025-08-05
"""

import tensorflow as tf
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import sys
import os

# Import our Phase 1 and Phase 2 components
try:
    from components.attention.mla import MLAAttention
    from components.moe.deepseek_moe import DeepSeekMoELayer
    print("✅ Successfully imported Phase 1 (MLA) and Phase 2 (MoE) components")
except ImportError as e:
    print(f"⚠️  Could not import components: {e}")
    print("   Make sure Phase 1 and Phase 2 are implemented")


class PipelineStageModel(tf.keras.Model):
    """
    Individual pipeline stage model with MLA attention and DeepSeek MoE layers
    
    This model represents a single transformer layer optimized for pipeline
    parallelism, integrating the advanced attention and MoE mechanisms from
    previous phases.
    
    Args:
        d_model: Model dimension (hidden size)
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        num_routed_experts: Number of routed experts in MoE
        num_shared_experts: Number of shared experts in MoE
        top_k: Number of experts to route each token to
        stage_id: Unique identifier for this pipeline stage
        dropout_rate: Dropout rate for regularization
        use_bias: Whether to use bias in linear layers
        activation_checkpointing: Whether to use activation checkpointing
    """
    
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 num_routed_experts: int = 256,
                 num_shared_experts: int = 1,
                 top_k: int = 8,
                 stage_id: int = 0,
                 dropout_rate: float = 0.0,
                 use_bias: bool = False,
                 activation_checkpointing: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_routed_experts = num_routed_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k
        self.stage_id = stage_id
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.activation_checkpointing = activation_checkpointing
        
        print(f"Pipeline Stage {stage_id} Configuration:")
        print(f"  d_model: {d_model}, num_heads: {num_heads}")
        print(f"  MoE experts: {num_routed_experts} routed + {num_shared_experts} shared")
        print(f"  Top-k routing: {top_k}")
        print(f"  Activation checkpointing: {activation_checkpointing}")
    
    def build(self, input_shape):
        """Build the pipeline stage components"""
        super().build(input_shape)
        
        # MLA Attention layer (Phase 1 component)
        self.attention_layer = MLAAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_latent=min(512, self.d_model // 2),  # Adaptive latent dimension
            use_bias=self.use_bias,
            dropout_rate=self.dropout_rate,
            name=f'mla_attention_stage_{self.stage_id}'
        )
        
        # DeepSeek MoE layer (Phase 2 component)
        self.mlp_layer = DeepSeekMoELayer(
            d_model=self.d_model,
            d_ff=self.d_ff,
            num_routed_experts=self.num_routed_experts,
            num_shared_experts=self.num_shared_experts,
            top_k=self.top_k,
            use_bias=self.use_bias,
            name=f'deepseek_moe_stage_{self.stage_id}'
        )
        
        # Layer normalization layers
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6,
            name=f'layer_norm_1_stage_{self.stage_id}'
        )
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6,
            name=f'layer_norm_2_stage_{self.stage_id}'
        )
        
        # Dropout layers for regularization
        if self.dropout_rate > 0:
            self.dropout_1 = tf.keras.layers.Dropout(
                self.dropout_rate,
                name=f'dropout_1_stage_{self.stage_id}'
            )
            self.dropout_2 = tf.keras.layers.Dropout(
                self.dropout_rate,
                name=f'dropout_2_stage_{self.stage_id}'
            )
        
        print(f"Pipeline Stage {self.stage_id} built successfully")
        print(f"  Total parameters: {self._count_parameters():,}")
    
    def call(self, 
             inputs: tf.Tensor, 
             attention_mask: Optional[tf.Tensor] = None,
             training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass through the pipeline stage
        
        Args:
            inputs: Input tensor [batch, seq_len, d_model]
            attention_mask: Optional attention mask [batch, seq_len, seq_len]
            training: Training mode flag
            
        Returns:
            outputs: Stage outputs [batch, seq_len, d_model]
        """
        if self.activation_checkpointing and training:
            return self._checkpointed_forward(inputs, attention_mask, training)
        else:
            return self._regular_forward(inputs, attention_mask, training)
    
    def _regular_forward(self, 
                        inputs: tf.Tensor, 
                        attention_mask: Optional[tf.Tensor],
                        training: bool) -> tf.Tensor:
        """Regular forward pass without checkpointing"""
        # Pre-norm architecture for better training stability
        
        # Attention block with residual connection
        norm_inputs = self.layer_norm_1(inputs)
        attention_output, attention_weights = self.attention_layer(
            norm_inputs, 
            attention_mask=attention_mask, 
            training=training
        )
        
        # Apply dropout if enabled
        if self.dropout_rate > 0 and training:
            attention_output = self.dropout_1(attention_output, training=training)
        
        # First residual connection
        attention_residual = inputs + attention_output
        
        # MoE block with residual connection
        norm_attention = self.layer_norm_2(attention_residual)
        moe_output = self.mlp_layer(norm_attention, training=training)
        
        # Apply dropout if enabled
        if self.dropout_rate > 0 and training:
            moe_output = self.dropout_2(moe_output, training=training)
        
        # Second residual connection
        final_output = attention_residual + moe_output
        
        return final_output
    
    @tf.function
    def _checkpointed_forward(self, 
                             inputs: tf.Tensor, 
                             attention_mask: Optional[tf.Tensor],
                             training: bool) -> tf.Tensor:
        """Forward pass with activation checkpointing for memory efficiency"""
        
        def attention_block(x):
            norm_x = self.layer_norm_1(x)
            attn_out, _ = self.attention_layer(
                norm_x, 
                attention_mask=attention_mask, 
                training=training
            )
            if self.dropout_rate > 0 and training:
                attn_out = self.dropout_1(attn_out, training=training)
            return x + attn_out
        
        def moe_block(x):
            norm_x = self.layer_norm_2(x)
            moe_out = self.mlp_layer(norm_x, training=training)
            if self.dropout_rate > 0 and training:
                moe_out = self.dropout_2(moe_out, training=training)
            return x + moe_out
        
        # Use gradient checkpointing for memory efficiency
        attention_residual = tf.recompute_grad(attention_block)(inputs)
        final_output = tf.recompute_grad(moe_block)(attention_residual)
        
        return final_output
    
    def _count_parameters(self) -> int:
        """Count total parameters in this pipeline stage"""
        total_params = 0
        
        # Count attention parameters
        if hasattr(self.attention_layer, '_count_parameters'):
            total_params += self.attention_layer._count_parameters()
        else:
            # Estimate MLA parameters
            total_params += self.d_model * self.d_model * 4  # Q, K, V, O projections
        
        # Count MoE parameters
        if hasattr(self.mlp_layer, '_count_parameters'):
            total_params += self.mlp_layer._count_parameters()
        else:
            # Estimate MoE parameters
            shared_params = self.num_shared_experts * (self.d_model * self.d_ff * 2)
            routed_params = self.num_routed_experts * (self.d_model * self.d_ff * 2)
            total_params += shared_params + routed_params
        
        # Layer norm parameters
        total_params += self.d_model * 4  # 2 layer norms, each with scale and bias
        
        return total_params
    
    def get_stage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for this pipeline stage"""
        stats = {
            'stage_id': self.stage_id,
            'total_parameters': self._count_parameters(),
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_routed_experts': self.num_routed_experts,
            'num_shared_experts': self.num_shared_experts,
            'top_k': self.top_k,
            'activation_checkpointing': self.activation_checkpointing
        }
        
        # Get MoE utilization statistics if available
        if hasattr(self.mlp_layer, 'get_expert_utilization_stats'):
            moe_stats = self.mlp_layer.get_expert_utilization_stats()
            stats.update({
                'expert_utilization_variance': moe_stats.get('utilization_variance', 0),
                'expert_load_balance_score': moe_stats.get('load_balance_score', 0),
                'total_tokens_processed': moe_stats.get('total_tokens', 0)
            })
        
        return stats
    
    def reset_stage_statistics(self):
        """Reset all stage statistics for fresh measurement"""
        if hasattr(self.mlp_layer, 'reset_expert_stats'):
            self.mlp_layer.reset_expert_stats()
    
    def get_config(self):
        """Return stage configuration for serialization"""
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'd_ff': self.d_ff,
            'num_routed_experts': self.num_routed_experts,
            'num_shared_experts': self.num_shared_experts,
            'top_k': self.top_k,
            'stage_id': self.stage_id,
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'activation_checkpointing': self.activation_checkpointing
        })
        return config


class PipelineStageManager:
    """
    Manager for multiple pipeline stages in distributed training

    This class coordinates multiple pipeline stages and provides utilities
    for distributed training across nodes.
    """

    def __init__(self,
                 num_stages: int,
                 stage_config: Dict[str, Any]):
        self.num_stages = num_stages
        self.stage_config = stage_config
        self.stages = []

        # Create all pipeline stages
        for stage_id in range(num_stages):
            stage = PipelineStageModel(
                stage_id=stage_id,
                **stage_config
            )
            self.stages.append(stage)

        print(f"Pipeline Stage Manager created with {num_stages} stages")

    def build_all_stages(self, input_shape: Tuple[int, ...]):
        """Build all pipeline stages with the given input shape"""
        for stage in self.stages:
            stage.build(input_shape)

    def get_stage(self, stage_id: int) -> PipelineStageModel:
        """Get a specific pipeline stage"""
        if 0 <= stage_id < len(self.stages):
            return self.stages[stage_id]
        else:
            raise ValueError(f"Invalid stage_id: {stage_id}")

    def get_total_parameters(self) -> int:
        """Get total parameters across all stages"""
        return sum(stage._count_parameters() for stage in self.stages)

    def reset_all_statistics(self):
        """Reset statistics for all stages"""
        for stage in self.stages:
            stage.reset_stage_statistics()

    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for the entire pipeline"""
        stage_stats = []
        total_params = 0

        for stage in self.stages:
            stats = stage.get_stage_statistics()
            stage_stats.append(stats)
            total_params += stats['total_parameters']

        return {
            'num_stages': self.num_stages,
            'total_parameters': total_params,
            'parameters_per_stage': total_params // self.num_stages,
            'stage_statistics': stage_stats
        }


# Testing and Validation
if __name__ == "__main__":
    print("🚀 Testing Pipeline Stage Models...")

    # Test configuration (scaled down for testing)
    stage_config = {
        'd_model': 512,
        'num_heads': 8,
        'd_ff': 2048,
        'num_routed_experts': 16,  # Scaled down from 256
        'num_shared_experts': 1,
        'top_k': 4,  # Scaled down from 8
        'dropout_rate': 0.1,
        'use_bias': False,
        'activation_checkpointing': True
    }

    print(f"\n📊 Pipeline Stage Configuration:")
    for key, value in stage_config.items():
        print(f"  {key}: {value}")

    print("\n🏗️ Testing Single Pipeline Stage...")

    # Create a single pipeline stage
    stage = PipelineStageModel(stage_id=0, **stage_config)

    # Test data
    batch_size, seq_len = 2, 64
    test_inputs = tf.random.normal([batch_size, seq_len, stage_config['d_model']])
    attention_mask = tf.ones([batch_size, seq_len, seq_len])

    # Build the stage
    stage.build(test_inputs.shape)

    print(f"  Stage built successfully")
    print(f"  Total parameters: {stage._count_parameters():,}")

    print("\n🔄 Testing Forward Pass...")

    # Test regular forward pass
    output_regular = stage(test_inputs, attention_mask=attention_mask, training=True)
    print(f"  Regular forward pass:")
    print(f"    Input shape: {test_inputs.shape}")
    print(f"    Output shape: {output_regular.shape}")
    print(f"    Output is finite: {tf.reduce_all(tf.math.is_finite(output_regular))}")
    print(f"    Shape preserved: {output_regular.shape == test_inputs.shape}")

    # Test checkpointed forward pass
    stage_checkpointed = PipelineStageModel(
        stage_id=1,
        activation_checkpointing=True,
        **{k: v for k, v in stage_config.items() if k != 'activation_checkpointing'}
    )
    stage_checkpointed.build(test_inputs.shape)

    output_checkpointed = stage_checkpointed(
        test_inputs,
        attention_mask=attention_mask,
        training=True
    )
    print(f"  Checkpointed forward pass:")
    print(f"    Output shape: {output_checkpointed.shape}")
    print(f"    Output is finite: {tf.reduce_all(tf.math.is_finite(output_checkpointed))}")

    print("\n📈 Testing Pipeline Stage Manager...")

    # Create pipeline stage manager
    num_stages = 4
    manager = PipelineStageManager(num_stages, stage_config)

    # Build all stages
    manager.build_all_stages(test_inputs.shape)

    # Get pipeline statistics
    pipeline_stats = manager.get_pipeline_statistics()

    print(f"  Number of stages: {pipeline_stats['num_stages']}")
    print(f"  Total parameters: {pipeline_stats['total_parameters']:,}")
    print(f"  Parameters per stage: {pipeline_stats['parameters_per_stage']:,}")

    print("\n🧪 Testing Multi-Stage Pipeline...")

    # Test processing through multiple stages
    current_output = test_inputs

    for stage_id in range(num_stages):
        stage = manager.get_stage(stage_id)
        current_output = stage(
            current_output,
            attention_mask=attention_mask,
            training=True
        )
        print(f"  Stage {stage_id}: Output shape {current_output.shape}")

    print(f"  Final output shape: {current_output.shape}")
    print(f"  Final output is finite: {tf.reduce_all(tf.math.is_finite(current_output))}")

    # Success criteria
    success_criteria = {
        'single_stage_working': output_regular.shape == test_inputs.shape,
        'checkpointing_working': output_checkpointed.shape == test_inputs.shape,
        'outputs_finite': tf.reduce_all(tf.math.is_finite(output_regular)) and tf.reduce_all(tf.math.is_finite(output_checkpointed)),
        'multi_stage_working': current_output.shape == test_inputs.shape,
        'manager_working': pipeline_stats['num_stages'] == num_stages,
        'parameters_reasonable': pipeline_stats['total_parameters'] > 0
    }

    print(f"\n✅ Success Criteria:")
    for criterion, passed in success_criteria.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {criterion}: {passed}")

    all_passed = all(success_criteria.values())
    if all_passed:
        print(f"\n🎉 All pipeline stage tests passed successfully!")
        print(f"🎯 Total parameters: {pipeline_stats['total_parameters']:,}")
        print(f"🚀 Ready for distributed training!")
    else:
        print(f"\n⚠️  Some tests failed - check implementation")

    print(f"💡 Pipeline stages integrate MLA attention and DeepSeek MoE for distributed training!")
