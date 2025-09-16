"""
Adaptive-Frequency-SSM: Main Model Architecture
S6-based State Space Model with Advanced Frequency Domain Compression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any
from einops import rearrange

from .layers import (
    AdaptiveFrequencyS6Block, 
    AdaptiveFrequencyResidualBlock, 
    MultiHeadAdaptiveFrequencyAttention
)


class AdaptiveFrequencySSMConfig:
    """Configuration class for Adaptive Frequency SSM"""
    
    def __init__(
        self,
        # Model architecture
        d_model: int = 512,
        n_layer: int = 12,
        vocab_size: int = 50304,
        
        # SSM parameters
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        
        # Spectral compression
        compression_ratio: float = 0.5,
        use_adaptive_mask: bool = True,
        use_spectral_norm: bool = True,
        
        # Training parameters
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        
        # Regularization
        dropout: float = 0.0,
        bias: bool = False,
        conv_bias: bool = True,
        
        # Task-specific
        num_classes: Optional[int] = None,
        max_seq_len: int = 2048,
        
        # Architecture variants
        use_hybrid_attention: bool = False,
        attention_layers: Optional[list] = None,  # Layers to use attention
        
        # Multi-scale processing
        use_multi_scale: bool = True,
        num_scales: int = 3,
        
        **kwargs
    ):
        self.d_model = d_model
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        
        self.compression_ratio = compression_ratio
        self.use_adaptive_mask = use_adaptive_mask
        self.use_spectral_norm = use_spectral_norm
        
        self.dt_rank = dt_rank
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        self.dt_init_floor = dt_init_floor
        
        self.dropout = dropout
        self.bias = bias
        self.conv_bias = conv_bias
        
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        
        self.use_hybrid_attention = use_hybrid_attention
        self.attention_layers = attention_layers or []
        
        self.use_multi_scale = use_multi_scale
        self.num_scales = num_scales
        
        # Store additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class AdaptiveFrequencySSMBackbone(nn.Module):
    """
    Spectral SSM Backbone Model
    Can be used for various tasks by adding appropriate heads
    """
    
    def __init__(self, config: AdaptiveFrequencySSMConfig, device=None, dtype=None):
        super().__init__()
        
        self.config = config
        factory_kwargs = {"device": device, "dtype": dtype}
        
        # Token embedding
        self.embedding = nn.Embedding(
            config.vocab_size, 
            config.d_model, 
            **factory_kwargs
        )
        
        # Positional encoding (learnable)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, config.max_seq_len, config.d_model, **factory_kwargs)
        )
        
        # Input dropout
        self.drop = nn.Dropout(config.dropout)
        
        # Spectral SSM layers
        self.layers = nn.ModuleList()
        for i in range(config.n_layer):
            if config.use_hybrid_attention and i in config.attention_layers:
                # Use attention layer at specified positions
                layer = MultiHeadAdaptiveFrequencyAttention(
                    d_model=config.d_model,
                    num_heads=config.d_model // 64,
                    dropout=config.dropout,
                    use_spectral=True,
                    compression_ratio=config.compression_ratio,
                )
            elif config.use_multi_scale:
                # Use residual block with multi-scale processing
                layer = AdaptiveFrequencyResidualBlock(
                    d_model=config.d_model,
                    d_state=config.d_state,
                    d_conv=config.d_conv,
                    expand=config.expand,
                    dt_rank=config.dt_rank,
                    dt_min=config.dt_min,
                    dt_max=config.dt_max,
                    dt_init=config.dt_init,
                    dt_scale=config.dt_scale,
                    dt_init_floor=config.dt_init_floor,
                    compression_ratio=config.compression_ratio,
                    use_adaptive_mask=config.use_adaptive_mask,
                    use_spectral_norm=config.use_spectral_norm,
                    dropout=config.dropout,
                    conv_bias=config.conv_bias,
                    bias=config.bias,
                    num_scales=config.num_scales,
                    **factory_kwargs,
                )
            else:
                # Standard spectral S6 block
                layer = AdaptiveFrequencyS6Block(
                    d_model=config.d_model,
                    d_state=config.d_state,
                    d_conv=config.d_conv,
                    expand=config.expand,
                    dt_rank=config.dt_rank,
                    dt_min=config.dt_min,
                    dt_max=config.dt_max,
                    dt_init=config.dt_init,
                    dt_scale=config.dt_scale,
                    dt_init_floor=config.dt_init_floor,
                    compression_ratio=config.compression_ratio,
                    use_adaptive_mask=config.use_adaptive_mask,
                    use_spectral_norm=config.use_spectral_norm,
                    dropout=config.dropout,
                    conv_bias=config.conv_bias,
                    bias=config.bias,
                    **factory_kwargs,
                )
            
            self.layers.append(layer)
        
        # Final layer norm
        self.norm_f = nn.LayerNorm(config.d_model, **factory_kwargs)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights following GPT-style initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: Not used in SSM, kept for compatibility
            position_ids: Optional position indices
            
        Returns:
            Dictionary with 'last_hidden_state' and other info
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token embedding
        hidden_states = self.embedding(input_ids)  # (B, L, D)
        
        # Add positional embedding
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        
        pos_emb = self.pos_embedding[:, :seq_len, :]
        hidden_states = hidden_states + pos_emb
        
        # Dropout
        hidden_states = self.drop(hidden_states)
        
        # Pass through layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        # Final norm
        hidden_states = self.norm_f(hidden_states)
        
        return {
            "last_hidden_state": hidden_states,
            "hidden_states": hidden_states,  # For compatibility
        }


class AdaptiveFrequencySSMForSequenceClassification(nn.Module):
    """
    Spectral SSM for sequence classification tasks
    """
    
    def __init__(self, config: AdaptiveFrequencySSMConfig, device=None, dtype=None):
        super().__init__()
        
        self.config = config
        self.num_labels = config.num_classes
        factory_kwargs = {"device": device, "dtype": dtype}
        
        # Backbone
        self.backbone = AdaptiveFrequencySSMBackbone(config, device, dtype)
        
        # Classification head
        self.classifier = nn.Linear(
            config.d_model, 
            config.num_classes, 
            **factory_kwargs
        )
        
        # Dropout for classifier
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for classification
        """
        # Get backbone outputs
        outputs = self.backbone(input_ids, attention_mask, **kwargs)
        hidden_states = outputs["last_hidden_state"]  # (B, L, D)
        
        # Global pooling (mean over sequence length)
        pooled_output = hidden_states.mean(dim=1)  # (B, D)
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)  # (B, num_classes)
        
        outputs = {
            "logits": logits,
            "hidden_states": hidden_states,
        }
        
        # Compute loss if labels provided
        if labels is not None:
            if self.num_labels == 1:
                # Regression
                loss = F.mse_loss(logits.squeeze(), labels.float())
            else:
                # Classification
                loss = F.cross_entropy(logits, labels)
            outputs["loss"] = loss
        
        return outputs


class AdaptiveFrequencySSMForLanguageModeling(nn.Module):
    """
    Spectral SSM for language modeling tasks
    """
    
    def __init__(self, config: AdaptiveFrequencySSMConfig, device=None, dtype=None):
        super().__init__()
        
        self.config = config
        factory_kwargs = {"device": device, "dtype": dtype}
        
        # Backbone
        self.backbone = AdaptiveFrequencySSMBackbone(config, device, dtype)
        
        # Language modeling head
        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
            **factory_kwargs
        )
        
        # Tie weights with embedding
        self.lm_head.weight = self.backbone.embedding.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for language modeling
        """
        # Get backbone outputs
        outputs = self.backbone(input_ids, attention_mask, **kwargs)
        hidden_states = outputs["last_hidden_state"]  # (B, L, D)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)  # (B, L, vocab_size)
        
        outputs = {
            "logits": logits,
            "hidden_states": hidden_states,
        }
        
        # Compute loss if labels provided
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for loss computation
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            outputs["loss"] = loss
        
        return outputs


class AdaptiveFrequencySSM(nn.Module):
    """
    Main Spectral SSM Model - unified interface
    """
    
    def __init__(self, config: AdaptiveFrequencySSMConfig, task: str = "classification", device=None, dtype=None):
        super().__init__()
        
        self.config = config
        self.task = task
        
        if task == "classification":
            self.model = AdaptiveFrequencySSMForSequenceClassification(config, device, dtype)
        elif task == "language_modeling":
            self.model = AdaptiveFrequencySSMForLanguageModeling(config, device, dtype)
        else:
            # Default to backbone only
            self.model = AdaptiveFrequencySSMBackbone(config, device, dtype)
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def get_num_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_flops_per_token(self, seq_len: int) -> int:
        """
        Estimate FLOPs per token for comparison with baselines
        This is an approximation based on the spectral compression
        """
        d_model = self.config.d_model
        d_state = self.config.d_state
        d_inner = d_model * self.config.expand
        compression_ratio = self.config.compression_ratio
        
        # Standard SSM FLOPs per layer
        standard_flops = seq_len * d_inner * d_state
        
        # Spectral SSM FLOPs per layer (compressed state)
        compressed_state = max(1, int(round(d_state * compression_ratio)))
        spectral_flops = (
            seq_len * d_inner * compressed_state +  # Compressed recurrence
            d_state * math.log2(d_state) * 2  # FFT operations
        )
        
        # Total for all layers
        total_flops = spectral_flops * self.config.n_layer
        
        return total_flops // seq_len  # Per token


def create_adaptive_frequency_ssm_model(
    task: str,
    d_model: int = 512,
    n_layer: int = 12,
    vocab_size: int = 50304,
    num_classes: Optional[int] = None,
    compression_ratio: float = 0.5,
    **kwargs
) -> AdaptiveFrequencySSM:
    """
    Factory function to create AdaptiveFrequencySSM models
    
    Args:
        task: One of ["classification", "language_modeling", "backbone"]
        d_model: Model dimension
        n_layer: Number of layers
        vocab_size: Vocabulary size
        num_classes: Number of classes (for classification)
        compression_ratio: Frequency domain compression ratio
        **kwargs: Additional config parameters
    
    Returns:
        AdaptiveFrequencySSM model
    """
    config = AdaptiveFrequencySSMConfig(
        d_model=d_model,
        n_layer=n_layer,
        vocab_size=vocab_size,
        num_classes=num_classes,
        compression_ratio=compression_ratio,
        **kwargs
    )
    
    return AdaptiveFrequencySSM(config, task=task)


