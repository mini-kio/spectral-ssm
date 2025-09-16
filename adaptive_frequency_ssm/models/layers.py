"""
Adaptive-Frequency-SSM Layers based on S6 Architecture
Implements advanced frequency domain state compression with S6 improvements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union
from einops import rearrange, repeat

from ..utils.fft_utils import (
    slice_low_frequencies,
    pad_and_reconstruct,
    AdaptiveFrequencyMask,
    frequency_dropout,
    spectral_norm_regularization
)


class AdaptiveFrequencyS6Block(nn.Module):
    """
    S6-based Spectral-Latent SSM Block with frequency domain compression
    
    Key improvements over standard S6:
    1. Frequency domain state compression
    2. Adaptive frequency selection
    3. Multi-scale processing
    4. Spectral regularization
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Union[int, str] = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        compression_ratio: float = 0.5,
        use_adaptive_mask: bool = True,
        use_spectral_norm: bool = True,
        dropout: float = 0.0,
        conv_bias: bool = True,
        bias: bool = False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.compression_ratio = compression_ratio
        # Fix: Ensure k is compatible with FFT output size
        freq_size = self.d_state // 2 + 1  # Real FFT output size
        target_k = int(round(self.d_state * compression_ratio))
        self.k = max(1, min(freq_size, target_k))  # Compressed state dimension
        self.use_adaptive_mask = use_adaptive_mask
        self.use_spectral_norm = use_spectral_norm
        
        # Input projection following S6
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        
        # Convolution layer for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        
        # S6 activation
        self.activation = "silu"
        self.act = nn.SiLU()
        
        # SSM parameters - modified for spectral domain
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        
        # Initialize dt projection
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
            
        # Initialize dt bias
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        
        # A parameter - frequency domain version
        state_dim = self.k if use_spectral_norm else self.d_state
        self.A_log = nn.Parameter(torch.randn(self.d_inner, state_dim, **factory_kwargs))

        # S6 structured matrix initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        with torch.no_grad():
            self.A_log.copy_(A_log[:, :state_dim].to(self.A_log.dtype))
        self.A_log._no_weight_decay = True

        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True
        
        # Frequency domain components
        if use_adaptive_mask:
            self.freq_mask = AdaptiveFrequencyMask(d_state, compression_ratio)
        else:
            self.freq_mask = None
            
        # Spectral gating for high-frequency information
        self.spectral_gate = nn.Linear(self.d_inner, self.d_inner, bias=bias, **factory_kwargs)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization for stability
        self.norm = nn.LayerNorm(self.d_inner, **factory_kwargs)
        
    def forward(self, hidden_states, inference_params=None):
        """
        Forward pass with spectral state compression
        
        Args:
            hidden_states: (B, L, D)
            inference_params: For compatibility with S6
        """
        batch, seqlen, dim = hidden_states.shape
        
        # Input projection
        xz = self.in_proj(hidden_states)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each (B, L, d_inner)
        
        # Convolution for local context
        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[..., :seqlen]  # Causal convolution
        x = rearrange(x, "b d l -> b l d")
        
        # Activation
        x = self.act(x)
        
        # SSM computation with spectral compression
        y = self.adaptive_frequency_ssm(x)
        
        # Gating mechanism
        z = self.act(z)
        y = y * z
        
        # Output projection
        output = self.out_proj(y)
        return output
    
    def adaptive_frequency_ssm(self, u):
        """
        Spectral SSM computation with frequency domain compression
        
        Args:
            u: Input tensor (B, L, d_inner)
        """
        batch, seqlen, d_inner = u.shape
        
        # Get SSM parameters
        delta, B, C = self.get_ssm_params(u)  # (B, L, d_inner), (B, L, d_state), (B, L, d_state)
        
        # Discretization
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        deltaA = torch.exp(torch.einsum('bld,dn->bldn', delta, A))  # (B, L, d_inner, d_state)
        deltaB_u = torch.einsum('bld,bln,bld->bldn', delta, B, u)  # (B, L, d_inner, d_state)
        
        # Spectral domain processing
        if self.training and hasattr(self, 'freq_mask') and self.freq_mask is not None:
            # Use adaptive frequency masking during training
            return self.adaptive_frequency_ssm_adaptive(deltaA, deltaB_u, C, u)
        else:
            # Use fixed compression ratio
            return self.adaptive_frequency_ssm_fixed(deltaA, deltaB_u, C, u)
    
    def adaptive_frequency_ssm_adaptive(self, deltaA, deltaB_u, C, u):
        """SSM with adaptive frequency selection"""
        batch, seqlen, d_inner, d_state = deltaA.shape

        # Convert to frequency domain and ensure consistent dimensions
        deltaA_freq = slice_low_frequencies(deltaA, self.k, dim=-1)
        deltaB_u_freq = slice_low_frequencies(deltaB_u, self.k, dim=-1)

        # Ensure both have the same compressed dimension
        compressed_dim = min(deltaA_freq.size(-1), deltaB_u_freq.size(-1))
        deltaA_freq = deltaA_freq[..., :compressed_dim]
        deltaB_u_freq = deltaB_u_freq[..., :compressed_dim]

        # Apply spectral normalization for stability
        if self.use_spectral_norm:
            deltaA_freq = spectral_norm_regularization(deltaA_freq)

        # Frequency dropout for regularization
        if self.training:
            deltaA_freq = frequency_dropout(deltaA_freq, p=0.1, training=True)

        # SSM recurrence in compressed frequency domain
        x_freq = self.parallel_scan_spectral(deltaA_freq, deltaB_u_freq)

        # Convert back to real domain for output projection
        if x_freq.dtype.is_complex:
            x_real = x_freq.real
        else:
            x_real = x_freq

        # Keep compressed state and compress C to match
        C_compressed = C[..., :compressed_dim]

        # Output projection in compressed domain
        y = torch.einsum('bldn,bln->bld', x_real, C_compressed)

        # Add skip connection
        y = y + u * self.D.unsqueeze(0).unsqueeze(0)

        return y
    
    def adaptive_frequency_ssm_fixed(self, deltaA, deltaB_u, C, u):
        """SSM with fixed compression ratio"""
        batch, seqlen, d_inner, d_state = deltaA.shape
        
        # Simple frequency domain compression
        deltaA_compressed = deltaA[..., :self.k]
        deltaB_u_compressed = deltaB_u[..., :self.k]
        C_compressed = C[..., :self.k]
        
        # SSM recurrence in compressed domain
        x_compressed = self.parallel_scan_spectral(deltaA_compressed, deltaB_u_compressed)
        
        # Output projection in compressed domain
        y = torch.einsum('bldn,bln->bld', x_compressed, C_compressed)
        
        # Add skip connection
        y = y + u * self.D.unsqueeze(0).unsqueeze(0)
        
        return y
    
    def parallel_scan_spectral(self, A, B):
        """
        Efficient parallel scan implementation using associative operations.
        Processes linear recurrence h[t] = A[t] * h[t-1] + B[t] in parallel.
        
        Args:
            A: (batch, seqlen, d_inner, d_state) transition matrices
            B: (batch, seqlen, d_inner, d_state) input terms
            
        Returns:
            x: (batch, seqlen, d_inner, d_state) scanned states
        """
        batch, seqlen, d_inner, d_state = A.shape
        
        # Use optimized implementation based on sequence length
        if seqlen <= 1024:
            return self._parallel_scan_associative(A, B)
        else:
            return self._parallel_scan_segmented(A, B, segment_size=1024)
    
    def _parallel_scan_associative(self, A, B):
        """Associative parallel scan using 2x2 matrix representation"""
        batch, seqlen, d_inner, d_state = A.shape

        # Ensure A and B have same dimensions
        assert A.shape == B.shape, f"A shape {A.shape} != B shape {B.shape}"

        # Convert to associative matrix form: [h[t], 1] = M[t] @ [h[t-1], 1]
        # where M[t] = [[A[t], B[t]], [0, 1]]
        matrices = torch.zeros(batch, seqlen, d_inner, d_state, 2, 2,
                              dtype=A.dtype, device=A.device)
        matrices[..., 0, 0] = A  # Transition component
        matrices[..., 0, 1] = B  # Input component
        matrices[..., 1, 1] = 1.0  # Identity for augmented dimension
        
        # Parallel prefix scan using tree reduction
        result_matrices = self._associative_scan(matrices)
        
        # Extract the accumulated B terms (states)
        return result_matrices[..., 0, 1]  # Shape: (batch, seqlen, d_inner, d_state)
    
    def _associative_scan(self, matrices):
        """Tree-based associative scan implementation"""
        batch, seqlen, d_inner, d_state, _, _ = matrices.shape
        
        # Fix: Add memory bounds checking to prevent exponential memory growth
        max_padded_len = min(2 ** 16, 4 * seqlen)  # Cap at 64K or 4x input size
        padded_len = 2 ** math.ceil(math.log2(seqlen)) if seqlen > 1 else 1

        if padded_len > max_padded_len:
            # Fall back to segmented processing for very long sequences
            return self._parallel_scan_segmented_safe(matrices, segment_size=8192)

        if padded_len > seqlen:
            # Pad with identity matrices
            padding = torch.zeros(batch, padded_len - seqlen, d_inner, d_state, 2, 2,
                                dtype=matrices.dtype, device=matrices.device)
            padding[..., 0, 0] = 1.0  # A = I
            padding[..., 1, 1] = 1.0  # Identity
            matrices = torch.cat([matrices, padding], dim=1)
        
        # Up-sweep phase: build reduction tree
        for step in range(int(math.log2(padded_len))):
            stride = 2 ** (step + 1)
            indices = torch.arange(stride - 1, padded_len, stride, device=matrices.device)
            
            if len(indices) > 0:
                left_idx = indices - stride // 2
                right_idx = indices
                
                # Matrix multiplication: matrices[right] = matrices[right] @ matrices[left]
                matrices[:, right_idx] = torch.matmul(
                    matrices[:, right_idx], 
                    matrices[:, left_idx]
                )
        
        # Down-sweep phase: distribute accumulated values
        # Set root to identity
        matrices[:, padded_len - 1, :, :, 0, 0] = 1.0
        matrices[:, padded_len - 1, :, :, 0, 1] = 0.0
        
        for step in range(int(math.log2(padded_len)) - 1, -1, -1):
            stride = 2 ** (step + 1)
            indices = torch.arange(stride - 1, padded_len, stride, device=matrices.device)
            
            if len(indices) > 0:
                left_idx = indices - stride // 2
                right_idx = indices
                
                # Save right nodes
                temp = matrices[:, right_idx].clone()
                
                # Move parent to right child
                matrices[:, right_idx] = matrices[:, left_idx].clone()
                
                # Update left child
                matrices[:, left_idx] = torch.matmul(temp, matrices[:, left_idx])
        
        # Return only the original sequence length
        return matrices[:, :seqlen]
    
    def _parallel_scan_segmented(self, A, B, segment_size=1024):
        """Memory-efficient segmented parallel scan for long sequences"""
        batch, seqlen, d_inner, d_state = A.shape
        
        segments = []
        running_state = torch.zeros(batch, d_inner, d_state, 
                                   dtype=A.dtype, device=A.device)
        
        for i in range(0, seqlen, segment_size):
            end_idx = min(i + segment_size, seqlen)
            
            # Process segment
            A_seg = A[:, i:end_idx]
            B_seg = B[:, i:end_idx].clone()
            
            # Incorporate running state into first element
            if i > 0:
                B_seg[:, 0] = A_seg[:, 0] * running_state + B_seg[:, 0]
            
            # Apply parallel scan to segment
            segment_result = self._parallel_scan_associative(A_seg, B_seg)
            segments.append(segment_result)
            
            # Update running state for next segment
            running_state = segment_result[:, -1]
        
        return torch.cat(segments, dim=1)

    def _parallel_scan_segmented_safe(self, matrices, segment_size=8192):
        """Safe segmented parallel scan with memory management for very long sequences"""
        batch, seqlen, d_inner, d_state, _, _ = matrices.shape

        segments = []
        # Initialize running state as identity matrix
        running_matrix = torch.eye(2, dtype=matrices.dtype, device=matrices.device)
        running_matrix = running_matrix.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(
            batch, d_inner, d_state, 2, 2
        )

        for i in range(0, seqlen, segment_size):
            end_idx = min(i + segment_size, seqlen)
            segment_len = end_idx - i

            # Process segment
            matrices_seg = matrices[:, i:end_idx]

            # Apply running state to first element of segment
            if i > 0:
                first_elem = matrices_seg[:, 0:1]  # Shape: (batch, 1, d_inner, d_state, 2, 2)
                # Matrix multiplication: first_elem = first_elem @ running_matrix
                matrices_seg = matrices_seg.clone()  # Avoid in-place modification
                matrices_seg[:, 0] = torch.matmul(first_elem[:, 0], running_matrix)

            # Apply associative scan to segment
            segment_result = self._associative_scan(matrices_seg)
            segments.append(segment_result)

            # Update running matrix for next segment
            if segment_len > 0:
                running_matrix = segment_result[:, -1]  # Last matrix of segment

        return torch.cat(segments, dim=1)
    
    def get_ssm_params(self, u):
        """Get SSM parameters following S6 design"""
        batch, seqlen, d_inner = u.shape
        
        # Project input to get dt, B, C
        x_dbl = self.x_proj(u)  # (B, L, dt_rank + 2*d_state)
        
        dt, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        
        # Process dt
        dt = self.dt_proj(dt)  # (B, L, d_inner)
        dt = F.softplus(dt + self.dt_proj.bias)
        
        return dt, B, C


class AdaptiveFrequencyResidualBlock(nn.Module):
    """
    Residual block with spectral processing for multi-scale information
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        compression_ratio: float = 0.5,
        num_scales: int = 3,
        **kwargs
    ):
        super().__init__()
        
        self.spectral_block = AdaptiveFrequencyS6Block(
            d_model=d_model,
            d_state=d_state,
            compression_ratio=compression_ratio,
            **kwargs
        )
        
        # Multi-scale processing
        self.num_scales = num_scales
        self.scale_weights = nn.Parameter(torch.ones(num_scales))
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(kwargs.get('dropout', 0.0)),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(kwargs.get('dropout', 0.0))
        )
    
    def forward(self, x):
        # Pre-norm for spectral block
        x_norm = self.norm1(x)
        spectral_out = self.spectral_block(x_norm)
        x = x + spectral_out
        
        # Pre-norm for FFN
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out
        
        return x


class MultiHeadAdaptiveFrequencyAttention(nn.Module):
    """
    Multi-head attention with spectral processing
    Can be used alongside spectral SSM for hybrid architectures
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_spectral: bool = True,
        compression_ratio: float = 0.5,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.use_spectral = use_spectral
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if use_spectral:
            self.spectral_processor = AdaptiveFrequencyMask(
                d_state=self.d_head, 
                compression_ratio=compression_ratio
            )
    
    def forward(self, x):
        batch, seqlen, d_model = x.shape
        
        # QKV projection
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head
        q = q.view(batch, seqlen, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(batch, seqlen, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(batch, seqlen, self.num_heads, self.d_head).transpose(1, 2)
        
        # Spectral processing for attention scores
        feature_dim = self.d_head
        if self.use_spectral and hasattr(self, 'spectral_processor'):
            q_freq, _ = self.spectral_processor(q, dim=-1)
            k_freq, _ = self.spectral_processor(k, dim=-1)
            # Fix: Properly handle complex to real conversion for attention
            q_real = torch.view_as_real(q_freq)  # Shape: (..., 2)
            k_real = torch.view_as_real(k_freq)  # Shape: (..., 2)
            # Flatten last two dimensions correctly
            q = q_real.reshape(batch, self.num_heads, seqlen, -1)
            k = k_real.reshape(batch, self.num_heads, seqlen, -1)
            feature_dim = q.shape[-1]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(feature_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch, seqlen, d_model)
        out = self.out_proj(out)
        
        return out

