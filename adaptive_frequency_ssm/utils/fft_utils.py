"""
FFT Utilities for Adaptive-Frequency-SSM
Implements advanced frequency domain operations for state compression
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


def _get_fft_dtype(input_dtype: torch.dtype) -> Tuple[torch.dtype, bool]:
    """
    Determine appropriate dtype for FFT operations based on input dtype.
    Returns: (fft_dtype, needs_conversion_back)
    """
    if input_dtype == torch.bfloat16:
        # bfloat16 not supported in FFT, use float32 for better precision
        return torch.float32, True
    elif input_dtype == torch.float16:
        # float16 may have precision issues, consider using float32 for FFT
        return torch.float32, True
    else:
        # float32, float64, etc. are supported natively
        return input_dtype, False


def _fft_with_dtype_handling(x: torch.Tensor, operation: str, *args, **kwargs) -> torch.Tensor:
    """Core FFT operation with proper dtype handling"""
    original_dtype = x.dtype
    fft_dtype, needs_conversion = _get_fft_dtype(original_dtype)

    # Convert to appropriate dtype for FFT if needed
    if needs_conversion:
        x = x.to(fft_dtype)

    # Perform FFT operation
    if operation == 'rfft':
        result = torch.fft.rfft(x, *args, **kwargs)
    elif operation == 'irfft':
        result = torch.fft.irfft(x, *args, **kwargs)
    elif operation == 'fft':
        result = torch.fft.fft(x, *args, **kwargs)
    elif operation == 'ifft':
        result = torch.fft.ifft(x, *args, **kwargs)
    else:
        raise ValueError(f"Unknown FFT operation: {operation}")

    # Convert back to original dtype if needed and result is real
    if needs_conversion and not result.dtype.is_complex:
        result = result.to(original_dtype)

    return result


def real_fft(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Real FFT with proper dtype handling for spectral analysis"""
    return _fft_with_dtype_handling(x, 'rfft', dim=dim, norm='ortho')


def real_ifft(x: torch.Tensor, n: Optional[int] = None, dim: int = -1) -> torch.Tensor:
    """Inverse real FFT with proper dtype handling"""
    return _fft_with_dtype_handling(x, 'irfft', n=n, dim=dim, norm='ortho')



def slice_low_frequencies(x: torch.Tensor, k: int, dim: int = -1) -> torch.Tensor:
    """
    Extract the first k low-frequency components after FFT.
    This serves as our primary compression method.
    """
    freqs = real_fft(x, dim=dim)
    freq_size = freqs.size(dim)

    # Ensure k doesn't exceed available frequencies
    k_safe = min(k, freq_size)

    if k_safe <= 0:
        raise ValueError(f"Invalid k={k} for frequency dimension size {freq_size}")

    return freqs.narrow(dim, 0, k_safe)


def pad_and_reconstruct(x_compressed: torch.Tensor, original_size: int, dim: int = -1) -> torch.Tensor:
    """
    Pad compressed frequencies and reconstruct signal
    """
    # Build the padded tensor for reconstruction
    pad_size = original_size // 2 + 1 - x_compressed.size(dim)  # rfft output size
    
    if pad_size > 0:
        # Create padding tensor
        pad_shape = list(x_compressed.shape)
        pad_shape[dim] = pad_size
        padding = torch.zeros(pad_shape, dtype=x_compressed.dtype, device=x_compressed.device)
        
        # Concatenate
        x_padded = torch.cat([x_compressed, padding], dim=dim)
    else:
        x_padded = x_compressed
    
    return real_ifft(x_padded, n=original_size, dim=dim)


class AdaptiveFrequencyMask(torch.nn.Module):
    """
    Learnable mask for adaptive frequency selection
    """
    def __init__(self, d_state: int, compression_ratio: float = 0.5):
        super().__init__()
        self.d_state = d_state
        freq_size = d_state // 2 + 1  # rfft output size
        target_k = int(round(d_state * compression_ratio))
        self.k = max(1, min(freq_size, target_k))
        
        # Learnable importance weights for each frequency
        self.freq_weights = torch.nn.Parameter(torch.ones(freq_size))
        self.temperature = torch.nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x: torch.Tensor, dim: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply adaptive frequency masking
        Returns: (compressed_freqs, selection_mask)
        """
        freqs = real_fft(x, dim=dim)
        
        # Compute importance scores
        importance = torch.sigmoid(self.freq_weights / self.temperature.abs())
        
        # Select the most important frequencies using learned weights
        _, indices = torch.topk(importance, self.k)
        indices = indices.sort().values  # Keep frequency order
        
        # Create selection mask
        mask = torch.zeros_like(importance, dtype=torch.bool)
        mask[indices] = True
        
        # Extract the selected frequency components
        if dim == -1:
            compressed_freqs = freqs[..., mask]
        else:
            # Fix: Proper mask broadcasting for arbitrary dimensions
            mask_shape = [1] * freqs.ndim
            mask_shape[dim] = mask.size(0)
            mask_expanded = mask.view(mask_shape)

            # Create boolean mask by expanding to match freqs dimensions
            bool_mask = mask_expanded.expand_as(freqs)

            # Use advanced indexing to select frequencies
            selected_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            compressed_freqs = torch.index_select(freqs, dim, selected_indices)
        
        return compressed_freqs, mask


def frequency_dropout(x: torch.Tensor, p: float = 0.1, training: bool = True) -> torch.Tensor:
    """
    Randomly drop frequency components during training for regularization
    """
    if not training or p == 0:
        return x
    
    mask = torch.rand_like(x.real) > p
    return x * mask.to(x.dtype)




def spectral_norm_regularization(A_freq: torch.Tensor, max_eigenval: float = 1.0) -> torch.Tensor:
    """
    Spectral normalization for frequency domain matrices
    Ensures stability by constraining largest eigenvalue
    """
    eigenvals = torch.abs(A_freq)
    max_eig = torch.max(eigenvals)
    
    if max_eig > max_eigenval:
        return A_freq * (max_eigenval / max_eig)
    return A_freq