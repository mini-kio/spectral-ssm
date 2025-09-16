"""
Training Script for Adaptive-Frequency-SSM
Supports distributed training, mixed precision, and comprehensive logging
"""

import os
import json
import math
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.amp import autocast

try:
    import wandb  # Optional; handled gracefully if missing
except Exception:  # pragma: no cover
    wandb = None
from tqdm import tqdm
import numpy as np

from .models.adaptive_frequency_ssm import (
    AdaptiveFrequencySSM, 
    AdaptiveFrequencySSMConfig, 
    create_adaptive_frequency_ssm_model
)


@dataclass
class TrainingConfig:
    """Training configuration"""
    
    # Model parameters
    model_name: str = "adaptive_frequency_ssm_base"
    d_model: int = 512
    n_layer: int = 12
    vocab_size: int = 50304
    compression_ratio: float = 0.5
    use_adaptive_mask: bool = True
    use_spectral_norm: bool = True
    
    # Training parameters
    batch_size: int = 32
    max_steps: int = 100000
    max_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip_norm: float = 1.0
    
    # Learning rate schedule
    warmup_steps: int = 1000
    lr_decay_steps: int = 80000
    min_lr_ratio: float = 0.1
    
    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "bfloat16"  # or "float16"
    
    # Regularization
    dropout: float = 0.1
    label_smoothing: float = 0.0
    
    # Data
    max_seq_len: int = 2048
    dataset_name: str = "lra"
    task_name: str = "listops"
    data_dir: str = "./data"
    
    # Logging and saving
    save_dir: str = "./checkpoints"
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    wandb_project: str = "adaptive_frequency_ssm"
    wandb_name: Optional[str] = None
    
    # Distributed training
    distributed: bool = False
    local_rank: int = 0
    world_size: int = 1
    
    # System
    device: str = "cuda"
    num_workers: int = 8
    pin_memory: bool = True
    compile_model: bool = True
    
    # Ablation study parameters
    ablation_mode: Optional[str] = None  # "compression_ratio", "adaptive_mask", etc.
    ablation_values: Optional[list] = None


class SpectralSSMTrainer:
    """Main trainer class for Spectral SSM"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float('inf') if 'loss' in config.task_name else 0.0
        
        # Setup distributed training
        if config.distributed:
            self.setup_distributed()
        
        # Setup device
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.set_device(config.local_rank)
            except Exception:
                pass
        else:
            self.device = torch.device("cpu")
        
        # Setup logging
        self.setup_logging()
        
        # Create model
        self.model = self.create_model()
        
        # Setup optimizer and scheduler
        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler()
        
        # Setup mixed precision
        if config.use_amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # Setup data loaders
        self.train_loader, self.val_loader = self.create_data_loaders()
        
        # Setup metrics tracking
        self.metrics = {}
        self.training_stats = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': [],
            'step_time': [],
            'throughput': [],
        }
        
        # Handle DDP-wrapped models when counting params
        base_model = self.model.module if isinstance(self.model, DDP) else self.model
        num_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        self.log_info(f"Trainer initialized with {num_params:,} parameters")
    
    def setup_distributed(self):
        """Setup distributed training"""
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            self.config.local_rank = int(os.environ['LOCAL_RANK'])
            self.config.world_size = int(os.environ['WORLD_SIZE'])
            rank = int(os.environ['RANK'])
            
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                rank=rank,
                world_size=self.config.world_size
            )
            
            # Adjust batch size for distributed training
            self.config.batch_size = self.config.batch_size // self.config.world_size
    
    def setup_logging(self):
        """Setup logging and wandb"""
        # Ensure save dir exists before creating FileHandler
        os.makedirs(self.config.save_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{self.config.save_dir}/training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize wandb (optional)
        self.wandb_enabled = False
        if self.config.local_rank == 0 and wandb is not None:
            try:
                wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.wandb_name,
                    config=asdict(self.config),
                    tags=["adaptive_frequency_ssm", self.config.task_name]
                )
                self.wandb_enabled = True
            except Exception:
                self.log_info("wandb not available; continuing without it.")
    
    def log_info(self, message: str):
        """Log info message"""
        if self.config.local_rank == 0:
            self.logger.info(message)
    
    def create_model(self) -> AdaptiveFrequencySSM:
        """Create and initialize model"""
        # Determine task type and number of classes
        if self.config.task_name in ['listops', 'text', 'retrieval', 'pathfinder', 'pathx']:
            task = "classification"
            num_classes = self.get_num_classes(self.config.task_name)
        else:
            task = "language_modeling"
            num_classes = None
        
        # Create model
        model = create_adaptive_frequency_ssm_model(
            task=task,
            d_model=self.config.d_model,
            n_layer=self.config.n_layer,
            vocab_size=self.config.vocab_size,
            num_classes=num_classes,
            compression_ratio=self.config.compression_ratio,
            use_adaptive_mask=self.config.use_adaptive_mask,
            use_spectral_norm=self.config.use_spectral_norm,
            dropout=self.config.dropout,
            max_seq_len=self.config.max_seq_len,
        )
        
        model = model.to(self.device)
        
        # Compile model for better performance
        if self.config.compile_model and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model)
                self.log_info("Model compiled successfully")
            except Exception as e:
                self.log_info(f"Model compilation failed: {e}")
        
        # Wrap with DDP for distributed training
        if self.config.distributed:
            model = DDP(model, device_ids=[self.config.local_rank])
        
        return model
    
    def get_num_classes(self, task_name: str) -> int:
        """Get number of classes for classification tasks"""
        task_classes = {
            'listops': 10,
            'text': 2,
            'retrieval': 2,
            'pathfinder': 2,
            'pathx': 2,
        }
        return task_classes.get(task_name, 2)
    
    def create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with proper weight decay handling"""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Don't apply weight decay to bias, layer norm, and embedding
                if any(nd in name for nd in ['bias', 'norm', 'embedding']):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optimizer_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        optimizer = torch.optim.AdamW(
            optimizer_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=1e-8
        )
        
        return optimizer
    
    def create_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        """Create learning rate scheduler with warmup and cosine decay"""
        def lr_lambda(step):
            # Warmup
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            
            # Cosine decay
            if step < self.config.lr_decay_steps:
                progress = (step - self.config.warmup_steps) / (self.config.lr_decay_steps - self.config.warmup_steps)
                return self.config.min_lr_ratio + 0.5 * (1 - self.config.min_lr_ratio) * (1 + math.cos(math.pi * progress))
            
            # Minimum learning rate
            return self.config.min_lr_ratio
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation data loaders"""
        # Prefer real datasets if available; otherwise, fall back to synthetic
        try:
            from .data.lra_datasets import create_lra_dataset
        except Exception:
            create_lra_dataset = None
        
        if create_lra_dataset is None:
            # Synthetic fallback dataset
            class _SyntheticDataset(torch.utils.data.Dataset):
                def __init__(self, num_samples=1024, seq_len=2048, vocab=50304, num_classes=2):
                    self.num_samples = num_samples
                    self.seq_len = seq_len
                    self.vocab = vocab
                    self.num_classes = num_classes
                def __len__(self):
                    return self.num_samples
                def __getitem__(self, idx):
                    x = torch.randint(0, self.vocab, (self.seq_len,), dtype=torch.long)
                    y = torch.randint(0, self.num_classes, (1,), dtype=torch.long).item()
                    return {"input_ids": x, "labels": torch.tensor(y, dtype=torch.long)}
            train_dataset = _SyntheticDataset(seq_len=self.config.max_seq_len)
            val_dataset = _SyntheticDataset(num_samples=256, seq_len=self.config.max_seq_len)
        else:
            train_dataset = create_lra_dataset(
                task_name=self.config.task_name,
                data_dir=self.config.data_dir,
                split='train',
                max_seq_len=self.config.max_seq_len,
            )
            val_dataset = create_lra_dataset(
                task_name=self.config.task_name,
                data_dir=self.config.data_dir,
                split='validation',
                max_seq_len=self.config.max_seq_len,
            )
        
        # Create samplers for distributed training
        train_sampler = DistributedSampler(train_dataset) if self.config.distributed else None
        val_sampler = DistributedSampler(val_dataset) if self.config.distributed else None
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=(self.config.num_workers > 0),
            drop_last=True,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=(self.config.num_workers > 0),
            drop_last=False,
        )
        
        return train_loader, val_loader
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        start_time = time.time()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device, non_blocking=True)
        labels = batch['labels'].to(self.device, non_blocking=True)
        
        # Forward pass with mixed precision
        device_type = self.device.type
        # Fix: Proper dtype selection based on device capabilities
        if device_type == 'cpu':
            # CPU doesn't support bfloat16, use float32
            amp_dtype = torch.float32
        elif device_type == 'cuda':
            # Check GPU capability for bfloat16
            if self.config.amp_dtype == 'bfloat16' and torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
            else:
                amp_dtype = torch.float16
        else:
            amp_dtype = torch.float32
        with torch.amp.autocast(device_type=device_type, enabled=self.config.use_amp, dtype=amp_dtype):
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs['loss']
            
            # Add label smoothing if specified
            if self.config.label_smoothing > 0:
                loss = self.apply_label_smoothing(outputs['logits'], labels, loss)
        
        # Backward pass
        if self.config.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        self.scheduler.step()
        
        # Calculate metrics
        step_time = time.time() - start_time
        throughput = input_ids.size(0) * input_ids.size(1) / step_time  # tokens per second
        
        return {
            'loss': loss.item(),
            'learning_rate': self.scheduler.get_last_lr()[0],
            'step_time': step_time,
            'throughput': throughput,
        }
    
    def apply_label_smoothing(self, logits: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor) -> torch.Tensor:
        """Apply label smoothing to loss"""
        confidence = 1.0 - self.config.label_smoothing
        smooth_loss = F.cross_entropy(logits, labels, reduction='mean')
        true_dist = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), confidence)
        true_dist += self.config.label_smoothing / logits.size(-1)
        kl_loss = F.kl_div(F.log_softmax(logits, dim=-1), true_dist, reduction='batchmean')
        return confidence * loss + self.config.label_smoothing * kl_loss
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating", disable=self.config.local_rank != 0):
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                
                device_type = self.device.type
                # Fix: Proper dtype selection based on device capabilities
                if device_type == 'cpu':
                    amp_dtype = torch.float32
                elif device_type == 'cuda':
                    if self.config.amp_dtype == 'bfloat16' and torch.cuda.is_bf16_supported():
                        amp_dtype = torch.bfloat16
                    else:
                        amp_dtype = torch.float16
                else:
                    amp_dtype = torch.float32
                with torch.amp.autocast(device_type=device_type, enabled=self.config.use_amp, dtype=amp_dtype):
                    outputs = self.model(input_ids=input_ids, labels=labels)
                    loss = outputs['loss']
                    logits = outputs['logits']
                
                total_loss += loss.item() * input_ids.size(0)
                
                # Calculate accuracy for classification tasks
                if len(logits.shape) > 1 and logits.size(-1) > 1:
                    predictions = torch.argmax(logits, dim=-1)
                    correct = (predictions == labels).sum().item()
                    total_correct += correct
                
                total_samples += input_ids.size(0)
        
        # Gather results from all processes
        if self.config.distributed:
            total_loss = self.gather_scalar(total_loss)
            total_correct = self.gather_scalar(total_correct)
            total_samples = self.gather_scalar(total_samples)
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        self.model.train()
        
        return {
            'val_loss': avg_loss,
            'val_accuracy': accuracy,
        }
    
    def gather_scalar(self, scalar: float) -> float:
        """Gather scalar across all processes"""
        tensor = torch.tensor(scalar, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.item()
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        if self.config.local_rank != 0:
            return
        
        checkpoint = {
            'model_state_dict': self.model.module.state_dict() if self.config.distributed else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_metric': self.best_metric,
            'config': asdict(self.config),
            'training_stats': self.training_stats,
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = Path(self.config.save_dir) / f"checkpoint_step_{self.global_step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = Path(self.config.save_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)
        
        # Keep only latest checkpoints
        self.cleanup_checkpoints()
    
    def cleanup_checkpoints(self, keep_last: int = 3):
        """Remove old checkpoints to save space"""
        checkpoint_dir = Path(self.config.save_dir)
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_step_*.pt"))
        
        if len(checkpoints) > keep_last:
            for checkpoint in checkpoints[:-keep_last]:
                checkpoint.unlink()
    
    def train(self):
        """Main training loop"""
        self.log_info("Starting training...")
        
        # Clear GPU cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Create save directory
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        # Save config
        if self.config.local_rank == 0:
            with open(Path(self.config.save_dir) / "config.json", 'w') as f:
                json.dump(asdict(self.config), f, indent=2)
        
        self.model.train()
        progress_bar = tqdm(
            total=self.config.max_steps, 
            desc="Training",
            disable=self.config.local_rank != 0
        )
        
        while self.global_step < self.config.max_steps and self.epoch < self.config.max_epochs:
            # Set epoch for distributed sampler
            if self.config.distributed and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(self.epoch)
            
            for batch in self.train_loader:
                if self.global_step >= self.config.max_steps:
                    break
                
                # Training step
                step_metrics = self.train_step(batch)
                
                # Update progress bar
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'loss': f"{step_metrics['loss']:.4f}",
                    'lr': f"{step_metrics['learning_rate']:.2e}",
                    'toks/s': f"{step_metrics['throughput']:.0f}",
                })
                
                # Log metrics
                if self.global_step % self.config.log_interval == 0:
                    self.log_metrics(step_metrics)
                
                # Evaluation
                if self.global_step % self.config.eval_interval == 0:
                    eval_metrics = self.evaluate()
                    self.log_metrics(eval_metrics)
                    
                    # Check if best model
                    is_best = False
                    if 'val_accuracy' in eval_metrics:
                        if eval_metrics['val_accuracy'] > self.best_metric:
                            self.best_metric = eval_metrics['val_accuracy']
                            is_best = True
                    
                    # Save checkpoint
                    if self.global_step % self.config.save_interval == 0 or is_best:
                        self.save_checkpoint(is_best)
                
                self.global_step += 1
            
            self.epoch += 1
        
        progress_bar.close()
        
        # Final evaluation and save
        final_metrics = self.evaluate()
        self.log_metrics(final_metrics)
        self.save_checkpoint()
        
        self.log_info("Training completed!")
        
        # Clear GPU cache after training
        if self.device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if self.config.local_rank == 0 and getattr(self, 'wandb_enabled', False):
            try:
                wandb.finish()
            except Exception:
                pass
    
    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to wandb and console"""
        if self.config.local_rank != 0:
            return
        
        # Add to training stats
        for key, value in metrics.items():
            if key in self.training_stats:
                self.training_stats[key].append(value)
        
        # Log to wandb
        if getattr(self, 'wandb_enabled', False):
            try:
                wandb.log({**metrics, 'step': self.global_step})
            except Exception:
                pass
        
        # Log to console
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.log_info(f"Step {self.global_step} | {metric_str}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train Spectral SSM")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="adaptive_frequency_ssm_base")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--compression_ratio", type=float, default=0.5)
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    
    # Data arguments
    parser.add_argument("--task_name", type=str, default="listops", 
                       choices=["listops", "text", "retrieval", "pathfinder", "pathx"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--max_seq_len", type=int, default=2048)
    
    # System arguments
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--wandb_project", type=str, default="adaptive_frequency_ssm")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    
    # Distributed training
    parser.add_argument("--distributed", action="store_true")
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(**vars(args))
    
    # Create trainer and start training
    trainer = SpectralSSMTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
