"""
Evaluation and Metrics for Spectral-Latent SSM
Comprehensive evaluation suite for LRA benchmark and ablation studies
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from .models.adaptive_frequency_ssm import AdaptiveFrequencySSM, AdaptiveFrequencySSMConfig
from .data.lra_datasets import create_lra_dataloader, LRABenchmark
from .train import TrainingConfig


@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""
    
    # Model and checkpoint
    checkpoint_path: str
    model_config_path: Optional[str] = None
    
    # Evaluation settings
    batch_size: int = 64
    device: str = "cuda"
    use_amp: bool = True
    
    # Data
    data_dir: str = "./data"
    tasks: List[str] = None  # None means all LRA tasks
    
    # Output
    output_dir: str = "./evaluation_results"
    save_predictions: bool = False
    save_plots: bool = True
    
    # Performance analysis
    measure_throughput: bool = True
    measure_memory: bool = True
    profile_layers: bool = False
    
    # Ablation studies
    compression_ratios: List[float] = None
    adaptive_mask_variants: List[bool] = None
    
    def __post_init__(self):
        if self.tasks is None:
            self.tasks = ['listops', 'text', 'retrieval', 'pathfinder']
        if self.compression_ratios is None:
            self.compression_ratios = [0.25, 0.5, 0.75]
        if self.adaptive_mask_variants is None:
            self.adaptive_mask_variants = [True, False]


class MetricsCalculator:
    """Calculate various metrics for evaluation"""
    
    @staticmethod
    def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate accuracy"""
        correct = (predictions == targets).sum().item()
        total = targets.size(0)
        return correct / total
    
    @staticmethod
    def top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
        """Calculate top-k accuracy"""
        _, top_k_preds = torch.topk(logits, k, dim=-1)
        targets_expanded = targets.unsqueeze(-1).expand_as(top_k_preds)
        correct = (top_k_preds == targets_expanded).any(dim=-1).sum().item()
        total = targets.size(0)
        return correct / total
    
    @staticmethod
    def f1_score(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int) -> Dict[str, float]:
        """Calculate F1 score (macro and weighted)"""
        from sklearn.metrics import f1_score, classification_report
        
        pred_np = predictions.cpu().numpy()
        target_np = targets.cpu().numpy()
        
        macro_f1 = f1_score(target_np, pred_np, average='macro', zero_division=0)
        weighted_f1 = f1_score(target_np, pred_np, average='weighted', zero_division=0)
        
        return {
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1
        }
    
    @staticmethod
    def confusion_matrix(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int) -> np.ndarray:
        """Calculate confusion matrix"""
        from sklearn.metrics import confusion_matrix
        
        pred_np = predictions.cpu().numpy()
        target_np = targets.cpu().numpy()
        
        return confusion_matrix(target_np, pred_np, labels=range(num_classes))
    
    @staticmethod
    def perplexity(logits: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate perplexity for language modeling"""
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return torch.exp(loss).item()


class PerformanceProfiler:
    """Profile model performance and efficiency"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.results = {}
    
    def measure_throughput(
        self, 
        model: nn.Module, 
        dataloader: DataLoader,
        num_batches: int = 50
    ) -> Dict[str, float]:
        """Measure inference throughput"""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= 5:  # 5 warmup batches
                    break
                input_ids = batch['input_ids'].to(self.device)
                _ = model(input_ids)
        
        # Measure
        if torch.cuda.is_available() and self.device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()
        total_tokens = 0
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                _ = model(input_ids)
                total_tokens += input_ids.numel()
        
        if torch.cuda.is_available() and self.device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        throughput = total_tokens / elapsed_time
        
        return {
            'tokens_per_second': throughput,
            'batches_per_second': num_batches / elapsed_time,
            'elapsed_time': elapsed_time,
            'total_tokens': total_tokens,
        }
    
    def measure_memory_usage(
        self, 
        model: nn.Module, 
        input_shape: Tuple[int, int],
        batch_sizes: List[int] = [1, 8, 16, 32, 64]
    ) -> Dict[str, Dict[str, float]]:
        """Measure memory usage for different batch sizes"""
        model.eval()
        results = {}

        if not (torch.cuda.is_available() and self.device.type == 'cuda'):
            return {f'batch_{b}': {'error': 'CUDA not available'} for b in batch_sizes}

        for batch_size in batch_sizes:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Create dummy input
            input_ids = torch.randint(0, 1000, (batch_size, input_shape[1]), device=self.device)
            
            try:
                with torch.no_grad():
                    _ = model(input_ids)
                
                peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
                current_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                
                results[f'batch_{batch_size}'] = {
                    'peak_memory_gb': peak_memory,
                    'current_memory_gb': current_memory,
                    'memory_per_sample_mb': (peak_memory * 1024) / batch_size,
                }
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    results[f'batch_{batch_size}'] = {'error': 'OOM'}
                else:
                    results[f'batch_{batch_size}'] = {'error': str(e)}
        
        return results
    
    def profile_layer_timing(
        self, 
        model: nn.Module, 
        input_ids: torch.Tensor,
        num_runs: int = 100
    ) -> Dict[str, float]:
        """Profile individual layer timing"""
        model.eval()
        
        # Hook to measure layer times
        layer_times = {}
        
        def forward_hook(name):
            def hook(module, input, output):
                if torch.cuda.is_available() and self.device.type == 'cuda':
                    torch.cuda.synchronize()
                layer_times[name] = time.time()
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                if torch.cuda.is_available() and self.device.type == 'cuda':
                    torch.cuda.synchronize()
                if name in layer_times:
                    layer_times[name] = time.time() - layer_times[name]
            return hook
        
        # Register hooks
        hooks = []
        if hasattr(model, 'model'):  # Wrapped model
            backbone = model.model.backbone if hasattr(model.model, 'backbone') else model.model
        else:
            backbone = model.backbone if hasattr(model, 'backbone') else model
        
        for i, layer in enumerate(backbone.layers):
            hook_handle = layer.register_forward_hook(forward_hook(f'layer_{i}'))
            hooks.append(hook_handle)
        
        # Run profiling
        total_times = {name: [] for name in layer_times.keys()}
        
        with torch.no_grad():
            for _ in range(num_runs):
                layer_times.clear()
                
                if torch.cuda.is_available() and self.device.type == 'cuda':
                    torch.cuda.synchronize()
                start_time = time.time()
                
                _ = model(input_ids)
                
                if torch.cuda.is_available() and self.device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                
                # Record times
                total_time = end_time - start_time
                for name, duration in layer_times.items():
                    total_times[name].append(duration)
        
        # Cleanup hooks
        for hook in hooks:
            hook.remove()
        
        # Calculate averages
        avg_times = {}
        for name, times in total_times.items():
            if times:
                avg_times[name] = {
                    'mean_ms': np.mean(times) * 1000,
                    'std_ms': np.std(times) * 1000,
                    'min_ms': np.min(times) * 1000,
                    'max_ms': np.max(times) * 1000,
                }
        
        return avg_times
    
    def calculate_flops(
        self, 
        model: nn.Module, 
        input_shape: Tuple[int, int]
    ) -> Dict[str, float]:
        """Estimate FLOPs for the model"""
        # This is a simplified estimation
        # In practice, you'd use tools like fvcore or ptflops
        
        if hasattr(model, 'get_flops_per_token'):
            flops_per_token = model.get_flops_per_token(input_shape[1])
            total_flops = flops_per_token * input_shape[0] * input_shape[1]
            
            return {
                'flops_per_token': flops_per_token,
                'total_flops': total_flops,
                'gflops': total_flops / 1e9,
            }
        else:
            return {'error': 'FLOPs calculation not available'}


class SpectralSSMEvaluator:
    """Main evaluator for Spectral SSM"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Initialize components
        self.model = self.load_model()
        self.metrics_calculator = MetricsCalculator()
        self.profiler = PerformanceProfiler(config.device)
        
        # Results storage
        self.results = {}
        
    def load_model(self) -> AdaptiveFrequencySSM:
        """Load model from checkpoint"""
        checkpoint = torch.load(self.config.checkpoint_path, map_location=self.device)
        
        # Load config
        if self.config.model_config_path:
            with open(self.config.model_config_path, 'r') as f:
                model_config = json.load(f)
        else:
            model_config = checkpoint.get('config', {})
        
        # Create model
        from .models.adaptive_frequency_ssm import create_adaptive_frequency_ssm_model
        
        model = create_adaptive_frequency_ssm_model(**model_config)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        self.logger.info(f"Loaded model with {model.get_num_params():,} parameters")
        
        return model
    
    def evaluate_task(self, task_name: str) -> Dict[str, Any]:
        """Evaluate model on a specific LRA task"""
        self.logger.info(f"Evaluating on {task_name}")
        
        # Create data loader
        benchmark = LRABenchmark(self.config.data_dir)
        task_config = benchmark.get_task_config(task_name)
        
        test_loader = create_lra_dataloader(
            task_name=task_name,
            data_dir=self.config.data_dir,
            split='test',
            batch_size=self.config.batch_size,
            max_seq_len=task_config['max_seq_len'],
            shuffle=False,
        )
        
        # Evaluate
        total_loss = 0
        all_predictions = []
        all_targets = []
        all_logits = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Evaluating {task_name}"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                device_type = self.device.type
                amp_dtype = torch.bfloat16 if (device_type == 'cpu') else torch.bfloat16
                with torch.amp.autocast(device_type=device_type, enabled=self.config.use_amp, dtype=amp_dtype):
                    outputs = self.model(input_ids=input_ids, labels=labels)
                    loss = outputs['loss']
                    logits = outputs['logits']
                
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.append(predictions.cpu())
                all_targets.append(labels.cpu())
                all_logits.append(logits.cpu())
        
        # Concatenate results
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        all_logits = torch.cat(all_logits)
        
        # Calculate metrics
        avg_loss = total_loss / len(test_loader)
        accuracy = self.metrics_calculator.accuracy(all_predictions, all_targets)
        
        results = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'num_samples': len(all_targets),
        }
        
        # Additional metrics for classification
        if len(all_logits.shape) > 1 and all_logits.size(-1) > 1:
            num_classes = all_logits.size(-1)
            
            # F1 scores
            f1_scores = self.metrics_calculator.f1_score(all_predictions, all_targets, num_classes)
            results.update(f1_scores)
            
            # Top-k accuracy
            if num_classes > 5:
                top5_acc = self.metrics_calculator.top_k_accuracy(all_logits, all_targets, k=5)
                results['top5_accuracy'] = top5_acc
            
            # Confusion matrix
            cm = self.metrics_calculator.confusion_matrix(all_predictions, all_targets, num_classes)
            results['confusion_matrix'] = cm.tolist()
        
        # Save predictions if requested
        if self.config.save_predictions:
            pred_file = Path(self.config.output_dir) / f"{task_name}_predictions.pt"
            torch.save({
                'predictions': all_predictions,
                'targets': all_targets,
                'logits': all_logits,
            }, pred_file)
        
        return results
    
    def run_performance_analysis(self) -> Dict[str, Any]:
        """Run comprehensive performance analysis"""
        self.logger.info("Running performance analysis")
        
        results = {}
        
        # Create a sample data loader for analysis
        sample_loader = create_lra_dataloader(
            task_name='listops',
            data_dir=self.config.data_dir,
            split='test',
            batch_size=self.config.batch_size,
            max_seq_len=2048,
            shuffle=False,
        )
        
        # Throughput analysis
        if self.config.measure_throughput:
            throughput_results = self.profiler.measure_throughput(self.model, sample_loader)
            results['throughput'] = throughput_results
        
        # Memory analysis
        if self.config.measure_memory:
            memory_results = self.profiler.measure_memory_usage(self.model, (1, 2048))
            results['memory'] = memory_results
        
        # Layer profiling
        if self.config.profile_layers:
            sample_batch = next(iter(sample_loader))
            input_ids = sample_batch['input_ids'][:1].to(self.device)  # Single sample
            layer_times = self.profiler.profile_layer_timing(self.model, input_ids)
            results['layer_timing'] = layer_times
        
        # FLOPs estimation
        flops_results = self.profiler.calculate_flops(self.model, (self.config.batch_size, 2048))
        results['flops'] = flops_results
        
        return results
    
    def run_ablation_study(self) -> Dict[str, Any]:
        """Run ablation study on compression ratios and other variants"""
        self.logger.info("Running ablation study")
        
        results = {}
        
        # Test different compression ratios
        for ratio in self.config.compression_ratios:
            self.logger.info(f"Testing compression ratio: {ratio}")
            
            # Modify model configuration
            original_ratio = self.model.config.compression_ratio
            self.model.config.compression_ratio = ratio
            
            # Update all spectral blocks
            self._update_compression_ratio(ratio)
            
            # Evaluate on a subset of tasks
            ratio_results = {}
            for task in ['listops', 'text']:  # Quick evaluation
                task_results = self.evaluate_task(task)
                ratio_results[task] = {
                    'accuracy': task_results['accuracy'],
                    'loss': task_results['loss']
                }
            
            results[f'compression_{ratio}'] = ratio_results
            
            # Restore original ratio
            self.model.config.compression_ratio = original_ratio
            self._update_compression_ratio(original_ratio)
        
        return results
    
    def _update_compression_ratio(self, ratio: float):
        """Update compression ratio in all spectral blocks"""
        def update_block(module):
            if hasattr(module, 'compression_ratio'):
                module.compression_ratio = ratio
            if hasattr(module, 'd_state') and hasattr(module, 'k'):
                ds = int(module.d_state)
                module.k = max(1, min(ds, int(round(ds * ratio))))
            if hasattr(module, 'freq_mask') and module.freq_mask is not None:
                fs = getattr(module.freq_mask, 'd_state', None)
                if fs is not None:
                    freq_size = fs // 2 + 1
                    module.freq_mask.k = max(1, min(freq_size, int(round(fs * ratio))))
        
        # Apply to all modules
        self.model.apply(update_block)
    
    def generate_plots(self):
        """Generate visualization plots"""
        if not self.config.save_plots:
            return
        
        self.logger.info("Generating plots")
        
        # LRA benchmark results plot
        if 'lra_results' in self.results:
            self._plot_lra_results()
        
        # Performance analysis plots
        if 'performance' in self.results:
            self._plot_performance_analysis()
        
        # Ablation study plots
        if 'ablation' in self.results:
            self._plot_ablation_study()
    
    def _plot_lra_results(self):
        """Plot LRA benchmark results"""
        lra_data = self.results['lra_results']
        
        tasks = list(lra_data.keys())
        accuracies = [lra_data[task]['accuracy'] for task in tasks]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(tasks, accuracies)
        plt.ylabel('Accuracy')
        plt.title('Spectral SSM Performance on LRA Benchmark')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(Path(self.config.output_dir) / 'lra_results.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_analysis(self):
        """Plot performance analysis results"""
        perf_data = self.results['performance']
        
        # Memory usage plot
        if 'memory' in perf_data:
            memory_data = perf_data['memory']
            batch_sizes = []
            peak_memories = []
            
            for key, value in memory_data.items():
                if 'error' not in value:
                    batch_size = int(key.split('_')[1])
                    batch_sizes.append(batch_size)
                    peak_memories.append(value['peak_memory_gb'])
            
            if batch_sizes:
                plt.figure(figsize=(8, 6))
                plt.plot(batch_sizes, peak_memories, 'o-')
                plt.xlabel('Batch Size')
                plt.ylabel('Peak Memory (GB)')
                plt.title('Memory Usage vs Batch Size')
                plt.grid(True)
                plt.savefig(Path(self.config.output_dir) / 'memory_usage.png', dpi=300, bbox_inches='tight')
                plt.close()
    
    def _plot_ablation_study(self):
        """Plot ablation study results"""
        ablation_data = self.results['ablation']
        
        # Compression ratio ablation
        ratios = []
        accuracies = []
        
        for key, value in ablation_data.items():
            if key.startswith('compression_'):
                ratio = float(key.split('_')[1])
                ratios.append(ratio)
                # Average accuracy across tasks
                task_accs = [task_data['accuracy'] for task_data in value.values()]
                accuracies.append(np.mean(task_accs))
        
        if ratios:
            plt.figure(figsize=(8, 6))
            plt.plot(ratios, accuracies, 'o-')
            plt.xlabel('Compression Ratio')
            plt.ylabel('Average Accuracy')
            plt.title('Compression Ratio vs Performance')
            plt.grid(True)
            plt.savefig(Path(self.config.output_dir) / 'compression_ablation.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation suite"""
        self.logger.info("Starting full evaluation")
        
        # LRA benchmark evaluation
        lra_results = {}
        for task in self.config.tasks:
            lra_results[task] = self.evaluate_task(task)
        
        self.results['lra_results'] = lra_results
        
        # Performance analysis
        if any([self.config.measure_throughput, self.config.measure_memory, self.config.profile_layers]):
            perf_results = self.run_performance_analysis()
            self.results['performance'] = perf_results
        
        # Ablation study
        if self.config.compression_ratios and len(self.config.compression_ratios) > 1:
            ablation_results = self.run_ablation_study()
            self.results['ablation'] = ablation_results
        
        # Generate plots
        self.generate_plots()
        
        # Save results
        self.save_results()
        
        self.logger.info("Evaluation completed")
        
        return self.results
    
    def save_results(self):
        """Save evaluation results"""
        results_file = Path(self.config.output_dir) / 'evaluation_results.json'
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(v) for v in obj]
            else:
                return obj
        
        json_results = convert_for_json(self.results)
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        self.logger.info(f"Results saved to {results_file}")
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("SPECTRAL SSM EVALUATION SUMMARY")
        print("="*60)
        
        if 'lra_results' in self.results:
            print("\nLRA Benchmark Results:")
            print("-" * 30)
            for task, results in self.results['lra_results'].items():
                acc = results['accuracy']
                loss = results['loss']
                print(f"{task:12s}: Acc={acc:.4f}, Loss={loss:.4f}")
            
            # Calculate average
            avg_acc = np.mean([r['accuracy'] for r in self.results['lra_results'].values()])
            print(f"{'Average':12s}: Acc={avg_acc:.4f}")
        
        if 'performance' in self.results:
            print("\nPerformance Analysis:")
            print("-" * 30)
            perf = self.results['performance']
            
            if 'throughput' in perf:
                tps = perf['throughput']['tokens_per_second']
                print(f"Throughput: {tps:.0f} tokens/sec")
            
            if 'flops' in perf and 'gflops' in perf['flops']:
                gflops = perf['flops']['gflops']
                print(f"GFLOPs: {gflops:.2f}")
        
        print("\n" + "="*60)


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate Spectral SSM")
    
    # Required arguments
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to model checkpoint")
    
    # Optional arguments
    parser.add_argument("--model_config_path", type=str, default=None,
                       help="Path to model config JSON")
    parser.add_argument("--data_dir", type=str, default="./data",
                       help="Data directory")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                       help="Output directory")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    
    # Task selection
    parser.add_argument("--tasks", nargs='+', 
                       choices=['listops', 'text', 'retrieval', 'pathfinder'],
                       help="Tasks to evaluate (default: all)")
    
    # Analysis options
    parser.add_argument("--measure_throughput", action="store_true",
                       help="Measure inference throughput")
    parser.add_argument("--measure_memory", action="store_true", 
                       help="Measure memory usage")
    parser.add_argument("--profile_layers", action="store_true",
                       help="Profile individual layers")
    
    # Ablation study
    parser.add_argument("--compression_ratios", nargs='+', type=float,
                       default=[0.25, 0.5, 0.75],
                       help="Compression ratios for ablation study")
    
    args = parser.parse_args()
    
    # Create config
    config = EvaluationConfig(**vars(args))
    
    # Run evaluation
    evaluator = SpectralSSMEvaluator(config)
    results = evaluator.run_full_evaluation()
    
    return results


if __name__ == "__main__":
    main()
