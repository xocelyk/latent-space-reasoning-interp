import torch
import numpy as np
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from typing import List, Tuple, Dict, Optional, Union
import seaborn as sns
import gc
import time
from tqdm.notebook import tqdm

class LightweightAnalyzer:
    """
    A more lightweight version of the residual stream analyzer for MacBook compatibility.
    """
    
    def __init__(self, model_name: str = "gpt2-small", use_cpu_only: bool = True):
        """
        Initialize the analyzer with a specific model.
        
        Args:
            model_name: Name of the model to analyze (any model supported by TransformerLens)
            use_cpu_only: Force CPU usage only (often more stable on MacBooks)
        """
        print(f"Initializing with model: {model_name}")
        
        # Force CPU for better stability
        if use_cpu_only:
            self.device = torch.device("cpu")
            print("Forcing CPU usage for better stability")
        else:
            # Determine the best available device
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("Using MPS (Metal Performance Shaders) for Apple Silicon")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("Using CUDA GPU")
            else:
                self.device = torch.device("cpu")
                print("Using CPU")
        
        # Load model with minimal components
        print("Loading model...")
        self.model = HookedTransformer.from_pretrained(
            model_name,
            device=self.device
        )
        
        self.d_model = self.model.cfg.d_model
        print(f"Model loaded with d_model = {self.d_model}")
        
        # Get unembedding matrix
        print("Getting unembedding matrix...")
        self.unembed = self.model.W_U.detach().cpu().numpy()  # Get as numpy directly
        print(f"Unembedding shape: {self.unembed.shape}")
        
        # Run garbage collection
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def analyze_projection_simple(self, variance_threshold: float = 0.95) -> Dict:
        """
        Perform a simplified analysis using NumPy operations which are more stable.
        
        Args:
            variance_threshold: Percentage of variance to capture
            
        Returns:
            Dictionary with results
        """
        print("Computing SVD with NumPy (this might take a few moments)...")
        start_time = time.time()
        
        # IMPORTANT CORRECTION:
        # We want directions in d_model space (768) that capture most of the variance
        # when projecting to vocabulary space
        # So we need to transpose the unembedding matrix first
        unembed_T = self.unembed.T  # Shape: [d_model, vocab_size]
        
        # Now SVD will find directions in d_model space
        U, S, _ = np.linalg.svd(unembed_T, full_matrices=False)
        
        print(f"SVD completed in {time.time() - start_time:.2f} seconds")
        
        # Calculate cumulative explained variance
        total_variance = np.sum(S**2)
        cumulative_variance = np.cumsum(S**2) / total_variance
        
        # Find number of components for threshold
        k = np.sum(cumulative_variance <= variance_threshold) + 1
        k = min(k, len(S))
        print(f"Number of components for {variance_threshold*100}% variance: {k}")
        
        # Calculate top-k components contribution
        topk_variance_ratio = np.sum(S[:k]**2) / total_variance
        print(f"Top-{k} components explain {topk_variance_ratio*100:.2f}% of variance")
        
        # Create simplified results
        results = {
            "singular_values": S,
            "cumulative_variance": cumulative_variance,
            "top_k": k,
            "topk_variance_ratio": topk_variance_ratio,
            "variance_threshold": variance_threshold,
            "singular_vectors": U  # Save the singular vectors for later use
        }
        
        return results
    
    def compute_projection_matrix(self, results: Dict) -> np.ndarray:
        """
        Compute the projection matrix for the token prediction subspace.
        
        Args:
            results: Results from analyze_projection_simple
            
        Returns:
            Projection matrix as numpy array
        """
        print("Computing projection matrix...")
        
        # Get the singular vectors saved from SVD
        U = results["singular_vectors"]
        
        # Get the important vectors corresponding to largest singular values
        k = results["top_k"]
        print(f"Using top {k} singular vectors from a total of {U.shape[1]}")
        
        # Get the directions in d_model space that matter most for token prediction
        V_selected = U[:, :k]  # Shape: [d_model, k]
        print(f"V shape: {V_selected.shape}")
        
        # Create projection matrix in d_model space
        P = V_selected.T @ V_selected  # Shape: [d_model, d_model]
        print(f"Projection matrix shape: {P.shape}")
        
        return P
    
    def visualize_singular_values(self, 
                                 results: Dict, 
                                 max_values: int = 100,
                                 figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Visualize the singular values and cumulative explained variance.
        
        Args:
            results: Results dictionary from analyze_projection_simple
            max_values: Maximum number of singular values to plot
            figsize: Figure size
        """
        S = results["singular_values"]
        cumulative_var = results["cumulative_variance"]
        variance_threshold = results["variance_threshold"]
        k = results["top_k"]
        
        max_values = min(max_values, len(S))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot singular values
        ax1.plot(range(1, max_values + 1), S[:max_values])
        ax1.set_xlabel("Index")
        ax1.set_ylabel("Singular Value")
        ax1.set_title("Top Singular Values of Unembedding Matrix")
        ax1.grid(True)
        
        # Plot cumulative explained variance
        ax2.plot(range(1, max_values + 1), cumulative_var[:max_values] * 100)
        threshold_line = variance_threshold * 100
        ax2.axhline(y=threshold_line, color='r', linestyle='--', alpha=0.7)
        ax2.text(max_values/2, threshold_line + 1, f"{threshold_line}% Variance", color='r')
        ax2.axvline(x=k, color='g', linestyle='--', alpha=0.7)
        ax2.text(k + 1, 50, f"k={k}", color='g')
        ax2.set_xlabel("Number of Components")
        ax2.set_ylabel("Cumulative Explained Variance (%)")
        ax2.set_title("Explained Variance by Singular Vectors")
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def compute_projection_percentage(self, 
                                     residual: np.ndarray, 
                                     projection_matrix: np.ndarray) -> np.ndarray:
        """
        Compute what percentage of the residual stream contributes to token prediction.
        
        Args:
            residual: Residual stream activations as numpy array
            projection_matrix: Projection matrix for token prediction subspace
            
        Returns:
            Projection percentage
        """
        print(f"Computing projection percentage: residual shape {residual.shape}, projection matrix shape {projection_matrix.shape}")
        
        # Ensure residual is 2D for batch processing
        original_shape = residual.shape
        if len(original_shape) == 1:
            # Single vector case
            residual = residual.reshape(1, -1)
        elif len(original_shape) > 2:
            # Batch case (e.g., [batch, seq, d_model])
            batch_size, seq_len, d_model = original_shape
            residual = residual.reshape(-1, d_model)
        
        # Check shapes
        if residual.shape[1] != projection_matrix.shape[0]:
            raise ValueError(f"Dimension mismatch: residual has last dim {residual.shape[1]}, "
                            f"but projection_matrix has first dim {projection_matrix.shape[0]}")
        
        # Compute projection using squared Frobenius norm
        # For each vector r, we want ||P·r||²/||r||²
        # This is equivalent to (r·P·P·r)/(r·r) but P·P = P for projection matrices
        
        # First compute the projected vectors
        residual_proj = residual @ projection_matrix  # Shape: [*, d_model]
        
        # Compute squared norms of original and projected vectors
        original_norm_sq = np.sum(residual**2, axis=1)  # Shape: [*]
        proj_norm_sq = np.sum(residual_proj**2, axis=1)  # Shape: [*]
        
        # Calculate percentage - adding safety for division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            projection_percentage = 100 * proj_norm_sq / original_norm_sq
            projection_percentage = np.nan_to_num(projection_percentage)
        
        # Reshape back to original dimensions except for the last one
        if len(original_shape) == 1:
            return projection_percentage[0]
        elif len(original_shape) > 2:
            return projection_percentage.reshape(batch_size, seq_len)
        else:
            return projection_percentage
    
    def analyze_residual_stream(self, 
                             text: str, 
                             percent_variance: float = 0.95,
                             max_seq_len: int = 50) -> Dict:
        """
        Analyze how much of the residual stream projects onto the token prediction subspace.
        
        Args:
            text: Input text to analyze
            percent_variance: Variance threshold for token prediction subspace
            max_seq_len: Maximum sequence length to process (for memory efficiency)
            
        Returns:
            Dictionary with results including projection percentages by layer
        """
        print(f"Analyzing residual stream for: '{text[:50]}...' (truncated for display)")
        
        # First, compute the token prediction subspace
        print("Computing token prediction subspace...")
        analysis_results = self.analyze_projection_simple(variance_threshold=percent_variance)
        projection_matrix = self.compute_projection_matrix(analysis_results)
        
        # Tokenize the input (limiting length for memory efficiency)
        print("Tokenizing input...")
        tokens = self.model.to_tokens(text)
        if tokens.shape[1] > max_seq_len:
            print(f"Truncating sequence from {tokens.shape[1]} to {max_seq_len} tokens for memory efficiency")
            tokens = tokens[:, :max_seq_len]
        
        print(f"Processing sequence of length {tokens.shape[1]}")
        
        # Dictionary to store results
        results = {
            "text": text,
            "tokens": tokens.cpu().numpy(),
            "projection_percentages": {},
            "token_prediction_subspace": analysis_results
        }
        
        # Get model dimension
        d_model = self.model.cfg.d_model
        
        # Hook function to capture residual stream
        def process_hook(tensor, hook):
            # Check for shape issues
            if tensor.shape[-1] != d_model:
                print(f"Warning: Unexpected shape in {hook.name}: {tensor.shape}, expected last dim to be {d_model}")
                return tensor
            
            # Move to CPU and convert to NumPy to save memory
            activation = tensor.detach().cpu().numpy()
            
            # Debug shapes
            print(f"Processing {hook.name}: activation shape {activation.shape}, projection matrix shape {projection_matrix.shape}")
            
            # Calculate projection percentage
            try:
                layer_percs = self.compute_projection_percentage(activation, projection_matrix)
                # Store results
                results["projection_percentages"][hook.name] = layer_percs
            except Exception as e:
                print(f"Error processing {hook.name}: {e}")
            
            # Free memory
            del activation
            gc.collect()
            
            return tensor
        
        # Set up hooks for targeted layers (fewer hooks to reduce memory usage)
        hook_points = []
        n_layers = self.model.cfg.n_layers
        sample_layers = [0, n_layers//4, n_layers//2, (3*n_layers)//4, n_layers-1, n_layers]
        
        for layer_idx in sample_layers:
            if layer_idx == 0:
                hook_name = "blocks.0.hook_resid_pre"
                hook_points.append((hook_name, process_hook))
            elif layer_idx == n_layers:
                hook_name = "ln_final.hook_normalized"
                hook_points.append((hook_name, process_hook))
            else:
                hook_name = f"blocks.{layer_idx}.hook_resid_post"
                hook_points.append((hook_name, process_hook))
        
        # Run the model with hooks
        print("Running model with hooks (this might take a moment)...")
        try:
            with torch.no_grad():  # Disable gradients for memory efficiency
                _ = self.model.run_with_hooks(tokens, fwd_hooks=hook_points)
            print("Analysis completed successfully")
        except Exception as e:
            print(f"Error running model with hooks: {e}")
            print("This might be related to memory limitations or shape mismatches.")
            print(f"Model dimension: {d_model}")
            print(f"Projection matrix shape: {projection_matrix.shape}")
            print(f"Unembedding matrix shape: {self.unembed.shape}")
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return results
    
    def compare_tasks(self, 
                     factual_text: str, 
                     reasoning_text: str,
                     percent_variance: float = 0.95,
                     max_seq_len: int = 50) -> None:
        """
        Compare projection percentages between factual recall and reasoning tasks.
        
        Args:
            factual_text: Text for factual recall task
            reasoning_text: Text for reasoning task
            percent_variance: Variance threshold for token prediction subspace
            max_seq_len: Maximum sequence length to process
        """
        print("=== Comparing Factual Recall vs. Reasoning Tasks ===")
        
        # Analyze both tasks
        print("\nAnalyzing factual recall task...")
        factual_results = self.analyze_residual_stream(
            factual_text, 
            percent_variance=percent_variance,
            max_seq_len=max_seq_len
        )
        
        print("\nAnalyzing reasoning task...")
        reasoning_results = self.analyze_residual_stream(
            reasoning_text, 
            percent_variance=percent_variance,
            max_seq_len=max_seq_len
        )
        
        # Extract final layer percentages
        final_layer = None
        for layer_name in factual_results["projection_percentages"].keys():
            if "ln_final" in layer_name:
                final_layer = layer_name
                break
        
        if final_layer is None:
            print("Warning: Final layer not found in results")
            return
        
        # Compare average percentages
        factual_pct = np.mean(factual_results["projection_percentages"][final_layer])
        reasoning_pct = np.mean(reasoning_results["projection_percentages"][final_layer])
        
        print("\n=== Results ===")
        print(f"Factual task: {factual_pct:.2f}% in token prediction space")
        print(f"Reasoning task: {reasoning_pct:.2f}% in token prediction space")
        print(f"Difference: {factual_pct - reasoning_pct:.2f}% more in token space for factual task")
        
        # Plot comparison
        self.plot_task_comparison(factual_results, reasoning_results)
        
    def plot_task_comparison(self, factual_results: Dict, reasoning_results: Dict) -> None:
        """
        Plot comparison between factual and reasoning tasks.
        
        Args:
            factual_results: Results from factual task
            reasoning_results: Results from reasoning task
        """
        # Get common hook points
        common_hooks = sorted(set(factual_results["projection_percentages"].keys()) & 
                             set(reasoning_results["projection_percentages"].keys()))
        
        # Extract percentages and layer names
        layer_indices = []
        layer_names = []
        factual_pcts = []
        reasoning_pcts = []
        
        for hook_name in common_hooks:
            # Parse layer index
            if "blocks" in hook_name:
                if "hook_resid_pre" in hook_name:
                    layer_idx = 0
                    layer_label = "Embedding"
                else:
                    layer_idx = int(hook_name.split(".")[1]) + 1
                    layer_label = f"Layer {layer_idx}"
            else:
                layer_idx = self.model.cfg.n_layers
                layer_label = "Final"
            
            layer_indices.append(layer_idx)
            layer_names.append(layer_label)
            
            # Get average percentages
            factual_pcts.append(np.mean(factual_results["projection_percentages"][hook_name]))
            reasoning_pcts.append(np.mean(reasoning_results["projection_percentages"][hook_name]))
        
        # Sort by layer index
        sorted_data = sorted(zip(layer_indices, layer_names, factual_pcts, reasoning_pcts))
        layer_names = [x[1] for x in sorted_data]
        factual_pcts = [x[2] for x in sorted_data]
        reasoning_pcts = [x[3] for x in sorted_data]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        x = np.arange(len(layer_names))
        width = 0.35
        
        plt.bar(x - width/2, factual_pcts, width, label='Factual Recall', color='skyblue')
        plt.bar(x + width/2, reasoning_pcts, width, label='Reasoning', color='orange')
        
        plt.xlabel('Layer')
        plt.ylabel('% in Token Prediction Space')
        plt.title('Projection Percentages by Layer: Factual vs. Reasoning')
        plt.xticks(x, layer_names, rotation=45)
        plt.legend()
        plt.ylim(0, 100)
        
        # Add value labels
        for i, v in enumerate(factual_pcts):
            plt.text(i - width/2, v + 2, f"{v:.1f}%", ha='center', fontsize=9)
        
        for i, v in enumerate(reasoning_pcts):
            plt.text(i + width/2, v + 2, f"{v:.1f}%", ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def is_token_focused(self, percent_variance: float = 0.95) -> None:
        """
        Analyze what percentage of the unembedding matrix variance is captured
        by a small number of directions, indicating token prediction focus.
        
        Args:
            percent_variance: Percent of variance to capture (default: 0.95)
        """
        results = self.analyze_projection_simple(variance_threshold=percent_variance)
        
        # Interpret results
        k = results["top_k"]
        d_model = self.d_model
        percent_dimensions = (k / d_model) * 100
        
        print("\n=== Token Prediction Space Analysis ===")
        print(f"Total embedding dimensions: {d_model}")
        print(f"Dimensions needed for {percent_variance*100:.0f}% variance: {k}")
        print(f"This represents {percent_dimensions:.2f}% of the embedding dimensions")
        
        if percent_dimensions < 25:
            print("\nInterpretation: The token prediction space is highly concentrated.")
            print("A small subset of directions in the embedding space accounts for most token prediction.")
            print("This suggests the residual stream likely serves dual purposes:")
            print("1. Token prediction (concentrated in few dimensions)")
            print("2. Persistent memory/reasoning (distributed across remaining dimensions)")
        elif percent_dimensions < 50:
            print("\nInterpretation: The token prediction space is moderately concentrated.")
            print("Token prediction uses a substantial but not dominant portion of the embedding space.")
        else:
            print("\nInterpretation: The token prediction space is widely distributed.")
            print("Most dimensions contribute significantly to token prediction,")
            print("suggesting less clear separation between token prediction and other functions.")
            
        # Visualize
        self.visualize_singular_values(results)