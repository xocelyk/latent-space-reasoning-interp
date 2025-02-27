import torch
import numpy as np
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from typing import List, Tuple, Dict, Optional, Union
import seaborn as sns

class ResidualStreamAnalyzer:
    """
    Analyzes the residual stream to determine what percentage contributes to token prediction
    versus persistent memory/latent reasoning.
    """
    
    def __init__(self, model_name: str = "gpt2-small"):
        """
        Initialize the analyzer with a specific model.
        
        Args:
            model_name: Name of the model to analyze (any model supported by TransformerLens)
        """
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
            
        self.model = HookedTransformer.from_pretrained(model_name)
        self.model.to(self.device)
        self.d_model = self.model.cfg.d_model
        self.unembed = self.model.W_U  # Shape: [vocab_size, d_model]
        
        # Cache for SVD results
        self.token_subspace_cache = {}
    
    def compute_token_subspace(self, 
                              variance_explained: float = 0.95, 
                              cache_key: str = "default") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the token prediction subspace via SVD of the unembedding matrix.
        
        Args:
            variance_explained: Percentage of variance to explain (default: 0.95)
            cache_key: Key to cache the results under
            
        Returns:
            Tuple of (projection_matrix, singular_values, cumulative_variance)
        """
        if cache_key in self.token_subspace_cache:
            return self.token_subspace_cache[cache_key]
        
        # Move unembedding matrix to appropriate device and use CPU for SVD
        # SVD on MPS can sometimes cause issues, so we'll use CPU for this computation
        unembed_cpu = self.unembed.cpu()
        
        # Compute SVD of unembedding matrix
        try:
            U, S, Vh = torch.linalg.svd(unembed_cpu)
        except Exception as e:
            print(f"SVD failed with error: {e}")
            print("Falling back to NumPy SVD implementation")
            # Fallback to NumPy implementation
            import numpy as np
            U_np, S_np, Vh_np = np.linalg.svd(unembed_cpu.numpy(), full_matrices=False)
            U = torch.from_numpy(U_np)
            S = torch.from_numpy(S_np)
            Vh = torch.from_numpy(Vh_np)
        
        # Calculate cumulative explained variance
        total_variance = torch.sum(S**2)
        cumulative_variance = torch.cumsum(S**2, dim=0) / total_variance
        
        # Determine how many singular values to keep
        k = torch.sum(cumulative_variance <= variance_explained).item() + 1
        k = min(k, len(S))  # Ensure k doesn't exceed the number of singular values
        
        # Get the important right singular vectors (columns of V)
        V = Vh.T[:, :k]
        
        # Create projection matrix
        P = V @ V.T  # Shape: [d_model, d_model]
        
        # Move results back to the device
        P = P.to(self.device)
        S = S.to(self.device)
        cumulative_variance = cumulative_variance.to(self.device)
        
        result = (P, S, cumulative_variance)
        self.token_subspace_cache[cache_key] = result
        
        return result
    
    def compute_projection_percentage(self, 
                                     residual: torch.Tensor, 
                                     projection_matrix: Optional[torch.Tensor] = None,
                                     variance_explained: float = 0.95) -> torch.Tensor:
        """
        Compute what percentage of the residual stream contributes to token prediction.
        
        Args:
            residual: Residual stream activations [batch, seq, d_model] or [d_model]
            projection_matrix: Pre-computed projection matrix (if None, will compute)
            variance_explained: Variance explained threshold (only used if projection_matrix is None)
            
        Returns:
            Tensor of projection percentages
        """
        # Ensure residual is on the correct device
        residual = residual.to(self.device)
        original_shape = residual.shape
        
        # Reshape to [*, d_model] for batch processing
        if len(original_shape) == 1:
            # Single vector case
            residual = residual.reshape(1, self.d_model)
        elif len(original_shape) > 2:
            # Batch case
            residual = residual.reshape(-1, self.d_model)
        
        # Get projection matrix if not provided
        if projection_matrix is None:
            projection_matrix, _, _ = self.compute_token_subspace(variance_explained)
        else:
            # Ensure projection matrix is on the correct device
            projection_matrix = projection_matrix.to(self.device)
        
        # Compute projected vectors
        residual_proj = residual @ projection_matrix  # [*, d_model]
        
        # Compute the ratio of squared norms
        original_norm_sq = torch.sum(residual**2, dim=-1)  # [*]
        projected_norm_sq = torch.sum(residual_proj**2, dim=-1)  # [*]
        
        # Avoid division by zero
        projection_percentage = torch.where(
            original_norm_sq > 0,
            100 * projected_norm_sq / original_norm_sq,
            torch.zeros_like(projected_norm_sq)
        )
        
        # Reshape to original dimensions except for the last one
        if len(original_shape) == 1:
            projection_percentage = projection_percentage[0]
        elif len(original_shape) > 2:
            projection_percentage = projection_percentage.reshape(original_shape[:-1])
            
        return projection_percentage
    
    def run_with_hooks(self, 
                      text: Union[str, List[str]], 
                      variance_explained: float = 0.95) -> Dict:
        """
        Run the model on text and collect residual stream activations and projection percentages.
        
        Args:
            text: Input text or list of texts
            variance_explained: Variance explained threshold
            
        Returns:
            Dictionary with results including residual activations and projection percentages
        """
        is_single_text = isinstance(text, str)
        if is_single_text:
            text = [text]
        
        # Tokenize input
        tokens = self.model.to_tokens(text)
        tokens = tokens.to(self.device)
        
        # Get projection matrix for token prediction subspace
        P, S, cumulative_var = self.compute_token_subspace(variance_explained=variance_explained)
        
        # Storage for residual activations
        residual_activations = {}
        projection_percentages = {}
        
        # Define hook function to capture residual stream at each layer
        def hook_fn(act, hook):
            layer_name = hook.name
            residual_activations[layer_name] = act.detach().clone()
            return act
        
        # Set up hooks for each layer's output
        hook_points = []
        for layer_idx in range(self.model.cfg.n_layers + 1):  # +1 for final residual
            if layer_idx == 0:
                hook_name = "blocks.0.hook_resid_pre"
            elif layer_idx == self.model.cfg.n_layers:
                hook_name = "ln_final.hook_normalized"
            else:
                hook_name = f"blocks.{layer_idx}.hook_resid_post"
            
            hook_points.append(hook_name)
        
        # Run the model with hooks
        try:
            logits = self.model.run_with_hooks(
                tokens, 
                fwd_hooks=[(hook_name, hook_fn) for hook_name in hook_points]
            )
            
            # Compute projection percentages for each layer
            for hook_name, activations in residual_activations.items():
                projection_percentages[hook_name] = self.compute_projection_percentage(
                    activations, projection_matrix=P, variance_explained=variance_explained
                )
            
            # Get token predictions
            token_strs = []
            for batch_idx in range(tokens.shape[0]):
                token_str = []
                for pos_idx in range(1, tokens.shape[1]):  # Start from 1 to align with predictions
                    pred_idx = torch.argmax(logits[batch_idx, pos_idx-1]).item()
                    pred_token = self.model.tokenizer.decode([pred_idx])
                    token_str.append(pred_token)
                token_strs.append(token_str)
            
            # Format results
            results = {
                "tokens": tokens,
                "token_strs": token_strs[0] if is_single_text else token_strs,
                "residual_activations": residual_activations,
                "projection_percentages": projection_percentages,
                "singular_values": S,
                "cumulative_variance": cumulative_var,
                "projection_matrix": P,
                "input_text": text[0] if is_single_text else text
            }
            
            return results
            
        except Exception as e:
            print(f"Error running model with hooks: {e}")
            print("This might be related to device compatibility issues.")
            print("If you're on Apple Silicon, you might need to use a smaller model or batch size.")
            raise
    
    def visualize_projection_percentages(self, 
                                        results: Dict, 
                                        title: str = "Residual Stream Projection to Token Space",
                                        figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Visualize the projection percentages across positions and layers.
        
        Args:
            results: Results dictionary from run_with_hooks
            title: Plot title
            figsize: Figure size
        """
        # Extract projection percentages and prepare for plotting
        layer_names = []
        layer_percs = []
        
        for hook_name, percs in results["projection_percentages"].items():
            # Extract layer number
            if "blocks" in hook_name:
                if "hook_resid_pre" in hook_name:
                    layer_idx = 0
                    layer_label = "Embedding"
                else:
                    layer_idx = int(hook_name.split(".")[1]) + 1
                    layer_label = f"Layer {layer_idx}"
            else:
                layer_idx = self.model.cfg.n_layers + 1
                layer_label = "Final Layer"
            
            layer_names.append((layer_idx, layer_label))
            # Ensure tensor is on CPU for visualization
            layer_percs.append(percs.cpu() if isinstance(percs, torch.Tensor) else percs)
        
        # Sort by layer index
        layer_data = sorted(zip(layer_names, layer_percs), key=lambda x: x[0][0])
        layer_names = [x[1] for x in layer_data]
        layer_percs = [x[1] for x in layer_data]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Convert to numpy for easier plotting
        layer_percs_np = [p.numpy() if isinstance(p, torch.Tensor) else np.array(p) for p in layer_percs]
        
        # Plot heatmap if we have multiple positions
        if layer_percs_np[0].ndim > 0 and layer_percs_np[0].size > 1:
            # Create a 2D array [layers, positions]
            data = np.vstack([p.reshape(1, -1) for p in layer_percs_np])
            
            # Get position labels
            pos_labels = [f"Pos {i}" for i in range(data.shape[1])]
            
            # Plot heatmap
            sns.heatmap(data, annot=True, fmt=".1f", cmap="viridis", 
                       xticklabels=pos_labels, yticklabels=layer_names, 
                       cbar_kws={'label': '% in Token Space'})
            
            plt.xlabel("Position")
            plt.ylabel("Layer")
            
        else:
            # Simple bar chart for single positions
            data = np.array([p.item() if isinstance(p, torch.Tensor) else p for p in layer_percs_np])
            
            sns.barplot(x=layer_names, y=data, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.set_ylabel("% in Token Space")
            ax.set_ylim(0, 100)
            
            # Add value labels on bars
            for i, v in enumerate(data):
                ax.text(i, v + 1, f"{v:.1f}%", ha='center')
        
        plt.title(title)
        plt.tight_layout()
        plt.show()
        
    def visualize_singular_values(self, 
                                 results: Dict, 
                                 max_values: int = 100,
                                 figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Visualize the singular values and cumulative explained variance.
        
        Args:
            results: Results dictionary from run_with_hooks
            max_values: Maximum number of singular values to plot
            figsize: Figure size
        """
        # Ensure tensors are on CPU for visualization
        if isinstance(results["singular_values"], torch.Tensor):
            S = results["singular_values"].cpu().numpy()
        else:
            S = np.array(results["singular_values"])
            
        if isinstance(results["cumulative_variance"], torch.Tensor):
            cumulative_var = results["cumulative_variance"].cpu().numpy()
        else:
            cumulative_var = np.array(results["cumulative_variance"])
        
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
        ax2.axhline(y=95, color='r', linestyle='--', alpha=0.7)
        ax2.text(max_values/2, 96, "95% Variance", color='r')
        ax2.set_xlabel("Number of Components")
        ax2.set_ylabel("Cumulative Explained Variance (%)")
        ax2.set_title("Explained Variance by Singular Vectors")
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()