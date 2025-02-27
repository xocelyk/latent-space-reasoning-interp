import torch
import numpy as np
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
import gc
from tqdm.notebook import tqdm

def compute_projection_auc(
    prompts,
    model_name="distilgpt2",
    n_points=30
):
    """
    Compute the AUC for token projection across different variance thresholds.
    
    Args:
        prompts: Dictionary of {prompt_name: prompt_text}
        model_name: Name of the model to use
        n_points: Number of points to sample for the curve
    
    Returns:
        Dictionary of results by prompt and layer
    """
    print(f"Loading model: {model_name}")
    model = HookedTransformer.from_pretrained(model_name, device="cpu")  # Force CPU
    d_model = model.cfg.d_model
    n_layers = model.cfg.n_layers
    
    print(f"Model loaded with {d_model} dimensions and {n_layers} layers")
    
    # Get unembedding matrix (W_U)
    print("Getting unembedding matrix...")
    W_U = model.W_U.detach().cpu().numpy()
    vocab_size, emb_dim = W_U.shape
    print(f"Unembedding matrix shape: {W_U.shape}")
    
    # Compute SVD once
    print("Computing SVD (this might take a moment)...")
    U, S, Vh = np.linalg.svd(W_U, full_matrices=False)
    
    # Compute cumulative explained variance
    total_variance = np.sum(S**2)
    cumulative_variance = np.cumsum(S**2) / total_variance
    
    # Define dimension fractions to test (logarithmically spaced for better resolution at lower values)
    dim_fractions = np.logspace(-2, 0, n_points)
    k_values = [max(1, int(d_model * f)) for f in dim_fractions]
    k_values = sorted(list(set(k_values)))  # Remove duplicates and sort
    
    # Store the token variance curve (same for all prompts and layers)
    token_variance_curve = []
    for k in k_values:
        if k <= len(cumulative_variance):
            variance_explained = cumulative_variance[k-1]
        else:
            variance_explained = 1.0
        token_variance_curve.append((k/d_model, variance_explained))
    
    # Compute AUC for token variance curve
    var_x = [p[0] for p in token_variance_curve]
    var_y = [p[1] for p in token_variance_curve]
    token_variance_auc = np.trapz(var_y, var_x)
    print(f"Token variance AUC: {token_variance_auc:.4f}")
    
    # Store results for all prompts and layers
    results = {prompt_name: {} for prompt_name in prompts}
    
    # Get all hook points we want to monitor
    hook_points = []
    
    # Add residual stream before each layer
    for layer_idx in range(n_layers):
        hook_points.append(f"blocks.{layer_idx}.hook_resid_pre")
    
    # Add residual stream after each layer
    for layer_idx in range(n_layers):
        hook_points.append(f"blocks.{layer_idx}.hook_resid_post")
    
    # Add final normalization
    hook_points.append("ln_final.hook_normalized")
    
    # Sort hook points to ensure they're in order
    sorted_hooks = sorted(hook_points, key=lambda x: (
        -2 if "ln_final" in x else
        -1 if "hook_resid_post" in x else
        int(x.split(".")[1].split("_")[0]) if "." in x else 0
    ))
    
    # Process each prompt
    for prompt_name, prompt_text in prompts.items():
        print(f"\nAnalyzing prompt: '{prompt_name}'")
        
        # Storage for this prompt's results
        prompt_results = {}
        
        # Tokenize
        tokens = model.to_tokens(prompt_text)
        
        # Define hook function that will be used for all layers
        def hook_fn(tensor, hook):
            # Get the activation and move to CPU
            act = tensor.detach().cpu().numpy()
            
            # Check shape
            if act.shape[-1] != d_model:
                print(f"Warning: Unexpected shape in {hook.name}")
                return tensor
            
            # Reshape for computation
            batch_size, seq_len, emb_dim = act.shape
            reshaped = act.reshape(-1, emb_dim)
            
            # Compute projection for each k value
            projection_curve = []
            
            for k in k_values:
                # Get projection matrix for this k
                token_directions = U[:, :k]
                projected = reshaped @ token_directions @ token_directions.T
                
                # Calculate projection percentage
                original_norm_sq = np.sum(reshaped**2, axis=1)
                projected_norm_sq = np.sum(projected**2, axis=1)
                
                # Handle division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    projection_ratios = projected_norm_sq / original_norm_sq
                    projection_ratios = np.nan_to_num(projection_ratios)
                
                # Average across all vectors
                avg_projection = np.mean(projection_ratios)
                projection_curve.append((k/d_model, avg_projection))
            
            # Calculate AUC for projection curve
            proj_x = [p[0] for p in projection_curve]
            proj_y = [p[1] for p in projection_curve]
            projection_auc = np.trapz(proj_y, proj_x)
            
            # Store the results
            prompt_results[hook.name] = {
                "projection_curve": projection_curve,
                "projection_auc": projection_auc,
                "token_variance_auc": token_variance_auc,
                "auc_ratio": projection_auc / token_variance_auc,
                "layer_name": hook.name
            }
            
            return tensor
        
        # Set up hooks for all the layers we want to monitor
        fwd_hooks = [(hook_name, hook_fn) for hook_name in sorted_hooks]
        
        # Run the model with hooks
        with torch.no_grad():
            _ = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
        
        # Store results for this prompt
        results[prompt_name] = prompt_results
        
        # Force garbage collection
        gc.collect()
    
    # Plot the results
    plot_auc_results(results, token_variance_curve)
    
    return results

def plot_auc_results(results, token_variance_curve):
    """
    Plot the AUC results.
    
    Args:
        results: Dictionary of results by prompt and layer
        token_variance_curve: The token variance curve for reference
    """
    # First, create a plot of AUC values across layers
    plt.figure(figsize=(14, 8))
    
    # Get all hook points in order
    hooks = []
    for prompt_name, prompt_results in results.items():
        for hook_name in prompt_results:
            if hook_name not in hooks:
                hooks.append(hook_name)
    
    # Sort hooks by layer
    hooks = sorted(hooks, key=lambda x: (
        1000 if "ln_final" in x else
        int(x.split(".")[1].split("_")[0])*2 + (1 if "post" in x else 0)
    ))
    
    # Create x-axis labels and positions
    x_positions = list(range(len(hooks)))
    x_labels = []
    for hook in hooks:
        if "ln_final" in hook:
            x_labels.append("Final")
        elif "hook_resid_pre" in hook:
            layer_num = int(hook.split(".")[1].split("_")[0])
            x_labels.append(f"L{layer_num}")
        elif "hook_resid_post" in hook:
            layer_num = int(hook.split(".")[1].split("_")[0])
            x_labels.append(f"L{layer_num}+")
    
    # Plot each prompt as a separate line
    markers = ['o', 's', '^', 'D', 'x', '*', 'p', 'h']
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
    
    for i, (prompt_name, prompt_results) in enumerate(results.items()):
        # Extract AUC ratios
        auc_ratios = []
        for hook in hooks:
            if hook in prompt_results:
                auc_ratios.append(prompt_results[hook]["auc_ratio"])
            else:
                auc_ratios.append(None)
        
        # Plot
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]
        plt.plot(x_positions, auc_ratios, marker=marker, color=color, label=prompt_name, linewidth=2)
    
    # Add a horizontal line at 1.0 (where projection AUC = token variance AUC)
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
    
    # Set up the plot
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Projection AUC / Token Variance AUC', fontsize=12)
    plt.title('Token Prediction Concentration Index Across Layers', fontsize=14)
    plt.xticks(x_positions, x_labels, rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    
    plt.show()
    
    # Now plot the full projection curves for the final layer
    plt.figure(figsize=(10, 8))
    
    # Get the token variance curve for reference
    var_x = [p[0] for p in token_variance_curve]
    var_y = [p[1] for p in token_variance_curve]
    plt.plot(var_x, var_y, 'k--', label='Token Variance', linewidth=2)
    
    # Get the final layer
    final_layer = "ln_final.hook_normalized"
    
    # Plot each prompt's final layer projection curve
    for i, (prompt_name, prompt_results) in enumerate(results.items()):
        if final_layer in prompt_results:
            projection_curve = prompt_results[final_layer]["projection_curve"]
            proj_x = [p[0] for p in projection_curve]
            proj_y = [p[1] for p in projection_curve]
            
            marker = markers[i % len(markers)]
            color = colors[i % len(colors)]
            plt.plot(proj_x, proj_y, marker=marker, color=color, label=f"{prompt_name} Projection", linewidth=2)
    
    # Set up the plot
    plt.xlabel('Fraction of Dimensions', fontsize=12)
    plt.ylabel('Fraction of Variance/Projection', fontsize=12)
    plt.title('Token Prediction Curves for Final Layer', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xscale('log')  # Use log scale for x-axis to better see the behavior at small dimension fractions
    plt.tight_layout()
    
    plt.show()
    
    # Print final layer AUC ratios
    print("\n=== Final Layer Concentration Indices ===")
    for prompt_name, prompt_results in results.items():
        if final_layer in prompt_results:
            auc_ratio = prompt_results[final_layer]["auc_ratio"]
            print(f"{prompt_name}: {auc_ratio:.4f}")

def compare_task_types():
    """Run AUC analysis comparing different types of tasks."""
    prompts = {
        "Factual (capitals)": "The capital of France is Paris. The capital of Japan is Tokyo. The capital of Italy is",
        "Factual (elements)": "The first elements of the periodic table are hydrogen, helium, lithium, beryllium, boron, carbon, nitrogen, oxygen, fluorine, and",
        "Reasoning (math)": "If 5x + 3 = 18, then we subtract 3 from both sides to get 5x = 15, then divide by 5 to get x equals",
        "Reasoning (logic)": "All humans are mortal. Socrates is human. Therefore, Socrates is"
    }
    
    return compute_projection_auc(prompts, model_name="gpt2-small")

if __name__ == "__main__":
    # Run the analysis
    compare_task_types()