import torch
import numpy as np
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
import gc

def analyze_token_projection(
    factual_prompt="The capital of France is Paris.",
    reasoning_prompt="If x + 5 = 10, then x equals 5.",
    model_name="distilgpt2",  # Use smaller model for MacBook
    variance_threshold=0.95
):
    """
    Minimal script to analyze projection percentages correctly.
    """
    print(f"Loading model: {model_name}")
    model = HookedTransformer.from_pretrained(model_name, device="cpu")  # Force CPU
    d_model = model.cfg.d_model
    n_layers = model.cfg.n_layers
    
    print(f"Model loaded with {d_model} dimensions")
    
    # Get unembedding matrix (W_U)
    print("Getting unembedding matrix...")
    W_U = model.W_U.detach().cpu().numpy()
    print(f"Unembedding matrix shape: {W_U.shape}")  # Should be [vocab_size, d_model]
    
    # CRITICAL FIX: Transpose unembedding matrix to get directions in embedding space
    print("Computing SVD (this might take a moment)...")
    # W_U_T = W_U.T  # Shape: [d_model, vocab_size]
    W_U_T = W_U
    # Compute SVD to find important directions in embedding space
    U, S, Vh = np.linalg.svd(W_U_T, full_matrices=False)
    print(f"SVD complete. U shape: {U.shape}, S shape: {S.shape}")
    
    # Find how many directions needed for threshold
    total_variance = np.sum(S**2)
    cumulative_variance = np.cumsum(S**2) / total_variance
    k = np.sum(cumulative_variance <= variance_threshold) + 1
    print(f"Dimensions needed for {variance_threshold*100}% variance: {k}")
    
    # Get the top k directions that matter for token prediction
    token_directions = U[:, :k]  # Shape: [d_model, k]
    print(f"Token directions shape: {token_directions.shape}")
    
    # Create projection matrix for token prediction subspace
    P = token_directions @ token_directions.T  # Shape: [d_model, d_model]
    print(f"Projection matrix shape: {P.shape}")
    assert P.shape == (d_model, d_model), "Projection matrix has wrong shape!"
    
    # Define hook function to capture residual stream
    results = {"factual": {}, "reasoning": {}}
    
    def hook_fn(tensor, hook, task_key):
        # Convert to numpy and reshape
        activation = tensor.detach().cpu().numpy()
        print(f"Processing {hook.name}: activation shape {activation.shape}")
        
        # Ensure activations are 2D array of shape [*, d_model]
        batch_size, seq_len, d_model = activation.shape
        reshaped = activation.reshape(-1, d_model)
        
        # Project onto token prediction subspace
        projected = reshaped @ P
        
        # Calculate projection percentage
        original_norm = np.sum(reshaped**2, axis=1)
        projected_norm = np.sum(projected**2, axis=1)
        percentages = 100 * projected_norm / original_norm
        
        # Reshape back to [batch, seq]
        percentages = percentages.reshape(batch_size, seq_len)
        
        # Store results
        results[task_key][hook.name] = percentages
        
        # Return the original tensor (hooks shouldn't modify tensors)
        return tensor
    
    # Process each prompt
    for task_key, prompt in [("factual", factual_prompt), ("reasoning", reasoning_prompt)]:
        print(f"\nAnalyzing {task_key} prompt: '{prompt}'")
        
        # Tokenize
        tokens = model.to_tokens(prompt)
        
        # Set up hooks for key layers
        hook_points = []
        # layers_to_hook = [0, n_layers//2, n_layers-1]
        layers_to_hook = [i for i in range(n_layers)]
        
        for layer_idx in layers_to_hook:
            if layer_idx == 0:
                hook_name = "blocks.0.hook_resid_pre"
            else:
                hook_name = f"blocks.{layer_idx}.hook_resid_post"
            
            # Create a hook that captures the task type
            hook_points.append((hook_name, lambda tensor, hook, task=task_key: hook_fn(tensor, hook, task)))
        
        # Hook the final normalized output too
        hook_points.append(("ln_final.hook_normalized", lambda tensor, hook, task=task_key: hook_fn(tensor, hook, task)))
        
        # Run the model with hooks
        with torch.no_grad():
            _ = model.run_with_hooks(tokens, fwd_hooks=hook_points)
        
        # Force garbage collection
        gc.collect()
    
    # Compare the results
    print("\n=== Results ===")
    
    # Calculate average percentages for final layer
    final_layer = "ln_final.hook_normalized"
    factual_pct = np.mean(results["factual"][final_layer])
    reasoning_pct = np.mean(results["reasoning"][final_layer])
    
    print(f"Factual task: {factual_pct:.2f}% in token prediction space")
    print(f"Reasoning task: {reasoning_pct:.2f}% in token prediction space")
    print(f"Difference: {factual_pct - reasoning_pct:.2f}% more in token space for factual task")
    
    # Plot comparison
    plot_comparison(results, model.cfg.n_layers)
    
    # Plot the singular values
    plot_singular_values(S, cumulative_variance, k, variance_threshold)
    
    return results

def plot_comparison(results, n_layers):
    """Plot comparison between factual and reasoning tasks."""
    # Get common hook points
    common_hooks = set(results["factual"].keys()) & set(results["reasoning"].keys())
    
    # Process hook names to get layer info
    data = []
    for hook_name in common_hooks:
        if "blocks" in hook_name:
            if "hook_resid_pre" in hook_name:
                layer_idx = 0
                layer_name = "Embedding"
            else:
                layer_idx = int(hook_name.split(".")[1]) + 1
                layer_name = f"Layer {layer_idx}"
        else:
            layer_idx = n_layers
            layer_name = "Final"
        
        factual_pct = np.mean(results["factual"][hook_name])
        reasoning_pct = np.mean(results["reasoning"][hook_name])
        
        data.append((layer_idx, layer_name, factual_pct, reasoning_pct))
    
    # Sort by layer
    data.sort()
    
    # Create plot
    plt.figure(figsize=(10, 6))
    x = np.arange(len(data))
    width = 0.35
    
    layer_names = [item[1] for item in data]
    factual_pcts = [item[2] for item in data]
    reasoning_pcts = [item[3] for item in data]
    
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

def plot_singular_values(S, cumulative_variance, k, variance_threshold, max_values=100):
    """Plot the singular values and cumulative variance."""
    max_values = min(max_values, len(S))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot singular values
    ax1.plot(range(1, max_values + 1), S[:max_values])
    ax1.set_xlabel("Index")
    ax1.set_ylabel("Singular Value")
    ax1.set_title("Top Singular Values of Unembedding Matrix")
    ax1.grid(True)
    
    # Plot cumulative explained variance
    ax2.plot(range(1, max_values + 1), cumulative_variance[:max_values] * 100)
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

if __name__ == "__main__":
    # Run the analysis
    analyze_token_projection()