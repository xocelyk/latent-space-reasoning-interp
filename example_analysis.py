import torch
import matplotlib.pyplot as plt

from residual_analysis import ResidualStreamAnalyzer

def compare_reasoning_vs_factual():
    """
    Compare projection percentages between factual recall and multi-step reasoning.
    """
    print(1)
    analyzer = ResidualStreamAnalyzer("gpt2-small")
    print(2)
    # Define example prompts
    factual_prompt = "The capital of France is"  # Simple factual recall
    reasoning_prompt = "If 5x + 3 = 18, then x equals"  # Simple reasoning task
    print(3)
    
    # Analyze both prompts
    factual_results = analyzer.run_with_hooks(factual_prompt)
    reasoning_results = analyzer.run_with_hooks(reasoning_prompt)
    print(4)
    
    # Display predicted tokens
    print("Factual prompt:", factual_prompt)
    print("Predicted next tokens:", " ".join(factual_results["token_strs"][:3]))
    
    print("\nReasoning prompt:", reasoning_prompt)
    print("Predicted next tokens:", " ".join(reasoning_results["token_strs"][:3]))
    
    # Visualize projection percentages
    analyzer.visualize_projection_percentages(
        factual_results, 
        title="Projection Percentages - Factual Recall"
    )
    
    analyzer.visualize_projection_percentages(
        reasoning_results, 
        title="Projection Percentages - Reasoning Task"
    )
    
    # Compare final layer percentages
    final_layer = "ln_final.hook_normalized"
    
    factual_percentages = factual_results["projection_percentages"][final_layer]
    reasoning_percentages = reasoning_results["projection_percentages"][final_layer]
    
    print("\nFinal Layer Projection Percentages:")
    print(f"Factual: {factual_percentages[0].item():.2f}%")
    print(f"Reasoning: {reasoning_percentages[0].item():.2f}%")
    
    # Visualize singular values
    analyzer.visualize_singular_values(factual_results)

def analyze_reasoning_chain():
    """
    Analyze a multi-step reasoning chain to see how projection percentages
    evolve throughout the reasoning process.
    """
    analyzer = ResidualStreamAnalyzer("gpt2-small")
    
    # Define a multi-step reasoning prompt
    reasoning_chain = """
    Problem: If a shirt costs $15 and is on sale for 20% off, what is the sale price?
    
    Step 1: Calculate the discount amount.
    Discount = Original price × Discount percentage
    Discount = $15 × 0.20
    Discount = $3
    
    Step 2: Subtract the discount from the original price.
    Sale price = Original price - Discount
    Sale price = $15 - $3
    Sale price = $12
    
    Therefore, the sale price of the shirt is $12.
    """
    
    # Run the analysis
    results = analyzer.run_with_hooks(reasoning_chain)
    
    # Visualize the projection percentages across the reasoning chain
    analyzer.visualize_projection_percentages(
        results,
        title="Projection Percentages Throughout Reasoning Chain"
    )
    
    # Find positions where specific reasoning steps start
    step1_pos = reasoning_chain.find("Step 1:") 
    step2_pos = reasoning_chain.find("Step 2:")
    conclusion_pos = reasoning_chain.find("Therefore")
    
    # Convert character positions to token positions (approximate)
    tokens = results["tokens"][0]
    token_strs = [analyzer.model.tokenizer.decode([t.item()]) for t in tokens]
    
    print("Analyzing projection percentages at key reasoning steps...")
    
    # Analyze the final layer
    final_layer = "ln_final.hook_normalized"
    percentages = results["projection_percentages"][final_layer][0].cpu().numpy()
    
    # Plot percentages over the sequence
    plt.figure(figsize=(12, 6))
    plt.plot(percentages)
    plt.title("Token Space Projection % Throughout Reasoning Chain")
    plt.xlabel("Token Position")
    plt.ylabel("% in Token Space")
    plt.grid(True)
    
    # Highlight transition points
    for name, pos in [("Problem", 0), 
                     ("Step 1", step1_pos), 
                     ("Step 2", step2_pos), 
                     ("Conclusion", conclusion_pos)]:
        plt.axvline(x=pos//4, color='r', linestyle='--', alpha=0.7)
        plt.text(pos//4, max(percentages) * 0.9, name, rotation=90)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Example 1: Comparing Factual Recall vs. Reasoning")
    compare_reasoning_vs_factual()
    
    print("\nExample 2: Analyzing a Multi-step Reasoning Chain")
    analyze_reasoning_chain()