# Residual Stream Analysis

This repository provides tools to analyze how much of a transformer's residual stream contributes to token prediction versus persistent memory/latent reasoning.

## Hypothesis

The core hypothesis being tested is that the residual stream serves a dual purpose:
1. Directly contributing to token prediction (via the unembedding matrix)
2. Maintaining persistent information for future token predictions (latent reasoning)

We quantify this by analyzing how much of the residual stream's "energy" projects onto the token prediction subspace defined by the unembedding matrix.

## Method

The approach uses Singular Value Decomposition (SVD) of the unembedding matrix:

1. Compute SVD of the unembedding matrix W_U = UΣV^T
2. Define the token prediction subspace using the top singular vectors in V
3. For any residual stream vector r, calculate the projection percentage:
   - Projection ratio = ||Pr||² / ||r||² × 100%
   
Where P is the projection matrix onto the token prediction subspace.

## Usage

```python
from residual_analysis import ResidualStreamAnalyzer

# Initialize the analyzer
analyzer = ResidualStreamAnalyzer("gpt2-small")

# Analyze text
results = analyzer.run_with_hooks("If 5x + 3 = 18, then x equals")

# Visualize results
analyzer.visualize_projection_percentages(results)
analyzer.visualize_singular_values(results)
```

## Example Analyses

The repository includes two example analyses:

1. **Comparing Factual vs. Reasoning Tasks**: Analyzes whether simple factual recall tasks use more of the residual stream for direct token prediction compared to reasoning tasks.

2. **Analyzing Reasoning Chains**: Examines how the projection percentages evolve throughout a multi-step reasoning process.

## Requirements

- transformer_lens
- torch
- matplotlib
- seaborn
- numpy

## Installation

```bash
git clone https://github.com/yourusername/residual-stream-analysis.git
cd residual-stream-analysis
pip install -r requirements.txt
```

## Next Steps

Potential extensions to explore:
1. Analyzing attention patterns in relation to projection percentages
2. Comparing results across models of different sizes
3. Causal intervention experiments to verify the role of the orthogonal subspace
4. Tracking information flow through specific reasoning traces