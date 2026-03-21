# NNDS 2025 - KV Cache Handling and Compression

This repository contains the end-of-term homework for the **Neural Networks for Data Science (NNDS 2025)** course, taught by Prof. Simone Scardapane at Sapienza University of Rome. 

**Course Link:** [NNDS 2025](https://www.sscardapane.it/teaching/nnds-2025/)  
**Author:** Leonardo Rocci

---

## Homework Objective

The core request of this homework was to implement KV Caching mechanisms and explore KV Cache Compression within autoregressive models. The entire project is written from scratch in JAX and Equinox. We were tasked with selecting a dataset, training a model, and extending the standard attention mechanisms to support long-context efficiency via compression strategies.

For this assignment, we utilized the **Rotten Tomatoes Critic Reviews Dataset**, framing the binary sentiment classification problem ("fresh" vs "rotten") as an autoregressive language modeling task. 

Specifically, the raw reviews are preprocessed into a strict sequence template to cleanly guide generation: `[SOS] <review text> [SOR] [RP]... [SEN] [FRE]/[ROT] [EOS]`. 
After digesting the `[SOS]` (Start of Sequence), the review text, and the `[SOR]` (Start of Reasoning) anchor, the model is tasked with autoregressively generating exactly **15 dummy `[RP]` (Reasoning Placeholder) tokens**. While simulating a chain-of-thought, this mechanic artificially extends the decoding window to effectively test latency improvements from KV Caching and forms an implicit counting task for stability benchmarking. Finally, the model outputs a `[SEN]` (Sentiment) token, signaling the end of reasoning, and outputs either a `[FRE]` (Fresh) or `[ROT]` (Rotten) classification.

---

## Project Narrative

1. **Training:** We designed and trained a custom Decoder-only Transformer Language Model (SLM) from scratch using JAX/Equinox. The model learns to read a review and autoregressively generate the corresponding sentiment token.
2. **KV Caching:** To optimize the autoregressive decoding phase, we implemented a stateful KV Cache embedded directly within the Causal Self-Attention mechanism, drastically reducing the redundant compute required for token generation.
3. **KV Cache Compression:** We implemented three distinct KV cache eviction algorithms:
   - **Random Compressor:** A baseline that randomly drops tokens.
   - **K-Norm Compressor:** Evicts tokens based on the lowest L2 norm of the key vectors.
   - **SnapKV Compressor:** Ranks token importance based on the observation attention scores of the final prompt tokens.
   
   We additionally introduced features to dynamically protect crucial structural tokens (such as `[SOS]` and `[SOR]`) from being evicted by the compressors.
4. **Eviction Analysis:** We conducted a rigorous analysis comparing how different algorithms balance the retention of *Structural Anchors* (like punctuation and special tags) versus *Semantic Signals* (highly polarized sentiment adjectives). This involved writing scripts and evaluation tools to formally measure how algorithms behave under extreme compression budgets.

---

## Repository Structure

The project is structured into modular Python files and auxiliary directories. 

### `/src` (Core Modules)
- **`model.py`**: Contains the core neural network architecture. It includes the Equinox definitions for the `DecoderSLM`, `TransformerBlock`, and the `CausalSelfAttention` layer (which manages the `KVCache` state).
- **`compression.py`**: Defines the `BaseKVCompressor` contract and implements the `RandomCompressor`, `KNormCompressor`, and `SnapKVCompressor`. Contains the logic for safely tracking cache indices and enforcing token protections.
- **`predict.py`**: Handles the inference loop. Includes `batched_generate`, which manages a JAX `lax.while_loop` for autoregressive generation while optionally passing context to the compressors.
- **`benchmark_compression.py`**: The benchmarking suite used to run inference across different configurations, measuring things like task accuracy and formatting failures.
- **`analyze_evictions.py`**: An analysis library designed to inspect the behavior of eviction algorithms. It computes dataset-wide polarity statistics, inspects raw token retention ratios, and evaluates the GT-Aligned Semantic Score and structural hoarding metrics.

### Root Files and Other Directories
- **`Rocci - NNDS HW.ipynb`**: The main Jupyter Notebook consolidating the theoretical explanations, step-by-step training narrative, and the inline visualizations of the results. 
- **`NNDS_2025_Original.ipynb`**: The original, unaltered homework template.
- **`data/`**: Contains the original csv dataset, the processed dataset files (`.npy`), tokenizer (`tokenizer.json`), and trained model weights (`.eqx`).
- **`assets/`**: Auxiliary folder used for storing plots and training histories.

---

## Reproducibility

The project was primarily developed and evaluated natively on **macOS (Apple Silicon)**, leveraging the **MPS (Metal Performance Shaders)** backend through JAX. 

To ensure exact execution alignment across systems, the environment dependencies are fully locked to be completely replicable using **`uv`**, leveraging the provided `pyproject.toml` and `uv.lock` files.

### 1. Install `uv`
If not yet installed, you can quickly download `uv` for your OS:
- **macOS and Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Windows:** `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

### 2. Setup and Run
With `uv` installed and the repository cloned, you do not need to manually create or manually activate a virtual environment. `uv` handles implicit resolution securely.

Simply execute any project scripts seamlessly using `uv run`. This command auto-discovers the `uv.lock` file, automatically resolves the environment, and triggers the script:
```bash
# Example: Run the eviction analysis tool
uv run python src/analyze_evictions.py --mode inspect

# Example: Process a sweep of compression budgets
uv run python src/analyze_evictions.py --mode sweep
```
