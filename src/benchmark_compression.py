import os
import time
import argparse
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from tokenizers import Tokenizer
import re
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.model import DecoderSLM
from src.predict import get_config, load_data, batched_generate, analyze_generation, prepare_batch_for_inference
from src.compression import BaseKVCompressor, RandomCompressor, KNormCompressor, SnapKVCompressor


def plot_compression_benchmarks(
    benchmark_dict, 
    metrics=["Sentiment Acc", "Cond. Acc", "Bad RP"], 
    use_balanced=True
):
    """
    Plots a facet grid of line charts for the specified benchmark metrics.
    Automatically constructs a semantic "Visual Grammar" legend, hiding redundant sections.
    
    Args:
        benchmark_dict: dict of { "Legend Label": dataframe }
        metrics: list of strings. Valid options: 
                 "Sentiment Acc", "Cond. Acc", "Missing EOS", "Missing SEN", "Bad RP"
        use_balanced: bool. If True, plots balanced accuracy and percentage for counts. 
                      If False, plots standard accuracy and raw counts.
    """
    column_mapping = {
        "Sentiment Acc": ("Sentiment Acc (%)", True),
        "Cond. Acc": ("Cond. Acc (%)", True),
        "Missing EOS": ("Missing EOS", False),
        "Missing SEN": ("Missing SEN", False),
        "Bad RP": ("Bad RP Count", False)
    }
    
    color_map = {
        "RandomCompressor": "#1f77b4", # Blue
        "KNormCompressor": "#ff7f0e",  # Orange
        "SnapKVCompressor": "#2ca02c", # Green
        "StreamingLLMCompressor": "#d62728" # Red
    }
    
    parsed_data = []
    style_info = {}
    
    # Track unique configurations to conditionally render the legend
    used_methods = set()
    used_sor = set()
    used_decode = set()
    
    for label, df in benchmark_dict.items():
        temp_df = df.copy()
        temp_df['Legend_Label'] = label
        
        first_row = temp_df.iloc[0]
        method_name = first_row.get("Method", "Unknown")
        
        def is_true(val):
            return val is True or str(val).strip() == "✓"
            
        protect_sor = is_true(first_row.get("protect_sor", False))
        apply_on_decode = is_true(first_row.get("apply_on_decode", False))
        
        used_methods.add(method_name)
        used_sor.add(protect_sor)
        used_decode.add(apply_on_decode)
        
        color = color_map.get(method_name, "#7f7f7f") 
        
        # Distinct bold styling
        linestyle = '-' if protect_sor else '--'
        linewidth = 2.5 if protect_sor else 1.5
        marker = 'D' if apply_on_decode else 'o' # Diamond vs Circle
        markersize = 8 if apply_on_decode else 7
        
        style_info[label] = {
            "color": color,
            "linestyle": linestyle,
            "linewidth": linewidth,
            "marker": marker,
            "markersize": markersize
        }
        
        for metric in metrics:
            if metric not in column_mapping:
                raise ValueError(f"Unknown metric '{metric}'. Valid options: {list(column_mapping.keys())}")
                
            col_name, is_accuracy = column_mapping[metric]
            
            def extract_value(val_str):
                if pd.isna(val_str):
                    return np.nan
                matches = re.findall(r"([0-9.]+|nan)", str(val_str))
                if not matches:
                    return np.nan
                if len(matches) >= 2:
                    return float(matches[1]) if use_balanced else float(matches[0])
                return float(matches[0])

            temp_df[metric] = temp_df[col_name].apply(extract_value)
            
        parsed_data.append(temp_df)
        
    final_df = pd.concat(parsed_data, ignore_index=True)
    
    num_metrics = len(metrics)
    
    # Calculate rows and columns for a max 3-column layout
    cols = min(num_metrics, 3)
    rows = math.ceil(num_metrics / 3)
    
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    
    # Flatten axes array for easy iteration regardless of grid shape
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
        
    budgets = sorted(final_df['budget'].unique(), reverse=True)
    budget_strs = [str(b) for b in budgets]
    
    # --- CHANGED: Added enumerate to track the axis index ---
    for i, (ax, metric) in enumerate(zip(axes[:num_metrics], metrics)):
        for label in benchmark_dict.keys():
            subset = final_df[final_df['Legend_Label'] == label]
            subset = subset.set_index('budget').reindex(budgets).reset_index()
            
            style = style_info[label]
            ax.plot(
                budget_strs, subset[metric], 
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                marker=style["marker"],
                markersize=style["markersize"],
                label=label
            )
            
        col_name, is_accuracy = column_mapping[metric]
        unit = "(Balanced %)" if is_accuracy and use_balanced else "(Standard %)" if is_accuracy else "(%)" if use_balanced else "(Raw Count)"
            
        ax.set_title(f"{metric} {unit}", fontsize=12, pad=10)
        ax.set_xlabel("Cache Budget (Tokens)", fontsize=10)
        ax.set_ylabel("Score / Value", fontsize=10)
        
        ax.invert_xaxis()
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # --- CHANGED: Target the last plot of the first row (index 0, 1, or 2) ---
        if i == min(num_metrics - 1, 2):
            legend_elements = []
            
            # Always show the actual configured labels first
            for label_name in benchmark_dict.keys():
                style = style_info[label_name]
                legend_elements.append(Line2D(
                    [0], [0], color=style["color"], linestyle=style["linestyle"],
                    linewidth=style["linewidth"], marker=style["marker"], 
                    markersize=style["markersize"], label=label_name
                ))
            
            # Check if we need the visual key
            show_methods = len(used_methods) > 1
            show_sor = len(used_sor) > 1
            show_phase = len(used_decode) > 1
            
            if show_methods or show_sor or show_phase:
                legend_elements.append(Line2D([0], [0], color='w', label=' '))
                legend_elements.append(Line2D([0], [0], color='w', label='--- VISUAL KEY ---'))
            
            if show_methods:
                legend_elements.append(Line2D([0], [0], color='w', label='METHODS (Color):'))
                for method in sorted(used_methods):
                    color = color_map.get(method, "#7f7f7f")
                    name = method.replace("Compressor", "")
                    legend_elements.append(Line2D([0], [0], color=color, lw=3, label=f"  {name}"))
                    
            if show_sor:
                if show_methods: legend_elements.append(Line2D([0], [0], color='w', label=' '))
                legend_elements.append(Line2D([0], [0], color='w', label='SOR PROTECTION (Line):'))
                legend_elements.append(Line2D([0], [0], color='gray', lw=2.5, linestyle='-', label='  Protected (Solid)'))
                legend_elements.append(Line2D([0], [0], color='gray', lw=1.5, linestyle='--', label='  Vanilla (Dashed)'))
                
            if show_phase:
                if show_methods or show_sor: legend_elements.append(Line2D([0], [0], color='w', label=' '))
                legend_elements.append(Line2D([0], [0], color='w', label='GENERATION PHASE (Shape):'))
                legend_elements.append(Line2D([0], [0], color='gray', lw=0, marker='o', markersize=7, label='  Prefill Only (Circle)'))
                legend_elements.append(Line2D([0], [0], color='gray', lw=0, marker='D', markersize=8, label='  On Decode (Diamond)'))
            
            ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False, fontsize=10)

    # Hide any remaining empty subplots in the grid
    for ax in axes[num_metrics:]:
        ax.set_visible(False)
            
    plt.tight_layout()
    plt.show()


def run_compression_benchmark(
    compressor_class,
    config_list,
    data_dir="data",
    num_samples=None, #8192,
    batch_size=256,
    use_cache=True,
    max_new_tokens=28,
    d_model=128,
    num_heads=4,
    num_layers=6,
    mlp_hidden=512,
    model_weights="best_model_50.eqx",
    print_throughput=False
):
    print(f"Loading tokenizer...")
    tokenizer = Tokenizer.from_file(os.path.join(data_dir, "tokenizer.json"))
    pad_id = tokenizer.token_to_id("[PAD]")
    eos_id = tokenizer.token_to_id("[EOS]")
    sor_id = tokenizer.token_to_id("[SOR]")
    sen_id = tokenizer.token_to_id("[SEN]")
    fre_id = tokenizer.token_to_id("[FRE]")
    rot_id = tokenizer.token_to_id("[ROT]")
    rp_id = tokenizer.token_to_id("[RP]")
    
    print(f"Loading config and initialized model...")
    cfg = get_config()
    key = jax.random.PRNGKey(0)
    
    model = DecoderSLM(
        vocab_size=cfg["vocab_size"],
        max_seq_len=cfg["max_seq_len"],
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        mlp_hidden=mlp_hidden,
        key=key
    )
    
    model_path = os.path.join(data_dir, model_weights)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Weights not found at {model_path}")
        
    model = eqx.tree_deserialise_leaves(model_path, model)
    print(f"Model weights loaded from {model_weights}.")
    
    test_x, test_y = load_data(data_dir, num_samples)
    print(f"Loaded {len(test_x)} test samples.")
    
    results_summary = []
    
    for c_idx, config in enumerate(config_list):
        print(f"\n[{c_idx+1}/{len(config_list)}] Benchmarking Config: {config}")
        
        comp_key = jax.random.PRNGKey(42 + c_idx)
        compressor = compressor_class(**config, key=comp_key)
        
        all_metrics = []
        total_generated_tokens = 0
        total_time = 0.0
        
        num_batches = int(np.ceil(len(test_x) / batch_size))
        pbar = tqdm(range(0, len(test_x), batch_size), desc="Inferring Batches")
        
        for i in pbar:
            batch_x = test_x[i:i+batch_size]
            batch_y = test_y[i:i+batch_size]
            
            modified_batch, sor_positions = prepare_batch_for_inference(batch_x, sor_id, pad_id)
            prompt_lens = sor_positions + 1
            
            start_time = time.time()
            
            outputs = batched_generate(
                model, 
                modified_batch, 
                sor_positions,
                max_new_tokens, 
                pad_id, 
                eos_id,
                use_cache=use_cache,
                compressor=compressor
            )
            
            outputs = jax.device_get(outputs)
            
            end_time = time.time()
            elapsed = end_time - start_time
            total_time += elapsed
            
            for j in range(len(batch_x)):
                generated = outputs[j, prompt_lens[j]:]
                valid_gen = len(generated[generated != pad_id])
                total_generated_tokens += valid_gen
            
            batch_df, _ = analyze_generation(
                outputs, prompt_lens, pad_id, eos_id, sor_id, sen_id, fre_id, rot_id, rp_id, batch_y, tokenizer
            )
            all_metrics.append(batch_df)
            
            pbar.set_postfix({"batch_time": f"{elapsed:.2f}s"})

        final_df = pd.concat(all_metrics, ignore_index=True)
        
        throughput = total_generated_tokens / total_time
        
        eos_fails = len(test_x) - final_df['has_eos'].sum()
        sen_fails = len(test_x) - final_df['has_sen'].sum()
        rp_fails = len(test_x) - final_df['is_perfect_rp'].sum()
        
        eos_pct = (eos_fails / len(test_x)) * 100
        sen_pct = (sen_fails / len(test_x)) * 100
        rp_pct = (rp_fails / len(test_x)) * 100
        
        sentiment_acc = final_df['is_correct'].mean() * 100
        
        pos_mask = final_df['gt'] == True
        neg_mask = final_df['gt'] == False
        tpr = final_df[pos_mask]['is_correct'].mean() * 100
        tnr = final_df[neg_mask]['is_correct'].mean() * 100
        sentiment_bal_acc = (tpr + tnr) / 2
        
        valid_df = final_df[final_df['valid_sentiment']]
        cond_acc = valid_df['is_correct'].mean() * 100
        
        v_pos_mask = valid_df['gt'] == True
        v_neg_mask = valid_df['gt'] == False
        v_tpr = valid_df[v_pos_mask]['is_correct'].mean() * 100 if valid_df[v_pos_mask].shape[0] > 0 else 0
        v_tnr = valid_df[v_neg_mask]['is_correct'].mean() * 100 if valid_df[v_neg_mask].shape[0] > 0 else 0
        cond_bal_acc = (v_tpr + v_tnr) / 2
        
        # Format config dictionary into individual distinct columns
        row_data = {
            "Method": compressor_class.__name__
        }
        for k, v in config.items():
            if isinstance(v, bool):
                row_data[k] = "✓" if v else " "
            else:
                row_data[k] = v
                
        if print_throughput:
            row_data["Throughput"] = throughput
            
        row_data.update({
            "Sentiment Acc (%)": f"{sentiment_acc:.1f} ({sentiment_bal_acc:.1f})",
            "Cond. Acc (%)": f"{cond_acc:.1f} ({cond_bal_acc:.1f})",
            "Missing EOS": f"{eos_fails} ({eos_pct:.1f}%)",
            "Missing SEN": f"{sen_fails} ({sen_pct:.1f}%)",
            "Bad RP Count": f"{rp_fails} ({rp_pct:.1f}%)"
        })
        
        results_summary.append(row_data)
        
    summary_df = pd.DataFrame(results_summary)
    print("\n" + "="*80)
    print("COMPRESSION BENCHMARK SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)
    
    return summary_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Compression Benchmarks")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate. If not provided, uses the entire 106k dataset.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--model_weights", type=str, default="best_model_50.eqx")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--mlp_hidden", type=int, default=512)
    parser.add_argument("--hide_throughput", action="store_true", help="Hide throughput from the output table")
    parser.add_argument("--compressor", type=str, choices=["random", "knorm", "snapkv"], default="random", help="Which compressor to test")
    args = parser.parse_args()

    # Define the experiment configs for the baseline RandomCompressor
    experiment_configs = [
        {"budget": k, "apply_on_decode": False, "protect_sor": p_sor} 
        for p_sor in [True, False]
        for k in [100, 30, 10, 3]
    ]

    compressor_map = {
        "random": RandomCompressor,
        "knorm": KNormCompressor,
        "snapkv": SnapKVCompressor
    }

    run_compression_benchmark(
        compressor_class=compressor_map[args.compressor],
        config_list=experiment_configs,
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_hidden=args.mlp_hidden,
        model_weights=args.model_weights,
        print_throughput=not args.hide_throughput
    )
