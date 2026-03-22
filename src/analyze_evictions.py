import os
import argparse
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from typing import Dict, Any, List, Tuple
from tqdm.auto import tqdm
from tokenizers import Tokenizer

from src.model import DecoderSLM
from src.predict import get_config, load_data, prepare_batch_for_inference
from src.compression import RandomCompressor, KNormCompressor, SnapKVCompressor

@eqx.filter_jit
def get_compression_indices(model: DecoderSLM, batch_x: jnp.ndarray, compressor: Any) -> jnp.ndarray:
    """Returns the indices of the tokens kept by the KV cache compressor.
    Shape: (num_layers, batch_size, budget)
    """
    logits, caches, indices = jax.vmap(
        lambda x: model(x, return_cache=True, compressor=compressor, return_indices=True)
    )(batch_x)
    return indices

def load_setup(data_dir: str = "data", num_samples: int = None, model_weights: str = "best_model_50.eqx", d_model: int = 128, num_heads: int = 4, num_layers: int = 6, mlp_hidden: int = 512) -> Dict[str, Any]:
    """Loads and prepares the dataset, tokenizer, and model for analysis. 
    Can be used in Jupyter Notebooks to avoid reloading data repeatedly."""
    print("Loading tokenizer and configuration...")
    tokenizer = Tokenizer.from_file(os.path.join(data_dir, "tokenizer.json"))
    pad_id = tokenizer.token_to_id("[PAD]")
    sor_id = tokenizer.token_to_id("[SOR]")
    cfg = get_config()
    cfg["d_model"] = d_model
    cfg["num_heads"] = num_heads
    cfg["num_layers"] = num_layers
    cfg["mlp_hidden"] = mlp_hidden
    
    print("Loading data...")
    test_x, test_y = load_data(data_dir, num_samples)
    
    print("Initializing model...")
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
    if os.path.exists(model_path):
        model = eqx.tree_deserialise_leaves(model_path, model)
        print(f"Loaded weights from {model_weights}.")
    else:
        print(f"WARNING: Weights not found at {model_path}. Using random weights.")
        
    return {
        "tokenizer": tokenizer,
        "cfg": cfg,
        "model": model,
        "test_x": test_x,
        "test_y": test_y,
        "pad_id": pad_id,
        "sor_id": sor_id
    }

def compute_dataset_statistics(test_x: np.ndarray, test_y: np.ndarray, vocab_size: int, pad_id: int, sor_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates token frequencies and the Balanced Sentiment Bias (bias_bal) over the dataset.
    Returns:
        word_totals (np.ndarray): Total occurrences of each word in the dataset.
        bias_bal (np.ndarray): The prior-adjusted sentiment bias of each word (-1.0 to 1.0).
    """
    def get_token_counts(x_data, y_labels, label_val):
        mask_label = y_labels == label_val
        subset_x = x_data[mask_label]
        sor_pos = np.argmax(subset_x == sor_id, axis=1)
        indices = np.arange(subset_x.shape[1])
        valid_mask = indices[None, :] <= sor_pos[:, None]
        return np.bincount(subset_x[valid_mask].flatten(), minlength=vocab_size)

    total_p = get_token_counts(test_x, test_y, 1)
    total_n = get_token_counts(test_x, test_y, 0)
    
    word_totals = total_p + total_n
    mask_v = word_totals > 0
    total_toks = np.sum(total_p) + np.sum(total_n)
    
    prior_p, prior_n = (0.5, 0.5)
    if total_toks > 0:
        prior_p = np.sum(total_p) / total_toks
        prior_n = np.sum(total_n) / total_toks
        
    bias_bal = np.zeros(vocab_size, dtype=float)
    if total_toks > 0:
        norm_p = (total_p[mask_v] / word_totals[mask_v]) / (prior_p + 1e-12)
        norm_n = (total_n[mask_v] / word_totals[mask_v]) / (prior_n + 1e-12)
        bias_bal[mask_v] = (norm_p - norm_n) / (norm_p + norm_n + 1e-12)
        
    return word_totals, bias_bal

def inspect_token_eviction(compressor_name: str = "random", budget: int = 18, protect_sor: bool = False, top_k: int = 5, order_by: str = "retention", aggregation: str = "layer", setup_data: Dict = None, data_dir: str = "data", batch_size: int = 128, num_samples: int = None):
    """
    Manual inspection function for analysis. Shows a tabular view of the top tokens 
    kept and evicted by a compressor, sortable by retention rate or absolute count.
    """
    if setup_data is None:
        setup_data = load_setup(data_dir=data_dir, num_samples=num_samples)
        
    tokenizer = setup_data["tokenizer"]
    model = setup_data["model"]
    test_x = setup_data["test_x"]
    test_y = setup_data["test_y"]
    pad_id = setup_data["pad_id"]
    sor_id = setup_data["sor_id"]
    vocab_size = setup_data["cfg"]["vocab_size"]
    num_layers = setup_data["cfg"]["num_layers"]
    
    comp_map = {"random": RandomCompressor, "knorm": KNormCompressor, "snapkv": SnapKVCompressor}
    compressor = comp_map[compressor_name](budget=budget, apply_on_decode=False, protect_sor=protect_sor, key=jax.random.PRNGKey(42))
    
    total_layer_slots = np.zeros(vocab_size, dtype=np.int64)
    total_layer_hits = np.zeros(vocab_size, dtype=np.int64)
    
    pbar = tqdm(range(0, len(test_x), batch_size), desc=f"Inspecting {compressor_name.upper()}")
    for start_idx in pbar:
        end_idx = min(start_idx + batch_size, len(test_x))
        batch_x = test_x[start_idx:end_idx]
        modified_batch, sor_pos = prepare_batch_for_inference(batch_x, sor_id, pad_id)
        
        layer_indices = get_compression_indices(model, modified_batch, compressor)
        layer_ids_np = np.array(layer_indices) # (layers, batch, budget)
        mask_within_sor = layer_ids_np <= sor_pos[None, :, None]
        
        indices = np.arange(modified_batch.shape[1])
        valid_mask = indices[None, :] <= sor_pos[:, None]
        
        # Track slots and hits based on aggregation strategy
        if aggregation in ["sequence_any", "sequence_all"]:
            kept_in_layers = np.zeros((num_layers, modified_batch.shape[0], modified_batch.shape[1]), dtype=bool)
            b_idx = np.arange(len(batch_x))[:, None]
            for l in range(num_layers):
                kept_in_layers[l, b_idx, layer_ids_np[l]] = True
                kept_in_layers[l, ~valid_mask] = False
                
            if aggregation == "sequence_any":
                final_kept_mask = np.any(kept_in_layers, axis=0)
            else: # sequence_all
                final_kept_mask = np.all(kept_in_layers, axis=0) & valid_mask
                
            total_layer_slots += np.bincount(modified_batch[valid_mask].flatten(), minlength=vocab_size)
            total_layer_hits += np.bincount(modified_batch[final_kept_mask].flatten(), minlength=vocab_size)
            
        else: # layer
            # Track presented slots (across all layers)
            batch_counts = np.bincount(modified_batch[valid_mask].flatten(), minlength=vocab_size)
            total_layer_slots += batch_counts * num_layers
            
            # Track layer hits
            kept_tokens = modified_batch[np.arange(len(batch_x))[None, :, None], layer_ids_np]
            for l in range(num_layers):
                valid_kept_l = kept_tokens[l][mask_within_sor[l]]
                total_layer_hits += np.bincount(valid_kept_l.flatten(), minlength=vocab_size)
            
    # Filter noise from rare words
    # Token must appear in at least 10 sequences (60 layer-slots)
    valid_mask = total_layer_slots >= (10 * num_layers)
    valid_idx = np.where(valid_mask)[0]
    
    retention_rates = np.zeros(vocab_size, dtype=float)
    retention_rates[valid_mask] = (total_layer_hits[valid_mask] / total_layer_slots[valid_mask]) * 100.0
    
    if order_by == "retention":
        sort_kept = valid_idx[np.argsort(-retention_rates[valid_idx])]
        sort_evicted = valid_idx[np.argsort(retention_rates[valid_idx])]
        metric_name = "Retention Rate"
    else:
        sort_kept = valid_idx[np.argsort(-total_layer_hits[valid_idx])]
        sort_evicted = valid_idx[np.argsort(-(total_layer_slots[valid_idx] - total_layer_hits[valid_idx]))]
        metric_name = "Abs Kept/Evicted Count"
        
    def print_table(title, sort_indices):
        print(f"\n{title}")
        print(f"| {'Rank':<4} | {'Token':<15} | {'Retained':<9} | {'Kept / Total':<15} |")
        print(f"|{'-'*6}|{'-'*17}|{'-'*11}|{'-'*17}|")
        for i, tok in enumerate(sort_indices[:top_k]):
            tok_str = tokenizer.decode([tok], skip_special_tokens=False).replace('\n', ' ')
            hits, slots = int(total_layer_hits[tok]), int(total_layer_slots[tok])
            fraction = f"{hits} / {slots}"
            print(f"| {i+1:>4} | {tok_str[:15]:<15} | {retention_rates[tok]:>8.2f}% | {fraction:>15} |")

    print_table(f"[{compressor_name.upper()} | Budget {budget}] TOP {top_k} MOST KEPT ({metric_name})", sort_kept)
    print_table(f"[{compressor_name.upper()} | Budget {budget}] TOP {top_k} MOST EVICTED ({metric_name})", sort_evicted)


def evaluate_eviction(compressors: List[str] = ["random", "knorm", "snapkv"], budgets: List[int] = [3, 10, 30, 100], k_sentiment: int = 100, k_frequent: int = 20, protect_sor: bool = False, aggregation: str = "layer", setup_data: Dict = None, data_dir: str = "data", batch_size: int = 128, num_samples: int = None) -> Dict:
    """
    Runs a parameter sweep calculating GT-Aligned Cache, Context-Aligned Sentiment Retention, 
    and Top-Frequent Retention metrics across compressors and budgets.
    """
    if setup_data is None:
        setup_data = load_setup(data_dir=data_dir, num_samples=num_samples)
        
    vocab_size = setup_data["cfg"]["vocab_size"]
    pad_id = setup_data["pad_id"]
    sor_id = setup_data["sor_id"]
    test_x = setup_data["test_x"]
    test_y = setup_data["test_y"]
    model = setup_data["model"]
    num_layers = setup_data["cfg"]["num_layers"]
    
    word_totals, bias_bal = compute_dataset_statistics(test_x, test_y, vocab_size, pad_id, sor_id)
    
    # 1. Sentiment Target Sets (k_sentiment PER class)
    sorted_bias = np.argsort(bias_bal)
    pos_mask = np.zeros(vocab_size, dtype=bool)
    neg_mask = np.zeros(vocab_size, dtype=bool)
    neg_mask[sorted_bias[:k_sentiment]] = True
    pos_mask[sorted_bias[-k_sentiment:]] = True
    
    # 2. Frequency Target Sets (matching total size: k_frequent)
    sorted_freq = np.argsort(word_totals)
    freq_mask = np.zeros(vocab_size, dtype=bool)
    freq_mask[sorted_freq[-k_frequent:]] = True
    
    metric_names = [
        "GT-Aligned Cache",
        f"Context-Aligned Sentiment Retention (%) [k={k_sentiment} per class]",
        f"Top-Frequent Retention (%) [k={k_frequent} total]"
    ]
    
    results = {m: {c: {} for c in compressors} for m in range(3)}
    comp_map = {"random": RandomCompressor, "knorm": KNormCompressor, "snapkv": SnapKVCompressor}
    
    print("\n" + "="*90)
    print(f"RUNNING METRICS SWEEP")
    print("="*90)
    
    for comp_name in compressors:
        for b in budgets:
            compressor = comp_map[comp_name](budget=b, apply_on_decode=False, protect_sor=protect_sor, key=jax.random.PRNGKey(42))
            
            m_gt_global = []
            s_kept, s_pres = 0.0, 0.0
            f_kept, f_pres = 0.0, 0.0
            
            pbar = tqdm(range(0, len(test_x), batch_size), desc=f"Eval {comp_name.upper()} (B={b})")
            for start_idx in pbar:
                end_idx = min(start_idx + batch_size, len(test_x))
                batch_x, batch_y = test_x[start_idx:end_idx], test_y[start_idx:end_idx]
                modified_batch, sor_pos = prepare_batch_for_inference(batch_x, sor_id, pad_id)
                
                layer_indices = get_compression_indices(model, modified_batch, compressor)
                layer_ids_np = np.array(layer_indices)
                mask_within_sor = layer_ids_np <= sor_pos[None, :, None]
                
                is_pos_seq = (batch_y == 1)
                is_neg_seq = (batch_y == 0)
                indices = np.arange(modified_batch.shape[1])
                v_mask = indices[None, :] <= sor_pos[:, None]
                batch_tokens = modified_batch
                
                # Presentations
                f_pres += np.sum(freq_mask[batch_tokens[v_mask]])
                if np.any(is_pos_seq):
                    s_pres += np.sum(pos_mask[batch_tokens[is_pos_seq[:, None] & v_mask]])
                if np.any(is_neg_seq):
                    s_pres += np.sum(neg_mask[batch_tokens[is_neg_seq[:, None] & v_mask]])
                    
                # Kept
                kept_tokens = modified_batch[np.arange(len(batch_x))[None, :, None], layer_ids_np]
                
                if aggregation in ["sequence_any", "sequence_all"]:
                    kept_in_layers = np.zeros((num_layers, batch_tokens.shape[0], batch_tokens.shape[1]), dtype=bool)
                    b_idx = np.arange(len(batch_x))[:, None]
                    for l in range(num_layers):
                        kept_in_layers[l, b_idx, layer_ids_np[l]] = True
                        kept_in_layers[l, ~v_mask] = False
                        
                    if aggregation == "sequence_any":
                        final_kept_mask = np.any(kept_in_layers, axis=0) # (batch, seq_len)
                    else:
                        final_kept_mask = np.all(kept_in_layers, axis=0) & v_mask
                        
                    f_kept += np.sum(freq_mask[batch_tokens[final_kept_mask]])
                    
                    if np.any(is_pos_seq):
                        s_kept += np.sum(pos_mask[batch_tokens[is_pos_seq[:, None] & final_kept_mask]])
                    if np.any(is_neg_seq):
                        s_kept += np.sum(neg_mask[batch_tokens[is_neg_seq[:, None] & final_kept_mask]])
                        
                    # GT Aligned Cache
                    l_sign = (batch_y * 2 - 1).astype(float)
                    b_bal_k = np.where(final_kept_mask, bias_bal[batch_tokens], 0.0)
                    s_m_gt = np.sum(b_bal_k * l_sign[:, None], axis=1) # (batch,)
                    n_k = np.sum(final_kept_mask, axis=1)       # (batch,)
                    m_gt_global.extend(np.where(n_k > 0, s_m_gt / (n_k + 1e-12), 0.0))
                else: # layer
                    m_f = freq_mask[kept_tokens] & mask_within_sor
                    f_kept += np.sum(m_f) / num_layers
                    
                    m_s = np.zeros_like(kept_tokens, dtype=bool)
                    if np.any(is_pos_seq):
                        m_s[:, is_pos_seq, :] = pos_mask[kept_tokens[:, is_pos_seq, :]]
                    if np.any(is_neg_seq):
                        m_s[:, is_neg_seq, :] = neg_mask[kept_tokens[:, is_neg_seq, :]]
                    m_s = m_s & mask_within_sor
                    s_kept += np.sum(m_s) / num_layers
                    
                    # GT Aligned Cache
                    l_sign = (batch_y * 2 - 1).astype(float)
                    b_bal_k = np.where(mask_within_sor, bias_bal[kept_tokens], 0.0)
                    s_m_gt = np.sum(b_bal_k * l_sign[None, :, None], axis=(0, 2)) / num_layers
                    n_k = np.sum(mask_within_sor, axis=(0, 2)) / num_layers
                    m_gt_global.extend(np.where(n_k > 0, s_m_gt / (n_k + 1e-12), 0.0))
                
            res_gt = np.mean(m_gt_global) * 100.0 if m_gt_global else 0.0
            res_s = (s_kept / s_pres * 100.0) if s_pres > 0 else 0.0
            res_f = (f_kept / f_pres * 100.0) if f_pres > 0 else 0.0
            
            results[0][comp_name][b] = res_gt
            results[1][comp_name][b] = res_s
            results[2][comp_name][b] = res_f
            
    print("\n" + "="*90)
    print("FINAL EXPERIMENT RESULTS")
    print("="*90)
    
    for m_idx, m_name in enumerate(metric_names):
        print(f"\n### {m_name}")
        col_width = 8
        header = f"| {'Compressor':<14} |" + "".join([f" {'B='+str(b):>{col_width}} |" for b in budgets])
        sep    = f"|{'-'*16}|" + "".join([f"{'-'*(col_width+2)}|" for _ in budgets])
        print(header)
        print(sep)
        for comp in compressors:
            row = f"| {comp:<14} |"
            for b in budgets:
                if m_idx == 0:
                    row += f" {results[m_idx][comp][b]:+8.2f} |"
                else:
                    row += f" {results[m_idx][comp][b]:8.2f} |"
            print(row)
    print("\n" + "="*90 + "\n")
    
    return results

def evaluate_sor_retention(compressors: List[str] = ["random", "knorm", "snapkv"], budgets: List[int] = [3, 10, 30, 100], aggregation: str = "layer", setup_data: Dict = None, data_dir: str = "data", batch_size: int = 128, num_samples: int = None) -> Dict:
    """
    Runs a parameter sweep calculating the retention rate of the [SOR] token across compressors and budgets
    with protect_sor=False.
    """
    if setup_data is None:
        setup_data = load_setup(data_dir=data_dir, num_samples=num_samples)
        
    sor_id = setup_data["sor_id"]
    pad_id = setup_data["pad_id"]
    test_x = setup_data["test_x"]
    model = setup_data["model"]
    num_layers = setup_data["cfg"]["num_layers"]
    
    comp_map = {"random": RandomCompressor, "knorm": KNormCompressor, "snapkv": SnapKVCompressor}
    results = {c: {} for c in compressors}
    
    print("\n" + "="*90)
    print(f"RUNNING [SOR] TOKEN RETENTION SWEEP")
    print("="*90)
    
    for comp_name in compressors:
        for b in budgets:
            compressor = comp_map[comp_name](budget=b, apply_on_decode=False, protect_sor=False, key=jax.random.PRNGKey(42))
            
            total_sor_hits = 0
            total_sor_slots = 0
            
            pbar = tqdm(range(0, len(test_x), batch_size), desc=f"Eval {comp_name.upper()} (B={b})")
            for start_idx in pbar:
                end_idx = min(start_idx + batch_size, len(test_x))
                batch_x = test_x[start_idx:end_idx]
                modified_batch, sor_pos = prepare_batch_for_inference(batch_x, sor_id, pad_id)
                
                layer_indices = get_compression_indices(model, modified_batch, compressor)
                layer_ids_np = np.array(layer_indices)
                mask_within_sor = layer_ids_np <= sor_pos[None, :, None]
                
                kept_sor_layers = np.any(layer_ids_np == sor_pos[None, :, None], axis=2)
                
                if aggregation == "sequence_any":
                    total_sor_hits += np.sum(np.any(kept_sor_layers, axis=0))
                    total_sor_slots += len(batch_x)
                elif aggregation == "sequence_all":
                    total_sor_hits += np.sum(np.all(kept_sor_layers, axis=0))
                    total_sor_slots += len(batch_x)
                else: # layer
                    total_sor_hits += np.sum(kept_sor_layers)
                    total_sor_slots += len(batch_x) * num_layers

            retention = (total_sor_hits / total_sor_slots * 100.0) if total_sor_slots > 0 else 0.0
            results[comp_name][b] = retention
            
    print("\n" + "="*90)
    print("FINAL EXPERIMENT RESULTS: [SOR] RETENTION")
    print("="*90)
    
    print(f"\n### [SOR] Token Retention (%)")
    col_width = 8
    header = f"| {'Compressor':<14} |" + "".join([f" {'B='+str(b):>{col_width}} |" for b in budgets])
    sep    = f"|{'-'*16}|" + "".join([f"{'-'*(col_width+2)}|" for _ in budgets])
    print(header)
    print(sep)
    for comp in compressors:
        row = f"| {comp:<14} |"
        for b in budgets:
            row += f" {results[comp][b]:8.2f} |"
        print(row)
    print("\n" + "="*90 + "\n")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Evictions Library")
    parser.add_argument("--mode", type=str, choices=["inspect", "sweep", "sor_sweep"], required=True, help="Mode of execution")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of test samples")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for analysis")
    parser.add_argument("--protect_sor", action="store_true", help="Protect [SOR] token from eviction")
    parser.add_argument("--aggregation", type=str, choices=["layer", "sequence_any", "sequence_all"], default="layer", help="Aggregation metric for retention tracking")
    
    # Sweep Args
    parser.add_argument("--budgets", type=int, nargs="+", default=[3, 10, 30, 100], help="Budgets for sweep")
    parser.add_argument("--compressors", type=str, nargs="+", default=["random", "knorm", "snapkv"], help="Compressors for sweep")
    parser.add_argument("--k_sentiment", type=int, default=100, help="k tokens PER SENTIMENT CLASS")
    parser.add_argument("--k_frequent", type=int, default=20, help="k frequent tokens IN TOTAL")
    
    # Inspect Args
    parser.add_argument("--compressor", type=str, default="snapkv", help="Single compressor for inspect")
    parser.add_argument("--budget", type=int, default=18, help="Single budget for inspect")
    parser.add_argument("--top_k", type=int, default=5, help="Top K to display in inspect")
    parser.add_argument("--order_by", type=str, choices=["retention", "absolute"], default="retention", help="Order inspect output by retention rate or absolute count")
    
    args = parser.parse_args()
    
    setup_data = load_setup(data_dir=args.data_dir, num_samples=args.num_samples)
    
    if args.mode == "inspect":
        inspect_token_eviction(
            compressor_name=args.compressor,
            budget=args.budget,
            protect_sor=args.protect_sor,
            top_k=args.top_k,
            order_by=args.order_by,
            aggregation=args.aggregation,
            setup_data=setup_data,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_samples=args.num_samples
        )
    elif args.mode == "sweep":
        evaluate_eviction(
            compressors=args.compressors,
            budgets=args.budgets,
            k_sentiment=args.k_sentiment,
            k_frequent=args.k_frequent,
            protect_sor=args.protect_sor,
            aggregation=args.aggregation,
            setup_data=setup_data,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_samples=args.num_samples
        )
    elif args.mode == "sor_sweep":
        evaluate_sor_retention(
            compressors=args.compressors,
            budgets=args.budgets,
            aggregation=args.aggregation,
            setup_data=setup_data,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_samples=args.num_samples
        )
