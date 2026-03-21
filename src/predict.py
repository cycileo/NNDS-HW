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
from src.model import DecoderSLM, KVCache

def load_data(data_dir, num_samples=None):
    """Loads the test dataset."""
    test_x = np.load(os.path.join(data_dir, "test_x.npy"))
    test_y = np.load(os.path.join(data_dir, "test_y.npy"))
    
    if num_samples is not None:
        test_x = test_x[:num_samples]
        test_y = test_y[:num_samples]
        
    return test_x, test_y

def get_config():
    """Returns the expected hyperparameters based on the training configuration."""
    return {
        "vocab_size": 5000,
        "max_seq_len": 127,
        "d_model": 256,
        "num_heads": 4,
        "num_layers": 4,
        "mlp_hidden": 1024
    }

def prepare_batch_for_inference(batch_x, sor_id, pad_id):
    """
    Finds [SOR] in each sequence of batch_x.
    Replaces everything AFTER [SOR] with pad_id.
    Returns the modified batch and an array of sor_positions.
    """
    batch_size, seq_len = batch_x.shape
    modified_batch = np.copy(batch_x)
    sor_positions = np.zeros(batch_size, dtype=np.int32)
    
    for i in range(batch_size):
        seq = batch_x[i]
        idx = np.where(seq == sor_id)[0]
        if len(idx) > 0:
            sor_pos = idx[0]
            modified_batch[i, sor_pos + 1:] = pad_id
            sor_positions[i] = sor_pos
        else:
            sor_positions[i] = seq_len - 1
            
    return modified_batch, sor_positions
@eqx.filter_jit
def _generate_all_compiled(model, init_seqs, init_positions, max_new_tokens, pad_id, eos_id, compressor=None):
    batch_size, seq_len = init_seqs.shape
    
    def cond_fn(val):
        seqs, current_positions, is_finished, step = val
        return jnp.logical_and(jnp.logical_not(jnp.all(is_finished)), step < max_new_tokens)
        
    def body_fn(val):
        seqs, current_positions, is_finished, step = val
        logits, _ = jax.vmap(lambda x: model(x, compressor=compressor))(seqs)
        batch_indices = jnp.arange(batch_size)
        step_logits = logits[batch_indices, current_positions - 1, :]
        next_tokens = jnp.argmax(step_logits, axis=-1)
        
        out_of_bounds = current_positions >= seq_len
        force_pad = is_finished | out_of_bounds
        next_tokens = jnp.where(force_pad, pad_id, next_tokens)
        
        @jax.vmap
        def update_seq(s, pos, tok, pad_flag):
            return jnp.where(pad_flag, s, s.at[pos].set(tok))
            
        seqs = update_seq(seqs, current_positions, next_tokens, force_pad)
        is_finished = is_finished | (next_tokens == eos_id) | out_of_bounds
        current_positions = current_positions + 1
        step = step + 1
        return seqs, current_positions, is_finished, step

    init_val = (init_seqs, init_positions, jnp.zeros(batch_size, dtype=bool), jnp.array(0))
    final_val = jax.lax.while_loop(cond_fn, body_fn, init_val)
    return final_val[0]

@eqx.filter_jit
def _generate_with_cache_compiled(model, init_seqs, init_positions, max_new_tokens, pad_id, eos_id, compressor=None):
    batch_size, seq_len = init_seqs.shape
    
    # 1. Prefill
    logits, caches = jax.vmap(lambda x: model(x, return_cache=True, compressor=compressor))(init_seqs)
    
    # Reset cache indices to the end of the prompts
    caches = [KVCache(c.k, c.v, init_positions, c.prompt_len) for c in caches]
    
    # Extract the first generated tokens
    batch_indices = jnp.arange(batch_size)
    step_logits = logits[batch_indices, init_positions - 1, :]
    next_tokens = jnp.argmax(step_logits, axis=-1)
    
    out_of_bounds = init_positions >= seq_len
    force_pad = out_of_bounds
    next_tokens = jnp.where(force_pad, pad_id, next_tokens)
    
    @jax.vmap
    def update_seq(s, pos, tok, pad_flag):
        return jnp.where(pad_flag, s, s.at[pos].set(tok))
        
    seqs = update_seq(init_seqs, init_positions, next_tokens, force_pad)
    is_finished = (next_tokens == eos_id) | out_of_bounds
    current_positions = init_positions + 1
    
    # 2. Decode Loop
    def cond_fn(val):
        seqs, current_positions, next_tokens, caches, is_finished, step = val
        return jnp.logical_and(jnp.logical_not(jnp.all(is_finished)), step < max_new_tokens - 1)
        
    def body_fn(val):
        seqs, current_positions, next_tokens, caches, is_finished, step = val
        
        last_tokens = next_tokens[:, None] # shape (batch_size, 1)
        
        def model_step(tok, c):
            return model(tok, kv_caches=c, compressor=compressor)
            
        logits, new_caches = jax.vmap(model_step)(last_tokens, caches)
        
        next_tokens_new = jnp.argmax(logits[:, -1, :], axis=-1)
        
        out_of_bounds = current_positions >= seq_len
        force_pad = is_finished | out_of_bounds
        next_tokens_new = jnp.where(force_pad, pad_id, next_tokens_new)
        
        seqs = update_seq(seqs, current_positions, next_tokens_new, force_pad)
        
        is_finished = is_finished | (next_tokens_new == eos_id) | out_of_bounds
        current_positions = current_positions + 1
        step = step + 1
        
        return seqs, current_positions, next_tokens_new, new_caches, is_finished, step

    init_val = (seqs, current_positions, next_tokens, caches, is_finished, jnp.array(0))
    final_val = jax.lax.while_loop(cond_fn, body_fn, init_val)
    return final_val[0]

def batched_generate(model, prompts, sor_positions, max_new_tokens, pad_id, eos_id, use_cache=False, compressor=None):
    """
    Generates tokens autoregressively.
    Fully implemented in pure JAX via `jax.lax.while_loop` and static @eqx.filter_jit compilation 
    to achieve extreme speeds globally without memory leaks.
    """
    init_seqs = jnp.array(prompts)
    init_positions = jnp.array(sor_positions) + 1
    
    if not use_cache:
        return _generate_all_compiled(model, init_seqs, init_positions, max_new_tokens, pad_id, eos_id, compressor=compressor)
    else:
        return _generate_with_cache_compiled(model, init_seqs, init_positions, max_new_tokens, pad_id, eos_id, compressor=compressor)

def analyze_generation(outputs, prompt_lens, pad_id, eos_id, sor_id, sen_id, fre_id, rot_id, rp_id, ground_truths, tokenizer):
    """
    Analyzes the structural and semantic metrics of the generated sequences.
    Also returns a list of raw decoded string examples where the RP token count was != 15
    OR if non-RP tokens were generated in the reviewing section.
    """
    batch_size = outputs.shape[0]
    metrics = []
    bad_rp_examples = []
    
    # We bring arrays back to CPU for easy loop analysis
    outputs = np.array(outputs)
    
    for i in range(batch_size):
        seq = outputs[i]
        p_len = prompt_lens[i]
        gt = ground_truths[i]
        
        generated_part = seq[p_len:] # Everything after [SOR]
        
        has_eos = eos_id in generated_part
        has_sen = sen_id in generated_part
        
        # 1. Identify Review Part Slice
        if has_sen:
            sen_idx = np.where(generated_part == sen_id)[0][0]
            rp_slice = generated_part[:sen_idx]
        else:
            if has_eos:
                end_idx = np.where(generated_part == eos_id)[0][0]
                rp_slice = generated_part[:end_idx]
            else:
                valid_generated = generated_part[generated_part != pad_id]
                rp_slice = valid_generated
                
        # The structure is ONLY flawless if the slice contains exclusively `rp_id` and exactly 15 of them.
        actual_rp_count = np.sum(rp_slice == rp_id)
        is_perfect_rp = (actual_rp_count == 15) and (len(rp_slice) == 15)
        
        if not is_perfect_rp:
            # Decode the full sequence (prompt + generation) dropping padding
            # Keep special tokens (skip_special_tokens=False) so we can debug structure
            valid_seq = seq[seq != pad_id]
            decoded_str = tokenizer.decode(valid_seq.tolist(), skip_special_tokens=False)
            bad_rp_examples.append(decoded_str)
                
        # 2. Check Valid Sentiment Token (FRE or ROT follows SEN immediately)
        valid_sentiment = False
        predicted_sentiment = None # True for Fresh, False for Rotten
        
        if has_sen:
            sen_idx = np.where(generated_part == sen_id)[0][0]
            if sen_idx + 1 < len(generated_part):
                sent_token = generated_part[sen_idx + 1]
                if sent_token == fre_id:
                    valid_sentiment = True
                    predicted_sentiment = True
                elif sent_token == rot_id:
                    valid_sentiment = True
                    predicted_sentiment = False
                    
        # 3. Accuracy matching
        is_correct = False
        if valid_sentiment and predicted_sentiment == gt:
            is_correct = True
            
        metrics.append({
            "has_eos": bool(has_eos),
            "has_sen": bool(has_sen),
            "num_rp_tokens": int(actual_rp_count),
            "is_perfect_rp": bool(is_perfect_rp),
            "valid_sentiment": bool(valid_sentiment),
            "is_correct": bool(is_correct),
            "gt": bool(gt)
        })
        
    return pd.DataFrame(metrics), bad_rp_examples


def run_inference(
    data_dir="data",
    num_samples=None,
    batch_size=256,
    use_cache=False,
    max_new_tokens=30,
    d_model=128,
    num_heads=4,
    num_layers=6,
    mlp_hidden=512,
    model_weights="best_model.eqx"
):
    """
    Runs autoregressive inference on the test dataset.
    Returns:
        final_df: pandas DataFrame containing the metrics per sequence.
    """
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
    cfg = get_config() # Keep vocab and seq_len from here for now
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
    
    all_metrics = []
    all_bad_rp_examples = []
    
    total_generated_tokens = 0
    total_time = 0.0
    
    num_batches = int(np.ceil(len(test_x) / batch_size))
    pbar = tqdm(range(0, len(test_x), batch_size), desc="Inferring Batches")
    
    for i in pbar:
        batch_x = test_x[i:i+batch_size]
        batch_y = test_y[i:i+batch_size]
        
        # Prepare fixed-shape batch with wiped out ground truth
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
            use_cache=use_cache
        )
        
        # block until computation is done for accurate timing
        outputs = jax.device_get(outputs)
        
        end_time = time.time()
        elapsed = end_time - start_time
        total_time += elapsed
        
        # Calculate generated tokens for throughput
        for j in range(len(batch_x)):
            generated = outputs[j, prompt_lens[j]:]
            # Count valid generated tokens before pad
            valid_gen = len(generated[generated != pad_id])
            total_generated_tokens += valid_gen
        
        batch_df, bad_examples = analyze_generation(
            outputs, prompt_lens, pad_id, eos_id, sor_id, sen_id, fre_id, rot_id, rp_id, batch_y, tokenizer
        )
        all_metrics.append(batch_df)
        all_bad_rp_examples.extend(bad_examples)
        
        pbar.set_postfix({"batch_time": f"{elapsed:.2f}s"})

    final_df = pd.concat(all_metrics, ignore_index=True)
    
    
    print("\n" + "="*50)
    print("INFERENCE RESULTS")
    print("="*50)
    print(f"Total Samples  : {len(test_x)}")
    print(f"Model File     : {model_weights}")
    print(f"KV Caching     : {'ENABLED' if use_cache else 'DISABLED'}")
    throughput = total_generated_tokens / total_time
    print(f"Throughput     : {throughput:.2f} tokens/sec")
    print("-" * 50)
    print("STRUCTURAL METRICS:")
    
    eos_fails = len(test_x) - final_df['has_eos'].sum()
    if eos_fails == 0:
        print("Generated [EOS]: ALL 100.0%")
    else:
        print(f"Generated [EOS]: {eos_fails} sequences ({(eos_fails/len(test_x))*100:.2f}%) failed missing EOS.")
        
    sen_fails = len(test_x) - final_df['has_sen'].sum()
    if sen_fails == 0:
        print("Generated [SEN]: ALL 100.0%")
    else:
        print(f"Generated [SEN]: {sen_fails} sequences ({(sen_fails/len(test_x))*100:.2f}%) failed missing SEN.")
    
    bad_rp_count = len(all_bad_rp_examples)
    if bad_rp_count == 0:
        print("Generated [RP] : ALL 100.0% exactly 15 tokens")
    else:
        bad_pct = (bad_rp_count / len(test_x)) * 100
        print(f"Generated [RP] : {bad_rp_count} sequences ({bad_pct:.2f}%) failed to form exactly 15 [RP] tokens.")
        
    print("-" * 50)
    
    pos_mask = final_df['gt'] == True
    neg_mask = final_df['gt'] == False
    tpr = final_df[pos_mask]['is_correct'].mean() * 100
    tnr = final_df[neg_mask]['is_correct'].mean() * 100
    bal_acc = (tpr + tnr) / 2
    
    valid_df = final_df[final_df['valid_sentiment']]
    v_pos_mask = valid_df['gt'] == True
    v_neg_mask = valid_df['gt'] == False
    v_tpr = valid_df[v_pos_mask]['is_correct'].mean() * 100 if valid_df[v_pos_mask].shape[0] > 0 else 0
    v_tnr = valid_df[v_neg_mask]['is_correct'].mean() * 100 if valid_df[v_neg_mask].shape[0] > 0 else 0
    cond_bal_acc = (v_tpr + v_tnr) / 2
    
    print(f"SENTIMENT ACC. : {final_df['is_correct'].mean() * 100:.2f}% ({bal_acc:.2f}% bal.) (over all samples)")
    print(f"CONDITIONAL ACC: {valid_df['is_correct'].mean() * 100:.2f}% ({cond_bal_acc:.2f}% bal.) (where valid sentiment exists)")
    print("="*50)
    
    if len(all_bad_rp_examples) > 0:
        print("\n=== EXAMPLES OF BAD [RP] TOKEN SEQUENCES ===")
        # Print up to 3 bad examples
        for idx, ex in enumerate(all_bad_rp_examples[:3]):
            print(f"\nExample {idx+1}:\n{ex}")
            print("-" * 30)
    
    return final_df, total_time, throughput

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autoregressive Inference & Evaluation")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of test samples to evaluate (default all)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--use_cache", action="store_true", help="Use KV Caching for faster generation")
    parser.add_argument("--max_new_tokens", type=int, default=30, help="Maximum tokens to generate after [SOR]")
    parser.add_argument("--d_model", type=int, default=128, help="Model hidden dimension")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--mlp_hidden", type=int, default=512, help="MLP hidden dimension")
    parser.add_argument("--model_weights", type=str, default="best_model.eqx", help="Filename of the `.eqx` model checkpoint inside data_dir")
    args = parser.parse_args()
    
    run_inference(
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        use_cache=args.use_cache,
        max_new_tokens=args.max_new_tokens,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_hidden=args.mlp_hidden,
        model_weights=args.model_weights
    )
