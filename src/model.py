import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, NamedTuple, Any
import math

from src.compression import BaseKVCompressor

class KVCache(NamedTuple):
    k: jax.Array
    v: jax.Array
    cache_index: jax.Array
    prompt_len: jax.Array

class CausalSelfAttention(eqx.Module):
    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    o_proj: eqx.nn.Linear
    num_heads: int
    head_dim: int

    def __init__(self, d_model, num_heads, key):
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads
        
        keys = jax.random.split(key, 4)
        self.q_proj = eqx.nn.Linear(d_model, d_model, use_bias=False, key=keys[0])
        self.k_proj = eqx.nn.Linear(d_model, d_model, use_bias=False, key=keys[1])
        self.v_proj = eqx.nn.Linear(d_model, d_model, use_bias=False, key=keys[2])
        self.o_proj = eqx.nn.Linear(d_model, d_model, use_bias=False, key=keys[3])

    def __call__(self, x, mask=None, pad_mask=None, kv_cache: Optional[KVCache] = None, return_cache: bool = False, compressor: Optional[BaseKVCompressor] = None, return_indices: bool = False):
        seq_len, _ = x.shape
        
        q = jax.vmap(self.q_proj)(x).reshape(seq_len, self.num_heads, self.head_dim)
        k = jax.vmap(self.k_proj)(x).reshape(seq_len, self.num_heads, self.head_dim)
        v = jax.vmap(self.v_proj)(x).reshape(seq_len, self.num_heads, self.head_dim)

        if pad_mask is not None:
            k = jnp.where(pad_mask[:, None, None], 0.0, k)
            v = jnp.where(pad_mask[:, None, None], 0.0, v)

        if kv_cache is not None:
            past_k, past_v, cache_index, prompt_len = kv_cache
            
            if compressor is not None:
                # Unpack both the write location and the new pointer state, passing past_k
                target_index, new_cache_index = compressor.compress_decode(cache_index, compressor.budget, k, past_k, prompt_len)
            else:
                target_index = cache_index
                new_cache_index = cache_index + 1
                
            new_k = jax.lax.dynamic_update_slice(past_k, k, (target_index, 0, 0))
            new_v = jax.lax.dynamic_update_slice(past_v, v, (target_index, 0, 0))
            
            new_cache = KVCache(new_k, new_v, new_cache_index, prompt_len)
            k_cache = new_k
            v_cache = new_v
            
            attn_scores = jnp.einsum("qhd,khd->hqk", q, k_cache) / math.sqrt(self.head_dim)
            
            # Mask out invalid/future keys
            max_seq_len = k_cache.shape[0]
            valid_keys = jnp.arange(max_seq_len) < new_cache_index
            
            if compressor is not None:
                # Dynamically restrict attention mass to populated block slots 
                # to prevent softmax leaking into zeroed padding.
                is_populated = jnp.any(k_cache != 0, axis=(1, 2))
                valid_keys = jnp.logical_and(valid_keys, is_populated)
                
            attn_scores = jnp.where(valid_keys[None, None, :], attn_scores, -jnp.inf)
            
            if seq_len > 1:
                # Apply causal mask inside the new tokens window
                q_idx = jnp.arange(seq_len)[:, None]
                k_idx = jnp.arange(max_seq_len)[None, :]
                causal = jnp.logical_or(k_idx < cache_index, k_idx - cache_index <= q_idx)
                attn_scores = jnp.where(causal[:, None, :], attn_scores, -jnp.inf)
                
            if mask is not None: # For prefill backwards compat
                # Custom mask logic can be added here if needed
                pass
                
            attn_weights = jax.nn.softmax(attn_scores, axis=-1)
            out = jnp.einsum("hqk,khd->qhd", attn_weights, v_cache)
            out = out.reshape(seq_len, -1)
            
            if return_indices:
                # In decode phase, compression indices are not returned
                return jax.vmap(self.o_proj)(out), new_cache, None
            return jax.vmap(self.o_proj)(out), new_cache
        else:
            if compressor is not None:
                if return_indices:
                    k_comp, v_comp, cache_len, prompt_len, comp_indices = compressor.compress_prefill(q, k, v, return_indices=True)
                else:
                    k_comp, v_comp, cache_len, prompt_len = compressor.compress_prefill(q, k, v)
                new_cache = KVCache(k_comp, v_comp, cache_len, prompt_len) if return_cache else None
            else:
                actual_len = jnp.sum(jnp.any(k != 0, axis=(1, 2)))
                new_cache = KVCache(k, v, jnp.array(seq_len), actual_len.astype(jnp.int32)) if return_cache else None
                comp_indices = jnp.arange(seq_len, dtype=jnp.int32) if return_indices else None
            
            attn_scores = jnp.einsum("qhd,khd->hqk", q, k) / math.sqrt(self.head_dim)
            if mask is not None:
                attn_scores = jnp.where(mask, attn_scores, -jnp.inf)
                
            attn_weights = jax.nn.softmax(attn_scores, axis=-1)
            out = jnp.einsum("hqk,khd->qhd", attn_weights, v)
            out = out.reshape(seq_len, -1)
            
            if return_indices:
                return jax.vmap(self.o_proj)(out), new_cache, comp_indices
            return jax.vmap(self.o_proj)(out), new_cache

class TransformerBlock(eqx.Module):
    ln_1: eqx.nn.LayerNorm
    attn: CausalSelfAttention
    ln_2: eqx.nn.LayerNorm
    mlp: eqx.nn.MLP

    def __init__(self, d_model, num_heads, mlp_hidden, key):
        self.ln_1 = eqx.nn.LayerNorm(d_model)
        attn_key, mlp_key = jax.random.split(key)
        self.attn = CausalSelfAttention(d_model, num_heads, attn_key)
        self.ln_2 = eqx.nn.LayerNorm(d_model)
        self.mlp = eqx.nn.MLP(d_model, d_model, mlp_hidden, depth=1, activation=jax.nn.gelu, key=mlp_key)

    def __call__(self, x, mask=None, pad_mask=None, kv_cache=None, return_cache=False, compressor: Optional[BaseKVCompressor] = None, return_indices: bool = False):
        if return_indices:
            attn_out, new_cache, comp_indices = self.attn(jax.vmap(self.ln_1)(x), mask=mask, pad_mask=pad_mask, kv_cache=kv_cache, return_cache=return_cache, compressor=compressor, return_indices=True)
        else:
            attn_out, new_cache = self.attn(jax.vmap(self.ln_1)(x), mask=mask, pad_mask=pad_mask, kv_cache=kv_cache, return_cache=return_cache, compressor=compressor)
        x = x + attn_out
        x = x + jax.vmap(self.mlp)(jax.vmap(self.ln_2)(x))
        if return_indices:
            return x, new_cache, comp_indices
        return x, new_cache

class DecoderSLM(eqx.Module):
    wte: eqx.nn.Embedding
    wpe: eqx.nn.Embedding
    blocks: list[TransformerBlock]
    ln_f: eqx.nn.LayerNorm
    lm_head: eqx.nn.Linear

    def __init__(self, vocab_size, max_seq_len, d_model, num_heads, num_layers, mlp_hidden, key):
        keys = jax.random.split(key, 3 + num_layers)
        
        self.wte = eqx.nn.Embedding(vocab_size, d_model, key=keys[0])
        self.wpe = eqx.nn.Embedding(max_seq_len, d_model, key=keys[1])
        
        self.blocks = [
            TransformerBlock(d_model, num_heads, mlp_hidden, keys[i + 2])
            for i in range(num_layers)
        ]
        
        self.ln_f = eqx.nn.LayerNorm(d_model)
        self.lm_head = eqx.nn.Linear(d_model, vocab_size, use_bias=False, key=keys[-1])

    def __call__(self, x, kv_caches=None, return_cache=False, compressor: Optional[BaseKVCompressor] = None, return_indices: bool = False):
        seq_len = x.shape[0]
        pad_mask = (x == 0)
        
        if kv_caches is None:
            causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
            mask = jnp.logical_and(causal_mask, jnp.logical_not(pad_mask)[None, :])
            positions = jnp.arange(seq_len)
            new_caches = [] if return_cache else None
        else:
            cache_index = kv_caches[0].cache_index
            mask = None
            positions = jnp.arange(seq_len) + cache_index
            # Safe clipping for WPE bounds
            max_pos = self.wpe.weight.shape[0] - 1
            positions = jnp.clip(positions, 0, max_pos)
            new_caches = []

        all_indices = [] if return_indices else None

        h = jax.vmap(self.wte)(x) + jax.vmap(self.wpe)(positions)
        
        for i, block in enumerate(self.blocks):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            if return_indices:
                h, c, comp_idx = block(h, mask=mask, pad_mask=pad_mask, kv_cache=layer_cache, return_cache=return_cache, compressor=compressor, return_indices=True)
                all_indices.append(comp_idx)
            else:
                h, c = block(h, mask=mask, pad_mask=pad_mask, kv_cache=layer_cache, return_cache=return_cache, compressor=compressor)
            if new_caches is not None:
                new_caches.append(c)
                
        h = jax.vmap(self.ln_f)(h)
        logits = jax.vmap(self.lm_head)(h) 
        if return_indices:
            return logits, new_caches, all_indices
        return logits, new_caches