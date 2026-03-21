import jax
import jax.numpy as jnp
import equinox as eqx

class BaseKVCompressor(eqx.Module):
    budget: int
    apply_on_decode: bool
    protect_sor: bool

    def compress_prefill(self, q, k, v, return_indices: bool = False):
        """
        Takes the full prompt sequence.
        Returns compacted k, v, and the resulting sequence length (to become the starting cache_index), and prompt_len.
        If return_indices is True, also returns final_indices.
        """
        raise NotImplementedError

    def compress_decode(self, cache_index, max_budget, k_new, past_k, prompt_len):
        """
        Returns (target_index, new_cache_index) indicating where the new token should be written
        and the updated length of the cache.
        """
        raise NotImplementedError

class RandomCompressor(BaseKVCompressor):
    budget: int
    apply_on_decode: bool
    protect_sor: bool
    base_key: jax.Array

    def __init__(self, budget: int, key: jax.Array, apply_on_decode: bool = False, protect_sor: bool = False):
        self.budget = budget
        self.apply_on_decode = apply_on_decode
        self.protect_sor = protect_sor
        self.base_key = key

    def compress_prefill(self, q, k, v, return_indices: bool = False):
        seq_len = k.shape[0]
        
        # Geometrically identify the true length of this specific prompt
        is_valid = jnp.any(k != 0, axis=(1, 2))
        actual_len = jnp.sum(is_valid)
        
        def do_compress():
            # Use data hashing for zero-communication intra-batch stochasticity 
            data_hash = jnp.sum(k).astype(jnp.int32)
            key = jax.random.fold_in(self.base_key, data_hash)
            
            # Generate random scores for every token
            noise = jax.random.uniform(key, shape=(seq_len,))
            
            # Padding trap: force padding noise to -inf so they are never selected
            is_pad = jnp.logical_not(is_valid)
            noise = jnp.where(is_pad, -jnp.inf, noise)
            
            # Always protect SOS by giving it the highest possible score (+inf)
            noise = noise.at[0].set(jnp.inf)
            
            # Conditionally protect SOR by giving it the highest possible score (+inf)
            sor_idx = actual_len - 1
            noise = jnp.where(self.protect_sor, noise.at[sor_idx].set(jnp.inf), noise)
            
            # Grab top 'budget' directly, natively handling all protections
            _, top_indices = jax.lax.top_k(noise, self.budget)
            final_indices = jnp.sort(top_indices)
            
            comp_k = k[final_indices]
            comp_v = v[final_indices]
            
            # Pad the compacted block back to max sequence length to satisfy JAX static shape constraints.
            k_padded = jnp.zeros_like(k).at[:self.budget].set(comp_k)
            v_padded = jnp.zeros_like(v).at[:self.budget].set(comp_v)
            
            if return_indices:
                return k_padded, v_padded, jnp.array(self.budget, dtype=jnp.int32), actual_len.astype(jnp.int32), final_indices
            return k_padded, v_padded, jnp.array(self.budget, dtype=jnp.int32), actual_len.astype(jnp.int32)

        def do_nothing():
            # If the prompt is shorter than the budget, skip compression and set pointer to actual_len
            if return_indices:
                return k, v, actual_len.astype(jnp.int32), actual_len.astype(jnp.int32), jnp.arange(self.budget, dtype=jnp.int32)
            return k, v, actual_len.astype(jnp.int32), actual_len.astype(jnp.int32)

        # Use the dynamic actual_len to safely branch, bypassing the Padding Avalanche
        return jax.lax.cond(
            actual_len > self.budget,
            do_compress,
            do_nothing
        )

    def compress_decode(self, cache_index, max_budget, k_new, past_k, prompt_len):
        def do_compress():
            data_hash = jnp.sum(k_new).astype(jnp.int32)
            key = jax.random.fold_in(self.base_key, data_hash + cache_index)
            
            noise = jax.random.uniform(key, shape=(self.budget,))
            noise = noise.at[0].set(-jnp.inf)
            current_sor_idx = jnp.minimum(prompt_len - 1, self.budget - 1)
            noise = jnp.where(self.protect_sor, noise.at[current_sor_idx].set(-jnp.inf), noise)
            
            target_index = jnp.argmax(noise)
            new_cache_index = cache_index
            return target_index, new_cache_index

        def do_nothing():
            return cache_index, cache_index + 1

        return jax.lax.cond(
            jnp.logical_and(self.apply_on_decode, cache_index >= self.budget),
            do_compress,
            do_nothing
        )


class KNormCompressor(BaseKVCompressor):
    budget: int
    apply_on_decode: bool
    protect_sor: bool

    def __init__(self, budget: int, key=None, apply_on_decode: bool = False, protect_sor: bool = False):
        self.budget = budget
        self.apply_on_decode = apply_on_decode
        self.protect_sor = protect_sor

    def compress_prefill(self, q, k, v, return_indices: bool = False):
        seq_len = k.shape[0]
        is_valid = jnp.any(k != 0, axis=(1, 2))
        actual_len = jnp.sum(is_valid)
        
        def do_compress():
            # Compute L2 norm of k along axis=-1, then average across heads
            norms = jnp.mean(jnp.linalg.norm(k, axis=-1), axis=1) # (seq_len,)
            
            # Identify padding
            is_pad = jnp.logical_not(is_valid)
            
            # Dynamically locate [SOR]
            sor_idx = actual_len - 1
            
            # Protection Trap: Force norms to jnp.inf for padding
            norms = jnp.where(is_pad, jnp.inf, norms)
            
            # Force [SOS] and [SOR] to -jnp.inf so they get picked up as the lowest norms
            norms = norms.at[0].set(-jnp.inf)
            norms = jnp.where(self.protect_sor, norms.at[sor_idx].set(-jnp.inf), norms)
            
            # Find the budget tokens with the lowest norms
            _, top_indices = jax.lax.top_k(-norms, self.budget)
            final_indices = jnp.sort(top_indices)
            
            comp_k = k[final_indices]
            comp_v = v[final_indices]
            
            # Pad the compacted tensors back to self.budget
            k_padded = jnp.zeros_like(k).at[:self.budget].set(comp_k)
            v_padded = jnp.zeros_like(v).at[:self.budget].set(comp_v)
            
            if return_indices:
                return k_padded, v_padded, jnp.array(self.budget, dtype=jnp.int32), actual_len.astype(jnp.int32), final_indices
            return k_padded, v_padded, jnp.array(self.budget, dtype=jnp.int32), actual_len.astype(jnp.int32)

        def do_nothing():
            if return_indices:
                return k, v, actual_len.astype(jnp.int32), actual_len.astype(jnp.int32), jnp.arange(self.budget, dtype=jnp.int32)
            return k, v, actual_len.astype(jnp.int32), actual_len.astype(jnp.int32)

        return jax.lax.cond(
            actual_len > self.budget,
            do_compress,
            do_nothing
        )

    def compress_decode(self, cache_index, max_budget, k_new, past_k, prompt_len):
        def do_compress():
            # Look only at valid cache
            current_k = past_k[:self.budget]
            
            # Compute L2 norm
            norms = jnp.mean(jnp.linalg.norm(current_k, axis=-1), axis=1) # (self.budget,)
            
            # Protect [SOS] and [SOR] tokens from eviction
            norms = norms.at[0].set(-jnp.inf)
            current_sor_idx = jnp.minimum(prompt_len - 1, self.budget - 1)
            norms = jnp.where(self.protect_sor, norms.at[current_sor_idx].set(-jnp.inf), norms)
            
            # Find the highest norm (weakest token to evict)
            target_index = jnp.argmax(norms)
            
            return target_index, cache_index  # pointer freezes

        def do_nothing():
            return cache_index, cache_index + 1

        return jax.lax.cond(
            jnp.logical_and(self.apply_on_decode, cache_index >= self.budget),
            do_compress,
            do_nothing
        )

class SnapKVCompressor(BaseKVCompressor):
    budget: int
    apply_on_decode: bool
    protect_sor: bool

    def __init__(self, budget: int, key=None, apply_on_decode: bool = False, protect_sor: bool = False):
        if apply_on_decode:
            raise ValueError("SnapKVCompressor only supports prefill compression. apply_on_decode must be False.")
        self.budget = budget
        self.apply_on_decode = False
        self.protect_sor = protect_sor

    def compress_prefill(self, q, k, v, return_indices: bool = False):
        seq_len = k.shape[0]
        is_valid = jnp.any(k != 0, axis=(1, 2))
        actual_len = jnp.sum(is_valid)
        
        def do_compress():
            # Identify padding
            is_pad = jnp.logical_not(is_valid)
            
            # Dynamically locate [SOR]
            sor_idx = actual_len - 1
            
            # Extract its Query (num_heads, head_dim)
            q_sor = q[sor_idx]
            
            # Calculate observation attention against all keys
            attn_scores = jnp.einsum("hd,thd->th", q_sor, k)
            
            # True SnapKV algorithm applies softmax here:
            # attn_scores = jax.nn.softmax(attn_scores, axis=0)
            
            # Average scores across heads to get (seq_len,)
            attn_scores = jnp.mean(attn_scores, axis=-1)
            
            # Protection Trap: Force scores to -jnp.inf for padding
            attn_scores = jnp.where(is_pad, -jnp.inf, attn_scores)
            
            # Force [SOS] and optionally [SOR] to jnp.inf so they receive the highest scores
            attn_scores = attn_scores.at[0].set(jnp.inf)
            attn_scores = jnp.where(self.protect_sor, attn_scores.at[sor_idx].set(jnp.inf), attn_scores)
            
            # Get indices of budget tokens with highest attention scores
            _, top_indices = jax.lax.top_k(attn_scores, self.budget)
            final_indices = jnp.sort(top_indices)
            
            comp_k = k[final_indices]
            comp_v = v[final_indices]
            
            # Pad the compacted tensors back to self.budget
            k_padded = jnp.zeros_like(k).at[:self.budget].set(comp_k)
            v_padded = jnp.zeros_like(v).at[:self.budget].set(comp_v)
            
            if return_indices:
                return k_padded, v_padded, jnp.array(self.budget, dtype=jnp.int32), actual_len.astype(jnp.int32), final_indices
            return k_padded, v_padded, jnp.array(self.budget, dtype=jnp.int32), actual_len.astype(jnp.int32)

        def do_nothing():
            if return_indices:
                return k, v, actual_len.astype(jnp.int32), actual_len.astype(jnp.int32), jnp.arange(self.budget, dtype=jnp.int32)
            return k, v, actual_len.astype(jnp.int32), actual_len.astype(jnp.int32)

        return jax.lax.cond(
            actual_len > self.budget,
            do_compress,
            do_nothing
        )

    # IMPORTANT: to make this actually work, the model needs to pass q_new to the compressor
    # consequently, we also need to update the compress_decode in the other compressors
    def compress_decode(self, cache_index, max_budget, k_new, past_k, prompt_len):
        def do_compress():
            # Only consider populated elements in past_k
            current_k = past_k[:self.budget]
            
            # We take the first (and only) token's query
            q_token = q_new[0]  # (num_heads, head_dim)
            
            # Calculate observation attention against past keys
            attn_scores = jnp.einsum("hd,thd->th", q_token, current_k)
            
            # True SnapKV algorithm applies softmax here:
            # attn_scores = jax.nn.softmax(attn_scores, axis=0)
            
            # Average scores across heads
            attn_scores = jnp.mean(attn_scores, axis=-1)  # (budget,)
            
            # Protect [SOS] and [SOR] tokens from eviction by giving them +inf
            attn_scores = attn_scores.at[0].set(jnp.inf)
            current_sor_idx = jnp.minimum(prompt_len - 1, self.budget - 1)
            attn_scores = jnp.where(self.protect_sor, attn_scores.at[current_sor_idx].set(jnp.inf), attn_scores)
            
            # Find the token with the lowest attention score to evict
            target_index = jnp.argmin(attn_scores)
            
            return target_index, cache_index
            
        def do_nothing():
            return cache_index, cache_index + 1

        return jax.lax.cond(
            jnp.logical_and(self.apply_on_decode, cache_index >= self.budget),
            do_compress,
            do_nothing
        )
