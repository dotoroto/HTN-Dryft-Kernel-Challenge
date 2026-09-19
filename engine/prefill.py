"""Expose the populated prefix of a fixed KV cache to native prefill layers."""


class PromptCache:
    """Cache adapter for one full-prompt, empty-prefix causal forward.

    Native SDPA sees K/V of exactly the prompt length, while the writes land
    directly in the fixed-address cache later used by decode graphs. This is
    deliberately not a Transformers StaticCache: that path constructs a
    full-capacity mask during prefill.
    """

    def __init__(self, static_cache):
        self.static_cache = static_cache

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        length = key_states.shape[-2]
        keys = self.static_cache.key_cache[layer_idx][:, :, :length, :]
        values = self.static_cache.value_cache[layer_idx][:, :, :length, :]
        keys.copy_(key_states)
        values.copy_(value_states)
        return keys, values
