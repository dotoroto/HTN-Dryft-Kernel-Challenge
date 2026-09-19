"""Static-cache Qwen3 engine with exact prompt-lookup speculation."""

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, StaticCache

from kernels.acceptance import argmax_and_accept
from kernels.gqa import grouped_query_attention
from speculation import make_speculative_inputs, resolve_verification


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    half = value.shape[-1] // 2
    return torch.cat((-value[..., half:], value[..., :half]), dim=-1)


@dataclass
class GraphBucket:
    input_ids: torch.Tensor
    positions: torch.Tensor
    targets: torch.Tensor
    matches: torch.Tensor
    graph: torch.cuda.CUDAGraph


class Engine:
    def __init__(self, model_path: str) -> None:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                local_files_only=True,
            )
            .eval()
            .to("cuda:0")
        )
        self.device = torch.device("cuda:0")
        self.cache = None
        self.graphs = {}
        self.runtime_shape = None

    def _allocate_runtime(self, batch: int, capacity: int) -> None:
        shape = (batch, capacity)
        if self.runtime_shape == shape:
            return
        if self.runtime_shape is not None:
            raise RuntimeError(
                "warmup and measured calls must use the same batch and total length"
            )
        self.cache = StaticCache(
            config=self.model.config,
            max_batch_size=batch,
            max_cache_len=capacity,
            device=self.device,
            dtype=torch.bfloat16,
        )
        self.runtime_shape = shape

    @torch.inference_mode()
    def _target_forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Run a native-formula 1/2/4-token target forward."""
        base = self.model.model
        config = self.model.config
        batch, query_length = input_ids.shape
        hidden = base.embed_tokens(input_ids)
        cos, sin = base.rotary_emb(hidden, positions.unsqueeze(0))
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        head_dim = config.head_dim
        query_heads = config.num_attention_heads
        kv_heads = config.num_key_value_heads

        for layer_index, layer in enumerate(base.layers):
            residual = hidden
            normalized = layer.input_layernorm(hidden)
            attention = layer.self_attn
            query = attention.q_proj(normalized).view(
                batch, query_length, query_heads, head_dim
            ).transpose(1, 2)
            key = attention.k_proj(normalized).view(
                batch, query_length, kv_heads, head_dim
            ).transpose(1, 2)
            value = attention.v_proj(normalized).view(
                batch, query_length, kv_heads, head_dim
            ).transpose(1, 2)
            query = attention.q_norm(query)
            key = attention.k_norm(key)
            query = query * cos + _rotate_half(query) * sin
            key = key * cos + _rotate_half(key) * sin

            key_cache = self.cache.key_cache[layer_index]
            value_cache = self.cache.value_cache[layer_index]
            key_cache.index_copy_(2, positions, key)
            value_cache.index_copy_(2, positions, value)
            attended = grouped_query_attention(
                query, key_cache, value_cache, positions, attention.scaling
            )
            attended = attended.transpose(1, 2).reshape(
                batch, query_length, query_heads * head_dim
            )
            hidden = residual + attention.o_proj(attended)
            residual = hidden
            hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))

        hidden = base.norm(hidden)
        return self.model.lm_head(hidden)

    def _verify(self, input_ids: torch.Tensor, positions: torch.Tensor):
        return argmax_and_accept(self._target_forward(input_ids, positions), input_ids)

    def _capture_graphs(self, batch: int, prompt_length: int, capacity: int) -> None:
        if self.graphs:
            return
        for query_length in (1, 2, 4):
            if prompt_length + query_length > capacity:
                continue
            input_ids = torch.zeros(
                (batch, query_length), dtype=torch.int64, device=self.device
            )
            positions = torch.arange(
                prompt_length, prompt_length + query_length,
                dtype=torch.int64, device=self.device,
            )
            for _ in range(2):
                self._verify(input_ids, positions)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                targets, matches = self._verify(input_ids, positions)
            self.graphs[query_length] = GraphBucket(
                input_ids, positions, targets, matches, graph
            )

    def _run_graph(self, rows: list[list[int]], cache_length: int):
        query_length = len(rows[0])
        bucket = self.graphs[query_length]
        bucket.input_ids.copy_(
            torch.tensor(rows, dtype=torch.int64)
        )
        bucket.positions.copy_(
            torch.tensor(
                list(range(cache_length, cache_length + query_length)),
                dtype=torch.int64,
            )
        )
        bucket.graph.replay()
        targets = bucket.targets.cpu().tolist()
        matches = (
            bucket.matches[:, : query_length - 1].cpu().tolist()
            if query_length > 1
            else [[] for _ in rows]
        )
        return targets, matches

    @torch.inference_mode()
    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        """Yield exactly ``max_new_tokens`` greedy IDs for every sequence."""
        if max_new_tokens <= 0:
            return
        batch = len(input_ids)
        prompt_length = len(input_ids[0])
        capacity = prompt_length + max_new_tokens
        self._allocate_runtime(batch, capacity)

        prompt = torch.tensor(input_ids, dtype=torch.int64, device=self.device)
        prompt_positions = torch.arange(
            prompt_length, dtype=torch.int64, device=self.device
        )
        output = self.model(
            input_ids=prompt,
            past_key_values=self.cache,
            cache_position=prompt_positions,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
        first = output.logits[:, -1, :].argmax(dim=-1).cpu().tolist()
        self._capture_graphs(batch, prompt_length, capacity)

        histories = [list(row) for row in input_ids]
        for index, token in enumerate(first):
            histories[index].append(token)
        pending = first
        cache_length = prompt_length
        produced = 1
        yield first

        while produced < max_new_tokens:
            remaining = max_new_tokens - produced
            query_length, rows = make_speculative_inputs(
                histories, pending, remaining
            )
            targets, matches = self._run_graph(rows, cache_length)

            emissions, committed = resolve_verification(rows, targets, matches)
            for tokens in emissions:
                for batch_index, token in enumerate(tokens):
                    histories[batch_index].append(token)
                produced += 1
                yield tokens
            cache_length += committed
            pending = emissions[-1]
