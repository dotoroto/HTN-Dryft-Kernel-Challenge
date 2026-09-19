"""Shape-based choices for prefill capture and decode attention tiling."""


def capture_prefill_graph(
    batch: int, prompt_length: int, output_length: int, mode: str = "auto"
) -> bool:
    """Capture only when the prompt graph has a bounded activation footprint."""
    if mode == "1":
        return True
    if mode == "0":
        return False
    if mode != "auto":
        raise ValueError("DRYFT_PREFILL_GRAPH must be 0, 1, or auto")
    return (
        batch >= 8
        and prompt_length <= 1024
        and batch * prompt_length <= 8192
        and output_length >= 64
    )


def gqa_block_size(capacity: int, query_length: int) -> int:
    """Use fewer key-block loop iterations for long K=1 attention."""
    return 128 if query_length == 1 and capacity >= 1536 else 64
