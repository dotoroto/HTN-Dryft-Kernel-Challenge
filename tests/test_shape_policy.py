import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

from shape_policy import capture_prefill_graph, gqa_block_size  # noqa: E402


def test_auto_prefill_capture_has_bounded_shape():
    assert capture_prefill_graph(16, 512, 128)
    assert capture_prefill_graph(8, 1024, 64)
    assert not capture_prefill_graph(4, 2048, 32)
    assert not capture_prefill_graph(32, 512, 128)
    assert not capture_prefill_graph(16, 512, 32)


def test_prefill_capture_override():
    assert capture_prefill_graph(1, 512, 32, "1")
    assert not capture_prefill_graph(16, 512, 128, "0")


def test_attention_tile_changes_only_for_long_single_token_decode():
    assert gqa_block_size(2080, 1) == 128
    assert gqa_block_size(2080, 4) == 64
    assert gqa_block_size(640, 1) == 64
