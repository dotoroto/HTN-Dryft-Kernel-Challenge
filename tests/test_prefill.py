import sys
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

from prefill import PromptCache  # noqa: E402


class FakeTensor:
    def __init__(self, shape):
        self.shape = shape
        self.selection = None
        self.source = None

    def __getitem__(self, selection):
        self.selection = selection
        return self

    def copy_(self, source):
        self.source = source


def test_prefill_writes_only_prompt_prefix_to_static_cache():
    keys = FakeTensor((4, 8, 2080, 128))
    values = FakeTensor((4, 8, 2080, 128))
    static = SimpleNamespace(key_cache=[keys], value_cache=[values])
    source_keys = FakeTensor((4, 8, 2048, 128))
    source_values = FakeTensor((4, 8, 2048, 128))

    result_keys, result_values = PromptCache(static).update(
        source_keys, source_values, 0, {"cache_position": object()}
    )

    assert (result_keys, result_values) == (keys, values)
    expected_prefix = (slice(None), slice(None), slice(None, 2048), slice(None))
    assert keys.selection == values.selection == expected_prefix
    assert keys.source is source_keys
    assert values.source is source_values
