import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

from speculation import (  # noqa: E402
    AdaptiveSpeculation,
    make_speculative_inputs,
    prompt_lookup,
    resolve_verification,
)


def test_prompt_lookup_prefers_longest_recent_suffix():
    assert prompt_lookup([1, 2, 3, 1, 2, 3], 2) == [1, 2]
    assert prompt_lookup([1, 2, 1], 1) == [2]
    assert prompt_lookup([1, 2, 3], 1) is None


def test_bucket_selection_requires_proposals_for_whole_batch():
    histories = [[1, 2, 3, 1]] * 7 + [[7, 8, 7]]
    pending = [1] * 7 + [7]
    bucket, rows = make_speculative_inputs(
        histories, pending, remaining=4
    )
    assert bucket == 2
    assert rows == [[1, 2]] * 7 + [[7, 8]]

    bucket, rows = make_speculative_inputs(
        [[1, 2, 3]] * 7 + [[7, 8, 9]], [3] * 7 + [9], remaining=4
    )
    assert bucket == 1
    assert rows == [[3]] * 7 + [[9]]


def test_small_batches_never_speculate():
    histories = [[1, 2, 3, 1]] * 7
    bucket, rows = make_speculative_inputs(histories, [1] * 7, remaining=4)
    assert bucket == 1
    assert rows == [[1]] * 7


def test_adaptive_policy_backs_off_and_reprobes():
    policy = AdaptiveSpeculation(16)
    history = [[1, 2, 3, 4, 1, 2, 3, 4, 1]] * 16
    pending = [1] * 16
    assert policy.inputs(history, pending, 4)[0] == 4
    policy.observe(4, 1)
    policy.observe(4, 1)
    assert policy.inputs(history, pending, 4)[0] == 2
    policy.observe(2, 1)
    policy.observe(2, 1)
    assert policy.inputs(history, pending, 4)[0] == 1
    for _ in range(8):
        policy.observe(1, 1)
    assert policy.inputs(history, pending, 4)[0] == 2
    for _ in range(4):
        policy.observe(2, 2)
    assert policy.inputs(history, pending, 4)[0] == 4
    assert policy.calls == {1: 8, 2: 6, 4: 2}


def test_full_acceptance_emits_proposals_and_bonus():
    rows = [[10, 11, 12, 13], [20, 21, 22, 23]]
    targets = [[11, 12, 13, 14], [21, 22, 23, 24]]
    matches = [[1, 1, 1], [1, 1, 1]]
    emissions, committed = resolve_verification(rows, targets, matches)
    assert emissions == [[11, 21], [12, 22], [13, 23], [14, 24]]
    assert committed == 4


def test_batch_rejection_emits_mixed_correction_and_rolls_back():
    rows = [[10, 11, 12, 13], [20, 21, 22, 23]]
    targets = [[11, 99, 88, 77], [21, 22, 23, 24]]
    matches = [[1, 0, 0], [1, 1, 1]]
    emissions, committed = resolve_verification(rows, targets, matches)
    assert emissions == [[11, 21], [99, 22]]
    assert committed == 2


def test_k1_is_ordinary_target_decode():
    emissions, committed = resolve_verification([[10], [20]], [[11], [21]], [[], []])
    assert emissions == [[11, 21]]
    assert committed == 1
