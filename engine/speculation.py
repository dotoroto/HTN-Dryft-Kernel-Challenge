"""CPU-side prompt lookup and speculative bucket selection."""

GRAPH_BUCKETS = (4, 2, 1)
MIN_SPECULATIVE_BATCH = 8


def prompt_lookup(history: list[int], count: int, max_ngram: int = 16):
    """Copy tokens following the newest earlier matching suffix."""
    if count <= 0:
        return []
    size = len(history)
    best_length = 0
    best = None
    # Search candidate suffix ends once rather than repeatedly slicing the
    # entire prompt for each possible n-gram length.
    for end in range(size - count - 1, -1, -1):
        if history[end] != history[-1]:
            continue
        matched = 1
        limit = min(max_ngram, end + 1, size)
        while (
            matched < limit
            and history[end - matched] == history[size - 1 - matched]
        ):
            matched += 1
        if matched > best_length:
            best_length = matched
            best = history[end + 1 : end + 1 + count]
            if matched == max_ngram:
                break
    return best


def make_speculative_inputs(
    histories: list[list[int]], pending: list[int], remaining: int,
    max_bucket: int = 4,
):
    """Return a K=1 decode below batch eight, otherwise the best bucket."""
    if len(histories) < MIN_SPECULATIVE_BATCH:
        return 1, [[token] for token in pending]
    for bucket in GRAPH_BUCKETS:
        if bucket > remaining or bucket > max_bucket:
            continue
        count = bucket - 1
        proposals = [prompt_lookup(history, count) for history in histories]
        if all(proposal is not None for proposal in proposals):
            rows = [[pending[i], *proposals[i]] for i in range(len(histories))]
            return bucket, rows
    raise RuntimeError("the K=1 fallback must always be available")


class AdaptiveSpeculation:
    """Back off when whole-batch verification rarely advances multiple tokens."""

    def __init__(self, batch: int):
        self.max_bucket = 4 if batch >= MIN_SPECULATIVE_BATCH else 1
        self.misses = 0
        self.k2_successes = 0
        self.cooldown = 0
        self.calls = {1: 0, 2: 0, 4: 0}
        self.emitted = {1: 0, 2: 0, 4: 0}

    def inputs(self, histories, pending, remaining):
        return make_speculative_inputs(
            histories, pending, remaining, max_bucket=self.max_bucket
        )

    def observe(self, bucket: int, output_steps: int) -> None:
        self.calls[bucket] += 1
        self.emitted[bucket] += output_steps
        if self.cooldown:
            self.cooldown -= 1
            if not self.cooldown:
                self.max_bucket = 2
            return
        if bucket == 4:
            self.misses = 0 if output_steps >= 3 else self.misses + 1
            if self.misses >= 2:
                self.max_bucket = 2
                self.misses = 0
                self.k2_successes = 0
        elif bucket == 2:
            if output_steps == 2:
                self.misses = 0
                self.k2_successes += 1
                if self.k2_successes >= 4:
                    self.max_bucket = 4
                    self.k2_successes = 0
            else:
                self.misses += 1
                self.k2_successes = 0
                if self.misses >= 2:
                    self.max_bucket = 1
                    self.cooldown = 8
                    self.misses = 0


def resolve_verification(
    rows: list[list[int]], targets: list[list[int]], matches: list[list[int]]
):
    """Resolve a batched verification while keeping every row in lockstep.

    Returns ``(emissions, committed_inputs)``.  Each item in ``emissions`` is
    one protocol step containing one token per batch row.  Cache slots after
    ``committed_inputs`` are speculative garbage and are logically rolled back.
    """
    batch = len(rows)
    query_length = len(rows[0])
    rejection = None
    for proposal_index in range(query_length - 1):
        if not all(row[proposal_index] for row in matches):
            rejection = proposal_index
            break

    if rejection is None:
        emissions = [
            [row[proposal_index + 1] for row in rows]
            for proposal_index in range(query_length - 1)
        ]
        emissions.append([row[-1] for row in targets])
        return emissions, query_length

    emissions = [
        [row[proposal_index + 1] for row in rows]
        for proposal_index in range(rejection)
    ]
    emissions.append(
        [
            rows[batch_index][rejection + 1]
            if matches[batch_index][rejection]
            else targets[batch_index][rejection]
            for batch_index in range(batch)
        ]
    )
    return emissions, rejection + 1
