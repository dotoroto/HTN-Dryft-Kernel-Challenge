"""CPU-side prompt lookup and speculative bucket selection."""

GRAPH_BUCKETS = (4, 2, 1)


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
    histories: list[list[int]], pending: list[int], remaining: int
):
    """Return ``(K, rows)`` for the largest usable 4/2/1 graph bucket."""
    for bucket in GRAPH_BUCKETS:
        if bucket > remaining:
            continue
        count = bucket - 1
        proposals = [prompt_lookup(history, count) for history in histories]
        if all(proposal is not None for proposal in proposals):
            rows = [[pending[i], *proposals[i]] for i in range(len(histories))]
            return bucket, rows
    raise RuntimeError("the K=1 fallback must always be available")


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
