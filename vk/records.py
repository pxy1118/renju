"""One explicit position schema shared by data production, storage and training.

A self-play position is not a tuple. It is a fixed set of named fields with a
declared dtype, so the producer, the shard writer, the replay buffer and the
loss cannot disagree about what they are looking at. VALUE_HEADS is frozen
here: the head order in the schema, in the network and in the objective are the
same tuple.

Field semantics (version 4):

    state            board planes, exactly what the network consumes
    policy           search target after noise correction and visit pruning;
                     all zeros when policy_valid is 0
    policy_valid     1 when this row may supervise the policy head
    policy_weight    per-row multiplier for the policy term (0 for cheap rows)
    value            one target per value head, NaN where that head is invalid
    value_valid      bitmask over VALUE_HEADS for the same question
    search_value     root Q of the search that produced this row (side to move)
    q_spread         max minus min Q over visited root children
    policy_surprise  KL(pruned target || unnoised prior) in nats
    value_surprise   |root Q - network value at the root|
    weight           surprise sampling weight, computed at generation time
    simulations      budget this move was actually given
    full_search      1 for the full budget (the policy target is trusted)
    game_id / ply    game index, stones on the board before the move
    winner           absolute result: +1 black, -1 white, 0 draw, 127 unknown
    source           0 self-play, 1 teacher(v4), 2 teacher normalised from v2/v3

Teacher rows additionally carry the engine analysis they came from; the fields
are sentinel-filled on self-play rows so a merged dataset needs no side channel.
"""
import numpy as np

SCHEMA_VERSION = 4
VALUE_HEADS = ("final", "mid", "short")
VALUE_SLOTS = len(VALUE_HEADS)

SOURCE_SELFPLAY = 0
SOURCE_TEACHER = 1
SOURCE_TEACHER_LEGACY = 2
SOURCE_NAMES = {SOURCE_SELFPLAY: "selfplay", SOURCE_TEACHER: "teacher",
                SOURCE_TEACHER_LEGACY: "teacher-legacy"}

ACTION_NONE = 255        # never a board action (0-224)
WINNER_UNKNOWN = 127     # teacher rows inherited from a format without results
TOPK_SLOTS = 5

BASE_FIELDS = ("state", "policy", "policy_valid", "policy_weight", "value", "value_valid",
               "search_value", "q_spread", "policy_surprise", "value_surprise", "weight",
               "simulations", "full_search", "game_id", "ply", "winner", "source")
TEACHER_FIELDS = ("teacher_best", "teacher_nodes", "teacher_topk_actions",
                  "teacher_topk_winrates")
FIELDS = BASE_FIELDS + TEACHER_FIELDS

DTYPE = np.dtype([
    ("state", np.uint8, (3, 15, 15)),
    ("policy", np.float16, (225,)),
    ("policy_valid", np.uint8),
    ("policy_weight", np.float16),
    ("value", np.float16, (VALUE_SLOTS,)),
    ("value_valid", np.uint8),
    ("search_value", np.float16),
    ("q_spread", np.float16),
    ("policy_surprise", np.float16),
    ("value_surprise", np.float16),
    ("weight", np.float32),
    ("simulations", np.uint32),
    ("full_search", np.uint8),
    ("game_id", np.uint32),
    ("ply", np.uint16),
    ("winner", np.int8),
    ("source", np.uint8),
    ("teacher_best", np.uint16),
    ("teacher_nodes", np.uint64),
    ("teacher_topk_actions", np.uint8, (TOPK_SLOTS,)),
    ("teacher_topk_winrates", np.float16, (TOPK_SLOTS,)),
])

# Shard field sets written before the schema above. They exist so a reader can
# recognise old files, not so the rest of the code can keep speaking old terms.
LEGACY_FIELDS = {
    2: ("state", "policy", "value", "game_id", "ply", "teacher_best", "teacher_nodes"),
    3: ("state", "policy", "value", "game_id", "ply", "teacher_best", "teacher_nodes",
        "teacher_topk_actions", "teacher_topk_winrates"),
}


def head_index(head):
    try:
        return VALUE_HEADS.index(head)
    except ValueError:
        raise ValueError(f"Unknown value head: {head!r} (known: {list(VALUE_HEADS)})") from None


def head_bit(head):
    return np.uint8(1 << head_index(head))


def head_mask(array, head):
    return (array["value_valid"] & head_bit(head)) != 0


def head_targets(array, head):
    """Targets for one head, with NaN wherever that head is invalid."""
    column = array["value"][:, head_index(head)].astype(np.float32)
    return np.where(head_mask(array, head), column, np.nan)


def set_head(array, head, targets, valid=None):
    """Write one head in place; valid defaults to 'not NaN'."""
    targets = np.asarray(targets, np.float32)
    index = head_index(head)
    array["value"][:, index] = targets
    keep = np.isfinite(targets) if valid is None else np.asarray(valid, bool)
    bit = head_bit(head)
    array["value_valid"] = np.where(keep, array["value_valid"] | bit,
                                    array["value_valid"] & np.uint8(~bit & 0xFF))
    return array


def value_valid_mask(array):
    """Boolean (rows, heads) view of the value_valid bitmask."""
    shifts = np.arange(VALUE_SLOTS, dtype=np.uint8)
    return ((np.asarray(array["value_valid"])[:, None] >> shifts) & 1) != 0


def blank(rows):
    """Rows records with every optional field explicitly unset."""
    out = np.zeros(int(rows), DTYPE)
    out["policy_valid"] = 0
    out["policy_weight"] = 0.0
    out["value"][:] = np.nan
    out["search_value"] = np.nan
    out["q_spread"] = np.nan
    out["policy_surprise"] = np.nan
    out["value_surprise"] = np.nan
    out["weight"] = 1.0
    out["winner"] = WINNER_UNKNOWN
    out["source"] = SOURCE_SELFPLAY
    out["teacher_topk_actions"] = ACTION_NONE
    out["teacher_topk_winrates"] = np.nan
    return out


def stack(rows):
    """Build one structured array from per-row mappings (teacher, tools, tests)."""
    out = blank(len(rows))
    for index, row in enumerate(rows):
        for name in FIELDS:
            if name in row:
                out[name][index] = row[name]
    return out


def concatenate(arrays):
    arrays = [item for item in arrays if item is not None and len(item)]
    if not arrays:
        return blank(0)
    return np.concatenate(arrays) if len(arrays) > 1 else arrays[0]


def validate(array):
    """Fail loudly on a malformed batch: the schema is not advisory."""
    if array.dtype.names != FIELDS:
        raise ValueError(f"Position records must use the v{SCHEMA_VERSION} schema; "
                         f"got fields {array.dtype.names}")
    valid = array["policy_valid"] != 0
    if valid.any():
        totals = array["policy"][valid].astype(np.float64).sum(axis=1)
        if not np.allclose(totals, 1.0, atol=1e-3):
            raise ValueError("Rows with policy_valid=1 must carry a normalised target")
    if (~valid).any() and np.abs(array["policy"][~valid].astype(np.float64)).sum() > 0:
        raise ValueError("Rows with policy_valid=0 must carry a zero policy target")
    finite = np.isfinite(array["value"])
    counted = ((array["value_valid"][:, None] >> np.arange(VALUE_SLOTS)) & 1) != 0
    if not np.array_equal(finite, counted):
        raise ValueError("value and value_valid disagree about which heads are set")
    if len(array) and np.isnan(array["weight"]).any():
        raise ValueError("Sampling weights must be finite")
    return array


def augment(state, policy, rotation, mirror):
    """One of the eight D4 transforms, applied to board and policy together."""
    state = np.rot90(state, rotation, axes=(-2, -1))
    policy = np.rot90(policy.reshape(15, 15), rotation)
    if mirror:
        state, policy = state[..., ::-1], policy[:, ::-1]
    return state.copy(), policy.reshape(225).copy()


def augment_batch(states, policies, rng):
    """Augment a batch of states/policies with independent random symmetries."""
    states = np.asarray(states, np.uint8)
    policies = np.asarray(policies, np.float32)
    out_states = np.empty_like(states)
    out_policies = np.empty_like(policies)
    for index in range(len(states)):
        rotation, mirror = int(rng.integers(4)), bool(rng.integers(2))
        out_states[index], out_policies[index] = augment(states[index], policies[index],
                                                         rotation, mirror)
    return out_states, out_policies


def normalize_legacy(arrays, version):
    """Bring a v2/v3 teacher shard onto the current schema.

    The old single value column was the engine winrate estimate of the
    position outcome. It becomes both the final-head target and the stored
    search value; the horizon heads stay invalid because those files never
    recorded a horizon. The source field marks the rows so reports can say so.
    """
    version = int(version)
    if version not in LEGACY_FIELDS:
        raise ValueError(f"Cannot normalise shard version {version!r} "
                         f"(known: {sorted(LEGACY_FIELDS)})")
    names = set(arrays)
    missing = set(LEGACY_FIELDS[version]) - names
    if missing:
        raise ValueError(f"Version-{version} shard is missing fields {sorted(missing)}")
    rows = len(arrays["state"])
    out = blank(rows)
    out["state"] = arrays["state"]
    out["policy"] = arrays["policy"]
    out["policy_valid"] = 1
    out["policy_weight"] = 1.0
    value = np.asarray(arrays["value"], np.float32)
    set_head(out, "final", value)
    out["search_value"] = value
    out["weight"] = 1.0
    out["simulations"] = np.asarray(arrays["teacher_nodes"], np.uint32)
    out["full_search"] = 1
    out["game_id"] = arrays["game_id"]
    out["ply"] = arrays["ply"]
    out["source"] = SOURCE_TEACHER_LEGACY
    out["teacher_best"] = arrays["teacher_best"]
    out["teacher_nodes"] = arrays["teacher_nodes"]
    if version >= 3:
        out["teacher_topk_actions"] = arrays["teacher_topk_actions"]
        out["teacher_topk_winrates"] = arrays["teacher_topk_winrates"]
    return out


def sentinel_topk(rows):
    """Top-k arrays for rows that carry no engine analysis."""
    return (np.full((rows, TOPK_SLOTS), ACTION_NONE, np.uint8),
            np.full((rows, TOPK_SLOTS), np.nan, np.float16))


def topk_valid(actions):
    """Element-wise mask: this top-k slot holds a real analysed move."""
    return np.asarray(actions) != ACTION_NONE


def topk_row_valid(actions):
    """Per row: the first slot holds a real analysed move."""
    return np.asarray(actions)[:, 0] != ACTION_NONE
