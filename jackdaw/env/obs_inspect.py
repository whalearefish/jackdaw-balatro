"""Human-readable inspection of encoded observations.

Decodes the numeric arrays produced by :func:`jackdaw.env.observation.encode_observation`
back into labelled values and, when the raw engine ``game_state`` is given,
shows what the game state actually says next to each feature.  Rows where the
decoded value disagrees with the raw state are flagged, which makes encoder
bugs easy to spot.

Every value has a dotted path that can be filtered with glob patterns::

    global.dollars                 global.hand_levels.flush.level
    hand.0.rank                    jokers.*.x_mult
    pack.*                         context.*        (raw-only, not encoded)

Python usage::

    from jackdaw.env.obs_inspect import inspect_observation

    report = inspect_observation(obs, gs, fields=["global.dollars", "jokers.*"])
    print(report)                     # aligned text table
    report.mismatches()               # only rows that disagree with gs
    report.to_records()               # list of dicts (for asserts / pandas)

``obs`` may be an :class:`Observation`, a :class:`GameObservation`, or the
padded dict returned by :class:`BalatroGymnasiumEnv` (in which case the
``entity_counts`` are used, so truncation shows up as a count mismatch).

CLI usage (runs a game and prints the state)::

    python -m jackdaw.env.obs_inspect --seed TEST1 --until-phase shop --fields "global.*" "jokers.*"
    python -m jackdaw.env.obs_inspect --seed TEST1 --steps 40 --mismatches
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import random
import sys
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from jackdaw.env import observation as O
from jackdaw.env.game_spec import GameObservation

# ---------------------------------------------------------------------------
# Inverse lookup tables (built from the encoder's own tables)
# ---------------------------------------------------------------------------

_ID_TO_CENTER: dict[int, str] = {i: k for k, i in O._CENTER_KEY_TO_ID.items()}
_IDX_TO_SUIT = {i: s for s, i in O._SUIT_IDX.items()}
_RANK_LABEL = {
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "10",
    11: "J",
    12: "Q",
    13: "K",
    14: "A",
}
_ENHANCEMENT_NAMES = [
    "none",
    "m_bonus",
    "m_mult",
    "m_wild",
    "m_glass",
    "m_steel",
    "m_stone",
    "m_gold",
    "m_lucky",
]
_EDITION_NAMES = ["none", "foil", "holo", "polychrome", "negative"]
_SEAL_NAMES = ["none", "Gold", "Red", "Blue", "Purple"]
_PHASE_NAMES = [str(p) for p, _ in sorted(O._PHASE_IDX.items(), key=lambda kv: kv[1])]
_BOD_NAMES = ["None", "Small", "Big", "Boss"]
_CARD_SET_NAMES = {
    0: "Default/none",
    1: "Enhanced",
    2: "Joker",
    3: "Tarot",
    4: "Planet",
    5: "Spectral",
    6: "Voucher",
    7: "Booster",
    8: "Back",
    9: "Edition",
}
_RARITY_NAMES = {1: "Common", 2: "Uncommon", 3: "Rare", 4: "Legendary"}
_JOKER_FLAG_NAMES = ["four_fingers", "shortcut", "smeared", "splash"]

with open(O._CENTERS_PATH, encoding="utf-8") as _f:
    _CENTERS: dict[str, Any] = json.load(_f)


def _snake(name: str) -> str:
    return name.lower().replace(" ", "_")


# ---------------------------------------------------------------------------
# Decoders: encoded float(s) -> human value
# ---------------------------------------------------------------------------


def _inv_log(v: float) -> float:
    """Inverse of observation._log_scale."""
    return 2.0**v - 1.0 if v >= 0 else 1.0 - 2.0 ** (-v)


def d_raw(v: float) -> float:
    return round(float(v), 4)


def d_bool(v: float) -> bool:
    return bool(round(float(v)))


def d_scaled(k: float, as_int: bool = True) -> Callable[[float], Any]:
    def f(v: float) -> Any:
        x = float(v) * k
        return int(round(x)) if as_int else round(x, 4)

    return f


def d_log(as_int: bool = True) -> Callable[[float], Any]:
    def f(v: float) -> Any:
        x = _inv_log(float(v))
        return int(round(x)) if as_int else round(x, 3)

    return f


def d_cat(names: list[str] | dict[int, str], k: float) -> Callable[[float], Any]:
    def f(v: float) -> Any:
        idx = int(round(float(v) * k))
        if isinstance(names, dict):
            return names.get(idx, f"<bad idx {idx}>")
        return names[idx] if 0 <= idx < len(names) else f"<bad idx {idx}>"

    return f


def d_center(v: float) -> str:
    idx = int(round(float(v) * O.NUM_CENTER_KEYS))
    return _ID_TO_CENTER.get(idx, f"<unknown id {idx}>")


def d_rank(v: float) -> str:
    rid = int(round(float(v) * 14))
    return _RANK_LABEL.get(rid, f"<none id {rid}>")


def s_onehot(names: list[str]) -> Callable[[np.ndarray], Any]:
    def f(arr: np.ndarray) -> Any:
        hot = [i for i, x in enumerate(arr) if x > 0.5]
        if len(hot) == 1:
            return names[hot[0]]
        return f"<invalid one-hot {hot}>"

    return f


def s_multihot(names: list[str]) -> Callable[[np.ndarray], Any]:
    def f(arr: np.ndarray) -> Any:
        return [names[i] for i, x in enumerate(arr) if x > 0.5]

    return f


# ---------------------------------------------------------------------------
# Raw-state helpers (ground truth)
# ---------------------------------------------------------------------------


def _phase_str(gs: dict) -> str:
    p = gs.get("phase")
    return str(p.value if hasattr(p, "value") else p)


def _blind(gs: dict) -> Any:
    return gs.get("blind")


def card_label(card: Any) -> str:
    """Short readable label for any card object in the game state."""
    base = getattr(card, "base", None)
    ck = getattr(card, "center_key", "?")
    if base is not None:
        txt = f"{_RANK_LABEL.get(base.id, base.rank.value)}{base.suit.value[0]}"
        mods = []
        if ck not in ("c_base", None):
            mods.append(ck)
    else:
        aset = card.ability.get("set", "") if getattr(card, "ability", None) else ""
        txt = f"{ck} ({aset})" if aset else ck
        mods = []
    ed = _edition_name(getattr(card, "edition", None))
    if ed != "none":
        mods.append(ed)
    if getattr(card, "seal", None):
        mods.append(f"{card.seal} seal")
    for sticker in ("eternal", "perishable", "rental"):
        if getattr(card, sticker, False):
            mods.append(sticker)
    if getattr(card, "debuff", False):
        mods.append("DEBUFFED")
    if getattr(card, "facing", "front") == "back":
        mods.append("face-down")
    return txt + (f" [{', '.join(mods)}]" if mods else "")


def _edition_name(edition: dict | None) -> str:
    return _EDITION_NAMES[O._edition_idx(edition)]


def _enh_name(card: Any) -> str:
    ck = card.center_key
    return ck if ck in _ENHANCEMENT_NAMES else "none"


def _base_attr(card: Any, fn: Callable[[Any], Any]) -> Any:
    return None if getattr(card, "base", None) is None else fn(card.base)


# ---------------------------------------------------------------------------
# Feature schemas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Feat:
    """One feature (or a slice of features) of an encoded vector.

    ``truth(gs)`` / ``truth(card, idx, gs)`` returns the raw-state value.
    ``compare=False`` shows the truth for reference without flagging it.
    """

    name: str
    start: int
    stop: int | None  # None = single index
    decode: Callable[[Any], Any]
    truth: Callable[..., Any] | None = None
    compare: bool = True


def _g(name, i, dec, truth=None, compare=True) -> Feat:
    return Feat(name, i, None, dec, truth, compare)


def _gs_slice(name, a, b, dec, truth=None, compare=True) -> Feat:
    return Feat(name, a, b, dec, truth, compare)


def _build_global_schema() -> list[Feat]:
    cr = lambda gs: gs.get("current_round", {})  # noqa: E731
    rr = lambda gs: gs.get("round_resets", {})  # noqa: E731

    def blind_chips(gs):
        b = _blind(gs)
        return getattr(b, "chips", 0) if b else 0

    feats: list[Feat] = [
        _gs_slice("phase", 0, 6, s_onehot(_PHASE_NAMES), _phase_str),
        _gs_slice(
            "blind_on_deck", 6, 10, s_onehot(_BOD_NAMES), lambda gs: str(gs.get("blind_on_deck"))
        ),
        _g("ante", 10, d_scaled(8), lambda gs: rr(gs).get("ante", 1)),
        _g("round", 11, d_scaled(30), lambda gs: gs.get("round", 0)),
        _g("dollars", 12, d_log(), lambda gs: gs.get("dollars", 0)),
        _g("hands_left", 13, d_scaled(10), lambda gs: cr(gs).get("hands_left", 0)),
        _g("discards_left", 14, d_scaled(10), lambda gs: cr(gs).get("discards_left", 0)),
        _g("hand_size", 15, d_scaled(15), lambda gs: gs.get("hand_size", 8)),
        _g("joker_slots", 16, d_scaled(10), lambda gs: gs.get("joker_slots", 5)),
        _g("consumable_slots", 17, d_scaled(5), lambda gs: gs.get("consumable_slots", 2)),
        _g("blind_chips", 18, d_log(), blind_chips),
        _g("chips_scored", 19, d_log(), lambda gs: gs.get("chips", 0)),
        _g(
            "chips_over_blind",
            20,
            d_scaled(10, as_int=False),
            lambda gs: round(gs.get("chips", 0) / max(blind_chips(gs), 1), 4),
        ),
        _g("reroll_cost", 21, d_scaled(10), lambda gs: cr(gs).get("reroll_cost", 5)),
        _g("free_rerolls", 22, d_scaled(5), lambda gs: cr(gs).get("free_rerolls", 0)),
        _g("interest_cap", 23, d_scaled(100), lambda gs: gs.get("interest_cap", 25)),
        _g("discount_percent", 24, d_scaled(50), lambda gs: gs.get("discount_percent", 0)),
        _g("skips", 25, d_scaled(10), lambda gs: gs.get("skips", 0)),
        _g(
            "blind_key",
            26,
            d_center,
            lambda gs: getattr(_blind(gs), "key", None) if _blind(gs) else "<unknown id 0>",
        ),
        _g("deck_remaining", 27, d_log(), lambda gs: len(gs.get("deck", []))),
        _g("discard_pile_size", 28, d_log(), lambda gs: len(gs.get("discard_pile", []))),
        _g(
            "joker_hand_flags",
            29,
            lambda v: [
                n for b, n in enumerate(_JOKER_FLAG_NAMES) if (int(round(float(v) * 15)) >> b) & 1
            ],
            lambda gs: [n for n in _JOKER_FLAG_NAMES if gs.get(n)],
        ),
    ]

    # Hand levels [30:90]
    for j, ht in enumerate(O._HAND_TYPES):
        off = 30 + j * 5
        hn = _snake(ht.value)

        def hs(gs, ht=ht):
            hl = gs.get("hand_levels")
            return hl.get_state(ht) if hl is not None else None

        feats += [
            _g(
                f"hand_levels.{hn}.level",
                off,
                d_scaled(20),
                lambda gs, hs=hs: hs(gs).level if hs(gs) else None,
            ),
            _g(
                f"hand_levels.{hn}.chips",
                off + 1,
                d_log(),
                lambda gs, hs=hs: hs(gs).chips if hs(gs) else None,
            ),
            _g(
                f"hand_levels.{hn}.mult",
                off + 2,
                d_log(),
                lambda gs, hs=hs: hs(gs).mult if hs(gs) else None,
            ),
            _g(
                f"hand_levels.{hn}.played",
                off + 3,
                d_log(),
                lambda gs, hs=hs: hs(gs).played if hs(gs) else None,
            ),
            _g(
                f"hand_levels.{hn}.visible",
                off + 4,
                d_bool,
                lambda gs, hs=hs: bool(hs(gs).visible) if hs(gs) else None,
            ),
        ]

    vb = O._D_BASE
    bb = vb + O.NUM_VOUCHERS
    feats += [
        _gs_slice(
            "vouchers_owned",
            vb,
            bb,
            s_multihot(O._VOUCHER_KEYS),
            lambda gs: sorted(k for k, v in gs.get("used_vouchers", {}).items() if v),
        ),
        _g(
            "blind.is_boss",
            bb,
            d_bool,
            lambda gs: bool(getattr(_blind(gs), "boss", False)) if _blind(gs) else False,
        ),
        _g(
            "blind.disabled",
            bb + 1,
            d_bool,
            lambda gs: bool(getattr(_blind(gs), "disabled", False)) if _blind(gs) else False,
        ),
        _g(
            "blind.mult",
            bb + 2,
            d_scaled(4, as_int=False),
            lambda gs: getattr(_blind(gs), "mult", None) if _blind(gs) else 0.0,
        ),
        _gs_slice(
            "blind.debuff_suit",
            bb + 3,
            bb + 7,
            lambda a: (s_multihot(list(O._DEBUFF_SUIT_IDX))(a) or [None])[0],
            lambda gs: (
                (getattr(_blind(gs), "debuff_config", {}) or {}).get("suit") if _blind(gs) else None
            ),
        ),
        _g(
            "blind.debuff_face",
            bb + 7,
            d_bool,
            lambda gs: (
                (getattr(_blind(gs), "debuff_config", {}) or {}).get("is_face") == "face"
                if _blind(gs)
                else False
            ),
        ),
    ]
    rb = bb + 8
    feats += [
        _gs_slice(
            "round_position",
            rb,
            rb + 3,
            lambda a: (s_multihot(["Small", "Big", "Boss"])(a) or [None])[0],
            lambda gs: rr(gs).get("blind_states", {}),
            compare=False,
        ),
        _g(
            "round_progress.hands_played",
            rb + 3,
            d_scaled(10),
            lambda gs: cr(gs).get("hands_played", 0),
        ),
        _g(
            "round_progress.discards_used",
            rb + 4,
            d_scaled(10),
            lambda gs: cr(gs).get("discards_used", 0),
        ),
    ]
    tb = rb + 5
    db = tb + O.NUM_TAGS
    feats += [
        _gs_slice(
            "tags_held",
            tb,
            db,
            s_multihot(O._TAG_KEYS),
            lambda gs: [
                t.get("key") for t in gs.get("awarded_tags", []) or [] if isinstance(t, dict)
            ],
        ),
    ]
    disc_names = [
        f"{_RANK_LABEL[r + 2]}{_IDX_TO_SUIT[s][0]}"
        for s in range(O.NUM_SUITS)
        for r in range(O.NUM_RANKS)
    ]
    sb = db + O.D_DISCARD_HISTOGRAM
    feats += [
        _gs_slice(
            "discard_pile",
            db,
            sb,
            lambda a: sorted(s_multihot(disc_names)(a)),
            lambda gs: sorted(
                f"{_RANK_LABEL.get(c.base.id, '?')}{c.base.suit.value[0]}"
                for c in gs.get("discard_pile", [])
                if getattr(c, "base", None)
            ),
        ),
    ]
    hb = sb + O.NUM_HAND_TYPES
    feats += [
        _gs_slice("hand_types_present", sb, hb, s_multihot([ht.value for ht in O._HAND_TYPES])),
        _g("best_hand.base_chips", hb, d_log()),
        _g("best_hand.base_mult", hb + 1, d_log()),
        _g("best_hand.base_score", hb + 2, d_log()),
        _g("score_to_blind_log_ratio", hb + 3, d_scaled(2, as_int=False)),
        _g("flush_proximity", hb + 4, d_scaled(5)),
        _g("straight_proximity", hb + 5, d_scaled(5)),
        _g("interest_fraction", hb + 6, d_raw),
        _g("flush_draw_outs", hb + 7, d_raw),
        _g("straight_draw_outs", hb + 8, d_raw),
        _g("hand_debuffed_fraction", hb + 9, d_raw),
        _g("spendable_above_interest", hb + 10, d_log()),
        _g("urgency", hb + 11, d_scaled(5, as_int=False)),
    ]
    return feats


def _build_playing_card_schema() -> list[Feat]:
    return [
        _g("rank", 0, d_rank, lambda c, i, gs: _base_attr(c, lambda b: _RANK_LABEL.get(b.id))),
        _g(
            "suit",
            1,
            d_cat(_IDX_TO_SUIT, 3),
            lambda c, i, gs: _base_attr(c, lambda b: b.suit.value),
        ),
        _g(
            "chip_value",
            2,
            d_scaled(11),
            lambda c, i, gs: _base_attr(c, lambda b: O._RANK_CHIPS.get(b.rank.value)),
        ),
        _g("enhancement", 3, d_cat(_ENHANCEMENT_NAMES, 8), lambda c, i, gs: _enh_name(c)),
        _g("edition", 4, d_cat(_EDITION_NAMES, 4), lambda c, i, gs: _edition_name(c.edition)),
        _g("seal", 5, d_cat(_SEAL_NAMES, 4), lambda c, i, gs: c.seal or "none"),
        _g("debuffed", 6, d_bool, lambda c, i, gs: bool(c.debuff)),
        _g("face_down", 7, d_bool, lambda c, i, gs: c.facing == "back"),
        _g(
            "is_face_card",
            8,
            d_bool,
            lambda c, i, gs: _base_attr(c, lambda b: b.id in (11, 12, 13)) or False,
        ),
        _g("is_scoring", 9, d_bool),
        _g(
            "bonus_chips",
            10,
            d_log(),
            lambda c, i, gs: c.ability.get("bonus", 0) + c.ability.get("perma_bonus", 0),
        ),
        _g(
            "times_played",
            11,
            d_log(),
            lambda c, i, gs: _base_attr(c, lambda b: b.times_played) or 0,
        ),
        _g("position", 12, d_scaled(20), lambda c, i, gs: i),
        _g("in_best_hand", 13, d_bool),
        _g("unused", 14, d_raw),
    ]


def _joker_extra(c: Any) -> Any:
    return c.ability.get("extra")


def _build_joker_schema() -> list[Feat]:
    return [
        _g("center_key", 0, d_center, lambda c, i, gs: c.center_key),
        _g(
            "rarity(cost proxy)",
            1,
            lambda v: f"base_cost~{int(round(float(v) * 20))}",
            lambda c, i, gs: _RARITY_NAMES.get(_CENTERS.get(c.center_key, {}).get("rarity")),
        ),
        _g("edition", 2, d_cat(_EDITION_NAMES, 4), lambda c, i, gs: _edition_name(c.edition)),
        _g("sell_value", 3, d_log(), lambda c, i, gs: c.sell_cost),
        _g("eternal", 4, d_bool, lambda c, i, gs: bool(c.eternal)),
        _g("perishable", 5, d_bool, lambda c, i, gs: bool(c.perishable)),
        _g(
            "perish_tally",
            6,
            d_scaled(5),
            lambda c, i, gs: c.perish_tally if c.perishable else None,
        ),
        _g("rental", 7, d_bool, lambda c, i, gs: bool(c.rental)),
        _g("debuffed", 8, d_bool, lambda c, i, gs: bool(c.debuff)),
        _g("position", 9, d_scaled(20), lambda c, i, gs: i),
        _g(
            "mult",
            10,
            d_log(as_int=False),
            lambda c, i, gs: c.ability.get("mult", 0) + c.ability.get("t_mult", 0),
        ),
        _g("x_mult", 11, d_raw, lambda c, i, gs: c.ability.get("x_mult", 1.0)),
        _g(
            "chips",
            12,
            d_log(as_int=False),
            lambda c, i, gs: c.ability.get("t_chips", 0) + c.ability.get("bonus", 0),
        ),
        _g(
            "extra(summed)",
            13,
            d_log(as_int=False),
            lambda c, i, gs: _joker_extra(c),
            compare=False,
        ),
        _g("condition_met", 14, d_bool),
    ]


def _build_consumable_schema() -> list[Feat]:
    def cfg(c):
        x = c.ability.get("consumeable", {})
        return x if isinstance(x, dict) else {}

    return [
        _g("center_key", 0, d_center, lambda c, i, gs: c.center_key),
        _g("card_set", 1, d_cat(_CARD_SET_NAMES, 9), lambda c, i, gs: c.ability.get("set")),
        _g("sell_value", 2, d_log(), lambda c, i, gs: c.sell_cost),
        _g("can_use", 3, d_bool),
        _g("needs_targets", 4, d_bool, lambda c, i, gs: cfg(c).get("max_highlighted", 0) > 0),
        _g("max_targets", 5, d_scaled(5), lambda c, i, gs: cfg(c).get("max_highlighted", 0)),
        _g("min_targets", 6, d_scaled(5), lambda c, i, gs: cfg(c).get("min_highlighted", 0)),
    ]


def _build_shop_schema() -> list[Feat]:
    return [
        _g("center_key", 0, d_center, lambda c, i, gs: c.center_key),
        _g("card_set", 1, d_cat(_CARD_SET_NAMES, 9), lambda c, i, gs: c.ability.get("set")),
        _g("cost", 2, d_log(), lambda c, i, gs: c.cost),
        _g("affordable", 3, d_bool, lambda c, i, gs: c.cost <= gs.get("dollars", 0)),
        _g("has_slot", 4, d_bool),
        _g("edition", 5, d_cat(_EDITION_NAMES, 4), lambda c, i, gs: _edition_name(c.edition)),
        _g("eternal", 6, d_bool, lambda c, i, gs: bool(c.eternal)),
        _g("perishable", 7, d_bool, lambda c, i, gs: bool(c.perishable)),
        _g("rental", 8, d_bool, lambda c, i, gs: bool(c.rental)),
    ]


GLOBAL_SCHEMA = _build_global_schema()
PLAYING_CARD_SCHEMA = _build_playing_card_schema()
JOKER_SCHEMA = _build_joker_schema()
CONSUMABLE_SCHEMA = _build_consumable_schema()
SHOP_SCHEMA = _build_shop_schema()


# Sanity: the schemas must cover every encoded index exactly.
def _covered(schema: list[Feat]) -> set[int]:
    out: set[int] = set()
    for f in schema:
        out |= set(range(f.start, f.stop)) if f.stop is not None else {f.start}
    return out


assert _covered(GLOBAL_SCHEMA) == set(range(O.D_GLOBAL)), "global schema out of sync"
assert _covered(PLAYING_CARD_SCHEMA) == set(range(O.D_PLAYING_CARD))
assert _covered(JOKER_SCHEMA) == set(range(O.D_JOKER))
assert _covered(CONSUMABLE_SCHEMA) == set(range(O.D_CONSUMABLE))
assert _covered(SHOP_SCHEMA) == set(range(O.D_SHOP))

# section name -> (entity key in GameObservation, schema, gs list getter)
ENTITY_SECTIONS: dict[str, tuple[str, list[Feat], Callable[[dict], list]]] = {
    "hand": ("hand_card", PLAYING_CARD_SCHEMA, lambda gs: gs.get("hand", [])),
    "jokers": ("joker", JOKER_SCHEMA, lambda gs: gs.get("jokers", [])),
    "consumables": ("consumable", CONSUMABLE_SCHEMA, lambda gs: gs.get("consumables", [])),
    "shop": (
        "shop_item",
        SHOP_SCHEMA,
        lambda gs: (
            gs.get("shop_cards", []) + gs.get("shop_vouchers", []) + gs.get("shop_boosters", [])
        ),
    ),
    "pack": ("pack_card", PLAYING_CARD_SCHEMA, lambda gs: gs.get("pack_cards", [])),
}

# Raw-state values the observation does NOT encode (shown for context only).
CONTEXT_FIELDS: dict[str, Callable[[dict], Any]] = {
    "stake": lambda gs: gs.get("stake"),
    "back": lambda gs: gs.get("selected_back_key"),
    "blind_choices": lambda gs: gs.get("round_resets", {}).get("blind_choices"),
    "skip_tags": lambda gs: gs.get("round_resets", {}).get("blind_tags"),
    "blind_states": lambda gs: gs.get("round_resets", {}).get("blind_states"),
    "blind_debuff_config": lambda gs: (
        getattr(_blind(gs), "debuff_config", None) if _blind(gs) else None
    ),
    "pack_type": lambda gs: gs.get("pack_type"),
    "pack_choices_remaining": lambda gs: gs.get("pack_choices_remaining"),
    "deck_remaining_cards": lambda gs: sorted(card_label(c) for c in gs.get("deck", [])),
}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class Row:
    path: str
    encoded: Any
    decoded: Any
    raw: Any = None
    has_raw: bool = False
    mismatch: bool = False
    note: str = ""


def _values_differ(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) != bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) > 1e-3 * max(1.0, abs(float(b)))
    return a != b


class Report:
    """Collection of decoded rows with filtering and formatting helpers."""

    def __init__(self, rows: list[Row], labels: dict[str, str]) -> None:
        self.rows = rows
        self.labels = labels  # "hand.0" -> "QH [m_glass]"

    def __iter__(self):
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __str__(self) -> str:
        return self.to_text()

    def get(self, path: str) -> Any:
        """Decoded value at an exact path (KeyError if absent)."""
        for r in self.rows:
            if r.path == path:
                return r.decoded
        raise KeyError(path)

    def filter(self, patterns: Iterable[str]) -> Report:
        pats = [p.lower() for p in patterns]
        rows = [r for r in self.rows if any(fnmatch.fnmatch(r.path.lower(), p) for p in pats)]
        return Report(rows, self.labels)

    def mismatches(self) -> Report:
        return Report([r for r in self.rows if r.mismatch], self.labels)

    def to_records(self) -> list[dict[str, Any]]:
        return [asdict(r) for r in self.rows]

    def to_json(self, **kw: Any) -> str:
        return json.dumps(self.to_records(), default=str, **kw)

    def to_text(self, width: int = 36) -> str:
        if not self.rows:
            return "(no matching fields)"
        out: list[str] = []
        section = None
        entity = None
        for r in self.rows:
            parts = r.path.split(".")
            sec = parts[0]
            ent = ".".join(parts[:2]) if sec in ENTITY_SECTIONS and parts[1].isdigit() else None
            if sec != section:
                out.append(f"\n== {sec} " + "=" * max(0, 86 - len(sec)))
                out.append(f"  {'field':<{width}} {'encoded':>9}  {'decoded':<22} raw state")
                section, entity = sec, None
            if ent and ent != entity:
                out.append(f"  -- {ent}: {self.labels.get(ent, '')}")
                entity = ent
            name = ".".join(parts[2:]) if ent else ".".join(parts[1:])
            indent = "    " if ent else "  "
            enc = r.encoded if isinstance(r.encoded, str) else f"{r.encoded:.4f}"
            raw = _short(r.raw) if r.has_raw else ""
            flag = "  <-- MISMATCH" if r.mismatch else ""
            note = f"  ({r.note})" if r.note else ""
            out.append(
                f"{indent}{name:<{width - len(indent) + 2}} {enc:>9}  "
                f"{_short(r.decoded):<22} {raw}{flag}{note}"
            )
        return "\n".join(out).lstrip("\n")


def _short(x: Any, n: int = 60) -> str:
    s = str(x)
    return s if len(s) <= n else s[: n - 3] + "..."


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _as_parts(obs: Any) -> tuple[np.ndarray, dict[str, np.ndarray], str]:
    """Normalise Observation / GameObservation / gym dict to (global, entities, source)."""
    if isinstance(obs, O.Observation):
        g = obs.to_game_observation()
        return g.global_context, g.entities, "Observation"
    if isinstance(obs, GameObservation):
        return obs.global_context, obs.entities, "GameObservation"
    if isinstance(obs, dict) and "global" in obs:
        counts = obs.get("entity_counts")
        ents: dict[str, np.ndarray] = {}
        for k, (ekey, _, _) in enumerate(ENTITY_SECTIONS.values()):
            arr = obs.get(ekey)
            if arr is None:
                continue
            n = int(counts[k]) if counts is not None else arr.shape[0]
            ents[ekey] = arr[:n]
        return obs["global"], ents, "gym dict (padded)"
    raise TypeError(f"Unsupported observation type: {type(obs).__name__}")


def _decode(f: Feat, vec: np.ndarray) -> tuple[Any, Any]:
    if f.stop is None:
        v = float(vec[f.start])
        return v, f.decode(v)
    return f"[{f.start}:{f.stop}]", f.decode(vec[f.start : f.stop])


def _row(path: str, f: Feat, vec: np.ndarray, truth_args: tuple | None) -> Row:
    enc, dec = _decode(f, vec)
    row = Row(path=path, encoded=enc, decoded=dec)
    if truth_args is not None and f.truth is not None:
        try:
            raw = f.truth(*truth_args)
        except Exception as e:  # truth extraction should never kill the report
            raw = f"<error: {e}>"
        row.raw, row.has_raw = raw, True
        if f.compare and raw is not None:
            row.mismatch = _values_differ(dec, raw)
    return row


def inspect_observation(
    obs: Any,
    gs: dict[str, Any] | None = None,
    fields: Iterable[str] | None = None,
    only_mismatches: bool = False,
    include_context: bool = True,
) -> Report:
    """Decode an observation into labelled rows, optionally checked against ``gs``.

    Parameters
    ----------
    obs:
        ``Observation``, ``GameObservation`` or the Gymnasium dict observation.
    gs:
        Raw engine game_state the observation was encoded from.  When given,
        each row shows the raw value and is flagged if it disagrees.
    fields:
        Glob patterns over dotted paths, e.g. ``["global.dollars", "jokers.*.x_mult"]``.
    only_mismatches:
        Keep only rows that disagree with ``gs``.
    include_context:
        Add ``context.*`` rows: raw-state values the encoder does not include.
    """
    gvec, entities, _src = _as_parts(obs)
    rows: list[Row] = []
    labels: dict[str, str] = {}

    for f in GLOBAL_SCHEMA:
        rows.append(_row(f"global.{f.name}", f, gvec, (gs,) if gs is not None else None))

    for sec, (ekey, schema, getter) in ENTITY_SECTIONS.items():
        arr = entities.get(ekey)
        n_enc = 0 if arr is None else arr.shape[0]
        raw_list = getter(gs) if gs is not None else None
        count_row = Row(path=f"{sec}.count", encoded="-", decoded=n_enc)
        if raw_list is not None:
            count_row.raw, count_row.has_raw = len(raw_list), True
            count_row.mismatch = n_enc != len(raw_list)
        rows.append(count_row)
        for i in range(n_enc):
            card = raw_list[i] if raw_list is not None and i < len(raw_list) else None
            if card is not None:
                labels[f"{sec}.{i}"] = card_label(card)
            # A face-down card's attributes are deliberately hidden from the
            # agent, so differences there are expected, not encoder bugs.
            hidden = (
                schema is PLAYING_CARD_SCHEMA
                and card is not None
                and getattr(card, "facing", "front") == "back"
            )
            for f in schema:
                targs = (card, i, gs) if card is not None else None
                row = _row(f"{sec}.{i}.{f.name}", f, arr[i], targs)
                if hidden and f.name not in ("face_down", "position") and row.mismatch:
                    row.mismatch = False
                    row.note = "hidden: card is face-down"
                rows.append(row)

    if include_context and gs is not None:
        for name, fn in CONTEXT_FIELDS.items():
            try:
                val = fn(gs)
            except Exception as e:
                val = f"<error: {e}>"
            rows.append(
                Row(
                    path=f"context.{name}",
                    encoded="-",
                    decoded="(not encoded)",
                    raw=val,
                    has_raw=True,
                )
            )

    report = Report(rows, labels)
    if fields:
        report = report.filter(fields)
    if only_mismatches:
        report = report.mismatches()
    return report


def print_observation(obs: Any, gs: dict[str, Any] | None = None, **kw: Any) -> None:
    """Convenience: ``print(inspect_observation(...))``."""
    print(inspect_observation(obs, gs, **kw))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run_cli(argv: list[str] | None = None) -> int:
    from jackdaw.env.action_space import factored_to_engine_action, get_action_mask
    from jackdaw.env.agents import RandomAgent
    from jackdaw.env.game_interface import DirectAdapter

    p = argparse.ArgumentParser(
        prog="python -m jackdaw.env.obs_inspect",
        description="Play a game with a scripted agent and print decoded observations.",
    )
    p.add_argument("--seed", default="INSPECT")
    p.add_argument("--back", default="b_red")
    p.add_argument("--stake", type=int, default=1)
    p.add_argument("--agent", choices=["smart", "random"], default="smart")
    p.add_argument("--steps", type=int, default=0, help="stop after N actions (default 0)")
    p.add_argument(
        "--until-phase",
        default=None,
        help="stop at the first state in this phase, e.g. shop, pack_opening",
    )
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--every", action="store_true", help="print every state along the way")
    p.add_argument("--fields", nargs="*", default=None, help="glob patterns over field paths")
    p.add_argument("--mismatches", action="store_true", help="only rows that disagree with gs")
    p.add_argument("--no-context", action="store_true", help="omit context.* rows")
    p.add_argument("--json", action="store_true", help="print JSON records instead of text")
    p.add_argument("--rng-seed", type=int, default=0, help="python RNG seed for --agent random")
    args = p.parse_args(argv)

    random.seed(args.rng_seed)
    ad = DirectAdapter()
    ad.reset(args.back, args.stake, args.seed)
    rand_agent = RandomAgent()

    def show(step: int) -> None:
        gs = ad.raw_state
        rep = inspect_observation(
            O.encode_observation(gs),
            gs,
            fields=args.fields,
            only_mismatches=args.mismatches,
            include_context=not args.no_context,
        )
        if args.json:
            print(
                json.dumps(
                    {"step": step, "phase": _phase_str(gs), "rows": rep.to_records()}, default=str
                )
            )
        else:
            print(
                f"\n######## step {step} | phase {_phase_str(gs)} | "
                f"ante {gs.get('round_resets', {}).get('ante')} | seed {args.seed}"
            )
            print(rep.to_text())

    target_steps = args.steps if args.until_phase is None else args.max_steps
    for step in range(target_steps + 1):
        gs = ad.raw_state
        hit_phase = args.until_phase is not None and step > 0 and _phase_str(gs) == args.until_phase
        if args.every or hit_phase or ad.done or step == target_steps:
            show(step)
        if hit_phase or ad.done:
            if ad.done and not hit_phase and args.until_phase:
                print(f"\n(game ended before reaching phase {args.until_phase!r})")
            return 0
        if step == target_steps:
            if args.until_phase:
                print(f"\n(phase {args.until_phase!r} not reached within --max-steps)")
            return 0
        if args.agent == "smart":
            from jackdaw.cli.smart_agent import smart_agent

            action = smart_agent(gs, ad.get_legal_actions())
        else:
            mask = get_action_mask(gs)
            fa = rand_agent.act({}, mask, {"raw_state": gs})
            action = factored_to_engine_action(fa, gs)
        ad.step(action)
    return 0


if __name__ == "__main__":
    sys.exit(_run_cli())
