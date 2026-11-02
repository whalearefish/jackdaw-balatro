"""Readable view of exactly what the Gym environment passes to the model.

``ReadableEnv`` wraps a :class:`BalatroGymnasiumEnv`.  ``reset``/``step``/
``action_masks`` and the spaces are passed straight through, unchanged, so it
can be used anywhere the original env is used (SB3, MaskablePPO).  After every
call it keeps the outputs, and on request decodes them into labelled values.

Only what the env returns is shown: the observation dict (``global``, the five
entity arrays, ``entity_counts``), the reward, ``terminated``/``truncated``,
``info`` and the action mask.  No game state is read.  Every element of every
returned array is mapped to a label (see ``Readable.coverage()``).

Information that is NOT passed is made visible instead of hidden:

* values the encoder left blank show ``NOT PASSED`` with the reason
  (face-down/blank card rows, catalog id 0 = no identity)
* entity counts at the padding cap are marked (extra items would be dropped)
* the meaning of each legal action number is listed, marked as not visible
  to the model (the model only receives the mask)
* ``env.missing()`` lists game information that has no field at all

Usage::

    from jackdaw.env import BalatroGymnasiumEnv, DirectAdapter
    from jackdaw.env.readable_env import ReadableEnv

    env = ReadableEnv(BalatroGymnasiumEnv(adapter_factory=DirectAdapter))
    obs, info = env.reset()                  # identical to the original env
    env.show()                               # everything, labelled
    env.show("global.dollars", "jokers.*")   # only matching fields
    env.query("hand.*.rank")                 # {path: value}
    env.get("global.hands_left")             # single value

    # Decode any observation dict, e.g. one sampled from a rollout buffer:
    from jackdaw.env.readable_env import decode_obs
    print(decode_obs(obs))
"""

from __future__ import annotations

import fnmatch
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

import gymnasium
import numpy as np

from jackdaw.env import obs_inspect as I
from jackdaw.env.action_space import ActionType
from jackdaw.env.game_spec import FactoredAction
from jackdaw.env.gymnasium_wrapper import _ENTITY_INFO

# obs-dict key -> (readable section name, feature schema)
_SECTIONS: dict[str, tuple[str, list[I.Feat]]] = {
    "hand_card": ("hand", I.PLAYING_CARD_SCHEMA),
    "joker": ("jokers", I.JOKER_SCHEMA),
    "consumable": ("consumables", I.CONSUMABLE_SCHEMA),
    "shop_item": ("shop", I.SHOP_SCHEMA),
    "pack_card": ("pack", I.PLAYING_CARD_SCHEMA),
}
_ENTITY_ORDER: list[str] = [name for name, _, _ in _ENTITY_INFO]
_MAX_COUNT: dict[str, int] = {name: mc for name, mc, _ in _ENTITY_INFO}
_FEAT_DIM: dict[str, int] = {name: d for name, _, d in _ENTITY_INFO}
assert set(_ENTITY_ORDER) == set(_SECTIONS), "entity layout changed; update _SECTIONS"

NOT_PASSED = "NOT PASSED"

# Structural facts about specific features (constant or duplicated values).
_FEATURE_NOTES: dict[str, str] = {
    "global.joker_hand_flags": "engine never sets these keys, so this is always empty",
    "hand.*.is_scoring": "equals 'not debuffed' (or splash flag)",
    "hand.*.unused": "unused slot, always 0",
    "pack.*.is_scoring": "equals 'not debuffed' (or splash flag)",
    "pack.*.unused": "unused slot, always 0",
    "jokers.*.condition_met": "equals 'not debuffed'",
    "jokers.*.rarity(cost proxy)": "base cost / 20, not the joker's real rarity",
    "jokers.*.extra(summed)": "sum of all numbers in the joker's 'extra' config",
}

# Game information with no field anywhere in the observation (hand-maintained;
# update when the encoder changes).
MISSING_FROM_OBS: list[tuple[str, str]] = [
    ("blind identity", "global.blind_key exists but is always id 0 (blind keys not in catalog)"),
    ("upcoming boss blind", "not encoded during blind select or before the boss"),
    ("skip tags offered", "tags for skipping Small/Big blind are not encoded"),
    ("pack type", "which kind of pack is open (Arcana, Celestial, ...) is not encoded"),
    ("pack picks remaining", "how many cards can still be taken from the pack"),
    ("non-playing pack cards", "Tarot/Planet/Spectral/Joker pack rows are blank (NOT PASSED)"),
    ("stake / deck back", "run settings are not encoded"),
    ("remaining deck contents", "only the deck size and draw odds are encoded"),
    ("discard duplicates", "discard_pile is yes/no per card, so duplicates are lost"),
    ("joker real rarity", "only a cost-based proxy is passed"),
    ("shop playing-card rank/suit", "shop rows only carry the catalog id"),
    ("action meanings", "the model gets a 500-slot mask, not what each slot does"),
    ("items beyond padding caps", "hand>8, jokers>5, consumables>2, shop>10, pack>5 are dropped"),
]


# ---------------------------------------------------------------------------
# Readable result
# ---------------------------------------------------------------------------


@dataclass
class Entry:
    path: str  # dotted name, e.g. "jokers.2.x_mult"
    value: Any  # readable value
    source: str  # where it came from, e.g. "joker[2,11]"
    encoded: Any = None  # the raw number(s) the model sees
    note: str = ""


@dataclass
class Readable:
    """Labelled view of one env output.  Query with glob patterns."""

    entries: list[Entry]
    labels: dict[str, str] = field(default_factory=dict)  # "hand.0" -> "AC"
    coverage_counts: dict[str, tuple[int, int]] = field(default_factory=dict)
    rows: dict[str, list[dict[str, Any]]] = field(default_factory=dict, repr=False)

    # --- querying -----------------------------------------------------------

    def filter(self, *patterns: str) -> Readable:
        pats = [p.lower() for p in _flatten(patterns)]
        if not pats:
            return self
        keep = [e for e in self.entries if any(fnmatch.fnmatch(e.path.lower(), p) for p in pats)]
        return Readable(keep, self.labels, self.coverage_counts, self.rows)

    def query(self, *patterns: str) -> dict[str, Any]:
        """``{path: value}`` for every entry matching any pattern (all if none)."""
        return {e.path: e.value for e in self.filter(*patterns).entries}

    def get(self, path: str) -> Any:
        for e in self.entries:
            if e.path == path:
                return e.value
        raise KeyError(path)

    def not_passed(self) -> Readable:
        """Entries whose value was left blank by the encoder."""
        keep = [e for e in self.entries if isinstance(e.value, str) and NOT_PASSED in e.value]
        return Readable(keep, self.labels, self.coverage_counts, self.rows)

    def paths(self) -> list[str]:
        return [e.path for e in self.entries]

    def coverage(self) -> dict[str, tuple[int, int]]:
        """``{array: (elements labelled, elements returned)}`` — should all be equal."""
        return dict(self.coverage_counts)

    # --- output -------------------------------------------------------------

    def to_records(self) -> list[dict[str, Any]]:
        return [asdict(e) for e in self.entries]

    def to_dict(self) -> dict[str, Any]:
        """Nested dict: ``{"global": {...}, "hand": [{...}, ...], ...}``."""
        out: dict[str, Any] = {}
        for e in self.entries:
            parts = e.path.split(".")
            sec = parts[0]
            if sec in _SECTION_NAMES and len(parts) > 2 and parts[1].isdigit():
                rows = out.setdefault(sec, {}).setdefault("rows", [])
                i = int(parts[1])
                while len(rows) <= i:
                    rows.append({})
                rows[i][".".join(parts[2:])] = e.value
            else:
                out.setdefault(sec, {})[".".join(parts[1:])] = e.value
        return out

    def to_text(self) -> str:
        if not self.entries:
            return "(no matching fields)"
        lines: list[str] = []
        section = entity = None
        for e in self.entries:
            parts = e.path.split(".")
            sec = parts[0]
            ent = ".".join(parts[:2]) if len(parts) > 2 and parts[1].isdigit() else None
            if sec != section:
                lines.append(f"\n== {sec} " + "=" * max(0, 80 - len(sec)))
                section, entity = sec, None
            if ent and ent != entity:
                lines.append(f"  -- {ent}: {self.labels.get(ent, '')}")
                entity = ent
            name = ".".join(parts[2:]) if ent else ".".join(parts[1:])
            indent = "      " if ent else "  "
            val = _short(e.value)
            src = e.source
            if isinstance(e.encoded, float):
                src += f"={e.encoded:.4g}"
            note = f"   # {e.note}" if e.note else ""
            lines.append(f"{indent}{name:<{34 - len(indent)}} {val:<34} {src}{note}")
        return "\n".join(lines).lstrip("\n")

    def __str__(self) -> str:
        return self.to_text()

    __repr__ = __str__


_SECTION_NAMES = {sec for sec, _ in _SECTIONS.values()}


def _flatten(patterns: Iterable[Any]) -> list[str]:
    out: list[str] = []
    for p in patterns:
        if isinstance(p, (list, tuple)):
            out += _flatten(p)
        elif p is not None:
            out.append(str(p))
    return out


def _short(x: Any, n: int = 70) -> str:
    s = str(x)
    return s if len(s) <= n else s[: n - 3] + "..."


def _note_for(path: str) -> str:
    for pat, note in _FEATURE_NOTES.items():
        if fnmatch.fnmatch(path, pat):
            return note
    return ""


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def _decode_feat(f: I.Feat, vec: np.ndarray) -> tuple[Any, Any]:
    if f.stop is None:
        v = float(vec[f.start])
        return v, f.decode(v)
    sl = np.asarray(vec[f.start : f.stop], dtype=float)
    return [round(float(x), 4) for x in sl], f.decode(sl)


def _src(arr: str, f: I.Feat, row: int | None = None) -> str:
    idx = f"{f.start}" if f.stop is None else f"{f.start}:{f.stop}"
    return f"{arr}[{idx}]" if row is None else f"{arr}[{row},{idx}]"


def _blank_marker(name: str, decoded: Any) -> Any:
    """Catalog id 0 means the encoder passed no identity."""
    if name in ("center_key", "blind_key") and str(decoded).startswith("<unknown id 0"):
        return f"{NOT_PASSED} (id 0: no identity)"
    return decoded


def _row_label(sec: str, row: dict[str, Any]) -> str:
    if sec in ("hand", "pack"):
        if row.get("face_down") is True:
            return "(face-down / blank row: no card info passed)"
        mods = [
            str(row[k])
            for k in ("enhancement", "edition", "seal")
            if row.get(k) not in (None, "none")
        ]
        if row.get("debuffed"):
            mods.append("DEBUFFED")
        txt = f"{row.get('rank')}{str(row.get('suit', '?'))[:1]}"
        return txt + (f" [{', '.join(mods)}]" if mods else "")
    ck = row.get("center_key", "?")
    extra = [str(row["card_set"])] if row.get("card_set") else []
    if row.get("edition") not in (None, "none"):
        extra.append(str(row["edition"]))
    return f"{ck}" + (f" ({', '.join(extra)})" if extra else "")


def _as_np(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):  # torch tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def decode_obs(
    obs: dict[str, Any],
    *,
    action_mask: Any = None,
    action_table: list[FactoredAction] | None = None,
    reward: float | None = None,
    terminated: bool | None = None,
    truncated: bool | None = None,
    info: dict[str, Any] | None = None,
    last_action: tuple[int, FactoredAction | None] | None = None,
    last_action_obs: Readable | None = None,
    env_index: int = 0,
) -> Readable:
    """Decode a Gym observation dict (plus optional step outputs) into a :class:`Readable`.

    ``obs`` may be batched (as from a VecEnv or a rollout buffer); ``env_index``
    picks the row.  Arrays can be numpy or torch.  ``last_action_obs`` is the
    decoded observation the action was chosen from (used to label its targets).
    """
    entries: list[Entry] = []
    labels: dict[str, str] = {}
    cov: dict[str, tuple[int, int]] = {}

    arrays = {k: _as_np(v) for k, v in obs.items()}
    if arrays["global"].ndim == 2:  # batched
        arrays = {k: v[env_index] for k, v in arrays.items()}

    # --- step outputs -------------------------------------------------------
    if last_action is not None:
        idx, fa = last_action
        ctx = (last_action_obs.labels, last_action_obs.rows) if last_action_obs else None
        entries.append(
            Entry(
                "step.action_taken",
                f"{idx}: {_describe(fa, ctx)}",
                "action argument",
                note="labels refer to the previous observation",
            )
        )
    if reward is not None:
        entries.append(Entry("step.reward", float(reward), "reward", float(reward)))
    if terminated is not None:
        entries.append(Entry("step.terminated", bool(terminated), "terminated"))
    if truncated is not None:
        entries.append(Entry("step.truncated", bool(truncated), "truncated"))

    # --- global -------------------------------------------------------------
    g = arrays["global"]
    for f in I.GLOBAL_SCHEMA:
        enc, dec = _decode_feat(f, g)
        path = f"global.{f.name}"
        entries.append(
            Entry(path, _blank_marker(f.name, dec), _src("global", f), enc, _note_for(path))
        )
    cov["global"] = (len(I._covered(I.GLOBAL_SCHEMA)), g.shape[0])

    # --- entity counts --------------------------------------------------------
    counts = arrays.get("entity_counts")
    n_by_key: dict[str, int] = {}
    for k, ekey in enumerate(_ENTITY_ORDER):
        sec = _SECTIONS[ekey][0]
        n = int(round(float(counts[k]))) if counts is not None else 0
        n_by_key[ekey] = n
        cap = _MAX_COUNT[ekey]
        note = f"at cap ({cap}): any extra {sec} would be dropped" if n >= cap else ""
        entries.append(Entry(f"counts.{sec}", n, f"entity_counts[{k}]", float(counts[k]), note))
    if counts is not None:
        cov["entity_counts"] = (len(_ENTITY_ORDER), counts.shape[0])

    # --- entity rows ----------------------------------------------------------
    row_values: dict[str, list[dict[str, Any]]] = {}
    for ekey in _ENTITY_ORDER:
        sec, schema = _SECTIONS[ekey]
        arr = arrays.get(ekey)
        if arr is None:
            continue
        n = n_by_key[ekey]
        rows: list[dict[str, Any]] = []
        for i in range(n):
            vec = arr[i]
            decoded_row = {f.name: _decode_feat(f, vec)[1] for f in schema}
            blank = schema is I.PLAYING_CARD_SCHEMA and decoded_row.get("face_down") is True
            shown: dict[str, Any] = {}
            for f in schema:
                enc, dec = _decode_feat(f, vec)
                path = f"{sec}.{i}.{f.name}"
                note = _note_for(path)
                if blank and f.name not in ("face_down", "position"):
                    dec = f"{NOT_PASSED} (row is face-down/blank)"
                else:
                    dec = _blank_marker(f.name, dec)
                shown[f.name] = dec
                entries.append(Entry(path, dec, _src(ekey, f, i), enc, note))
            labels[f"{sec}.{i}"] = _row_label(sec, decoded_row)
            rows.append(decoded_row)
        row_values[sec] = rows
        # padding rows must be all zero; report if not
        pad = arr[n:]
        if pad.size and np.any(pad != 0):
            bad = [int(r) + n for r in np.nonzero(np.any(pad != 0, axis=1))[0]]
            entries.append(
                Entry(
                    f"{sec}.padding",
                    f"NONZERO padding rows {bad}",
                    f"{ekey}[{n}:]",
                    note="unexpected: padding should be zeros",
                )
            )
        cov[ekey] = (arr.shape[0] * len(I._covered(schema)), arr.size)

    # --- unrecognised obs keys (shown raw so nothing is hidden) ---------------
    known = {"global", "entity_counts", *_ENTITY_ORDER}
    for k, v in arrays.items():
        if k not in known:
            entries.append(Entry(f"other.{k}", v.tolist(), k, note="unrecognised obs key"))
            cov[k] = (v.size, v.size)

    # --- action mask ------------------------------------------------------------
    if action_mask is not None:
        m = np.asarray(action_mask, dtype=bool)
        legal = np.flatnonzero(m).tolist()
        entries.append(
            Entry("actions.legal_count", len(legal), "action_mask", note=f"of {m.shape[0]} slots")
        )
        entries.append(Entry("actions.legal_slots", _ranges(legal), "action_mask"))
        cov["action_mask"] = (m.shape[0], m.shape[0])
        if action_table is not None:
            for i in legal:
                fa = action_table[i] if i < len(action_table) else None
                entries.append(
                    Entry(
                        f"actions.{i}",
                        _describe(fa, (labels, row_values)),
                        f"action_mask[{i}]",
                        note="meaning not passed to model",
                    )
                )

    # --- info -------------------------------------------------------------------
    if info:
        for k, v in info.items():
            if k == "action_mask":
                n_legal = int(np.asarray(v, dtype=bool).sum())
                entries.append(
                    Entry(
                        "info.action_mask",
                        f"{n_legal} legal (same as actions.*)",
                        "info['action_mask']",
                    )
                )
            else:
                val = v.tolist() if isinstance(v, np.ndarray) else v
                entries.append(Entry(f"info.{k}", val, f"info[{k!r}]"))

    return Readable(entries, labels, cov, row_values)


def _ranges(idx: list[int]) -> str:
    if not idx:
        return "none"
    out, start, prev = [], idx[0], idx[0]
    for i in idx[1:] + [None]:
        if i is not None and i == prev + 1:
            prev = i
            continue
        out.append(f"{start}" if start == prev else f"{start}-{prev}")
        if i is not None:
            start = prev = i
    return ", ".join(out)


def _describe(fa: FactoredAction | None, ctx: tuple | None) -> str:
    """Readable meaning of a factored action, labelled from the decoded obs."""
    if fa is None:
        return "?"
    name = ActionType(fa.action_type).name
    labels, rows = ctx if ctx else ({}, {})

    def lab(sec: str, i: int) -> str:
        return labels.get(f"{sec}.{i}", "?")

    parts = [name]
    t = fa.action_type
    e = fa.entity_target
    if e is not None:
        if t in (ActionType.SellJoker, ActionType.SwapJokersLeft, ActionType.SwapJokersRight):
            parts.append(f"jokers.{e} {lab('jokers', e)}")
        elif t in (ActionType.SellConsumable, ActionType.UseConsumable):
            parts.append(f"consumables.{e} {lab('consumables', e)}")
        elif t == ActionType.PickPackCard:
            parts.append(f"pack.{e} {lab('pack', e)}")
        elif t in (ActionType.SwapHandLeft, ActionType.SwapHandRight):
            parts.append(f"hand.{e} {lab('hand', e)}")
        elif t in (ActionType.BuyCard, ActionType.RedeemVoucher, ActionType.OpenBooster):
            parts.append(_shop_target(t, e, labels, rows.get("shop", [])))
        else:
            parts.append(f"target {e}")
    if fa.card_target:
        cards = ", ".join(lab("hand", i) for i in fa.card_target)
        parts.append(f"cards {list(fa.card_target)} [{cards}]")
    return "  ".join(parts)


def _shop_target(t: int, e: int, labels: dict[str, str], shop_rows: list[dict]) -> str:
    """BuyCard/RedeemVoucher/OpenBooster index into their own list; map to a shop row."""
    want = {ActionType.RedeemVoucher: "Voucher", ActionType.OpenBooster: "Booster"}.get(t)

    def in_group(r: dict) -> bool:
        if want:
            return r.get("card_set") == want
        return r.get("card_set") not in ("Voucher", "Booster")

    group = [i for i, r in enumerate(shop_rows) if in_group(r)]
    if e < len(group):
        i = group[e]
        return f"shop.{i} {labels.get(f'shop.{i}', '?')}"
    return f"shop target {e}"


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


class ReadableEnv(gymnasium.Wrapper):
    """Pass-through wrapper that records env outputs and decodes them on request.

    Behaviour of ``reset``/``step``/``action_masks`` is unchanged; decoding only
    runs when ``readable()``/``show()``/``query()``/``get()`` is called.
    """

    def __init__(self, env: gymnasium.Env) -> None:
        super().__init__(env)
        self._last: dict[str, Any] = {}

    # --- identical env API ----------------------------------------------------

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        self._record(obs, info=info)
        return obs, info

    def step(self, action: Any):
        table = self._action_table()
        idx = int(action)
        taken = table[idx] if idx < len(table) else None
        prev_obs = self._last.get("obs")
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._record(
            obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            last_action=(idx, taken),
            prev_obs=prev_obs,
        )
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return self.env.get_wrapper_attr("action_masks")()

    # --- readable API -----------------------------------------------------------

    def readable(self) -> Readable:
        """Decode the most recent reset/step output."""
        if not self._last:
            raise RuntimeError("call reset() first")
        L = self._last
        prev = decode_obs(L["prev_obs"]) if L.get("prev_obs") is not None else None
        return decode_obs(
            L["obs"],
            action_mask=L["mask"],
            action_table=L["table"],
            reward=L.get("reward"),
            terminated=L.get("terminated"),
            truncated=L.get("truncated"),
            info=L.get("info"),
            last_action=L.get("last_action"),
            last_action_obs=prev,
        )

    def show(self, *patterns: str) -> None:
        """Print the latest output; optional glob patterns select fields."""
        print(self.readable().filter(*patterns).to_text())

    def query(self, *patterns: str) -> dict[str, Any]:
        return self.readable().query(*patterns)

    def get(self, path: str) -> Any:
        return self.readable().get(path)

    def legal_actions(self) -> dict[int, str]:
        """``{action number: meaning}`` for the current legal actions."""
        r = self.readable()
        return {
            int(e.path.split(".")[1]): e.value
            for e in r.entries
            if e.path.startswith("actions.") and e.path.split(".")[1].isdigit()
        }

    @staticmethod
    def missing() -> None:
        """Print game information that has no field in the observation."""
        for what, why in MISSING_FROM_OBS:
            print(f"  {what:<28} {why}")

    # --- internals ----------------------------------------------------------------

    def _action_table(self) -> list[FactoredAction]:
        return list(getattr(self.env.unwrapped, "_action_table", []))

    def _record(self, obs: dict[str, Any], **kw: Any) -> None:
        self._last = {
            "obs": {k: np.array(v, copy=True) for k, v in obs.items()},
            "mask": np.array(self.action_masks(), copy=True),
            "table": self._action_table(),
            **kw,
        }
