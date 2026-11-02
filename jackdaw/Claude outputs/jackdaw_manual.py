"""Manual-play helpers for jackdaw.  Run:  python -i jackdaw_manual.py"""
import copy
import sys

sys.path.insert(0, r"C:\haystackHold\balatroModel\jackdawFork")  # folder that contains jackdaw\

from jackdaw.engine import *  # noqa: F401,F403  (initialize_run, step, get_legal_actions, PlayHand, ...)

RANK = {"10": "10", "Jack": "J", "Queen": "Q", "King": "K", "Ace": "A"}
SKIP = (SwapHandLeft, SwapHandRight, SwapJokersLeft, SwapJokersRight)


def card_str(c):
    if c.base is None:
        return c.center_key
    s = RANK.get(c.base.rank, c.base.rank) + c.base.suit[0]
    extras = [x for x in (c.center_key if c.center_key != "c_base" else None,
                          c.seal and f"{c.seal} seal",
                          c.edition and next(iter(c.edition)),
                          c.debuff and "DEBUFFED",
                          c.facing == "back" and "face-down") if x]
    return s + (f"({', '.join(extras)})" if extras else "")


def legal(gs, swaps=False):
    """Numbered legal actions (swap actions hidden unless swaps=True)."""
    acts = [a for a in get_legal_actions(gs) if swaps or not isinstance(a, SKIP)]
    for i, a in enumerate(acts):
        hint = "   <- use play(gs, i, j, ...)" if isinstance(a, PlayHand) else (
               "   <- use discard(gs, i, j, ...)" if isinstance(a, Discard) else "")
        print(f"  [{i}] {a}{hint}")


def show(gs, swaps=False):
    """Print the state you'd see on screen, then the legal actions."""
    rr, cr, ph = gs["round_resets"], gs["current_round"], gs["phase"]
    print(f"\n== {ph.value} | ante {rr['ante']} round {gs['round']} | ${gs['dollars']}")
    if ph == GamePhase.BLIND_SELECT:
        print(f"   on deck: {gs['blind_on_deck']} | blinds {rr['blind_choices']} | skip tags {rr['blind_tags']}")
    if ph == GamePhase.SELECTING_HAND:
        b = gs["blind"]
        print(f"   {b.name}: {gs['chips']}/{b.chips} | hands {cr['hands_left']} | discards {cr['discards_left']} | deck {len(gs['deck'])}")
    if gs["hand"]:
        print("   hand:", "  ".join(f"{i}:{card_str(c)}" for i, c in enumerate(gs["hand"])))
    if gs["jokers"]:
        print("   jokers:", [c.center_key for c in gs["jokers"]])
    if gs["consumables"]:
        print("   consumables:", [c.center_key for c in gs["consumables"]])
    if ph == GamePhase.SHOP:
        print("   shop:", [(c.center_key, f"${c.cost}") for c in gs["shop_cards"]])
        print("   vouchers:", [(c.center_key, f"${c.cost}") for c in gs["shop_vouchers"]],
              "| boosters:", [(c.center_key, f"${c.cost}") for c in gs["shop_boosters"]],
              f"| reroll ${cr['reroll_cost']}")
    if ph == GamePhase.PACK_OPENING:
        print(f"   {gs['pack_type']} pack:", [c.center_key for c in gs["pack_cards"]],
              f"| picks left {gs['pack_choices_remaining']}")
    if ph == GamePhase.GAME_OVER:
        print("   WON" if gs["won"] else "   LOST")
        return
    print("   legal actions:")
    legal(gs, swaps)


def act(gs, action):
    """Apply an action (an Action object, or an index from the last show()), then show the result."""
    if isinstance(action, int):
        acts = [a for a in get_legal_actions(gs) if not isinstance(a, SKIP)]
        if not 0 <= action < len(acts):
            print(f"  no action [{action}]; pick 0-{len(acts) - 1}")
            return
        action = acts[action]
    try:
        step(gs, action)
    except IllegalActionError as e:
        print("  ILLEGAL:", e)
        return
    r = gs.get("last_score_result")
    if isinstance(action, PlayHand) and r is not None:
        print(f"  -> {r.hand_type}: {r.chips:g} x {r.mult:g} = {r.total}")
    show(gs)


def play(gs, *idx):    return act(gs, PlayHand(card_indices=tuple(idx)))
def discard(gs, *idx): return act(gs, Discard(card_indices=tuple(idx)))


def preview(gs, action):
    """Try an action on a copy; the real game is untouched."""
    g2 = copy.deepcopy(gs)
    act(g2, action)
    return g2


if __name__ == "__main__":
    gs = initialize_run("b_red", 1, "WALKTHRU")
    show(gs)
