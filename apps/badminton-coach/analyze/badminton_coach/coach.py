"""Turn measurements into things a player can act on.

The target ranges here are coaching heuristics -- what badminton coaches say
about technique, written as numbers so they can be checked automatically. They
are not clinical norms, they are not tuned to any one player, and a good player
will break several of them on purpose. Treat a cue as "worth looking at the video
for", not as a verdict; every threshold is in :data:`RULES` and can be edited.

Each rule states a mechanism, because a cue without one is a scolding: "elbow at
118 degrees" means nothing, "the arm never straightened, so the shuttle was
struck below full reach" tells the player what to change.

Mirrors ``web/js/core/coach.js``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

NAN = float("nan")


def grade(value: float, target: dict) -> str:
    """Grade a value against a target band."""
    if value is None or not math.isfinite(value):
        return "unknown"
    lo = target.get("min", -math.inf)
    hi = target.get("max", math.inf)
    warn_lo = target.get("warn_min", lo)
    warn_hi = target.get("warn_max", hi)
    if lo <= value <= hi:
        return "good"
    if value < warn_lo or value > warn_hi:
        return "bad"
    return "warn"


def _is_overhead(s):
    return s["height"] == "overhead"


def _is_underarm(s):
    return s["height"] == "underarm"


def _is_drive(s):
    return s["height"] == "drive"


def _is_forehand(s):
    return s["side"] in ("forehand", "roundhead")


def _is_backhand(s):
    return s["side"] == "backhand"


@dataclass(frozen=True)
class Rule:
    id: str
    applies: object
    measure: object
    unit: str
    label: str
    why: str
    target: dict = field(default_factory=dict)

    def bands(self) -> dict:
        return {
            "min": self.target.get("min", -math.inf),
            "max": self.target.get("max", math.inf),
            "warn_min": self.target.get("warn_min", self.target.get("min", -math.inf)),
            "warn_max": self.target.get("warn_max", self.target.get("max", math.inf)),
        }

    def target_range(self) -> dict:
        """The numeric band, for a UI that phrases it in its own language."""
        b = self.bands()
        return {
            "min": b["min"] if math.isfinite(b["min"]) else None,
            "max": b["max"] if math.isfinite(b["max"]) else None,
        }

    def target_text(self) -> str:
        b = self.bands()
        has_min, has_max = math.isfinite(b["min"]), math.isfinite(b["max"])
        def n(v):
            return f"{v:.0f}" if abs(v) >= 10 else f"{v:.2f}"
        if has_min and has_max:
            return f"{n(b['min'])}–{n(b['max'])}{self.unit}"
        if has_min:
            return f"at least {n(b['min'])}{self.unit}"
        if has_max:
            return f"at most {n(b['max'])}{self.unit}"
        return ""


RULES = [
    Rule(
        id="overhead-elbow-extension",
        applies=lambda s: _is_overhead(s) and _is_forehand(s),
        measure=lambda s: s["contact"]["elbow"],
        unit="°",
        target={"min": 150, "warn_min": 135},
        label="Arm extension at contact",
        why="An overhead is struck at full reach. A bent arm lowers the contact point, "
            "which flattens the angle into the opponent's court and costs racket-head speed.",
    ),
    Rule(
        id="overhead-contact-height",
        applies=_is_overhead,
        measure=lambda s: s["contact"]["hand"]["height"],
        unit=" trunk lengths above the shoulders",
        target={"min": 0.30, "warn_min": 0.15},
        label="Contact height",
        why="Taking the shuttle high buys the steep downward angle a smash needs, and "
            "gives the opponent less time.",
    ),
    Rule(
        id="overhead-contact-in-front",
        applies=_is_overhead,
        measure=lambda s: s["contact"]["hand"]["forward"],
        unit=" trunk lengths in front of the chest",
        target={"min": 0.15, "warn_min": -0.05},
        label="Contact point in front",
        why="Letting the shuttle drift behind the head turns a smash into a defensive "
            "push, and loads the shoulder in its weakest position.",
    ),
    Rule(
        id="overhead-free-arm",
        applies=_is_overhead,
        measure=lambda s: s["backswing"]["max_off_arm_elevation"],
        unit="°",
        target={"min": 120, "warn_min": 90},
        label="Free arm raised in preparation",
        why="Pointing at the shuttle with the non-racket arm turns the shoulders into the "
            "shot and keeps the body balanced. It is the most visible difference between "
            "a coached and an uncoached overhead.",
    ),
    Rule(
        id="overhead-body-rotation",
        applies=_is_overhead,
        measure=lambda s: abs(s["backswing"]["max_separation"]),
        unit="°",
        target={"min": 20, "warn_min": 12},
        label="Shoulder turn against the hips",
        why="Power comes from unwinding the trunk. Without separation between the "
            "shoulders and the hips there is nothing stored to release, and the shot is "
            "played with the arm alone.",
    ),
    Rule(
        id="overhead-backswing-load",
        applies=_is_overhead,
        measure=lambda s: s["backswing"]["min_elbow"],
        unit="°",
        target={"max": 110, "warn_max": 135},
        label="Elbow bend in the backswing",
        why="The racket should drop behind the back before it accelerates. An arm that "
            "stays straight throughout has no throwing action to release.",
    ),
    Rule(
        id="backhand-elbow-extension",
        applies=lambda s: _is_backhand(s) and (_is_overhead(s) or _is_drive(s)),
        measure=lambda s: s["contact"]["elbow"],
        unit="°",
        target={"min": 140, "warn_min": 120},
        label="Arm extension at contact",
        why="A backhand clear is driven by the elbow straightening and the forearm "
            "rotating. Contact with a folded arm is the usual reason a backhand will not "
            "reach the back of the court.",
    ),
    Rule(
        id="backhand-contact-in-front",
        applies=_is_backhand,
        measure=lambda s: s["contact"]["hand"]["forward"],
        unit=" trunk lengths in front of the chest",
        target={"min": 0.20, "warn_min": 0.05},
        label="Contact point in front",
        why="The backhand has to be taken early. Once the shuttle is level with the body "
            "the arm can no longer extend into it.",
    ),
    Rule(
        id="net-lunge-knee",
        applies=_is_underarm,
        measure=lambda s: s["window"]["min_knee"],
        unit="°",
        target={"min": 95, "max": 145, "warn_min": 80, "warn_max": 160},
        label="Front knee bend in the lunge",
        why="Lunging with a bent knee lowers the body to the shuttle and lets the leg push "
            "back out. Reaching with a straight leg leaves the player stranded at the net.",
    ),
    Rule(
        id="net-trunk-upright",
        applies=_is_underarm,
        measure=lambda s: abs(s["contact"]["trunk_lean"]["forward"]),
        unit="°",
        target={"max": 45, "warn_max": 60},
        label="Chest position in the lunge",
        why="Falling forward over the front foot makes the recovery push impossible and "
            "puts the load on the knee rather than the hip.",
    ),
    Rule(
        id="drive-contact-in-front",
        applies=_is_drive,
        measure=lambda s: s["contact"]["hand"]["forward"],
        unit=" trunk lengths in front of the chest",
        target={"min": 0.10, "warn_min": -0.05},
        label="Contact point in front",
        why="Flat exchanges are won by taking the shuttle early. A late contact turns a "
            "drive into a lift.",
    ),
    Rule(
        id="drive-arm-extension",
        applies=_is_drive,
        measure=lambda s: s["contact"]["elbow"],
        unit="°",
        target={"min": 120, "max": 170, "warn_min": 100, "warn_max": 178},
        label="Arm extension at contact",
        why="A drive is a compact action: neither folded against the body nor locked out "
            "straight, which leaves no room to accelerate the racket.",
    ),
]


def coach_stroke(stroke: dict, min_confidence: float = 0.15, rules=None) -> list[dict]:
    """Apply every relevant rule to one stroke.

    Strokes whose type is a coin-flip are skipped: coaching the wrong shot type is
    worse than saying nothing.
    """
    if stroke["confidence"] < min_confidence:
        return []
    cues = []
    for rule in (rules or RULES):
        if not rule.applies(stroke):
            continue
        value = rule.measure(stroke)
        if value is None or not math.isfinite(value):
            continue
        cues.append({
            "id": rule.id,
            "stroke_index": stroke["index"],
            "shot": stroke["shot"],
            "label": rule.label,
            "why": rule.why,
            "value": value,
            "unit": rule.unit,
            "target": rule.target_text(),
            "target_range": rule.target_range(),
            "status": grade(value, rule.bands()),
        })
    return cues


SESSION_RULES = {
    "ready_knee": {"min": 145, "max": 168, "warn_min": 130, "warn_max": 175},
    "stance_width": {"min": 1.1, "max": 2.2, "warn_min": 0.9, "warn_max": 2.6},
    "recovery_seconds": {"max": 1.2, "warn_max": 1.8},
    "recovered_fraction": {"min": 0.7, "warn_min": 0.5},
}


def coach_session(frames, strokes, recovery=None, quiet: float = 0.4) -> list[dict]:
    """Session-level cues: posture between shots, and movement if the court is set up.

    "Between shots" means frames at least ``quiet`` seconds from any contact,
    which is where a player's default posture actually shows.
    """
    cues = []
    contact_times = [s["t"] for s in strokes]
    idle = [f for f in frames if all(abs(f["t"] - t) > quiet for t in contact_times)]

    if len(idle) >= 15:
        knees = [max(f["metrics"]["knee_left"], f["metrics"]["knee_right"]) for f in idle]
        knees = [k for k in knees if math.isfinite(k)]
        if knees:
            value = sum(knees) / len(knees)
            cues.append({
                "id": "ready-knee-bend",
                "label": "Knee bend between shots",
                "why": "A badminton ready position keeps the knees soft so the first step "
                       "can be pushed rather than fallen into. Standing tall between "
                       "rallies costs a fraction of a second on every shuttle.",
                "value": value, "unit": "°", "target": "145–168°",
                "target_range": {"min": 145, "max": 168},
                "status": grade(value, SESSION_RULES["ready_knee"]),
            })
        stances = [f["metrics"]["stance_width"] for f in idle]
        stances = [s for s in stances if math.isfinite(s)]
        if stances:
            value = sum(stances) / len(stances)
            cues.append({
                "id": "ready-stance-width",
                "label": "Stance width between shots",
                "why": "A base wider than the shoulders gives something to push against in "
                       "either direction; feet together means the first move is a stumble.",
                "value": value, "unit": "× shoulder width",
                "target": "1.1–2.2× shoulder width",
                "target_range": {"min": 1.1, "max": 2.2},
                "status": grade(value, SESSION_RULES["stance_width"]),
            })

    if recovery:
        times = [r["recovery_seconds"] for r in recovery if r["recovery_seconds"] is not None]
        if times:
            value = sorted(times)[len(times) // 2]
            cues.append({
                "id": "recovery-time",
                "label": "Median time back to base",
                "why": "Recovering to the middle is what makes the next shuttle reachable. "
                       "A slow return is felt two shots later, not on the shot itself.",
                "value": value, "unit": " s", "target": "under 1.2 s",
                "target_range": {"min": None, "max": 1.2},
                "status": grade(value, SESSION_RULES["recovery_seconds"]),
            })
        fraction = sum(1 for r in recovery if r["recovered"]) / len(recovery)
        cues.append({
            "id": "recovery-rate",
            "label": "Shots followed by a return to base",
            "why": "Staying where the last shot was played is the most common reason a "
                   "rally is lost from a winning position.",
            "value": fraction, "unit": " of shots", "target": "at least 0.70 of shots",
            "target_range": {"min": 0.7, "max": None},
            "status": grade(fraction, SESSION_RULES["recovered_fraction"]),
        })

    return cues


def rank_cues(cues) -> list[dict]:
    """Group cues by rule so the recurring faults come first."""
    groups: dict[str, dict] = {}
    for cue in cues:
        if cue["status"] not in ("bad", "warn"):
            continue
        g = groups.setdefault(cue["id"], {
            "id": cue["id"], "label": cue["label"], "why": cue["why"],
            "unit": cue["unit"], "target": cue["target"],
            "target_range": cue.get("target_range"),
            "count": 0, "bad": 0, "values": [],
        })
        g["count"] += 1
        if cue["status"] == "bad":
            g["bad"] += 1
        g["values"].append(cue["value"])
    out = []
    for g in groups.values():
        g["mean"] = sum(g["values"]) / len(g["values"])
        out.append(g)
    # Most-often-wrong first, and among equals the clearly wrong before the
    # borderline: the order a coach would raise them in.
    out.sort(key=lambda g: (g["bad"], g["count"]), reverse=True)
    return out
