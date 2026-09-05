"""Turn a frame of pose landmarks into badminton-relevant joint angles.

Mirrors ``web/js/core/biomech.js``, which carries the full rationale. In brief:

* Angles come from MediaPipe's *world* landmarks -- metric, hip-centred, and
  independent of where in the frame the player is, which matters when the player
  runs from the net to the back of the court.
* :func:`to_up_frame` flips MediaPipe's y-down, z-away convention into a
  right-handed frame with **y up** and **z towards the camera**.
* Technique angles are expressed in a frame attached to the torso, so they do not
  change meaning when the player turns to face a different way.
* One caveat: MediaPipe's world frame is aligned to the camera, not to gravity.
  Angles between body parts are unaffected; trunk lean, which is quoted against
  vertical, assumes the phone is held roughly upright.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import landmarks as L
from .geometry import (
    angle_between, cross, distance, dot, joint_angle, length, midpoint,
    normalize, reject, signed_angle, sub,
)

NAN = float("nan")


def to_up_frame(p):
    """One MediaPipe world landmark, in the y-up, z-towards-camera frame."""
    return (p[0], -p[1], -p[2])


def world_to_up_frame(points):
    return [to_up_frame(p) for p in points]


@dataclass(frozen=True)
class BodyFrame:
    """An orthonormal basis attached to the player.

    ``right`` points out of the player's right shoulder, ``up`` runs mid-hip to
    mid-shoulder, ``forward`` comes out of the chest. ``up`` is taken as primary
    and ``right`` orthogonalised against it, because the shoulder line is the
    noisier of the two while an arm is swinging.
    """

    origin: tuple
    mid_shoulder: tuple
    mid_hip: tuple
    right: tuple
    up: tuple
    forward: tuple


def body_frame(world) -> BodyFrame:
    mid_shoulder = midpoint(world[L.LEFT_SHOULDER], world[L.RIGHT_SHOULDER])
    mid_hip = midpoint(world[L.LEFT_HIP], world[L.RIGHT_HIP])
    up = normalize(sub(mid_shoulder, mid_hip))
    shoulder_axis = sub(world[L.RIGHT_SHOULDER], world[L.LEFT_SHOULDER])
    right = normalize(reject(shoulder_axis, up))
    # cross(up, right) points out of the chest in a right-handed y-up frame.
    forward = normalize(cross(up, right))
    return BodyFrame(mid_shoulder, mid_shoulder, mid_hip, right, up, forward)


def to_body(frame: BodyFrame, point):
    """Express a world point in the body frame, in metres."""
    d = sub(point, frame.origin)
    return (dot(d, frame.right), dot(d, frame.up), dot(d, frame.forward))


@dataclass(frozen=True)
class BodyScale:
    trunk: float
    shoulder_width: float
    hip_width: float
    arm_left: float
    arm_right: float


def body_scale(world) -> BodyScale:
    """Body-scale estimates used to normalise lengths.

    Normalising by the player's own trunk makes thresholds transferable between a
    tall adult and a junior, and between a clip shot from 5 m and one from 15 m.
    """
    mid_shoulder = midpoint(world[L.LEFT_SHOULDER], world[L.RIGHT_SHOULDER])
    mid_hip = midpoint(world[L.LEFT_HIP], world[L.RIGHT_HIP])
    return BodyScale(
        trunk=distance(mid_shoulder, mid_hip),
        shoulder_width=distance(world[L.LEFT_SHOULDER], world[L.RIGHT_SHOULDER]),
        hip_width=distance(world[L.LEFT_HIP], world[L.RIGHT_HIP]),
        arm_left=distance(world[L.LEFT_SHOULDER], world[L.LEFT_ELBOW])
        + distance(world[L.LEFT_ELBOW], world[L.LEFT_WRIST]),
        arm_right=distance(world[L.RIGHT_SHOULDER], world[L.RIGHT_ELBOW])
        + distance(world[L.RIGHT_ELBOW], world[L.RIGHT_WRIST]),
    )


def elbow_angle(world, arm: str) -> float:
    """Elbow flexion: 180 is a straight arm."""
    return joint_angle(
        world[L.side("SHOULDER", arm)],
        world[L.side("ELBOW", arm)],
        world[L.side("WRIST", arm)],
    )


def knee_angle(world, leg: str) -> float:
    """Knee flexion: 180 is a straight leg."""
    return joint_angle(
        world[L.side("HIP", leg)],
        world[L.side("KNEE", leg)],
        world[L.side("ANKLE", leg)],
    )


def shoulder_elevation(frame: BodyFrame, world, arm: str) -> float:
    """How far the upper arm is raised from the side of the body, in degrees.

    0 is the arm hanging down the flank, 90 horizontal, 180 straight overhead --
    the number a coach means by "get your elbow up".
    """
    upper_arm = sub(world[L.side("ELBOW", arm)], world[L.side("SHOULDER", arm)])
    down = (-frame.up[0], -frame.up[1], -frame.up[2])
    return angle_between(upper_arm, down)


def shoulder_azimuth(frame: BodyFrame, world, arm: str, racket_arm: str | None = None) -> float:
    """Where the upper arm points in the horizontal plane, in degrees.

    Measured from straight ahead of the chest, signed towards the racket side:
    near 0 the arm is in front, +90 straight out to the racket side, negative once
    the arm has crossed the body -- the geometry of a backhand.
    """
    racket_arm = racket_arm or arm
    upper_arm = sub(world[L.side("ELBOW", arm)], world[L.side("SHOULDER", arm)])
    flat = reject(upper_arm, frame.up)
    raw = signed_angle(frame.forward, flat, frame.up)
    # signed_angle is positive towards the player's left about `up`; flip so that
    # positive always means "towards the racket side" for either handedness.
    return raw * (-1 if racket_arm == "right" else 1)


def trunk_lean(frame: BodyFrame, racket_arm: str = "right") -> dict:
    """Trunk lean away from vertical, split into forwards and sideways.

    The decomposition is taken against *world-horizontal* axes derived from which
    way the player faces -- not against the torso's own axes, which are orthogonal
    to its up-vector by construction and would make every component zero.
    """
    up = frame.up
    vertical = (0.0, 1.0, 0.0)
    total = angle_between(up, vertical)

    heading = normalize(reject(frame.forward, vertical))
    if length(heading) < 1e-6:
        # Chest pointing straight up or down (a dive, or a bad frame): fall back
        # to the shoulder line so the axes stay defined.
        heading = normalize(reject(frame.right, vertical))
    # With y up, cross(heading, vertical) points out of the player's right side.
    right_axis = normalize(cross(heading, vertical))

    upright = dot(up, vertical)
    forward = math.degrees(math.atan2(dot(up, heading), upright))
    lateral = math.degrees(math.atan2(dot(up, right_axis), upright))
    return {
        "total": total,
        "forward": forward,
        "lateral": lateral * (1 if racket_arm == "right" else -1),
        "heading": heading,
    }


def hip_shoulder_separation(frame: BodyFrame, world, racket_arm: str = "right") -> float:
    """Shoulder-hip separation ("X-factor"), in degrees.

    The twist stored between pelvis and ribcage during the backswing is where an
    overhead's power comes from; unwinding it is what a coach means by "hit with
    your body, not your arm".
    """
    shoulder_line = reject(sub(world[L.RIGHT_SHOULDER], world[L.LEFT_SHOULDER]), frame.up)
    hip_line = reject(sub(world[L.RIGHT_HIP], world[L.LEFT_HIP]), frame.up)
    raw = signed_angle(hip_line, shoulder_line, frame.up)
    return raw * (1 if racket_arm == "right" else -1)


def stance_width(world, scale: BodyScale) -> float:
    """Stance width as a multiple of shoulder width."""
    feet = distance(world[L.LEFT_ANKLE], world[L.RIGHT_ANKLE])
    return feet / scale.shoulder_width if scale.shoulder_width > 1e-6 else NAN


def hand_position(frame: BodyFrame, world, arm: str, trunk: float,
                  arm_length: float, racket_arm: str | None = None) -> dict:
    """The racket hand's position, in body-frame trunk lengths.

    ``height`` is 0 at the shoulder line and positive above it; ``lateral`` is
    positive on the racket side and negative once the hand has crossed the
    midline, which is the single number separating forehand from backhand;
    ``forward`` is positive in front of the chest; ``extension`` is how much of
    the arm's length is being used, 0..1.
    """
    racket_arm = racket_arm or arm
    wrist = to_body(frame, world[L.side("WRIST", arm)])
    unit = trunk if trunk > 1e-6 else NAN
    towards_racket_side = 1 if racket_arm == "right" else -1
    reach = distance(world[L.side("WRIST", arm)], world[L.side("SHOULDER", arm)])
    return {
        "height": wrist[1] / unit,
        "lateral": (wrist[0] * towards_racket_side) / unit,
        "forward": wrist[2] / unit,
        "extension": min(1.5, reach / arm_length) if arm_length > 1e-6 else NAN,
        "body": wrist,
    }


def core_visibility(image, indices=None) -> float:
    """Mean visibility of the landmarks the angles depend on."""
    indices = L.CORE if indices is None else indices
    values = [image[i][3] for i in indices if len(image[i]) > 3]
    values = [v for v in values if v is not None and math.isfinite(v)]
    return sum(values) / len(values) if values else 0.0


def frame_metrics(world, racket_arm: str = "right", trunk: float | None = None,
                  arm_length: float | None = None) -> dict:
    """The full metric set for one frame.

    ``trunk`` and ``arm_length`` should be stable estimates (a running median),
    not this frame's raw values.
    """
    frame = body_frame(world)
    scale = body_scale(world)
    trunk_len = trunk if (trunk is not None and math.isfinite(trunk) and trunk > 1e-6) else scale.trunk
    if arm_length is not None and math.isfinite(arm_length) and arm_length > 1e-6:
        arm_len = arm_length
    else:
        arm_len = scale.arm_right if racket_arm == "right" else scale.arm_left
    off = L.other_arm(racket_arm)

    return {
        "frame": frame,
        "scale": scale,
        "racket_arm": racket_arm,
        "elbow": elbow_angle(world, racket_arm),
        "elbow_off": elbow_angle(world, off),
        "shoulder_elevation": shoulder_elevation(frame, world, racket_arm),
        "shoulder_elevation_off": shoulder_elevation(frame, world, off),
        "shoulder_azimuth": shoulder_azimuth(frame, world, racket_arm, racket_arm),
        "trunk_lean": trunk_lean(frame, racket_arm),
        "separation": hip_shoulder_separation(frame, world, racket_arm),
        "knee_left": knee_angle(world, "left"),
        "knee_right": knee_angle(world, "right"),
        "stance_width": stance_width(world, scale),
        "hand": hand_position(frame, world, racket_arm, trunk_len, arm_len, racket_arm),
        "trunk_length": trunk_len,
        "arm_length": arm_len,
    }
