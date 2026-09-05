# What is measured, and what each number means

Every number the app reports comes from MediaPipe Pose landmarks and nothing
else. There is no shuttle tracking, no racket detection, and no machine-learned
scoring of technique — the analysis is geometry applied to a skeleton, which
makes it inspectable and makes its limits easy to state.

## Coordinate conventions

MediaPipe returns two sets of landmarks per frame:

| | what it is | used for |
|---|---|---|
| **image landmarks** | x, y in 0..1 of the frame, plus a visibility score | drawing the overlay, and the feet position that the court mapping needs |
| **world landmarks** | x, y, z in **metres**, origin at the mid-hip | every joint angle |

World landmarks are used for angles because they are metric and hip-centred: the
same elbow angle reads the same whether the player is at the net or the back of
the court, which is not true of image coordinates.

MediaPipe's world frame has **y pointing down** and **z pointing away from the
camera**. The first thing the code does is flip both, giving a right-handed frame
with **y up** and **z towards the camera**. Everything below assumes that frame.

## The torso frame

Angles that describe technique — is the arm across the body? is the chest
turned? — only mean anything relative to the player, so an orthonormal basis is
built on the torso each frame:

- `up` = mid-hip → mid-shoulder
- `right` = the shoulder line, orthogonalised against `up`
- `forward` = `up × right`, pointing out of the chest

`up` is taken as primary because the shoulder line is the noisier of the two
while an arm is swinging.

## The measurements

| name | definition | reads as |
|---|---|---|
| **Elbow angle** | interior angle shoulder–elbow–wrist | 180° straight, 90° a right angle |
| **Arm elevation** | angle between the upper arm and the line down the flank | 0° arm down, 90° horizontal, 180° straight overhead |
| **Arm direction (azimuth)** | where the upper arm points in the horizontal plane, from straight ahead, signed towards the racket side | 0° in front, +90° out to the racket side, negative once it has crossed the body |
| **Trunk lean** | tilt of the trunk axis from vertical, split into `forward` (the way the player faces) and `lateral` (towards the racket side) | 0° upright |
| **Trunk twist** | signed angle between the shoulder line and the hip line in the horizontal plane — the "X-factor" | 0° square, larger means more wound up |
| **Knee angle** | interior angle hip–knee–ankle | 180° straight |
| **Stance width** | ankle separation ÷ shoulder width | ~1 standing, larger in a ready stance or a lunge |
| **Contact height / lateral / forward** | racket wrist in the torso frame, divided by trunk length | height 0 at the shoulder line; lateral positive on the racket side; forward positive in front of the chest |
| **Wrist speed** | speed of the racket wrist *relative to the hips*, in trunk lengths per second | also reported in m/s |

### Why trunk lengths

Distances are divided by the player's own trunk length so that one set of
thresholds works for a junior and an adult, and for a clip filmed from 5 m and
one from 15 m. The trunk length used is a running median over about 1.5 seconds,
not the current frame's value — per-frame estimates wander by ±10% on a distant
subject, and dividing by a noisy number makes every normalised measurement noisy.

## Detecting a stroke

A stroke is a local maximum of racket-wrist speed above a threshold, with a
refractory window so that one swing cannot be counted twice. Peak wrist speed is
the standard stand-in for the moment of contact in racket-sports analysis: true
contact is within a frame or two of it, and unlike the shuttle, the wrist is
something a pose model can actually see.

Around each peak the code walks outwards to where the arm goes quiet again, which
gives the backswing and the follow-through. The deepest elbow bend and the
largest trunk twist *before* contact are reported separately from the contact
itself, because those are what the arm had available to release.

## Naming a shot

At contact the racket hand is expressed in the torso frame:

- **height** above the shoulder line separates *overhead*, *drive* and *underarm*
- **lateral** position, signed so positive is always the racket side, separates
  *forehand* from *backhand* — a backhand being precisely the shot where the
  racket hand has crossed the body's midline
- the band in between, when overhead, is called *round-the-head*

Each shot carries a **confidence**: how far the hand sits from the nearest
decision boundary. Shots near a boundary are the ones a human would hesitate over
too, and they are not coached.

## What is assumed, and where it breaks

- **The phone is roughly upright.** MediaPipe's world frame is aligned to the
  camera, not to gravity. Angles *between body parts* — elbow, knee, arm-to-trunk,
  trunk twist — are unaffected by a tilted phone. Trunk lean is quoted against
  vertical and is only as upright as the phone was.
- **The whole body is in shot, including the feet.** Court position comes from the
  ankles.
- **For court measurements, the camera does not move after calibration**, and the
  feet are on the floor. During a jump smash the mapped position creeps towards
  the camera; those frames are excluded rather than silently used.
- **Contact is a proxy.** The shuttle is not tracked, so a swing that misses the
  shuttle entirely still counts as a stroke.
- **Grip and forearm rotation are invisible.** MediaPipe gives a wrist point, not
  a racket face, so nothing here can tell a thumb-up backhand grip from a panhandle
  one.
