# Filming a clip that actually analyses well

The single biggest factor in the quality of the results is how the phone is
placed. None of this needs equipment.

## The essentials

**Get the whole body in frame, feet included.** Court position, stance width and
knee angles all come from the lower body. A clip framed on the upper body gives
arm angles and nothing else.

**Prop the phone somewhere still.** A bag, a bench, the net post. Hand-held works
for arm angles, but the court mapping assumes a fixed camera and will drift.

**Fill more of the frame with the player.** The player should be at least about a
fifth of the frame height. Closer is better: the app crops and upscales around
the player before running the pose model, which recovers a lot, but it cannot
invent detail that was never recorded.

**Film from the side or the back corner, not down the tramlines.** A view straight
along the court foreshortens everything: forehand and backhand become hard to tell
apart because the arm's sideways motion is pointing at the camera.

**Landscape or portrait both work.** The app matches its layout to the clip.

## If you want court positions and recovery times

Set the court up by tapping four corners of the near half, in this order:

1. net corner, **left**
2. net corner, **right**
3. back corner, **right**
4. back corner, **left**

All four corners must be visible in the frame. This is the usual reason the court
feature cannot be used on a clip: a camera placed at the side of a hall often has
the near corners running off the bottom of the picture. If that happens, move
further back or raise the phone.

Left and right are as *you* see them on screen, not as the player sees them. Once
the four taps are in, the app draws the court model back onto the video: if the
yellow lines do not sit on the painted lines, the calibration is wrong and worth
redoing. If left and right came out swapped, one button fixes it without
re-tapping.

## Lighting and background

Indoor hall lighting is fine. What causes trouble:

- **Other players close behind the subject.** Tap the player you want to track;
  the app locks on to them and stays locked. Without a tap it picks the largest
  person in frame, which is usually right and occasionally is somebody walking
  past the camera.
- **Very strong backlight** (a doorway or window behind the player) silhouettes
  the body and costs landmark accuracy.

## Length

A minute is plenty to start with. The offline pipeline handles long clips, but
pose extraction runs at roughly video speed, so a twenty-minute recording is a
twenty-minute wait.
