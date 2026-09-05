/**
 * Turn measurements into things a player can act on.
 *
 * ## What these numbers are, and are not
 *
 * The target ranges below are coaching heuristics -- the things badminton
 * coaches say about technique, written down as numbers so they can be checked
 * automatically. They are not clinical norms, they are not tuned to any one
 * player, and a good player will break several of them on purpose. Treat a cue
 * as "worth looking at the video for", not as a verdict. Every threshold lives
 * in {@link RULES} and can be edited or switched off.
 *
 * ## Why each rule exists
 *
 * Every rule states a mechanism, because a cue without one is just a scolding.
 * "Elbow at 118 degrees" means nothing; "the arm never straightened, so the
 * shuttle was struck below full reach and the steep angle was not available"
 * tells the player what to change.
 */

/** Grade a value against a target band. */
export function grade(value, target) {
  if (!Number.isFinite(value)) return 'unknown';
  const { min = -Infinity, max = Infinity, warnMin = -Infinity, warnMax = Infinity } = target;
  if (value >= min && value <= max) return 'good';
  if (value < warnMin || value > warnMax) return 'bad';
  return 'warn';
}

const isOverhead = (s) => s.height === 'overhead';
const isUnderarm = (s) => s.height === 'underarm';
const isDrive = (s) => s.height === 'drive';
const isForehand = (s) => s.side === 'forehand' || s.side === 'roundhead';
const isBackhand = (s) => s.side === 'backhand';

/**
 * Per-stroke rules.
 *
 * `measure` returns the number being judged; `target` is the band a coach would
 * be happy with; values outside `warnMin`/`warnMax` are called out firmly.
 */
export const RULES = [
  {
    id: 'overhead-elbow-extension',
    applies: (s) => isOverhead(s) && isForehand(s),
    measure: (s) => s.contact.elbow,
    unit: '°',
    target: { min: 150, warnMin: 135 },
    label: 'Arm extension at contact',
    why: 'An overhead is struck at full reach. A bent arm lowers the contact point, '
       + 'which flattens the angle into the opponent\'s court and costs racket-head speed.',
  },
  {
    id: 'overhead-contact-height',
    applies: isOverhead,
    measure: (s) => s.contact.hand.height,
    unit: ' trunk lengths above the shoulders',
    target: { min: 0.30, warnMin: 0.15 },
    label: 'Contact height',
    why: 'Taking the shuttle high buys the steep downward angle a smash needs, and '
       + 'gives the opponent less time.',
  },
  {
    id: 'overhead-contact-in-front',
    applies: isOverhead,
    measure: (s) => s.contact.hand.forward,
    unit: ' trunk lengths in front of the chest',
    target: { min: 0.15, warnMin: -0.05 },
    label: 'Contact point in front',
    why: 'Letting the shuttle drift behind the head turns a smash into a defensive '
       + 'push, and loads the shoulder in its weakest position.',
  },
  {
    id: 'overhead-free-arm',
    applies: isOverhead,
    measure: (s) => s.backswing.maxOffArmElevation,
    unit: '°',
    target: { min: 120, warnMin: 90 },
    label: 'Free arm raised in preparation',
    why: 'Pointing at the shuttle with the non-racket arm turns the shoulders into '
       + 'the shot and keeps the body balanced. It is the single most visible '
       + 'difference between a coached and an uncoached overhead.',
  },
  {
    id: 'overhead-body-rotation',
    applies: isOverhead,
    measure: (s) => Math.abs(s.backswing.maxSeparation),
    unit: '°',
    target: { min: 20, warnMin: 12 },
    label: 'Shoulder turn against the hips',
    why: 'Power comes from unwinding the trunk. Without separation between the '
       + 'shoulders and the hips there is nothing stored to release, and the shot '
       + 'is played with the arm alone.',
  },
  {
    id: 'overhead-backswing-load',
    applies: isOverhead,
    measure: (s) => s.backswing.minElbow,
    unit: '°',
    target: { max: 110, warnMax: 135 },
    label: 'Elbow bend in the backswing',
    why: 'The racket should drop behind the back before it accelerates. An arm that '
       + 'stays straight throughout has no throwing action to release.',
  },
  {
    id: 'backhand-elbow-extension',
    applies: (s) => isBackhand(s) && (isOverhead(s) || isDrive(s)),
    measure: (s) => s.contact.elbow,
    unit: '°',
    target: { min: 140, warnMin: 120 },
    label: 'Arm extension at contact',
    why: 'A backhand clear is driven by the elbow straightening and the forearm '
       + 'rotating. Contact with a folded arm is the usual reason a backhand will '
       + 'not reach the back of the court.',
  },
  {
    id: 'backhand-contact-in-front',
    applies: isBackhand,
    measure: (s) => s.contact.hand.forward,
    unit: ' trunk lengths in front of the chest',
    target: { min: 0.20, warnMin: 0.05 },
    label: 'Contact point in front',
    why: 'The backhand has to be taken early. Once the shuttle is level with the '
       + 'body the arm can no longer extend into it.',
  },
  {
    id: 'net-lunge-knee',
    applies: isUnderarm,
    measure: (s) => s.window.minKnee,
    unit: '°',
    target: { min: 95, max: 145 },
    warnBand: true,
    warn: { min: 80, max: 160 },
    label: 'Front knee bend in the lunge',
    why: 'Lunging with a bent knee lowers the body to the shuttle and lets the leg '
       + 'push back out. Reaching with a straight leg leaves the player stranded '
       + 'at the net.',
  },
  {
    id: 'net-trunk-upright',
    applies: isUnderarm,
    measure: (s) => Math.abs(s.contact.trunkLean.forward),
    unit: '°',
    target: { max: 45, warnMax: 60 },
    label: 'Chest position in the lunge',
    why: 'Falling forward over the front foot makes the recovery push impossible '
       + 'and puts the load on the knee rather than the hip.',
  },
  {
    id: 'drive-contact-in-front',
    applies: isDrive,
    measure: (s) => s.contact.hand.forward,
    unit: ' trunk lengths in front of the chest',
    target: { min: 0.10, warnMin: -0.05 },
    label: 'Contact point in front',
    why: 'Flat exchanges are won by taking the shuttle early. A late contact turns '
       + 'a drive into a lift.',
  },
  {
    id: 'drive-arm-extension',
    applies: isDrive,
    measure: (s) => s.contact.elbow,
    unit: '°',
    target: { min: 120, max: 170 },
    warn: { min: 100, max: 178 },
    warnBand: true,
    label: 'Arm extension at contact',
    why: 'A drive is a compact action: neither folded up against the body nor '
       + 'locked out straight, which leaves no room to accelerate the racket.',
  },
];

/** Normalise the two ways a rule can express its bands. */
function bandsOf(rule) {
  if (rule.warnBand && rule.warn) {
    return {
      min: rule.target.min ?? -Infinity,
      max: rule.target.max ?? Infinity,
      warnMin: rule.warn.min ?? -Infinity,
      warnMax: rule.warn.max ?? Infinity,
    };
  }
  return {
    min: rule.target.min ?? -Infinity,
    max: rule.target.max ?? Infinity,
    warnMin: rule.target.warnMin ?? rule.target.min ?? -Infinity,
    warnMax: rule.target.warnMax ?? rule.target.max ?? Infinity,
  };
}

/** The numeric band a rule wants, for a UI that phrases it in its own language. */
export function targetRange(rule) {
  const b = bandsOf(rule);
  return {
    min: Number.isFinite(b.min) ? b.min : null,
    max: Number.isFinite(b.max) ? b.max : null,
  };
}

/** Human-readable statement of what the rule wants, in English. */
export function targetText(rule) {
  const b = bandsOf(rule);
  const hasMin = Number.isFinite(b.min);
  const hasMax = Number.isFinite(b.max);
  const n = (v) => (Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(2));
  if (hasMin && hasMax) return `${n(b.min)}–${n(b.max)}${rule.unit}`;
  if (hasMin) return `at least ${n(b.min)}${rule.unit}`;
  if (hasMax) return `at most ${n(b.max)}${rule.unit}`;
  return '';
}

/**
 * Apply every relevant rule to one stroke.
 *
 * @param {object} stroke a record from `strokes.detectStrokes`
 * @param {object} [options]
 * @param {number} [options.minConfidence=0.15] skip strokes whose shot type is a
 *   coin-flip; coaching the wrong shot type is worse than saying nothing
 * @returns {Array} one cue per applicable rule
 */
export function coachStroke(stroke, { minConfidence = 0.15, rules = RULES } = {}) {
  if (stroke.confidence < minConfidence) return [];
  const cues = [];
  for (const rule of rules) {
    if (!rule.applies(stroke)) continue;
    const value = rule.measure(stroke);
    if (!Number.isFinite(value)) continue;
    const status = grade(value, bandsOf(rule));
    cues.push({
      id: rule.id,
      strokeIndex: stroke.index,
      shot: stroke.shot,
      label: rule.label,
      why: rule.why,
      value,
      unit: rule.unit,
      target: targetText(rule),
      targetRange: targetRange(rule),
      status,
    });
  }
  return cues;
}

/** Session-level rules that need more than one stroke to say anything. */
export const SESSION_RULES = {
  readyKnee: { min: 145, max: 168, warnMin: 130, warnMax: 175 },
  stanceWidth: { min: 1.1, max: 2.2, warnMin: 0.9, warnMax: 2.6 },
  recoverySeconds: { max: 1.2, warnMax: 1.8 },
  recoveredFraction: { min: 0.7, warnMin: 0.5 },
};

/**
 * Look at the whole session: posture between shots, and movement if the court
 * has been calibrated.
 *
 * "Between shots" means frames at least `quiet` seconds away from any detected
 * contact, which is where a player's default posture actually shows.
 */
export function coachSession(frames, strokes, { recovery = null, quiet = 0.4 } = {}) {
  const cues = [];
  const contactTimes = strokes.map((s) => s.t);
  const idle = frames.filter((f) => contactTimes.every((t) => Math.abs(f.t - t) > quiet));

  if (idle.length >= 15) {
    const knees = idle
      .map((f) => Math.max(f.metrics.kneeLeft, f.metrics.kneeRight))
      .filter(Number.isFinite);
    if (knees.length) {
      const value = knees.reduce((a, b) => a + b, 0) / knees.length;
      cues.push({
        id: 'ready-knee-bend',
        label: 'Knee bend between shots',
        why: 'A badminton ready position keeps the knees soft so the first step can '
           + 'be pushed rather than fallen into. Standing tall between rallies costs '
           + 'a fraction of a second on every shuttle.',
        value,
        unit: '°',
        target: '145–168°',
        targetRange: { min: 145, max: 168 },
        status: grade(value, SESSION_RULES.readyKnee),
      });
    }

    const stances = idle.map((f) => f.metrics.stanceWidth).filter(Number.isFinite);
    if (stances.length) {
      const value = stances.reduce((a, b) => a + b, 0) / stances.length;
      cues.push({
        id: 'ready-stance-width',
        label: 'Stance width between shots',
        why: 'A base wider than the shoulders gives something to push against in '
           + 'either direction; feet together means the first move is a stumble.',
        value,
        unit: '× shoulder width',
        target: '1.1–2.2× shoulder width',
        targetRange: { min: 1.1, max: 2.2 },
        status: grade(value, SESSION_RULES.stanceWidth),
      });
    }
  }

  if (recovery && recovery.length) {
    const times = recovery.map((r) => r.recoverySeconds).filter(Number.isFinite);
    if (times.length) {
      const sorted = times.slice().sort((a, b) => a - b);
      const value = sorted[Math.floor(sorted.length / 2)];
      cues.push({
        id: 'recovery-time',
        label: 'Median time back to base',
        why: 'Recovering to the middle is what makes the next shuttle reachable. '
           + 'A slow return is felt two shots later, not on the shot itself.',
        value,
        unit: ' s',
        target: 'under 1.2 s',
        targetRange: { min: null, max: 1.2 },
        status: grade(value, SESSION_RULES.recoverySeconds),
      });
    }
    const fraction = recovery.filter((r) => r.recovered).length / recovery.length;
    cues.push({
      id: 'recovery-rate',
      label: 'Shots followed by a return to base',
      why: 'Staying where the last shot was played is the most common reason a '
         + 'rally is lost from a winning position.',
      value: fraction,
      unit: ' of shots',
      target: 'at least 0.70 of shots',
      targetRange: { min: 0.7, max: null },
      status: grade(fraction, SESSION_RULES.recoveredFraction),
    });
  }

  return cues;
}

/** Group cues by rule so the summary can show the recurring faults first. */
export function rankCues(cues) {
  const groups = new Map();
  for (const cue of cues) {
    if (cue.status !== 'bad' && cue.status !== 'warn') continue;
    const g = groups.get(cue.id) || {
      id: cue.id,
      label: cue.label,
      why: cue.why,
      unit: cue.unit,
      target: cue.target,
      targetRange: cue.targetRange,
      count: 0,
      bad: 0,
      values: [],
    };
    g.count += 1;
    if (cue.status === 'bad') g.bad += 1;
    g.values.push(cue.value);
    groups.set(cue.id, g);
  }
  return [...groups.values()]
    .map((g) => ({
      ...g,
      mean: g.values.reduce((a, b) => a + b, 0) / g.values.length,
    }))
    // Most-often-wrong first, and among equals the ones that were clearly wrong
    // rather than borderline: that is the order a coach would raise them in.
    .sort((a, b) => b.bad - a.bad || b.count - a.count);
}
