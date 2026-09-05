/**
 * Thai and English strings.
 *
 * Thai first: the app is for a Thai club court, and reading a coaching cue in
 * your own language between rallies is the difference between acting on it and
 * ignoring it.
 */

export const STRINGS = {
  th: {
    title: 'ตัววิเคราะห์ท่าตีแบดมินตัน',
    tagline: 'วัดมุมแขนและท่าทางขณะตีโฟร์แฮนด์และแบ็คแฮนด์',
    start_camera: 'ใช้กล้อง',
    start_video: 'เปิดไฟล์วิดีโอ',
    stop: 'หยุด',
    settings: 'ตั้งค่า',
    racket_hand: 'มือที่ถือไม้',
    right: 'ขวา',
    left: 'ซ้าย',
    auto: 'อัตโนมัติ',
    quality: 'คุณภาพโมเดล',
    quality_lite: 'เร็ว (lite)',
    quality_full: 'สมดุล (full)',
    quality_heavy: 'ละเอียด (heavy)',
    sensitivity: 'ความไวในการจับจังหวะตี',
    tab_live: 'สด',
    tab_shots: 'ลูกที่ตี',
    tab_court: 'สนาม',
    tab_summary: 'สรุป',
    hint_tap_player: 'แตะที่ตัวผู้เล่นเพื่อล็อกเป้าหมาย',
    hint_calibrate: 'แตะมุมสนาม 4 จุดตามลำดับ',
    corner_netLeft: 'มุมซ้ายติดเน็ต',
    corner_netRight: 'มุมขวาติดเน็ต',
    corner_backRight: 'มุมขวาหลังสนาม',
    corner_backLeft: 'มุมซ้ายหลังสนาม',
    calibrate: 'ตั้งค่าสนาม',
    calibrate_redo: 'ตั้งค่าสนามใหม่',
    calibrate_clear: 'ล้างค่าสนาม',
    swap_sides: 'สลับซ้าย–ขวา',
    court_ready: 'ตั้งค่าสนามแล้ว',
    court_none: 'ยังไม่ได้ตั้งค่าสนาม',
    court_invalid: 'จุดที่แตะไม่ถูกต้อง กรุณาแตะใหม่ตามลำดับ',
    m_elbow: 'มุมข้อศอก',
    m_elevation: 'การยกแขน',
    m_azimuth: 'ทิศแขน',
    m_lean: 'การเอนตัว',
    m_separation: 'การบิดลำตัว',
    m_knee: 'มุมเข่า',
    m_stance: 'ความกว้างของขา',
    m_speed: 'ความเร็วข้อมือ',
    m_hand_height: 'ความสูงจุดปะทะ',
    tracking: 'กำลังติดตาม',
    searching: 'กำลังค้นหาผู้เล่น',
    no_shots: 'ยังไม่พบการตี',
    shots_found: 'พบการตี',
    forehand: 'โฟร์แฮนด์',
    backhand: 'แบ็คแฮนด์',
    roundhead: 'ราวด์เดอะเฮด',
    overhead: 'เหนือศีรษะ',
    drive: 'ระดับลำตัว',
    underarm: 'ใต้มือ',
    good: 'ดี',
    warn: 'พอใช้',
    bad: 'ควรแก้ไข',
    target: 'เป้าหมาย',
    export: 'บันทึกข้อมูล (JSON)',
    reset: 'เริ่มใหม่',
    summary_shots: 'จำนวนลูกที่ตี',
    summary_speed: 'ความเร็วข้อมือสูงสุด',
    summary_faults: 'สิ่งที่ควรแก้ไขบ่อยที่สุด',
    summary_court: 'การเคลื่อนที่ในสนาม',
    distance_covered: 'ระยะทางที่วิ่ง',
    recovery: 'เวลากลับจุดกึ่งกลาง',
    zone_time: 'เวลาในแต่ละโซน',
    times: 'ครั้ง',
    of_shots: 'จากลูกที่ตี',
    loading_model: 'กำลังโหลดโมเดล…',
    error_camera: 'เปิดกล้องไม่ได้ กรุณาอนุญาตการใช้กล้อง',
    error_model: 'โหลดโมเดลไม่สำเร็จ ต้องต่ออินเทอร์เน็ตในการใช้ครั้งแรก',
    fps: 'เฟรม/วินาที',
    coverage: 'ตรวจจับได้',
    camera_hint: 'วางมือถือให้นิ่ง เห็นตัวเต็มตัวและเท้าติดพื้น',
    tilt_warning: 'ถือมือถือให้ตั้งตรง มุมการเอนตัวจะแม่นขึ้น',

    // Coaching cues. The rule ids come from js/core/coach.js; the English text
    // lives with the rules themselves and is used when a translation is missing.
    unit_deg: '°',
    unit_trunk_above: ' เท่าของลำตัว (สูงกว่าระดับไหล่)',
    unit_trunk_front: ' เท่าของลำตัว (หน้าอก)',
    unit_shoulder_width: ' เท่าของความกว้างไหล่',
    unit_seconds: ' วินาที',
    unit_of_shots: ' ของลูกที่ตี',
    target_at_least: 'อย่างน้อย',
    target_at_most: 'ไม่เกิน',

    'cue.overhead-elbow-extension.label': 'การเหยียดแขนตอนปะทะลูก',
    'cue.overhead-elbow-extension.why': 'ลูกเหนือศีรษะต้องตีที่จุดสูงสุดเท่าที่เอื้อมถึง ถ้างอศอก จุดปะทะจะต่ำลง ทำให้ตีลงมุมชันไม่ได้และเสียความเร็วหัวไม้',
    'cue.overhead-contact-height.label': 'ความสูงของจุดปะทะ',
    'cue.overhead-contact-height.why': 'ยิ่งรับลูกสูง ยิ่งได้มุมตีลงที่ชัน ซึ่งเป็นสิ่งที่ลูกตบต้องการ และให้เวลาคู่แข่งน้อยลง',
    'cue.overhead-contact-in-front.label': 'จุดปะทะอยู่หน้าลำตัว',
    'cue.overhead-contact-in-front.why': 'ถ้าปล่อยให้ลูกเลยหัวไปข้างหลัง ลูกตบจะกลายเป็นลูกดันตั้งรับ และหัวไหล่ต้องรับแรงในท่าที่อ่อนแรงที่สุด',
    'cue.overhead-free-arm.label': 'การยกแขนอีกข้างขึ้นชี้ลูก',
    'cue.overhead-free-arm.why': 'การชี้ลูกด้วยแขนข้างที่ไม่ถือไม้ ช่วยหมุนไหล่เข้าหาลูกและทรงตัว เป็นจุดต่างที่เห็นชัดที่สุดระหว่างคนที่ผ่านการฝึกกับคนที่ยังไม่ได้ฝึก',
    'cue.overhead-body-rotation.label': 'การบิดไหล่สวนกับสะโพก',
    'cue.overhead-body-rotation.why': 'พลังมาจากการคลายการบิดของลำตัว ถ้าไหล่กับสะโพกไม่บิดสวนกัน ก็ไม่มีแรงสะสมให้ปล่อย และจะกลายเป็นการตีด้วยแขนอย่างเดียว',
    'cue.overhead-backswing-load.label': 'การงอศอกช่วงเงื้อไม้',
    'cue.overhead-backswing-load.why': 'หัวไม้ควรตกลงหลังหลังก่อนจะเร่งขึ้น แขนที่เหยียดตรงตลอดไม่มีจังหวะขว้างให้ปล่อยแรง',
    'cue.backhand-elbow-extension.label': 'การเหยียดแขนตอนปะทะลูก',
    'cue.backhand-elbow-extension.why': 'แบ็คแฮนด์ตีไกลได้ด้วยการเหยียดศอกและหมุนแขนท่อนล่าง การปะทะขณะแขนยังพับอยู่คือสาเหตุที่ทำให้ตีไม่ถึงหลังสนาม',
    'cue.backhand-contact-in-front.label': 'จุดปะทะอยู่หน้าลำตัว',
    'cue.backhand-contact-in-front.why': 'แบ็คแฮนด์ต้องรับลูกเร็ว เมื่อลูกมาถึงระดับลำตัวแล้ว แขนจะเหยียดเข้าหาลูกไม่ได้อีก',
    'cue.net-lunge-knee.label': 'การงอเข่าขาหน้าเวลาพุ่ง',
    'cue.net-lunge-knee.why': 'การพุ่งโดยงอเข่าช่วยลดตัวลงหาลูกและใช้ขาถีบกลับได้ ถ้าเอื้อมด้วยขาตรง จะติดอยู่หน้าเน็ตกลับไม่ทัน',
    'cue.net-trunk-upright.label': 'ท่าลำตัวเวลาพุ่งหน้าเน็ต',
    'cue.net-trunk-upright.why': 'การทิ้งตัวไปข้างหน้าเลยเท้าหน้า ทำให้ถีบกลับไม่ได้ และลงน้ำหนักที่เข่าแทนที่จะเป็นสะโพก',
    'cue.drive-contact-in-front.label': 'จุดปะทะอยู่หน้าลำตัว',
    'cue.drive-contact-in-front.why': 'การตีโต้ระดับลำตัวชนะกันที่ใครรับลูกได้เร็วกว่า ปะทะช้าไปลูกจะกลายเป็นลูกงัด',
    'cue.drive-arm-extension.label': 'การเหยียดแขนตอนปะทะลูก',
    'cue.drive-arm-extension.why': 'ลูกระดับลำตัวใช้วงสวิงสั้น ไม่พับชิดตัวและไม่เหยียดสุดจนล็อก เพราะจะไม่เหลือระยะให้เร่งหัวไม้',
    'cue.ready-knee-bend.label': 'การย่อเข่าระหว่างรอลูก',
    'cue.ready-knee-bend.why': 'ท่าเตรียมของแบดมินตันต้องย่อเข่าไว้ เพื่อให้ก้าวแรกเป็นการถีบออกไม่ใช่การทิ้งตัว การยืนตัวตรงระหว่างรอทำให้ช้าลงเสี้ยววินาทีทุกลูก',
    'cue.ready-stance-width.label': 'ความกว้างของขาระหว่างรอลูก',
    'cue.ready-stance-width.why': 'ฐานที่กว้างกว่าช่วงไหล่ทำให้มีแรงถีบไปได้ทั้งสองทาง ถ้าเท้าชิดกัน ก้าวแรกจะกลายเป็นการเซ',
    'cue.recovery-time.label': 'เวลาเฉลี่ยกลับจุดกึ่งกลาง',
    'cue.recovery-time.why': 'การกลับมากลางสนามคือสิ่งที่ทำให้ไปถึงลูกถัดไปได้ การกลับช้าจะเห็นผลอีกสองลูกถัดไป ไม่ใช่ที่ลูกนั้น',
    'cue.recovery-rate.label': 'สัดส่วนลูกที่กลับจุดกึ่งกลาง',
    'cue.recovery-rate.why': 'การยืนค้างอยู่ตรงที่ตีลูกที่แล้ว เป็นสาเหตุที่พบบ่อยที่สุดของการเสียแต้มทั้งที่กำลังได้เปรียบ',
  },
  en: {
    title: 'Badminton posture coach',
    tagline: 'Arm angles and body position on forehands and backhands',
    start_camera: 'Use camera',
    start_video: 'Open a video',
    stop: 'Stop',
    settings: 'Settings',
    racket_hand: 'Racket hand',
    right: 'Right',
    left: 'Left',
    auto: 'Auto',
    quality: 'Model quality',
    quality_lite: 'Fast (lite)',
    quality_full: 'Balanced (full)',
    quality_heavy: 'Accurate (heavy)',
    sensitivity: 'Swing sensitivity',
    tab_live: 'Live',
    tab_shots: 'Shots',
    tab_court: 'Court',
    tab_summary: 'Summary',
    hint_tap_player: 'Tap the player to lock on',
    hint_calibrate: 'Tap the four court corners in order',
    corner_netLeft: 'Net corner, left',
    corner_netRight: 'Net corner, right',
    corner_backRight: 'Back corner, right',
    corner_backLeft: 'Back corner, left',
    calibrate: 'Set up court',
    calibrate_redo: 'Redo court set-up',
    calibrate_clear: 'Clear court',
    swap_sides: 'Swap left / right',
    court_ready: 'Court is set up',
    court_none: 'Court not set up',
    court_invalid: 'Those corners do not make a court. Tap them again in order.',
    m_elbow: 'Elbow angle',
    m_elevation: 'Arm elevation',
    m_azimuth: 'Arm direction',
    m_lean: 'Trunk lean',
    m_separation: 'Trunk twist',
    m_knee: 'Knee angle',
    m_stance: 'Stance width',
    m_speed: 'Wrist speed',
    m_hand_height: 'Contact height',
    tracking: 'Tracking',
    searching: 'Looking for the player',
    no_shots: 'No shots detected yet',
    shots_found: 'Shots detected',
    forehand: 'Forehand',
    backhand: 'Backhand',
    roundhead: 'Round-the-head',
    overhead: 'Overhead',
    drive: 'Drive',
    underarm: 'Underarm',
    good: 'Good',
    warn: 'Borderline',
    bad: 'Work on this',
    target: 'Target',
    export: 'Export data (JSON)',
    reset: 'Start over',
    summary_shots: 'Shots played',
    summary_speed: 'Fastest wrist speed',
    summary_faults: 'Most frequent things to fix',
    summary_court: 'Court movement',
    distance_covered: 'Distance covered',
    recovery: 'Time back to base',
    zone_time: 'Time by zone',
    times: 'times',
    of_shots: 'of shots',
    loading_model: 'Loading the pose model…',
    error_camera: 'Could not open the camera. Please allow camera access.',
    error_model: 'Could not load the pose model. The first run needs an internet connection.',
    fps: 'fps',
    coverage: 'detected',
    camera_hint: 'Prop the phone still, with the whole body and the feet in shot',
    tilt_warning: 'Hold the phone upright for accurate trunk-lean angles',

    unit_deg: '°',
    unit_trunk_above: ' trunk lengths above the shoulders',
    unit_trunk_front: ' trunk lengths in front of the chest',
    unit_shoulder_width: '× shoulder width',
    unit_seconds: ' s',
    unit_of_shots: ' of shots',
    target_at_least: 'at least',
    target_at_most: 'at most',
  },
};

/** Shot-name pieces, so `forehand-overhead` reads properly in both languages. */
export function shotName(shot, lang) {
  const table = STRINGS[lang] || STRINGS.en;
  const [side, height] = shot.split('-');
  const sideText = table[side] || side;
  const heightText = table[height] || height;
  if (lang === 'th') return `${sideText}${heightText}`;
  // "Forehand overhead", not "Forehand Overhead": only the first word leads.
  return `${sideText} ${heightText.charAt(0).toLowerCase()}${heightText.slice(1)}`;
}

let current = 'th';

export function setLanguage(lang) {
  current = STRINGS[lang] ? lang : 'en';
  // Guarded so the string table can be unit-tested in Node, where there is no
  // document to update.
  if (typeof document !== 'undefined') {
    document.documentElement.lang = current;
    applyTranslations();
  }
  return current;
}

export const getLanguage = () => current;

/** Look up a key in the active language, falling back to English then the key. */
export function t(key) {
  return STRINGS[current]?.[key] ?? STRINGS.en[key] ?? key;
}

/** True when the active language has its own text for this key. */
const has = (key) => Object.prototype.hasOwnProperty.call(STRINGS[current] || {}, key);

/**
 * A coaching cue in the reader's language.
 *
 * The rules in `core/coach.js` carry English text so that the analysis core
 * stays usable on its own -- in Node, in the Python port, in a report. The app
 * translates by rule id where a translation exists and shows the rule's own
 * words where it does not, so a newly added rule appears in English rather than
 * as a missing-key placeholder.
 */
export function localiseCue(cue) {
  const label = has(`cue.${cue.id}.label`) ? t(`cue.${cue.id}.label`) : cue.label;
  const why = has(`cue.${cue.id}.why`) ? t(`cue.${cue.id}.why`) : cue.why;
  return {
    ...cue,
    label,
    why,
    unit: localiseUnit(cue.unit),
    target: formatTarget(cue),
  };
}

const UNIT_KEYS = new Map([
  ['°', 'unit_deg'],
  [' trunk lengths above the shoulders', 'unit_trunk_above'],
  [' trunk lengths in front of the chest', 'unit_trunk_front'],
  ['× shoulder width', 'unit_shoulder_width'],
  [' s', 'unit_seconds'],
  [' of shots', 'unit_of_shots'],
]);

const localiseUnit = (unit) => (UNIT_KEYS.has(unit) ? t(UNIT_KEYS.get(unit)) : unit);

const formatNumber = (v) => (Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(2));

/**
 * Phrase a rule's target band in the reader's language.
 *
 * The rules carry an English sentence *and* the numbers behind it. Translating
 * the sentence would mean translating "at least" three times over; formatting
 * from the numbers keeps one phrase per language.
 */
function formatTarget(cue) {
  const range = cue.targetRange;
  const unit = localiseUnit(cue.unit);
  if (!range) return cue.target;
  const { min, max } = range;
  if (min !== null && max !== null) return `${formatNumber(min)}–${formatNumber(max)}${unit}`;
  if (min !== null) return `${t('target_at_least')} ${formatNumber(min)}${unit}`;
  if (max !== null) return `${t('target_at_most')} ${formatNumber(max)}${unit}`;
  return cue.target;
}

/** Fill every `[data-i18n]` element in the document. */
export function applyTranslations(root = (typeof document === 'undefined' ? null : document)) {
  if (!root) return;
  for (const el of root.querySelectorAll('[data-i18n]')) {
    el.textContent = t(el.dataset.i18n);
  }
  for (const el of root.querySelectorAll('[data-i18n-label]')) {
    el.setAttribute('aria-label', t(el.dataset.i18nLabel));
  }
}
