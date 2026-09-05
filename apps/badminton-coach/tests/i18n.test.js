/**
 * The string tables.
 *
 * A missing key shows up in the app as an untranslated identifier in the middle
 * of a Thai sentence, which is the sort of thing nobody notices until a user
 * points at it, so it is checked here instead.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { STRINGS, setLanguage, t, localiseCue, shotName } from '../web/js/i18n.js';
import { RULES, targetText, targetRange } from '../web/js/core/coach.js';

test('both languages define the same keys', () => {
  const th = new Set(Object.keys(STRINGS.th));
  const en = new Set(Object.keys(STRINGS.en));
  const missingThai = [...en].filter((k) => !th.has(k) && !k.startsWith('cue.'));
  const missingEnglish = [...th].filter((k) => !en.has(k) && !k.startsWith('cue.'));
  assert.deepEqual(missingThai, []);
  assert.deepEqual(missingEnglish, []);
});

test('every coaching rule has Thai text', () => {
  const missing = RULES
    .flatMap((r) => [`cue.${r.id}.label`, `cue.${r.id}.why`])
    .filter((key) => !(key in STRINGS.th));
  assert.deepEqual(missing, [], 'rules without a Thai translation');
});

test('the session rules have Thai text too', () => {
  for (const id of ['ready-knee-bend', 'ready-stance-width', 'recovery-time', 'recovery-rate']) {
    assert.ok(`cue.${id}.label` in STRINGS.th, `${id} has no Thai label`);
    assert.ok(`cue.${id}.why` in STRINGS.th, `${id} has no Thai reason`);
  }
});

test('no string is left as an empty placeholder', () => {
  for (const [lang, table] of Object.entries(STRINGS)) {
    for (const [key, value] of Object.entries(table)) {
      assert.ok(typeof value === 'string' && value.trim().length > 0, `${lang}.${key} is empty`);
    }
  }
});

test('an unknown key falls back rather than throwing', () => {
  setLanguage('th');
  assert.equal(t('definitely-not-a-key'), 'definitely-not-a-key');
});

test('a rule with no translation keeps its own English words', () => {
  setLanguage('th');
  const cue = localiseCue({
    id: 'a-rule-added-tomorrow', label: 'Brand new', why: 'Because.',
    unit: '°', target: 'at least 5°', targetRange: { min: 5, max: null }, value: 1,
  });
  assert.equal(cue.label, 'Brand new');
  assert.equal(cue.target, 'อย่างน้อย 5.00°');
});

test('targets are phrased in the reader language', () => {
  const rule = RULES.find((r) => r.id === 'overhead-elbow-extension');
  const cue = {
    id: rule.id, label: rule.label, why: rule.why, unit: rule.unit,
    target: targetText(rule), targetRange: targetRange(rule), value: 120,
  };
  setLanguage('en');
  assert.equal(localiseCue(cue).target, 'at least 150°');
  setLanguage('th');
  assert.equal(localiseCue(cue).target, 'อย่างน้อย 150°');
});

test('two-sided targets render as a range', () => {
  const rule = RULES.find((r) => r.id === 'drive-arm-extension');
  const cue = { id: rule.id, unit: rule.unit, target: targetText(rule), targetRange: targetRange(rule) };
  setLanguage('en');
  assert.equal(localiseCue(cue).target, '120–170°');
});

test('shot names read naturally in both languages', () => {
  assert.equal(shotName('backhand-overhead', 'en'), 'Backhand overhead');
  assert.equal(shotName('forehand-underarm', 'th'), 'โฟร์แฮนด์ใต้มือ');
  assert.equal(shotName('roundhead-overhead', 'en'), 'Round-the-head overhead');
});
