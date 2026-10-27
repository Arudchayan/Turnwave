"""Offline replay for the reversible floor policy.

This is a policy/timing diagnostic, not an acoustic or model benchmark. It
drives the same FloorPolicy the app uses with a small, frozen set of authored
barge-in shapes and reports where hard cancellation and reversible ducking
disagree. It opens no microphone, speaker, model or network connection.

Run from the repository root:

    python -m src.synth.floor_replay --out outputs/floor_replay.json
"""
import argparse
import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from floor import HARD, REVERSIBLE, FloorPolicy

STEP_MS = 20
CONFIRM_MS = 160       # DuplexFrontend's 3-of-5 32 ms confirmation window
ENDPOINT_MS = 480      # The app's default silence endpoint

CASES = [
    # id, total user speech, expected floor take, note
    ('backchannel-180', 180, False, 'short acknowledgement'),
    ('backchannel-240', 240, False, 'longer acknowledgement'),
    ('cough-120', 120, False, 'non-speech burst above the VAD floor'),
    ('command-300', 300, True, 'short command such as stop'),
    ('floor-take-900', 900, True, 'sustained user turn'),
    ('mid-pause-400', 400, False, 'complete-sounding prefix; known EOT limitation'),
    ('echo-only', 0, False, 'no user speech at all'),
]


def _onset_ms(speech_ms):
    if speech_ms < 96:
        return None
    return min(CONFIRM_MS, speech_ms)


def replay_case(case, mode):
    case_id, speech_ms, expected_cancel, note = case
    policy = FloorPolicy(mode=mode)
    now = 0.0
    events = []
    ducked = resumed = cancelled = False
    cancel_from_onset = cancel_from_speech = None

    def step(speaking, started=False, ended=False, onset_ms=None):
        nonlocal now, ducked, resumed, cancelled, cancel_from_onset, cancel_from_speech
        action = policy.update(now, speaking=speaking, started=started,
                               ended=ended, block_s=STEP_MS / 1000.0,
                               onset_ms=onset_ms, assistant_active=True,
                               audible=True)
        at_ms = round(now * 1000.0, 1)
        if action.duck and not ducked:
            ducked = True
            events.append(dict(kind='duck', at_ms=at_ms))
        if action.resume and not resumed:
            resumed = True
            events.append(dict(kind='resume', at_ms=at_ms,
                               reason=action.reason))
        if action.cancel and not cancelled:
            cancelled = True
            cancel_from_onset = at_ms
            cancel_from_speech = round(at_ms + (_onset_ms(speech_ms) or 0), 1)
            events.append(dict(kind='commit', at_ms=at_ms,
                               reason=action.reason,
                               speech_ms=round(action.speech_ms, 1)))
        now += STEP_MS / 1000.0
        return action

    onset = _onset_ms(speech_ms)
    if onset is None:
        # Too short for the 3-of-5 confirmation window; the frontend never
        # emits started, so no floor decision exists.
        return dict(id=case_id, mode=mode, expected_cancel=expected_cancel,
                    note=note, ducked=False, resumed=False, cancelled=False,
                    false_cancel=False, missed_cancel=False,
                    cancel_from_onset_ms=None, cancel_from_speech_ms=None,
                    events=[], limitation='below VAD confirmation')

    # The frontend confirms after the onset window. The onset credit is the
    # speech already inside that window; post-confirmation blocks follow.
    step(speaking=True, started=True, onset_ms=onset)
    post = max(0, speech_ms - onset)
    sent = 0
    while sent < post and not cancelled:
        step(speaking=True)
        sent += STEP_MS

    if not cancelled:
        # Silence until the configured endpoint. The policy sees no speech in
        # these blocks, so only the endpoint decision can fire.
        silent = 0
        while silent < ENDPOINT_MS and not cancelled:
            last = silent + STEP_MS >= ENDPOINT_MS
            step(speaking=False, ended=last)
            silent += STEP_MS

    false_cancel = bool(cancelled and not expected_cancel)
    missed_cancel = bool(expected_cancel and not cancelled)
    return dict(id=case_id, mode=mode, expected_cancel=expected_cancel,
                note=note, ducked=ducked, resumed=resumed, cancelled=cancelled,
                false_cancel=false_cancel, missed_cancel=missed_cancel,
                cancel_from_onset_ms=cancel_from_onset,
                cancel_from_speech_ms=cancel_from_speech,
                events=events,
                limitation=('known EOT limitation' if case_id == 'mid-pause-400'
                            else None))


def summarize(rows):
    hard = [row for row in rows if row['mode'] == HARD]
    rev = [row for row in rows if row['mode'] == REVERSIBLE]
    def counts(items):
        return dict(
            cases=len(items),
            ducked=sum(row['ducked'] for row in items),
            resumed=sum(row['resumed'] for row in items),
            cancelled=sum(row['cancelled'] for row in items),
            false_cancels=sum(row['false_cancel'] for row in items),
            missed_cancels=sum(row['missed_cancel'] for row in items),
            known_limitation_false_cancels=sum(
                row['false_cancel'] and row['limitation'] == 'known EOT limitation'
                for row in items),
            # Count only authored vocalisations, not the echo-only control or
            # the mid-pause case that neither arm is expected to solve.
            backchannels_preserved=sum(
                (not row['expected_cancel']) and (not row['cancelled'])
                and row['id'] not in ('echo-only', 'mid-pause-400')
                for row in items),
            cancel_latencies_ms=[row['cancel_from_onset_ms'] for row in items
                                 if row['cancelled']],
        )
    def median(values):
        if not values:
            return None
        values = sorted(values)
        mid = len(values) // 2
        return values[mid] if len(values) % 2 else round((values[mid - 1] + values[mid]) / 2, 1)
    hard_summary, rev_summary = counts(hard), counts(rev)
    hard_summary['median_cancel_from_onset_ms'] = median(hard_summary.pop('cancel_latencies_ms'))
    rev_summary['median_cancel_from_onset_ms'] = median(rev_summary.pop('cancel_latencies_ms'))
    # The mid-pause case is a deliberate known limitation, not evidence for
    # either arm. Count it separately so it cannot be hidden.
    hard_unknown = hard_summary['false_cancels'] - hard_summary['known_limitation_false_cancels']
    rev_unknown = rev_summary['false_cancels'] - rev_summary['known_limitation_false_cancels']
    accepted = bool(rev_unknown < hard_unknown
                    and rev_summary['missed_cancels'] <= hard_summary['missed_cancels']
                    and (rev_summary['median_cancel_from_onset_ms'] or 0)
                    <= (hard_summary['median_cancel_from_onset_ms'] or 0) + 200)
    return dict(
        hard=hard_summary,
        reversible=rev_summary,
        non_limitation_false_cancels=dict(hard=hard_unknown, reversible=rev_unknown),
        decision=('candidate for real-device A/B' if accepted else 'not accepted'),
        criteria=('fewer non-limitation false cancels; no increase in missed floor takes; '
                  'median commit latency within 200 ms of hard cancellation'),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(ROOT / 'outputs' / 'floor_replay.json'))
    args = parser.parse_args()
    rows = []
    for mode in (HARD, REVERSIBLE):
        for case in CASES:
            rows.append(replay_case(case, mode))
    source = {}
    for name in ('src/floor.py', 'src/duplex.py', 'src/app.py'):
        path = ROOT / name
        source[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = dict(
        schema_version=1,
        generated_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        evidence_class='offline policy/timing replay; no microphone, model or device',
        step_ms=STEP_MS,
        confirm_ms=CONFIRM_MS,
        endpoint_ms=ENDPOINT_MS,
        source_sha256=source,
        cases=rows,
        summary=summarize(rows),
        limits=('Authored case shapes, not human conversation. The mid-pause case '
                'is a known end-of-turn limitation and is reported separately. '
                'Commit latency is measured from the confirmed onset; these '
                'numbers exclude device, model and playback latency.'),
    )
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    summary = report['summary']
    print(json.dumps(dict(out=str(out), hard=summary['hard'],
                          reversible=summary['reversible'],
                          decision=summary['decision']), indent=2))


if __name__ == '__main__':
    main()
