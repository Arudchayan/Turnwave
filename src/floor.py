"""Reversible floor control for speech during assistant playback.

The original app treats any confirmed speech onset as a floor take: it aborts
the reply, stops playback and resets the conversation. That is safe but it also
cancels on backchannels, coughs and residual echo. This policy separates the
cheap, reversible action (duck playback) from the expensive, irreversible one
(cancel the turn).

It is deliberately not a turn-completion detector. A mid-turn pause can still
commit a turn; that needs the separate end-of-turn work. The measured claim
here is narrower: a short vocalisation can be observed without destroying the
reply, while sustained speech still takes the floor.
"""
from dataclasses import dataclass

IDLE = 'idle'
CANDIDATE = 'candidate'
FLOOR = 'floor'
BACKCHANNEL = 'backchannel'

HARD = 'hard'
REVERSIBLE = 'reversible'
MODES = (HARD, REVERSIBLE)


@dataclass(frozen=True)
class FloorAction:
    """One policy decision. The app applies only the fields that are set."""

    state: str = IDLE
    duck: bool = False
    resume: bool = False
    cancel: bool = False
    promote: bool = False
    speech_ms: float = 0.0
    wall_ms: float = 0.0
    reason: str = ''


class FloorPolicy:
    """Decide whether observed speech during playback is a backchannel or a turn.

    ``hard`` preserves the original immediate cancellation. ``reversible``
    ducks on onset, then waits for either enough speech to commit or a short
    ending to classify as a backchannel. The policy only acts while an
    assistant turn is active; ordinary user turns are left to the app.
    """

    def __init__(self, mode=HARD, backchannel_max_ms=250,
                 floor_confirm_ms=300, duck_gain=.25):
        if mode not in MODES:
            raise ValueError('Unknown floor mode: ' + str(mode))
        backchannel_max_ms = float(backchannel_max_ms)
        floor_confirm_ms = float(floor_confirm_ms)
        if backchannel_max_ms < 0:
            raise ValueError('backchannel_max_ms must be non-negative')
        if floor_confirm_ms <= 0:
            raise ValueError('floor_confirm_ms must be positive')
        if floor_confirm_ms < backchannel_max_ms:
            raise ValueError('floor confirmation must not precede backchannel cutoff')
        self.mode = mode
        self.backchannel_max_ms = backchannel_max_ms
        self.floor_confirm_ms = floor_confirm_ms
        self.duck_gain = max(0.0, min(1.0, float(duck_gain)))
        self.reset()

    def reset(self):
        self.state = IDLE
        self._t0 = None
        self._last = None
        self._speech_s = 0.0
        self.reason = ''

    def _begin(self, now, cancel, reason, speech_s=0.0):
        self._t0 = self._last = float(now)
        # Count speech evidence already inside the confirmation window. The
        # callback fires after several hot frames, not at the first sample.
        self._speech_s = max(0.0, float(speech_s))
        self.reason = reason
        if cancel:
            self.state = FLOOR
            return FloorAction(state=FLOOR, cancel=True, reason=reason)
        self.state = CANDIDATE
        return FloorAction(state=CANDIDATE, duck=True, reason=reason)

    def _elapsed_ms(self, now):
        return max(0.0, (float(now) - self._t0) * 1000.0)

    def _speech_ms(self):
        return max(0.0, self._speech_s * 1000.0)

    def update(self, now, speaking, started=False, ended=False, rms=0.0,
               assistant_active=True, audible=True, block_s=0.02, onset_ms=None):
        """Consume one audio block and return the action to apply.

        ``now`` is a monotonic clock in seconds. ``ended`` is the frontend's
        silence endpoint, not a model judgement. ``audible`` is False while the
        assistant is still thinking or has already stopped, where ducking has
        no meaning and the safe action is the original cancellation.
        """
        now = float(now)
        speaking = bool(speaking)

        if not assistant_active:
            if self.state == CANDIDATE:
                # The reply ended while the user was still speaking. Keep the
                # capture alive as an ordinary turn; there is nothing left to
                # duck or cancel.
                self.state = IDLE
                return FloorAction(state=IDLE, promote=True,
                                   speech_ms=self._speech_ms(),
                                   wall_ms=self._elapsed_ms(now),
                                   reason='assistant-ended')
            self.state = IDLE
            return FloorAction(state=IDLE)

        if started and self.state in (IDLE, BACKCHANNEL):
            initial = block_s if onset_ms is None else max(0.0, float(onset_ms)) / 1000.0
            if initial <= 0 and speaking:
                initial = block_s
            if self.mode == HARD or not audible:
                return self._begin(now, True, 'hard' if self.mode == HARD else 'not-audible',
                                   initial)
            return self._begin(now, False, 'duck', initial)

        if self.state != CANDIDATE:
            return FloorAction(state=self.state)

        if speaking:
            self._speech_s += max(0.0, float(block_s))
        self._last = now
        wall_ms = self._elapsed_ms(now)
        speech_ms = self._speech_ms()

        if speaking and speech_ms >= self.floor_confirm_ms:
            self.state = FLOOR
            self.reason = 'sustained-speech'
            return FloorAction(state=FLOOR, cancel=True, speech_ms=speech_ms,
                               wall_ms=wall_ms, reason=self.reason)

        if ended:
            if speech_ms < self.backchannel_max_ms:
                self.state = IDLE
                self.reason = 'backchannel'
                return FloorAction(state=BACKCHANNEL, resume=True,
                                   speech_ms=speech_ms, wall_ms=wall_ms,
                                   reason=self.reason)
            self.state = FLOOR
            self.reason = 'ended-speech'
            return FloorAction(state=FLOOR, cancel=True, speech_ms=speech_ms,
                               wall_ms=wall_ms, reason=self.reason)

        return FloorAction(state=CANDIDATE, speech_ms=speech_ms,
                           wall_ms=wall_ms, reason=self.reason)

    def snapshot(self):
        return dict(mode=self.mode, state=self.state,
                    speech_ms=round(self._speech_ms(), 1),
                    wall_ms=round(self._elapsed_ms(self._last), 1) if self._last is not None else 0.0,
                    reason=self.reason)
