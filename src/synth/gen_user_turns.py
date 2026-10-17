"""Synthetic user-turn generator: renders scripted user lines in the US MALE
voice (contrast with the assistant's UK female) via server TTS.
Run:  C:/Python312/python src/synth/gen_user_turns.py
Needs: GPU server on :8080. Writes data/synth/user_*.wav (16 kHz mono).
"""
import base64
import io
import json
import os
import struct
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "data", "synth")
os.makedirs(OUT, exist_ok=True)

LINES = [
    ("greet", "Hello, can you hear me clearly?"),
    ("question", "What is the time right now?"),
    ("fact", "Tell me a short fact about the moon."),
    ("thanks", "Thanks, that is all I needed."),
    ("backchannel", "Mm hmm."),
    ("barge", "Wait, stop, let me think for a second."),
    ("correction", "No, I meant the other one, not that."),
    ("long", "I am planning a trip to the mountains next weekend and I wonder what I should pack for cold weather."),
]

TTS_SYS = "Perform TTS. Use the US male voice."


def tts(text, attempts=3):
    import numpy as np
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from client import ServerClient, Turn
    client = ServerClient()
    for _ in range(attempts):
        parts = []
        turn = Turn()
        client._stream(client._payload(TTS_SYS, text, True, 256), turn, parts.append)
        if parts and not turn.error:
            audio = np.concatenate(parts)
            # Reject runaway TTS instead of accepting an unusable fixture.
            if .3 <= len(audio) / 24000 <= 2 + .6 * len(text.split()):
                return audio
    return np.zeros(0, np.float32)


def to_16k_wav(a24):
    import numpy as np
    idx = (np.arange(int(len(a24) * 16000 / 24000)) * 24000 / 16000).astype(int)
    b = (np.clip(a24[idx], -1, 1) * 32767).astype(np.int16)
    import io as _io
    buf = _io.BytesIO()
    buf.write(b"RIFF....WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, 16000,
                                                32000, 2, 16) + b"data" +
              struct.pack("<I", len(b) * 2) + b.tobytes())
    wav = buf.getvalue()
    return wav[:4] + struct.pack("<I", len(wav) - 8) + wav[8:]


def main():
    import numpy as np
    manifest = []
    for name, text in LINES:
        a = tts(text)
        dur = len(a) / 24000 if len(a) else 0
        path = os.path.join(OUT, "user_%s.wav" % name)
        if len(a):
            open(path, "wb").write(to_16k_wav(a))
        manifest.append({"name": name, "text": text, "seconds": round(dur, 2),
                         "file": os.path.basename(path) if len(a) else None})
        print("%-12s %.1fs %s" % (name, dur, "OK" if len(a) else "NO-AUDIO"))
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=2)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(BASE, "src"))
    from backend_session import backend_session
    with backend_session(test=True):
        main()
