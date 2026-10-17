"""Backend and frontend synthetic battery (application integration is in tests/) (headless: no speakers/mic needed).

Drives ServerClient + DuplexFrontend with rendered user WAVs and measures:
 T1 turn-taking: audio presence, TTFT, total, degenerate flag (per line)
 T2 barge-in: assistant reply streaming + user 'barge' injected at +1.0 s
    through frontend (far-end = assistant audio) -> stop-decision latency
 T4 echo immunity: assistant audio as mic input, no user -> must not trigger
Gates (v1): presence 100%, ttft_p50 < 1.2 s, barge < 1.5 s, echo FP = 0.
Run:  C:/Python312/python src/synth/run_suite.py
"""
import io
import json
import os
import struct
import sys
import threading
import time

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, "src"))
from client import ServerClient, Turn, wav16k_bytes, extract_audio
from duplex import DuplexFrontend

SYN = os.path.join(BASE, "data", "synth")
OUT = os.path.join(BASE, "outputs", "synth_report_streaming.json")


def read_wav16(path):
    import wave
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        n = w.getnframes()
        import numpy as np
        return np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32767.0


def t1_turns(client):
    man = json.load(open(os.path.join(SYN, "manifest.json")))
    rows = []
    for m in man:
        if not m["file"]:
            rows.append({"name": m["name"], "skipped": "no-audio-render"})
            continue
        wav = open(os.path.join(SYN, os.path.basename(m["file"])), "rb").read()
        t = Turn()
        got = [0]
        t0 = time.time()
        client.chat_turn(wav, t, True, max_tokens=512,
                         on_audio=lambda a: got.__setitem__(0, got[0] + len(a)))
        dt = time.time() - t0
        rows.append({"name": m["name"], "ttft": round(t.ttft or -1, 2),
                     "total": round(dt, 2), "audio_s": round(got[0] / 24000, 2),
                     "text": t.text[:120], "degenerate": t.degenerate,
                     "presence": got[0] > 0, "error": t.error})
        print("T1 %-12s ttft=%s total=%.1fs audio=%.1fs degen=%s %s" % (
            m["name"], t.ttft, dt, got[0] / 24000, t.degenerate,
            "OK" if got[0] > 0 else "NO-AUDIO"))
    return rows


def resample(a, sr_in, sr_out=16000):
    import numpy as np
    if sr_in == sr_out:
        return np.asarray(a, dtype=np.float32)
    idx = (np.arange(int(len(a) * sr_out / sr_in)) * sr_in / sr_out).astype(int)
    return np.asarray(a, dtype=np.float32)[np.clip(idx, 0, len(a) - 1)]


def first_speech(a, sr=16000, ms=30, thresh=0.02):
    import numpy as np
    w = int(sr * ms / 1000)
    for i in range(0, len(a) - w, w):
        if float(np.sqrt(np.mean(a[i:i + w] ** 2))) > thresh:
            return i
    return 0


def stream_assistant(client, wav, collect, stop):
    t = Turn()
    client.chat_turn(wav, t, True, max_tokens=512,
                     on_audio=lambda a: collect.append(a))
    stop.append(True)


def t2_barge(client, front, awav):
    """Assistant speaks (streamed); user barge line injected at +1.0 s,
    pumped in realtime (32 ms steps). Far-end resampled to 16 kHz.
    Latency measured from first speech energy in the barge line."""
    import numpy as np
    collect, done = [], []
    th = threading.Thread(target=stream_assistant,
                          args=(client, awav, collect, done), daemon=True)
    th.start()
    time.sleep(1.0)
    ask16 = read_wav16(os.path.join(SYN, "user_barge.wav"))
    start = first_speech(ask16)
    ask = ask16[start:]
    t0 = time.time()
    decision = None
    # pump: assistant audio = far-end (resampled), user = mic, realtime
    while not collect and time.time() - t0 < 5:
        time.sleep(0.05)
    ai = resample(np.concatenate(collect), 24000) if collect else np.zeros(0, np.float32)
    n = max(len(ask), 16000)
    steps = n // 512
    t1 = time.time()
    for i in range(steps):
        seg = ask[i * 512:(i + 1) * 512]
        if len(seg) < 512:
            seg = np.pad(seg, (0, 512 - len(seg)))
        fseg = ai[i * 512:(i + 1) * 512] if len(ai) >= (i + 1) * 512 else np.zeros(512, np.float32)
        front.push_far(fseg)
        step_t = time.time()
        sp, started, ended, clean = front.update(seg)
        if started:
            decision = (time.time() - t1) + (start / 16000)
            break
        dt = 0.032 - (time.time() - step_t)
        if dt > 0:
            time.sleep(dt)
    th.join(timeout=20)
    print("T2 barge decision latency: %s" %
          ("%.2fs" % decision if decision else "NEVER"))
    return decision


def t4_echo(front, awav):
    """Assistant-only audio as mic input (properly resampled to 16 kHz).
    The mic is delayed 20 ms vs the reference to simulate acoustic
    propagation (speaker -> air -> mic); with zero delay the AEC violates
    causality and cannot converge — an unrealistic bench-only condition.
    Must never trigger."""
    import numpy as np
    import wave
    with wave.open(awav, "rb") as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    a = resample(a, sr, 16000)
    md = int(16000 * 0.020)
    mic = np.concatenate([np.zeros(md, np.float32), (0.5 * a)[:len(a) - md]])
    trig = False
    for i in range(0, len(a) - 512, 512):
        front.push_far(a[i:i + 512])
        sp, started, ended, clean = front.update(mic[i:i + 512])
        if started:
            trig = True
            break
    print("T4 echo false-trigger: %s" % trig)
    return trig


def main():
    import numpy as np
    # Individual frontends warm before timed probes.
    client = ServerClient()
    rep = {"t1": t1_turns(client)}
    pres = [r for r in rep["t1"] if r.get("presence") is not None]
    ttfts = sorted(r["ttft"] for r in rep["t1"] if isinstance(r.get("ttft"), float) and r["ttft"] >= 0)
    rep["gates"] = {
        "presence": "%d/%d" % (sum(1 for r in pres if r["presence"]), len(pres)),
        "ttft_p50": round(ttfts[len(ttfts) // 2], 2) if ttfts else None,
        "degenerate": sum(1 for r in rep["t1"] if r.get("degenerate")),
    }
    parts = []
    probe = Turn()
    client.tts('The moon is about three hundred and eighty four thousand kilometers from Earth, and it influences our ocean tides every day.',
               probe, max_tokens=512, on_audio=parts.append)
    if probe.error:
        rep['assistant_render_error'] = probe.error
    if parts:
        a = np.concatenate(parts)
        import wave as _w
        ap = os.path.join(SYN, "_assistant.wav")
        with _w.open(ap, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())
        awav = open(os.path.join(SYN, "user_greet.wav"), "rb").read() if os.path.exists(
            os.path.join(SYN, "user_greet.wav")) else None
        if awav:
            front = DuplexFrontend()
            front.warmup()
            rep["t2_barge_latency"] = t2_barge(client, front, awav)
            echo_front = DuplexFrontend()
            echo_front.warmup()
            rep["t4_echo_fp"] = t4_echo(echo_front, ap)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rep['passed'] = bool(
        len(pres) == len(rep['t1']) and all(r['presence'] and not r.get('error') for r in pres)
        and rep['gates']['ttft_p50'] is not None and rep['gates']['ttft_p50'] < 1.2
        and rep['gates']['degenerate'] == 0
        and rep.get('t2_barge_latency') is not None and rep['t2_barge_latency'] < 1.5
        and rep.get('t4_echo_fp') is False)
    json.dump(rep, open(OUT, "w"), indent=2)
    print("GATES:", json.dumps(rep["gates"]))
    print("wrote", OUT)
    if not rep["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    from backend_session import backend_session
    with backend_session(test=True):
        main()
