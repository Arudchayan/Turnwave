"""TTFT breakdown profiler: where does first-audio latency go?

Three short prompts against the GPU server (default http://localhost:8080):
  1. wav5s : 5.0 s user audio WAV (first 5 s of data/synth/user_long.wav)
  2. wav2s : 2.06 s user audio WAV (data/synth/user_greet.wav as-is)
  3. tts_ok: pure-text TTS request ("OK", UK female voice, no audio upload)

Chat requests use modalities ["text","audio"]; TTS uses ["audio"].
Each case uses a fresh
context (reset_context=true) so the three probes are comparable.

Per case we measure (client-side clocks, seconds since request send):
  (a) upload_bytes      HTTP JSON body size (audio arrives base64'd)
  (b) t_first_any       first SSE data frame of any kind
  (c) t_first_text      first frame carrying delta.content text
  (d) t_first_audio     first frame carrying decodable audio samples
  (e) server encode ms  NOT accessible (no server log / no timing fields
                        observed in SSE deltas) -> client-side only, plus
                        first-chunk client decode ms for scale
  (f) t_total           stream open -> [DONE]/close
  (g) text_chars + audio_s (decoded output volume)

Saves outputs/ttft_profile_streaming.json and prints a table.
Run:  C:/Python312/python src/synth/ttft_profile.py
Load: 3 short probe requests. No hidden retries.
Historical buffered measurements are retained in outputs/ttft_profile.json.
"""
import base64
import json
import os
import sys
import time
import urllib.request
import wave

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, "src"))
from client import ServerClient, CHAT_SYS, TTS_VOICE, extract_audio, sse_events

SYN = os.path.join(BASE, "data", "synth")
OUT = os.path.join(BASE, "outputs", "ttft_profile_streaming.json")
BASE_URL = os.environ.get("TTFT_BASE", "http://localhost:8080")


def read_wav16_mono(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1, path
        raw = w.readframes(w.getnframes())
    import numpy as np
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0


def wav16_bytes_from_f32(samples):
    import numpy as np
    import io
    import struct
    b = (np.clip(np.asarray(samples, dtype=np.float32), -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    buf.write(b"RIFF....WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16)
              + b"data" + struct.pack("<I", len(b) * 2) + b.tobytes())
    wav = buf.getvalue()
    return wav[:4] + struct.pack("<I", len(wav) - 8) + wav[8:]


def audio_payload(wav_bytes, max_tokens=512):
    return {"model": "", "messages": [
        {"role": "system", "content": CHAT_SYS},
        {"role": "user", "content": [{
            "type": "input_audio",
            "input_audio": {"data": base64.b64encode(wav_bytes).decode(),
                            "format": "wav"}}]}],
        "stream": True, "max_tokens": max_tokens,
        "reset_context": True, "modalities": ["text", "audio"]}


def tts_payload(text, max_tokens=128):
    return {"model": "", "messages": [
        {"role": "system", "content": TTS_VOICE},
        {"role": "user", "content": text}],
        "stream": True, "max_tokens": max_tokens,
        "reset_context": True, "modalities": ["audio"]}


def profile_case(client, name, payload):
    """Measure each SSE event as it arrives, without 64 KiB read batching."""
    body = json.dumps(payload).encode()
    row = dict(case=name, upload_bytes=len(body), server_timing_exposed=None,
               retried=False, transport='read1 SSE', t_first_any=None,
               t_first_text=None, t_first_audio=None, text_chars=0,
               audio_s=0, n_frames_any=0, n_text_frames=0, n_audio_frames=0,
               first_audio_chunk_samples=0, first_chunk_client_decode_ms=None)
    samples = 0
    keys = set()
    start = time.perf_counter()
    try:
        with client.request_lock:
            with client._post(payload) as response:
                for event in sse_events(response):
                    elapsed = time.perf_counter() - start
                    if row['t_first_any'] is None:
                        row['t_first_any'] = elapsed
                    row['n_frames_any'] += 1
                    choices = event.get('choices') or []
                    if not choices:
                        continue
                    delta = choices[0].get('delta') or {}
                    keys.update('delta.' + k for k in delta)
                    keys.update(k for k in event if k != 'choices')
                    for k in ('timings', 'timing', 'metrics', 'usage'):
                        if k in delta or k in event:
                            row['server_timing_exposed'] = {k: delta.get(k, event.get(k))}
                    if delta.get('content'):
                        if row['t_first_text'] is None:
                            row['t_first_text'] = elapsed
                        row['n_text_frames'] += 1
                        row['text_chars'] += len(delta['content'])
                    decode_start = time.perf_counter()
                    audio = extract_audio(delta)
                    if audio is not None and len(audio):
                        if row['t_first_audio'] is None:
                            row['t_first_audio'] = elapsed
                            row['first_audio_chunk_samples'] = len(audio)
                            row['first_chunk_client_decode_ms'] = round((time.perf_counter() - decode_start) * 1000, 3)
                        samples += len(audio)
                        row['n_audio_frames'] += 1
    except Exception as exc:
        row['error'] = str(exc)
    row['t_total'] = round(time.perf_counter() - start, 3)
    row['audio_s'] = round(samples / 24000, 3)
    row['sse_extra_keys'] = sorted(keys)
    for key in ('t_first_any', 't_first_text', 't_first_audio'):
        if row[key] is not None:
            row[key] = round(row[key], 3)
    if row['t_first_audio'] is not None and row['t_first_any'] is not None:
        row['d_audio_after_any'] = round(row['t_first_audio'] - row['t_first_any'], 3)
    return row


def main():
    import numpy as np
    client = ServerClient(base=BASE_URL)
    long5 = read_wav16_mono(os.path.join(SYN, "user_long.wav"))[:5 * 16000]
    wav5 = wav16_bytes_from_f32(long5)
    wav2 = open(os.path.join(SYN, "user_greet.wav"), "rb").read()
    short_duration = len(read_wav16_mono(os.path.join(SYN, "user_greet.wav"))) / 16000
    cases = [("wav5s (5.0s user audio)", audio_payload(wav5)),
             ("wav_short (%.2fs user audio)" % short_duration, audio_payload(wav2)),
             ("tts_ok (text 'OK')", tts_payload("OK"))]
    rows = [profile_case(client, name, p) for name, p in cases]
    rep = {"base": client.base, "modalities": {"chat": ["text", "audio"], "tts": ["audio"]},
           "note": ("server log not accessible; no timing fields observed in "
                    "SSE deltas, so all timings are client-side. "
                    "t0 = just before request send."),
           "cases": rows}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(rep, open(OUT, "w"), indent=2)
    hdr = ("%-24s %8s %8s %8s %8s %8s %6s %7s" % (
        "case", "upl KB", "1st-any", "1st-txt", "1st-aud", "total", "chars", "aud s"))
    print(hdr)
    for r in rows:
        if "error" in r:
            print("%-24s ERROR %s" % (r["case"], r["error"]))
            continue
        print("%-24s %8.1f %8.3f %8s %8s %8.3f %6d %7.2f" % (
            r["case"], r["upload_bytes"] / 1024,
            r["t_first_any"] or -1,
            ("%.3f" % r["t_first_text"]) if r["t_first_text"] is not None else "n/a",
            ("%.3f" % r["t_first_audio"]) if r["t_first_audio"] is not None else "n/a",
            r["t_total"], r["text_chars"], r["audio_s"]))
        xtra = "  audio-after-any=%ss" % r.get("d_audio_after_any", "?")
        if "d_audio_after_text" in r:
            xtra += ", audio-after-text=%ss" % r["d_audio_after_text"]
        if r.get("retried"):
            xtra += " (zero-audio retry used)"
        print(xtra + "  first-chunk-decode=%sms/%d samples  keys=%s" % (
            r.get("first_chunk_client_decode_ms"),
            r.get("first_audio_chunk_samples"), r.get("sse_extra_keys")))
    print("wrote", OUT)
    if any(r.get('error') or not r['audio_s'] for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    from backend_session import backend_session
    with backend_session(BASE_URL, test=True):
        main()
