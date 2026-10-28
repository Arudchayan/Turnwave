"""Small public ASR check for LFM3-Audio-2B.

Uses the LibriSpeech dummy fixture through a local llama.cpp server. This is a
quality check, not a full LibriSpeech benchmark: 20 clean read-speech clips,
one quantization, one machine.
"""
import argparse
import base64
import io
import json
import pathlib
import re
import sys
import time
import urllib.request
import wave

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def wav16k(samples, rate):
    if rate != 16000:
        positions = np.arange(round(len(samples) * 16000 / rate)) * rate / 16000
        samples = np.interp(positions, np.arange(len(samples)), samples)
    pcm = (np.clip(samples, -1, 1) * 32767).astype('<i2').tobytes()
    buf = io.BytesIO()
    buf.write(b'RIFF' + (36 + len(pcm)).to_bytes(4, 'little') + b'WAVEfmt ')
    buf.write((16).to_bytes(4, 'little') + (1).to_bytes(2, 'little')
              + (1).to_bytes(2, 'little') + (16000).to_bytes(4, 'little')
              + (32000).to_bytes(4, 'little') + (2).to_bytes(2, 'little')
              + (16).to_bytes(2, 'little'))
    buf.write(b'data' + len(pcm).to_bytes(4, 'little') + pcm)
    return buf.getvalue()


def normalize(text):
    return re.sub(r'[^A-Z0-9 ]+', ' ', text.upper()).split()


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, hyp in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (ref != hyp)))
        previous = current
    return previous[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8091')
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--out', default=str(ROOT / 'outputs' / 'lfm3_asr_bench.json'))
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download
    path = hf_hub_download('hf-internal-testing/librispeech_asr_dummy',
                           'clean/validation-00000-of-00001.parquet',
                           repo_type='dataset')
    frame = pd.read_parquet(path).head(args.limit)
    rows = []
    total_edits = total_words = 0
    for _, row in frame.iterrows():
        audio, rate = sf.read(io.BytesIO(row['audio']['bytes']), dtype='float32')
        wav = wav16k(audio, rate)
        payload = {
            'model': '',
            'messages': [
                {'role': 'system', 'content': 'Perform ASR in English.'},
                {'role': 'user', 'content': [{'type': 'input_audio',
                                              'input_audio': {
                                                  'data': base64.b64encode(wav).decode(),
                                                  'format': 'wav'}}]},
            ],
            'stream': False,
            'max_tokens': 128,
            'temperature': 0,
            'modalities': ['text'],
        }
        request = urllib.request.Request(
            args.url.rstrip('/') + '/v1/chat/completions',
            data=json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json'})
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=300) as response:
            body = json.load(response)
        hypothesis = (body.get('choices') or [{}])[0].get('message', {}).get('content', '')
        reference = row['text']
        ref_words, hyp_words = normalize(reference), normalize(hypothesis)
        edits = edit_distance(ref_words, hyp_words)
        total_edits += edits
        total_words += len(ref_words)
        rows.append(dict(id=row['id'], reference=reference, hypothesis=hypothesis,
                         words=len(ref_words), edits=edits,
                         wer=round(edits / max(1, len(ref_words)), 4),
                         elapsed_s=round(time.perf_counter() - started, 3)))
        print(json.dumps(rows[-1]), flush=True)
    report = dict(
        schema_version=1,
        evidence_class='small public read-speech ASR check; one quantization and machine',
        dataset=dict(repo='hf-internal-testing/librispeech_asr_dummy',
                     file='clean/validation-00000-of-00001.parquet',
                     clips=len(rows)),
        model=dict(repo='LiquidAI/LFM3-Audio-2B-GGUF',
                   file='LFM3-Audio-2B-Q4_K_M.gguf',
                   quantization='Q4_K_M', backend='llama.cpp b11960 CPU'),
        corpus_wer=round(total_edits / max(1, total_words), 4),
        sentence_error_rate=round(sum(row['edits'] > 0 for row in rows)
                                  / max(1, len(rows)), 4),
        rows=rows,
        limits=('Twenty clean LibriSpeech dummy clips, read speech, one speaker '
                'subset, Q4_K_M on CPU. Not a full LibriSpeech or noisy-speech '
                'benchmark. Punctuation/case are normalized.'),
    )
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8',
                   newline='\n')
    print(json.dumps(dict(out=str(out), corpus_wer=report['corpus_wer'],
                          sentence_error_rate=report['sentence_error_rate']),
                     indent=2))


if __name__ == '__main__':
    main()
