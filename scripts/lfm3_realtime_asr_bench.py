"""Measure the app's realtime ASR on a small, pinned public speech fixture.

Use a *dedicated* local server, never an app's active conversation endpoint.
This reports numerical errors and timings, not references or transcriptions.
It is a clean read-speech screen, not an evaluation of a particular person's voice.
"""
import argparse
import io
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from backend_session import backend_session
from compare_lfm_variants import file_sha256, validate_server
from lfm3_client import LFM3Client
from synth.text_metrics import edit_distance, normalize_text

DATASET = 'hf-internal-testing/librispeech_asr_dummy'
REVISION = '5be91486e11a2d616f4ec5db8d3fd248585ac07a'
DATA_FILE = 'clean/validation-00000-of-00001.parquet'


def score(reference, hypothesis):
    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    return {'words': len(ref), 'edits': edit_distance(ref, hyp), 'empty': not bool(hyp)}


def summary(rows):
    ok = [row for row in rows if 'error' not in row]
    words = sum(row['words'] for row in ok)
    return {'clips': len(rows), 'completed': len(ok), 'errors': len(rows)-len(ok),
            'reference_words': words, 'word_edits': sum(row['edits'] for row in ok),
            'corpus_wer': sum(row['edits'] for row in ok) / words if words else None,
            'sentences_with_errors': sum(row['edits'] > 0 for row in ok),
            'empty': sum(row['empty'] for row in ok)}


def run(url, model, model_path, out, limit):
    if out.exists():
        raise FileExistsError('Choose a new evidence file; existing evidence is never overwritten')
    from huggingface_hub import hf_hub_download
    import pandas as pd
    import soundfile as sf
    from lfm3_asr_bench import wav16k
    if not 1 <= limit <= 20:
        raise ValueError('The pinned screen contains 1 to 20 clips')
    dataset = hf_hub_download(DATASET, DATA_FILE, repo_type='dataset', revision=REVISION)
    frame = pd.read_parquet(dataset).head(limit)
    if len(frame) != limit:
        raise ValueError('The pinned fixture did not contain the requested number of clips')
    before = file_sha256(model)
    with backend_session(url):
        validate_server(url, model_path)
        report = dict(dataset=DATASET, revision=REVISION, dataset_file=DATA_FILE,
                      model_sha256=before, server_model_path=model_path, rows=[], completed=False,
                      scope='Pinned clean read speech; this does not measure a particular microphone or person.')
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('x', encoding='utf8') as stream:
            json.dump(report, stream, indent=2)
        client = LFM3Client(url, timeout=180)
        try:
            for _, row in frame.iterrows():
                item = {'id': row['id']}
                started = time.monotonic()
                try:
                    audio, rate = sf.read(io.BytesIO(row['audio']['bytes']), dtype='float32')
                    result = client.asr_text(wav16k(audio, rate))
                    item.update(score(row['text'], result))
                    item['seconds'] = round(time.monotonic() - started, 2)
                except Exception as exc:
                    item['error'] = type(exc).__name__
                report['rows'].append(item)
                report['summary'] = summary(report['rows'])
                out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
                print(f"{len(report['rows'])}/{len(frame)}", flush=True)
        finally:
            client.close()
        validate_server(url, model_path)
        if file_sha256(model) != before:
            raise RuntimeError('Model file changed during evaluation')
        report['completed'] = True
        out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
        return report['summary']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--model', required=True, type=Path, help='Local filesystem path to hash')
    parser.add_argument('--server-model-path', required=True, help='Exact model path reported by /props')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--limit', type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(run(args.url, args.model, args.server_model_path, args.out, args.limit), indent=2))


if __name__ == '__main__':
    main()
