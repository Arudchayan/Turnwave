"""Run an optional faster-whisper CPU ASR screen without saving transcripts.

Install faster-whisper separately and provide a previously downloaded local
CTranslate2 checkpoint. This script neither downloads models nor touches the
live app's voice context.
"""
import argparse
import io
import json
from pathlib import Path
import time

from lfm3_realtime_asr_bench import DATASET, DATA_FILE, REVISION, score, summary
from compare_lfm_variants import file_sha256


def run(model_path, out, limit):
    if out.exists():
        raise FileExistsError('Choose a new evidence file; existing evidence is never overwritten')
    if not 1 <= limit <= 20:
        raise ValueError('The pinned screen contains 1 to 20 clips')
    from faster_whisper import WhisperModel
    from huggingface_hub import hf_hub_download
    import pandas as pd
    import soundfile as sf
    dataset = hf_hub_download(DATASET, DATA_FILE, repo_type='dataset', revision=REVISION)
    frame = pd.read_parquet(dataset).head(limit)
    if len(frame) != limit:
        raise ValueError('The pinned fixture did not contain the requested number of clips')
    model_hash = file_sha256(model_path / 'model.bin')
    model = WhisperModel(str(model_path), device='cpu', compute_type='int8', cpu_threads=2)
    report = dict(dataset=DATASET, revision=REVISION, dataset_file=DATA_FILE,
                  model_path=str(model_path), model_sha256=model_hash,
                  device='cpu', compute_type='int8', cpu_threads=2,
                  language='en', beam_size=5, vad_filter=False, rows=[], completed=False,
                  scope='Pinned clean read speech; this does not measure a particular microphone or person.')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf8') as stream:
        json.dump(report, stream, indent=2)
    for _, row in frame.iterrows():
        item = {'id': row['id']}
        started = time.monotonic()
        try:
            audio, rate = sf.read(io.BytesIO(row['audio']['bytes']), dtype='float32')
            if rate != 16000:
                raise ValueError('Expected 16 kHz dataset audio')
            segments, _ = model.transcribe(audio, language='en', beam_size=5, vad_filter=False)
            result = ' '.join(segment.text for segment in segments)
            item.update(score(row['text'], result))
            item['seconds'] = round(time.monotonic() - started, 2)
        except Exception as exc:
            item['error'] = type(exc).__name__
        report['rows'].append(item)
        report['summary'] = summary(report['rows'])
        out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
        print(f"{len(report['rows'])}/{len(frame)}", flush=True)
    if file_sha256(model_path / 'model.bin') != model_hash:
        raise RuntimeError('Model file changed during evaluation')
    report['completed'] = True
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    return report['summary']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True, type=Path, help='Previously downloaded local checkpoint')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--limit', type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(run(args.model_dir, args.out, args.limit), indent=2))


if __name__ == '__main__':
    main()
