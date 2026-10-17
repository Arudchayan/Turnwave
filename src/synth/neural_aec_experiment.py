"""Offline DTLN-AEC comparison; never installs an AEC into the running app.

Two-stage inference follows upstream run_aec.py (MIT, saved with each model).
The streaming adapter uses 128-sample hops, 512-sample windows and a 128-sample
output reserve. Its nominal alignment delay is 512 samples / 32 ms at 16 kHz.
This delay is retained in detection measurements rather than compensated away.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from duplex import DuplexFrontend


class NeuralAEC:
    def __init__(self, models):
        from ai_edge_litert.interpreter import Interpreter
        self.models = [Interpreter(model_path=str(models / f'dtln_aec_128_{i}.tflite'),
                                   num_threads=1) for i in (1, 2)]
        for model in self.models:
            model.allocate_tensors()
        self.inputs = [model.get_input_details() for model in self.models]
        self.outputs = [model.get_output_details() for model in self.models]
        self.states = [np.zeros(details[1]['shape'], np.float32) for details in self.inputs]
        self.near = np.zeros(0, np.float32)
        self.far = np.zeros(0, np.float32)
        self.near_window = np.zeros(512, np.float32)
        self.far_window = np.zeros(512, np.float32)
        self.overlap = np.zeros(512, np.float32)
        self.ready = np.zeros(128, np.float32)
        self.hop_seconds = []
        self.sample_count = 0
        self.stream_delay_ms = 0
        # Match the upstream file runner's leading 384 zero samples, then
        # discard that prefix. Keep only the streaming output reserve.
        self.process(np.zeros(384, np.float32), np.zeros(384, np.float32))
        self.ready = np.zeros(128, np.float32)
        self.hop_seconds = []
        self.sample_count = 0

    def _infer(self, stage, signal, far):
        model, inputs, outputs = self.models[stage], self.inputs[stage], self.outputs[stage]
        for detail, value in zip(inputs, (signal, self.states[stage], far)):
            model.set_tensor(detail['index'], value)
        model.invoke()
        self.states[stage] = model.get_tensor(outputs[1]['index'])
        return model.get_tensor(outputs[0]['index'])

    def process(self, near, far):
        count = len(near)
        self.sample_count += count
        self.near = np.concatenate((self.near, near))
        self.far = np.concatenate((self.far, far))
        while len(self.near) >= 128:
            started = time.perf_counter()
            self.near_window[:-128] = self.near_window[128:]
            self.near_window[-128:] = self.near[:128]
            self.far_window[:-128] = self.far_window[128:]
            self.far_window[-128:] = self.far[:128]
            self.near, self.far = self.near[128:], self.far[128:]
            spectrum = np.fft.rfft(self.near_window).astype(np.complex64)
            far_spectrum = np.fft.rfft(self.far_window).astype(np.complex64)
            magnitude = np.abs(spectrum).reshape(1, 1, -1).astype(np.float32)
            far_magnitude = np.abs(far_spectrum).reshape(1, 1, -1).astype(np.float32)
            mask = self._infer(0, magnitude, far_magnitude)
            estimate = np.fft.irfft(spectrum * mask).reshape(1, 1, -1).astype(np.float32)
            block = self._infer(1, estimate, self.far_window.reshape(1, 1, -1))
            self.overlap[:-128] = self.overlap[128:]
            self.overlap[-128:] = 0
            self.overlap += block.ravel()
            self.ready = np.concatenate((self.ready, self.overlap[:128]))
            self.hop_seconds.append(time.perf_counter() - started)
        if len(self.ready) < count:
            raise RuntimeError('Streaming adapter output underrun')
        result, self.ready = self.ready[:count].copy(), self.ready[count:]
        return result


def verify_adapter(folder):
    """Numerically compare with the saved, unmodified upstream file runner."""
    import importlib.util
    import types
    import soundfile as sf
    from ai_edge_litert.interpreter import Interpreter
    from synth.acoustic_stress import read
    tensorflow = types.ModuleType('tensorflow')
    lite = types.ModuleType('tensorflow.lite')
    lite.Interpreter = Interpreter
    tensorflow.lite = lite
    sys.modules['tensorflow'] = tensorflow
    sys.modules['tensorflow.lite'] = lite
    spec = importlib.util.spec_from_file_location('upstream_aec', folder / 'upstream/run_aec.py')
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    target = folder / 'adapter-verification'
    target.mkdir(exist_ok=False)
    far = np.tile(read('_assistant.wav'), 2)[:48000]
    near = .3 * np.roll(far, 1280)
    user = read('user_barge.wav')
    count = min(len(user), len(near) - 16000)
    near[16000:16000+count] += .3 * user[:count]
    sf.write(target / 'test_mic.wav', near, 16000)
    sf.write(target / 'test_lpb.wav', far, 16000)
    near, _ = sf.read(target / 'test_mic.wav', dtype='float32')
    far, _ = sf.read(target / 'test_lpb.wav', dtype='float32')
    models = folder / 'upstream/pretrained_models'
    interpreters = [Interpreter(model_path=str(models / f'dtln_aec_128_{i}.tflite'), num_threads=1)
                    for i in (1, 2)]
    for interpreter in interpreters:
        interpreter.allocate_tensors()
    upstream.process_file(*interpreters, str(target / 'test_mic.wav'), str(target / 'upstream.wav'))
    reference, _ = sf.read(target / 'upstream.wav', dtype='float32')
    adapter = NeuralAEC(models)
    near = np.pad(near, (0, 640)); far = np.pad(far, (0, 640))
    chunks = [adapter.process(near[i:i+160], far[i:i+160]) for i in range(0, len(near), 160)]
    streamed = np.concatenate(chunks)[128:128+len(reference)]
    sf.write(target / 'streamed-aligned.wav', streamed, 16000)
    error = float(np.max(np.abs(streamed - reference)))
    report = dict(passed=error <= 1/32768 + 1e-6, max_absolute_error=error,
                  tolerance=1/32768 + 1e-6, output_reserve_samples=128,
                  note='Compared with unmodified upstream runner; tolerance allows its PCM16 output quantization.')
    (target / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Adapter verification:', json.dumps(report), flush=True)
    if not report['passed']:
        raise RuntimeError('Streaming adapter differs from the upstream reference')


def main(folder, human):
    from synth import acoustic_stress, replay_human_acoustics
    from synth.archive_experiment import capture_files

    models = folder / 'upstream/pretrained_models'
    instances = []

    class Candidate(DuplexFrontend):
        def __init__(self, **kwargs):
            kwargs['aec'] = False
            super().__init__(**kwargs)
            self.aec = NeuralAEC(models)
            instances.append(self.aec)

        def _echo_similarity(self):
            # Separate experimental policy: trust neural residual suppression.
            # Raw correlation blocks quiet speech despite successful AEC.
            return 0.0

    # Patches affect only this offline process, never the running UI.
    acoustic_stress.DuplexFrontend = Candidate
    replay_human_acoustics.DuplexFrontend = Candidate
    if (folder / 'acoustic.json').exists():
        raise FileExistsError('Preserve earlier results; use a new experiment directory')
    capture_files(folder / 'source')
    verify_adapter(folder)
    acoustic_stress.run(folder / 'acoustic.json')
    replay_human_acoustics.main(human, filename='h3.json', output_folder=folder)
    hops = np.array([value for instance in instances for value in instance.hop_seconds])
    metrics = dict(hops=len(hops), audio_seconds=sum(i.sample_count for i in instances) / 16000,
                   inference_seconds=float(hops.sum()), hop_p50_ms=float(np.median(hops) * 1000),
                   hop_p95_ms=float(np.quantile(hops, .95) * 1000),
                   hop_max_ms=float(hops.max() * 1000), hop_budget_ms=8,
                   nominal_streaming_delay_ms=32, deployed=False,
                   policy='DTLN-AEC 128, no raw correlation veto; existing residual floor retained',
                   limitation='Offline recordings and seeded mixtures; no claim of live-room performance.')
    from importlib.metadata import version
    metrics['packages'] = {name: version(name) for name in ('ai-edge-litert', 'soundfile', 'numpy')}
    (folder / 'runtime.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('human', type=Path)
    args = parser.parse_args()
    # Failed runs are evidence too. Save the traceback before making the copy.
    # Refuse an existing result without modifying or recopying that experiment.
    if (args.folder / 'acoustic.json').exists():
        parser.error('Use a new experiment folder; earlier results are preserved')
    try:
        main(args.folder, args.human)
    except Exception:
        import traceback
        (args.folder / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    finally:
        from synth.checkpoint_experiment import checkpoint
        print('Verified backup:', checkpoint(args.folder), flush=True)
