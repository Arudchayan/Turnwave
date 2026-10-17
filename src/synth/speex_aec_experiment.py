"""Offline SpeexDSP linear-cancellation comparison, isolated from the app."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from duplex import DuplexFrontend
from synth.archive_experiment import capture_files
from synth.checkpoint_experiment import checkpoint


class SpeexAEC:
    def __init__(self, dll):
        self.dll = ctypes.CDLL(str(dll.resolve()))
        ptr = ctypes.POINTER(ctypes.c_int16)
        self.dll.speex_echo_state_init.argtypes = [ctypes.c_int, ctypes.c_int]
        self.dll.speex_echo_state_init.restype = ctypes.c_void_p
        self.dll.speex_echo_cancellation.argtypes = [ctypes.c_void_p, ptr, ptr, ptr]
        self.dll.speex_echo_cancellation.restype = None
        self.dll.speex_echo_ctl.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self.dll.speex_echo_ctl.restype = ctypes.c_int
        self.dll.speex_echo_state_destroy.argtypes = [ctypes.c_void_p]
        self.dll.speex_echo_state_destroy.restype = None
        self.state = self.dll.speex_echo_state_init(160, 8000)
        if not self.state: raise MemoryError('Speex echo state allocation failed')
        rate = ctypes.c_int(16000)
        if self.dll.speex_echo_ctl(self.state, 24, ctypes.byref(rate)):
            self.close(); raise RuntimeError('Speex sampling-rate configuration failed')
        self.clean_history = np.zeros(0, np.float32)
        self.records = []
        self.costs = []
        self.stream_delay_ms = 0  # No external-delay API; the 500 ms filter learns it.

    def process(self, near, far):
        if len(near) != 160 or len(far) != 160: raise ValueError('Speex requires 10 ms frames')
        before = time.perf_counter()
        microphone = np.rint(np.clip(near, -1, 32767/32768)*32768).astype(np.int16)
        render = np.rint(np.clip(far, -1, 32767/32768)*32768).astype(np.int16)
        output = np.zeros(160, np.int16)
        pointer = ctypes.POINTER(ctypes.c_int16)
        self.dll.speex_echo_cancellation(self.state, microphone.ctypes.data_as(pointer),
            render.ctypes.data_as(pointer), output.ctypes.data_as(pointer))
        clean = output.astype(np.float32)/32768
        self.costs.append(time.perf_counter()-before)
        self.clean_history = np.concatenate((self.clean_history, clean))[-1600:]
        self.records.append((near.copy(), far.copy(), clean.copy()))
        return clean

    def close(self):
        if getattr(self, 'state', None):
            self.dll.speex_echo_state_destroy(self.state)
            self.state = None

    def __del__(self):
        self.close()


def run(folder, human):
    from synth import acoustic_stress, replay_human_acoustics
    if (folder/'acoustic.json').exists(): raise FileExistsError('Preserve previous results')
    with (folder/'experiment-started.json').open('x') as marker: marker.write('{}\n')
    capture_files(folder/'source')
    fixtures=folder/'fixtures'; fixtures.mkdir(exist_ok=False)
    for name in ['_assistant.wav', 'user_barge.wav']:
        shutil.copy2(ROOT/'data/synth'/name, fixtures/name)
    instances=[]
    class Candidate(DuplexFrontend):
        def __init__(self, **kwargs):
            kwargs['aec']=False
            super().__init__(**kwargs)
            self.aec=SpeexAEC(folder/'speexdsp.dll'); instances.append(self.aec)
        def _echo_similarity(self):
            return DuplexFrontend._echo_similarity(SimpleNamespace(
                _near_history=self.aec.clean_history, _far_history=self._far_history))
    acoustic_stress.DuplexFrontend=Candidate
    replay_human_acoustics.DuplexFrontend=Candidate
    try:
        acoustic_stress.run(folder/'acoustic.json')
        replay_human_acoustics.main(human, filename='h3.json', output_folder=folder)
    finally:
        # Preserve actual AEC inputs and output, including failed cases, losslessly.
        for index, instance in enumerate(instances):
            if instance.records:
                arrays=[np.concatenate([record[column] for record in instance.records]) for column in range(3)]
                np.savez_compressed(folder/f'aec-{index:02d}.npz', near=arrays[0], far=arrays[1], clean=arrays[2])
                with wave.open(str(folder/f'aec-{index:02d}-clean.wav'), 'wb') as writer:
                    writer.setnchannels(1); writer.setsampwidth(2); writer.setframerate(16000)
                    writer.writeframes(np.rint(arrays[2]*32768).astype('<i2').tobytes())
            instance.close()
    acoustic=json.loads((folder/'acoustic.json').read_text()); h3=json.loads((folder/'h3.json').read_text())
    costs=np.array([value for instance in instances for value in instance.costs])
    passes=sum(row['passed'] for row in acoustic['cases'])
    false=sum(len(row['quiet_h3_onsets']) for row in h3['cases'] if row['residual_floor']==.005)
    report=dict(acoustic_passed=passes, acoustic_total=len(acoustic['cases']), guarded_h3_false_onsets=false,
        decision='rejected; not deployed' if passes!=len(acoustic['cases']) or false else 'preliminary pass; held-out validation required',
        policy='SpeexDSP 1.2.1, 10 ms frames, 500 ms filter, cleaned-signal correlation; existing residual guard.',
        frame_p50_ms=float(np.median(costs)*1000), frame_p95_ms=float(np.quantile(costs,.95)*1000),
        dll_sha256=hashlib.sha256((folder/'speexdsp.dll').read_bytes()).hexdigest(), deployed=False,
        recording_index='AEC files 00-21 follow acoustic.cases order; 22-25 follow h3.cases order.',
        limitation='Seeded synthetic acoustics and an offline recorded quiet control; no live-room validation.')
    (folder/'decision.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('folder',type=Path); parser.add_argument('human',type=Path); args=parser.parse_args()
    if (args.folder/'acoustic.json').exists() or (args.folder/'experiment-started.json').exists(): parser.error('Use a new experiment folder')
    try:
        run(args.folder,args.human)
    except Exception:
        import traceback
        (args.folder/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        print('Verified backup:',checkpoint(args.folder),flush=True)
