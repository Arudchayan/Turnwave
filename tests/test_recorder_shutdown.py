"""No audio devices: callbacks queued during shutdown must retain timestamps."""
import json
from pathlib import Path
import sys
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from synth import record_human_audio as recorder


def test_shutdown_flush_indexes_every_saved_frame(tmp_path, monkeypatch):
    class Stream:
        def __init__(self, callback):
            self.callback = callback

        def emit(self):
            self.callback(b'\x01\x00'*16, 16,
                          dict(input_buffer_adc_time=1., current_time=1.), 0)

        def start_stream(self):
            self.emit()

        def stop_stream(self):
            self.emit()

        def close(self):
            pass

    class Audio:
        def get_default_wasapi_device(self, **kwargs):
            return dict(defaultSampleRate=16000, maxInputChannels=1, index=1)

        get_default_wasapi_loopback = get_default_wasapi_device

        def open(self, **kwargs):
            return Stream(kwargs['stream_callback'])

        def terminate(self):
            pass

    monkeypatch.setattr(recorder.pa, 'PyAudio', Audio)
    # Expired deadline forces all callback audio through the shutdown drain.
    recorder.run(tmp_path, minutes=0)
    rows=[json.loads(s) for s in (tmp_path/'audio-timeline.jsonl').read_text().splitlines()]
    finished=json.loads((tmp_path/'audio-finished.json').read_text())
    for track in ['microphone', 'speaker']:
        entries=[r for r in rows if r['track']==track]
        assert [r['offset_frames'] for r in entries]==[0,16]
        assert sum(r['frames'] for r in entries)==finished['frames'][track]==32
        assert all(r['device_timing']['input_buffer_adc_time']==1. for r in entries)
        with wave.open(str(tmp_path/(track+'.wav')),'rb') as reader:
            assert reader.getnframes()==32
