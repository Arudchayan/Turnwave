"""End-of-turn scoring and a bounded re-query policy.

Smart Turn v3.2 is an 8M-parameter Whisper-Tiny classifier that scores the last
8 seconds of user audio. It is a research component here, not part of the
default capture path: the app still uses its silence endpoint and the reversible
floor policy. The class and policy are small enough to unit-test without a
model, which keeps the eventual integration honest.
"""
import numpy as np


class SmartTurnScorer:
    """Lazy ONNX scorer; imports heavy dependencies only when constructed."""

    def __init__(self, model_path=None, session=None, feature_extractor=None):
        if session is not None and feature_extractor is not None:
            self.session = session
            self.features = feature_extractor
            self.model_path = model_path
            return
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import WhisperFeatureExtractor
        if model_path is None:
            model_path = hf_hub_download('pipecat-ai/smart-turn-v3',
                                         'smart-turn-v3.2-cpu.onnx')
        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, sess_options=options)
        self.features = WhisperFeatureExtractor(chunk_length=8)
        self.model_path = model_path

    def score(self, audio, sample_rate=16000):
        audio = np.asarray(audio, dtype=np.float32).ravel()
        if sample_rate != 16000:
            positions = np.arange(round(len(audio) * 16000 / sample_rate)) * sample_rate / 16000
            audio = np.interp(positions, np.arange(len(audio)), audio).astype(np.float32)
        limit = 8 * 16000
        if len(audio) > limit:
            audio = audio[-limit:]
        elif len(audio) < limit:
            audio = np.pad(audio, (limit - len(audio), 0), mode='constant')
        inputs = self.features(
            audio, sampling_rate=16000, return_tensors='np',
            padding='max_length', max_length=limit, truncation=True,
            do_normalize=True)
        features = inputs.input_features.squeeze(0).astype(np.float32)[None]
        output = self.session.run(None, {'input_features': features})[0]
        return float(output[0][0])


class CompletionPolicy:
    """Re-query a completion scorer while silence lasts, then time out.

    ``update`` is driven by the app's audio clock. It returns ``None`` while
    there is nothing to decide, ``'hold'`` to keep waiting, or ``'commit'`` to
    release the turn. A scorer is queried at most once per ``step_ms`` so a
    20 ms capture loop does not run a model on every block.
    """

    def __init__(self, scorer, threshold=.7, first_threshold=.9,
                 first_query_ms=300, step_ms=200, timeout_ms=1000):
        if not 0 <= threshold <= 1 or not 0 <= first_threshold <= 1:
            raise ValueError('thresholds must be in [0, 1]')
        if first_query_ms < 0 or step_ms <= 0 or timeout_ms < first_query_ms:
            raise ValueError('invalid completion timing')
        self.scorer = scorer
        # The first query is deliberately stricter: a mid-turn pause is most
        # likely to look complete near its beginning. Later queries may use the
        # lower threshold because the silence itself is now evidence.
        self.threshold = float(threshold)
        self.first_threshold = float(first_threshold)
        self.first_query_ms = int(first_query_ms)
        self.step_ms = int(step_ms)
        self.timeout_ms = int(timeout_ms)
        self.reset()

    def reset(self):
        self._last_query_ms = None
        self.last_score = None
        self.last_query_ms = None
        self.queries = 0

    def due(self, silence_ms):
        """True when the policy would query at this silence duration."""
        silence_ms = max(0, int(silence_ms))
        if silence_ms < self.first_query_ms:
            return False
        return (self._last_query_ms is None
                or silence_ms - self._last_query_ms >= self.step_ms)

    def update(self, audio, silence_ms, speaking=False):
        if speaking:
            self.reset()
            return None
        silence_ms = max(0, int(silence_ms))
        if silence_ms < self.first_query_ms:
            return None
        if self._last_query_ms is not None and silence_ms - self._last_query_ms < self.step_ms:
            return 'hold'
        self._last_query_ms = silence_ms
        self.last_query_ms = silence_ms
        self.last_score = float(self.scorer.score(audio))
        limit = self.first_threshold if self.queries == 0 else self.threshold
        self.queries += 1
        if self.last_score >= limit:
            return 'commit'
        if silence_ms >= self.timeout_ms:
            return 'commit'
        return 'hold'

    def snapshot(self):
        return dict(threshold=self.threshold,
                    first_threshold=self.first_threshold,
                    last_score=self.last_score,
                    last_query_ms=self.last_query_ms, queries=self.queries,
                    timeout_ms=self.timeout_ms)
