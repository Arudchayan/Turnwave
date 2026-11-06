"""Explicit, pre-started loopback LFM3 voice variants.

The model endpoint reports an identifier, not an independently verified hash.
This module neither launches nor stops a server.
"""
import json
from pathlib import PurePosixPath
import urllib.error
import urllib.parse
import urllib.request

MAX_MODELS_RESPONSE = 65536


def local_base(value):
    try:
        parsed = urllib.parse.urlsplit(value)
        port = parsed.port
    except (TypeError, ValueError):
        raise ValueError('LFM3 variant servers must be loopback HTTP URLs with a port') from None
    if (parsed.scheme != 'http' or parsed.hostname not in ('localhost', '127.0.0.1')
            or parsed.username is not None or parsed.password is not None
            or port is None or parsed.path not in ('', '/') or parsed.query or parsed.fragment):
        raise ValueError('LFM3 variant servers must be loopback HTTP URLs with a port')
    return f'http://127.0.0.1:{port}'


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class VoiceVariants:
    def __init__(self, standard_url, research_url,
                 standard_model='LFM3-Omni-2B-Q4_K_M.gguf', research_model='research-q4.gguf', opener=None):
        self.endpoints = {'standard': local_base(standard_url), 'research': local_base(research_url)}
        self.expected = {'standard': standard_model, 'research': research_model}
        if self.endpoints['standard'] == self.endpoints['research']:
            raise ValueError('Standard and research voice servers must use different endpoints')
        if any(not isinstance(x, str) or not x.strip() or len(x) > 2048 for x in self.expected.values()):
            raise ValueError('Each variant needs a nonempty expected model identifier')
        if standard_model.casefold() == research_model.casefold():
            raise ValueError('Standard and research must have distinct expected model identifiers')
        self.opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    def verify(self, variant):
        if variant not in self.endpoints:
            raise ValueError('Choose standard or research')
        request = urllib.request.Request(self.endpoints[variant] + '/v1/models', headers={'Accept': 'application/json'})
        try:
            with self.opener.open(request, timeout=2) as response:
                raw = response.read(MAX_MODELS_RESPONSE + 1)
            if len(raw) > MAX_MODELS_RESPONSE:
                raise ValueError('Model identity response was too large')
            data = json.loads(raw)
            entries = data['data']
            if not isinstance(entries, list) or len(entries) != 1 or not isinstance(entries[0], dict):
                raise ValueError('Voice server must report exactly one model')
            model_id = entries[0]['id']
            if not isinstance(model_id, str) or not model_id.strip():
                raise ValueError('Voice server did not report a model identifier')
            reported = model_id.replace(chr(92), '/')
            expected = self.expected[variant].replace(chr(92), '/')
            if ('/' in expected and reported != expected) or ('/' not in expected and PurePosixPath(reported).name != expected):
                raise ValueError(f'{variant.title()} server reported an unexpected model identifier')
            return model_id
        except (OSError, KeyError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f'{variant.title()} model identity is unavailable') from exc


class GuardedVoiceClient:
    """Check the last confirmed server-reported ID before every voice request."""
    def __init__(self, client, config, variant, model_id, on_failure):
        self._client = client
        self._config = config
        self._variant = variant
        self._model_id = model_id
        self._on_failure = on_failure

    def __getattr__(self, name):
        return getattr(self._client, name)

    def _guard(self):
        try:
            if self._config.verify(self._variant) != self._model_id:
                raise ValueError('The voice server reported a different model identifier')
        except ValueError as exc:
            self._on_failure(str(exc))
            raise RuntimeError('Voice model verification failed: ' + str(exc)) from exc

    def chat_turn(self, *args, **kwargs):
        self._guard()
        return self._client.chat_turn(*args, **kwargs)

    def chat_text(self, *args, **kwargs):
        self._guard()
        return self._client.chat_text(*args, **kwargs)

    def asr_text(self, *args, **kwargs):
        self._guard()
        return self._client.asr_text(*args, **kwargs)

    def tts(self, *args, **kwargs):
        self._guard()
        return self._client.tts(*args, **kwargs)
