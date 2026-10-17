import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from runner_selection import selected_runner
from synth.activate_abort_linux import switch


def selection(tmp_path):
    binary=tmp_path/'bin/llama-liquid-audio-server';binary.parent.mkdir();binary.write_bytes(b'candidate')
    library=tmp_path/'lib';library.mkdir();shared=library/'libggml.so';shared.write_bytes(b'library')
    record=dict(format=1,binary=str(binary),library_directory=str(library),
        files={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in [binary,shared]})
    config=tmp_path/'runner.json';config.write_text(json.dumps(record))
    return config,binary,shared,record


def test_selection_requires_intact_binary_and_libraries(tmp_path):
    config,binary,shared,record=selection(tmp_path)
    assert selected_runner(config)==(str(binary.parent.resolve()),str(shared.parent.resolve()))
    shared.write_bytes(b'changed')
    with pytest.raises(ValueError,match='file changed'):selected_runner(config)


def test_selection_rejects_omitted_and_new_runtime_files(tmp_path):
    config,binary,shared,record=selection(tmp_path)
    original=json.dumps(record)
    del record['files'][str(shared.resolve())];config.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='omits'):selected_runner(config)
    config.write_text(original);(shared.parent/'new.so').write_bytes(b'untracked runtime')
    with pytest.raises(ValueError,match='omits'):selected_runner(config)


def test_missing_selection_preserves_default_and_changed_binary_fails(tmp_path):
    assert selected_runner(tmp_path/'absent.json') is None
    config,binary,shared,record=selection(tmp_path);binary.write_bytes(b'other')
    with pytest.raises(ValueError,match='file changed'):selected_runner(config)


@pytest.mark.parametrize('previous',[None,b'{"old":"selection"}\n'])
def test_activation_rolls_back_exact_previous_selection(tmp_path,monkeypatch,previous):
    monkeypatch.setattr(Path,'home',classmethod(lambda cls:tmp_path))
    candidate,binary,shared,record=selection(tmp_path)
    (tmp_path/'candidate-selection.json').write_bytes(candidate.read_bytes())
    active=tmp_path/'.config/localvoice/runner.json';active.parent.mkdir(parents=True)
    if previous is not None:active.write_bytes(previous)
    (tmp_path/'selection-rollback.json').write_text(json.dumps(dict(path=str(active),previous_hex=previous.hex() if previous is not None else None)))
    switch(tmp_path)
    assert selected_runner(active)==(str(binary.parent.resolve()),str(shared.parent.resolve()))
    switch(tmp_path,True)
    assert (active.read_bytes() if active.exists() else None)==previous


def test_activation_refuses_to_overwrite_an_external_selection_change(tmp_path,monkeypatch):
    monkeypatch.setattr(Path,'home',classmethod(lambda cls:tmp_path))
    active=tmp_path/'.config/localvoice/runner.json';active.parent.mkdir(parents=True);active.write_bytes(b'changed elsewhere')
    (tmp_path/'candidate-selection.json').write_bytes(b'candidate')
    (tmp_path/'selection-rollback.json').write_text(json.dumps(dict(path=str(active),previous_hex=None)))
    with pytest.raises(RuntimeError,match='outside this transaction'):switch(tmp_path)
    assert active.read_bytes()==b'changed elsewhere'
