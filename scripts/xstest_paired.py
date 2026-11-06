"""Run an isolated, blinded XSTest pilot; private outputs contain model responses.

XSTest data: Röttger et al., NAACL 2024, CC BY 4.0,
https://github.com/paul-rottger/xstest (pinned below).
Review responses against the XSTest three-class rubric before reporting scores.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import socket
import subprocess
import time
import urllib.request

REVISION = 'd7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d'
URL = f'https://raw.githubusercontent.com/paul-rottger/xstest/{REVISION}/xstest_prompts.csv'


def request(url, body=None, timeout=5):
    payload = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_cases(profile='original'):
    raw = urllib.request.urlopen(URL, timeout=30).read()
    rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    groups = sorted({row['type'] for row in rows})
    rng = random.Random(421337)
    original = [rng.choice([row for row in rows if row['type'] == group]) for group in groups]
    if profile == 'original':
        return original, hashlib.sha256(raw).hexdigest()
    if profile not in ('concise', 'expanded'):
        raise ValueError('Unknown evaluation profile')
    # Independently selected follow-up cases, distinct from the first pilot.
    rng = random.Random(99488)
    selected_groups = sorted(g for g in groups if not g.startswith('contrast_'))[:4] + sorted(g for g in groups if g.startswith('contrast_'))[:4]
    excluded_ids = {row['id'] for row in original}
    concise = [rng.choice([row for row in rows if row['type'] == g and row['id'] not in excluded_ids]) for g in selected_groups]
    if profile == 'concise':
        return concise, hashlib.sha256(raw).hexdigest()
    # Two fresh cases from every XSTest category; exclude both earlier pilots.
    excluded_ids.update(row['id'] for row in concise)
    rng = random.Random(817321)
    cases = []
    for group in groups:
        available = [row for row in rows if row['type'] == group and row['id'] not in excluded_ids]
        if len(available) < 2:
            raise ValueError('Not enough independent cases in category: ' + group)
        cases.extend(rng.sample(available, 2))
    return cases, hashlib.sha256(raw).hexdigest()


def run_arm(args, variant, model, cases, output):
    log = output / (variant + '-server.log')
    with log.open('x', encoding='utf-8') as stream:
        proc = subprocess.Popen([str(args.runner), '-m', str(model), '-mm', str(args.projector),
            '--host', '127.0.0.1', '--port', str(args.port), '-ngl', '0', '-t', '2',
            '-np', '1', '-c', '4096', '--reasoning', 'off'], stdout=stream, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            url = f'http://127.0.0.1:{args.port}'
            for _ in range(90):
                if proc.poll() is not None:
                    raise RuntimeError('Isolated model server exited; inspect private server log')
                try:
                    props = request(url + '/props')
                    if Path(props['model_path']).resolve() != model.resolve():
                        raise RuntimeError('Server reported an unexpected model')
                    break
                except (OSError, ValueError):
                    time.sleep(1)
            else:
                raise RuntimeError('Isolated server failed to become ready')
            rows = []
            for case in cases:
                row = {'id': case['id']}
                try:
                    messages = ([{'role':'system', 'content':'Please answer in two sentences.'}]
                                if args.profile in ('concise', 'expanded') else [])
                    messages.append({'role':'user','content':case['prompt']})
                    data = request(url + '/v1/chat/completions', {'model':'local',
                        'messages':messages, 'seed':42, 'temperature':0,
                        'max_tokens':512 if args.profile in ('concise', 'expanded') else 256, 'stream':False}, timeout=300)
                    choice = data['choices'][0]
                    row.update(response=choice['message'].get('content') or '', finish_reason=choice.get('finish_reason'))
                except Exception as exc:
                    row['error_type'] = type(exc).__name__
                rows.append(row)
                (output / (variant + '.json')).write_text(json.dumps(rows, indent=2), encoding='utf-8')
                print(f'{variant}: {len(rows)}/{len(cases)}', flush=True)
            return rows
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('runner', 'standard', 'research', 'projector', 'out'):
        p.add_argument('--' + flag, type=Path, required=True)
    p.add_argument('--port', type=int, default=18125)
    p.add_argument('--profile', choices=('original', 'concise', 'expanded'), default='original')
    args = p.parse_args()
    if args.port in (8080, 18087) or not 1024 <= args.port <= 65535:
        p.error('Choose a separate unprivileged evaluation port')
    # Fail before starting a server when the requested port is already occupied.
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', args.port))
    args.out.mkdir(parents=True, exist_ok=False)
    cases, digest = load_cases(args.profile)
    hashes = {name:sha256_file(path) for name, path in [('standard', args.standard),
              ('research', args.research), ('projector', args.projector), ('runner', args.runner)]}
    arms = {name: run_arm(args, name, path, cases, args.out) for name, path in
            [('standard', args.standard), ('research', args.research)]}
    if any(sha256_file(path) != hashes[name] for name, path in [('standard', args.standard),
           ('research', args.research), ('projector', args.projector), ('runner', args.runner)]):
        raise RuntimeError('A model or runner changed during evaluation')
    blind = []
    for arm, rows in arms.items():
        for case, row in zip(cases, rows):
            blind.append({'source_arm':arm, 'case_id':case['id'], 'prompt':case['prompt'],
                          'response':row.get('response'), 'finish_reason':row.get('finish_reason'),
                          'error_type':row.get('error_type')})
    random.Random(260611).shuffle(blind)
    key = {}
    for i, item in enumerate(blind, 1):
        key[str(i)] = {k:item.pop(k) for k in ('source_arm', 'case_id')}
        item['blind_id'] = i
    (args.out / 'blind.json').write_text(json.dumps(blind, indent=2), encoding='utf-8')
    (args.out / 'key.json').write_text(json.dumps(key, indent=2), encoding='utf-8')
    (args.out / 'manifest.json').write_text(json.dumps({'dataset_revision':REVISION,
        'dataset_sha256':digest, 'selection_seed':817321 if args.profile == 'expanded' else (99488 if args.profile == 'concise' else 421337), 'blind_seed':260611,
        'artifact_sha256':hashes, 'profile':args.profile,
        'max_tokens':512 if args.profile in ('concise', 'expanded') else 256, 'seed':42, 'temperature':0,
        'cases':[{'id':c['id'], 'type':c['type'], 'label':c['label']} for c in cases]}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
