"""Build private before/after server executables without rebuilding live libraries.

Run inside WSL. Runtime tests set LOCALVOICE_ABORT_PROBE_FAIL_ONCE to exercise
the real failed-write branch exactly once. That identical transport test hook
exists in both executables and is separate from the candidate stop fix.
"""
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run(private,output):
    native=(Path.home()/'liquid-build');build=native/'build';cwd=build/'tools/liquid-audio'
    private.mkdir(parents=True,exist_ok=False);(private/'lib').mkdir();(private/'sources').mkdir()
    snapshot=output/'source-input'
    expected=json.loads((snapshot/'summary.json').read_text())['native_source_sha256']
    if digest(native/'tools/liquid-audio/server.cpp')!=expected:raise RuntimeError('Native source changed since the captured regression probe')
    manifest=[]
    def copy_file(path,destination):
        shutil.copy2(path,destination);manifest.append(dict(source=str(path),copy=str(destination.relative_to(private)),sha256=digest(destination)))
    for path in (build/'bin').glob('*.so*'):
        real=path.resolve();destination=private/'lib'/real.name
        if not destination.exists():copy_file(real,destination)
        alias=private/'lib'/path.name
        if not alias.exists():alias.symlink_to(real.name)
    for name in ['runner.h','runner.cpp']:
        copy_file(native/'tools/liquid-audio'/name,private/'sources'/name)
    # Retain the actual dirty tracked sources, including the existing fixes.
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=native).decode().split('\0')
    with tarfile.open(private/'native-source.tar.gz','x:gz') as archive:
        for name in tracked:
            if name and (native/name).is_file():archive.add(native/name,arcname=name,recursive=False)
    (private/'native-working.patch').write_bytes(subprocess.check_output(['git','diff','--binary','HEAD'],cwd=native))
    (private/'native-head.txt').write_bytes(subprocess.check_output(['git','rev-parse','HEAD'],cwd=native))
    flags=(cwd/'CMakeFiles/llama-liquid-audio-server.dir/flags.make').read_text()
    values={line.split(' = ',1)[0]:line.split(' = ',1)[1] for line in flags.splitlines() if ' = ' in line}
    compile_flags=sum((shlex.split(values[key]) for key in ['CXX_DEFINES','CXX_INCLUDES','CXX_FLAGS']),[])
    link=shlex.split((cwd/'CMakeFiles/llama-liquid-audio-server.dir/link.txt').read_text())
    commands=[]
    for version in ['before','after']:
        destination=private/version;destination.mkdir()
        server=(snapshot/f'server-{version}.cpp').read_text()
        needle='if (!sink.write(data.c_str(), data.size())) {'
        if server.count(needle)!=1:raise RuntimeError('Expected one stream write error branch')
        injected='''static std::atomic<bool> localvoice_probe_fail_once{true};
                        const bool localvoice_probe_injected = std::getenv("LOCALVOICE_ABORT_PROBE_FAIL_ONCE") != nullptr && localvoice_probe_fail_once.exchange(false);
                        if (localvoice_probe_injected || !sink.write(data.c_str(), data.size())) {
                            if (localvoice_probe_injected) { LOG_INF("LOCALVOICE_PROBE: injecting one failed stream write\\n"); }'''
        server=server.replace(needle,injected)
        (destination/'server.cpp').write_text('#include <cstdlib>\n'+server)
        command=['/usr/bin/c++','-I'+str(private/'sources'),*compile_flags,'-c',str(destination/'server.cpp'),'-o',str(destination/'server.o')]
        commands.append(command)
        with (destination/'compile.log').open('w') as log:subprocess.run(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,check=True)
        command=[];skip=False
        for index,arg in enumerate(link):
            if skip:skip=False;continue
            if arg=='-o':command+=['-o',str(destination/'llama-liquid-audio-server')];skip=True
            elif arg.endswith('server.cpp.o'):command.append(str(destination/'server.o'))
            elif arg.startswith('-Wl,-rpath,'):command.append('-Wl,-rpath,'+str(private/'lib'))
            elif arg.endswith('.a'):
                path=(cwd/arg).resolve();target=private/'lib'/path.name
                if not target.exists():copy_file(path,target)
                command.append(str(target))
            elif '.so' in arg and not arg.startswith('-'):command.append(str(private/'lib'/Path(arg).name))
            else:command.append(arg)
        commands.append(command)
        with (destination/'link.log').open('w') as log:subprocess.run(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,check=True)
        manifest.append(dict(source='private build',copy=str((destination/'llama-liquid-audio-server').relative_to(private)),sha256=digest(destination/'llama-liquid-audio-server')))
    (private/'manifest.json').write_text(json.dumps(dict(files=manifest,commands=commands,source_sha256=expected,
        native_checkout_modified=False,live_libraries_rebuilt=False,
        injection='Identical test-only failed-write branch trigger in both builds. First emitted stream chunk is rejected once per process when the explicit environment variable is present.'),indent=2))
    print('Private before/after builds complete:',private,flush=True)


def clean_build(private,output,lifecycle=False):
    destination=private/('lifecycle' if lifecycle else 'clean');destination.mkdir(exist_ok=False)
    source=output/'source-input/server-after.cpp'
    data=source.read_bytes()
    if lifecycle:
        text=data.decode()
        old='auto check_abort = [&req, output, &runner, &is_server_running]() {'
        if text.count(old)!=1:raise RuntimeError('Expected request-reference capture')
        text=text.replace(old,'auto connection_closed = req.is_connection_closed;\n            auto check_abort = [connection_closed, output, &runner, &is_server_running]() {')
        old='if (!should_abort && req.is_connection_closed) {\n                    should_abort = req.is_connection_closed();'
        if text.count(old)!=1:raise RuntimeError('Expected connection check')
        text=text.replace(old,'if (!should_abort && connection_closed) {\n                    should_abort = connection_closed();')
        old='return true;\n                });'
        if text.count(old)!=1:raise RuntimeError('Expected streaming provider end')
        text=text.replace(old,'''return true;
                }, [output](bool) {
                    // Response teardown can follow a disconnect without another
                    // provider call. Notify only this work item; its worker will
                    // stop the runner, avoiding a late stop of a newer request.
                    if (!output->done.load()) {
                        if (!output->aborted.exchange(true)) {
                            LOG_INF("Aborting unfinished response\\n");
                        }
                        output->cv.notify_all();
                    }
                });''')
        data=text.encode()
    if b'LOCALVOICE_ABORT_PROBE' in data or b'LOCALVOICE_PROBE' in data:raise RuntimeError('Clean source contains a transport test hook')
    (destination/'server.cpp').write_bytes(data)
    manifest=json.loads((private/'manifest.json').read_text())
    commands=[[arg.replace(str(private/'after')+'/',str(destination)+'/') for arg in command] for command in manifest['commands'][-2:]]
    for index,command in enumerate(commands):
        with (destination/f'build-{index}.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    binary=destination/'llama-liquid-audio-server'
    if b'LOCALVOICE_ABORT_PROBE_FAIL_ONCE' in binary.read_bytes():raise RuntimeError('Clean binary contains the injection environment hook')
    report=dict(source_sha256=digest(destination/'server.cpp'),binary_sha256=digest(binary),commands=commands,
        transport_test_hook=False,lifecycle_cleanup=lifecycle,live_checkout_modified=False,libraries_manifest_sha256=digest(private/'manifest.json'))
    (destination/'manifest.json').write_text(json.dumps(report,indent=2));shutil.copy2(destination/'manifest.json',output/'clean-manifest.json')
    print('Clean candidate built without fault injection:',binary,flush=True)


if __name__=='__main__':
    private=Path(sys.argv[1]);output=Path(sys.argv[2])
    try:
        if '--clean' in sys.argv[3:] or '--lifecycle' in sys.argv[3:]:clean_build(private,output,'--lifecycle' in sys.argv[3:])
        else:run(private,output)
    except Exception:
        import traceback
        (output/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        if private.exists():
            with tarfile.open(output/'runtime-build.tar.gz','x:gz') as archive:archive.add(private,arcname='runtime')
            if (private/'manifest.json').exists():shutil.copy2(private/'manifest.json',output/'manifest.json')
