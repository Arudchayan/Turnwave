"""Explicit human-test recorder: microphone plus WASAPI speaker loopback.

Independent shared-mode capture; never resets inference or plays sound.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import queue
import threading
import time
import wave
import pyaudiowpatch as pa


def run(folder,minutes):
    folder.mkdir(parents=True,exist_ok=True)
    stop=threading.Event();chunks=queue.Queue(maxsize=2048);streams=[];writers={};counts={};drops={};statuses={};devices={}
    timeline=None
    audio=pa.PyAudio();started=time.time();deadline=time.monotonic()+minutes*60
    def write_chunk(item):
        track,data,frames,wall,monotonic,timing,status=item
        offset=counts[track]
        writers[track].writeframes(data);counts[track]+=frames
        timeline.write(json.dumps(dict(track=track,offset_frames=offset,frames=frames,wall_ts=wall,monotonic=monotonic,device_timing=timing,status=status))+'\n')
    try:
        devices={'microphone':audio.get_default_wasapi_device(d_in=True),'speaker':audio.get_default_wasapi_loopback()}
        for name,device in devices.items():
            rate=int(device['defaultSampleRate']);channels=int(device['maxInputChannels'])
            writer=wave.open(str(folder/(name+'.wav')),'wb');writer.setnchannels(channels);writer.setsampwidth(2);writer.setframerate(rate)
            writers[name]=writer;counts[name]=0;drops[name]=0;statuses[name]=0
            def callback(data,frames,timing,status,track=name):
                if status:statuses[track]+=1
                try:chunks.put_nowait((track,data,frames,time.time(),time.perf_counter(),timing,status))
                except queue.Full:drops[track]+=1
                return (None,pa.paContinue)
            stream=audio.open(format=pa.paInt16,channels=channels,rate=rate,input=True,input_device_index=int(device['index']),frames_per_buffer=1024,stream_callback=callback,start=False)
            streams.append(stream)
        metadata=dict(started_utc=datetime.now(timezone.utc).isoformat(),started_ts=started,minutes=minutes,devices=devices,scope='Raw microphone and actual system speaker loopback; can include room/system sounds. Recording begins now, not retroactively.')
        (folder/'audio-session.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        timeline=(folder/'audio-timeline.jsonl').open('a',encoding='utf-8')
        for stream in streams:stream.start_stream()
        print('RECORDING',folder,flush=True);print(json.dumps(devices),flush=True)
        last_status=0
        while time.monotonic()<deadline and not (folder/'STOP_AUDIO').exists():
            try:item=chunks.get(timeout=.5)
            except queue.Empty:continue
            write_chunk(item)
            if time.monotonic()-last_status>1:
                timeline.flush();last_status=time.monotonic()
                report=dict(observed_ts=time.time(),frames=counts,queue_drops=drops,device_status_events=statuses,seconds={name:counts[name]/int(devices[name]['defaultSampleRate']) for name in counts})
                (folder/'recording-status.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    finally:
        for stream in streams:
            stream.stop_stream();stream.close()
        while not chunks.empty():
            write_chunk(chunks.get_nowait())
        if timeline is not None:timeline.close()
        for writer in writers.values():writer.close()
        audio.terminate()
        (folder/'audio-finished.json').write_text(json.dumps(dict(finished_utc=datetime.now(timezone.utc).isoformat(),frames=counts,queue_drops=drops,device_status_events=statuses),indent=2),encoding='utf-8')
        print('Saved audio',folder,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--folder',type=Path,required=True);parser.add_argument('--minutes',type=float,default=60);args=parser.parse_args()
    if not 0<args.minutes<=120:parser.error('Use 0-120 minutes')
    if (args.folder/'audio-session.json').exists():parser.error('Choose a new folder; existing audio must not be overwritten')
    run(args.folder,args.minutes)
    from checkpoint_experiment import checkpoint
    print('Verified audio backup:',checkpoint(args.folder),flush=True)
