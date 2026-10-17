"""Create aligned copies of explicit test recordings without changing originals.

WASAPI loopback can omit idle periods. Device timestamps reconstruct those gaps
so a replay does not compress time between replies. Run after recording stops.
"""
import argparse
import json
from pathlib import Path
import wave


def align(folder):
    if not (folder/'audio-finished.json').exists():raise RuntimeError('Stop the audio recorder normally before aligning its files.')
    session=json.loads((folder/'audio-session.json').read_text(encoding='utf-8'))
    tracks={name:[] for name in session['devices']}
    for line in (folder/'audio-timeline.jsonl').read_text(encoding='utf-8').splitlines():
        event=json.loads(line);tracks[event['track']].append(event)
    report={}
    for name,events in tracks.items():
        source=folder/(name+'.wav');destination=folder/(name+'-aligned.wav')
        if destination.exists():raise FileExistsError('Aligned file already exists: '+str(destination))
        inserted=0;segments=0;overlaps=0;written=0;indexed_end=0;segment_start=0;segment_offset=0;last_device_end=None
        with wave.open(str(source),'rb') as reader,wave.open(str(destination),'wb') as writer:
            rate=reader.getframerate();channels=reader.getnchannels();width=reader.getsampwidth()
            writer.setnchannels(channels);writer.setsampwidth(width);writer.setframerate(rate)
            source_frames=reader.getnframes()
            for event in events:
                timing=event['device_timing'];adc=timing.get('input_buffer_adc_time',0);current=timing.get('current_time',0)
                # Retain sub-frame timing uncertainty in metadata. Only jump
                # to a new segment for a real gap (>50 ms), not clock jitter.
                new_segment=last_device_end is None or abs(adc-last_device_end)>.05
                if new_segment:
                    segment_start=max(0,round((event['wall_ts']-(current-adc)-session['started_ts'])*rate))
                    segment_offset=event['offset_frames'];segments+=1
                target=segment_start+event['offset_frames']-segment_offset
                if target>written:
                    remaining=target-written;inserted+=remaining
                    while remaining:
                        count=min(remaining,rate);writer.writeframesraw(bytes(count*channels*width));written+=count;remaining-=count
                reader.setpos(event['offset_frames']);data=reader.readframes(event['frames'])
                if target<written:
                    skip=min(written-target,len(data)//(channels*width));overlaps+=skip;data=data[skip*channels*width:]
                writer.writeframesraw(data);written+=len(data)//(channels*width)
                indexed_end=max(indexed_end,event['offset_frames']+event['frames'])
                last_device_end=adc+event['frames']/rate
            # The first recorder version drained its last queued frames after
            # closing the timeline. Keep those bytes and label timing as unknown.
            tail=max(0,source_frames-indexed_end)
            if tail:
                reader.setpos(indexed_end);data=reader.readframes(tail);writer.writeframesraw(data);written+=len(data)//(channels*width)
        report[name]=dict(source_frames=source_frames,aligned_frames=written,inserted_silence_frames=inserted,
                          segments=segments,overlap_frames_trimmed=overlaps,unindexed_tail_frames=tail,
                          rate=rate,channels=channels,output=destination.name,
                          timing_note='Device/callback timestamps approximate capture time; this is not a calibrated physical latency measurement.')
    (folder/'alignment.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path);args=parser.parse_args()
    print(json.dumps(align(args.folder),indent=2))
