"""Summarize structured application metrics without inferring user intent."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]


def summarize(paths, since=0, pid=None):
    records=[];malformed=0
    for path in paths:
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                try:
                    record=json.loads(line)
                    if not isinstance(record,dict) or not isinstance(record.get('ts'),(int,float)):raise ValueError()
                except (ValueError,TypeError):
                    malformed+=1;continue
                if record['ts']>=since and (pid is None or record.get('pid')==pid):records.append(record)
    records.sort(key=lambda item:item['ts'])
    ends=[r for r in records if r.get('event')=='playback_end']
    generations=[r for r in records if r.get('event')=='generation_end']
    health=[r for r in records if r.get('event')=='health']
    def distribution(rows,key):
        values=[float(r[key]) for r in rows if isinstance(r.get(key),(int,float)) and np.isfinite(r[key])]
        return dict(count=len(values),median=round(float(np.median(values)),3),p95=round(float(np.percentile(values,95)),3),maximum=round(max(values),3)) if values else None
    errors=[dict(ts=r['ts'],pid=r.get('pid'),event=r.get('event'),turn_id=r.get('turn_id'),error=r.get('error')) for r in records if r.get('error')]
    underflows=sum(r.get('playback',{}).get('underflows',0) for r in ends)
    # Capture counters are cumulative per process, not independent samples.
    process_counters={}
    for row in health:
        process_id=str(row.get('pid'));entry=process_counters.setdefault(process_id,dict(mic_drops=0,capture_errors=0))
        for key in entry:entry[key]=max(entry[key],row.get(key,0))
    return dict(since=since,pid=pid,events=len(records),malformed_lines=malformed,processes=len({r.get('pid') for r in records}),
                latest_event_ts=records[-1]['ts'] if records else None,
                first_audio_s=distribution(generations,'first_audio_s'),
                submit_to_play_s=distribution([r for r in records if r.get('event')=='playback_start'],'submit_to_play_s'),
                playback_completed=sum(not r.get('cancelled') and not r.get('error') for r in ends),
                playback_cancelled=sum(bool(r.get('cancelled')) for r in ends),
                playback_underflows=underflows,capture_counter_maxima=process_counters,
                clipping_observations=sum(r.get('clipped_fraction',0)>0 for r in health),
                interruptions=dict(Counter(r.get('reason','unknown') for r in records if r.get('event')=='interrupt')),
                errors=errors,
                note='Barge-in metrics do not distinguish real speech, background speech or false echo. Clipping definitions changed when unity gain was deployed. Counter maxima are not additive across device replacements.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--hours',type=float,default=1)
    parser.add_argument('--since',type=float,help='Unix timestamp; overrides --hours')
    parser.add_argument('--pid',type=int,help='Only this application process')
    parser.add_argument('--output',default=str(ROOT/'outputs/log_summary.json'));args=parser.parse_args()
    if not 0<args.hours<=168:parser.error('Use greater than zero and at most 168 hours')
    paths=list((ROOT/'outputs/logs').glob('app.jsonl*'))
    report=summarize(paths,args.since if args.since is not None else time.time()-args.hours*3600,args.pid)
    Path(args.output).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
