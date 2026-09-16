"""Retain scalar calibration evidence without publishing calculation caches.

Copies report records, not particle arrays or snapshots. Reports from different
implementations retain their individual identities. This does not install or
promote any candidate to an active default.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error('Choose a new evidence archive path')
    rows=[]
    for path in args.reports:
        raw=path.read_bytes()
        document=json.loads(raw)
        if not isinstance(document,dict) or not isinstance(document.get('records'),list):
            parser.error(f'Expected a scalar calibration report: {path}')
        rows.append(dict(report=path.name,sha256=hashlib.sha256(raw).hexdigest(),document=document))
    output=dict(schema='assembly-illumination-evidence-v1',status='NOT_INSTALLED',
        scope='Scalar particle illumination evidence only; no wave/image acceptance',
        reports=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(f'Saved {len(rows)} scalar reports to {args.output}')


if __name__=='__main__':
    main()
