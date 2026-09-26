"""Verify the archive and reconstruct the complete original report, losslessly."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'manifest.json').read_text())
for name, expected in manifest['files'].items():
    raw = (root / name).read_bytes()
    assert len(raw) == expected['bytes'], f'Size mismatch: {name}'
    assert hashlib.sha256(raw).hexdigest() == expected['sha256'], f'Hash mismatch: {name}'
report = json.load(gzip.open(root / manifest['metadata_file'], 'rt'))
report['episodes'] = []
for name in manifest['episode_files']:
    report['episodes'].extend(json.load(gzip.open(root / name, 'rt'))['episodes'])
payload = {k:v for k,v in report.items() if k not in {'timestamp_utc','elapsed_seconds','provenance','timings'}}
canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
digest = hashlib.sha256(canonical(payload)).hexdigest()
assert digest == manifest['reconstructed_semantic_sha256'], 'Semantic report mismatch'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, help='Optional new .json.gz file; existing files are never replaced')
args = parser.parse_args()
if args.output:
    if not args.output.name.endswith('.json.gz'): parser.error('Output must end in .json.gz')
    with args.output.open('xb') as handle:
        handle.write(gzip.compress(canonical(report)+b'\n',mtime=0))
print(json.dumps({'episodes':len(report['episodes']),'semantic_sha256':digest,'verified':True},indent=2))
