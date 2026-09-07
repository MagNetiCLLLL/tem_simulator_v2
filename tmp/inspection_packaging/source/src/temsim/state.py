from temsim.detector.recording_system import restore_recording_system

from temsim.detector.recording_system import ensure_recording_system

import json

from pathlib import Path

from temsim.optics.model import State

def save(s,p):Path(p).write_text(json.dumps(s.to_dict(),indent=2),encoding='utf-8')

def load(p):return ensure_recording_system(State.from_dict(json.loads(Path(p).read_text(encoding='utf-8'))))
