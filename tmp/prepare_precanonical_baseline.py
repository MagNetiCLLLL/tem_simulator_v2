from pathlib import Path
import subprocess
out=Path("tmp/precanonical_baseline")
out.mkdir(exist_ok=True)
for source,name in (("src/temsim/physics/core.py","core.py"),("src/temsim/optics/direct_alignment.py","direct_alignment.py")):
    data=subprocess.run(["git","show","HEAD:"+source],capture_output=True,check=True).stdout
    (out/name).write_bytes(data)
