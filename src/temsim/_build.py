"""Wheel build isolation; imported by setuptools without the runtime package.

Keep previous build trees out of release inputs. Removed source modules and
assets must not return through an incremental build or interrupted wheel stage.
Only this command's newly allocated temporary directory is cleaned up.
"""
from pathlib import Path
from tempfile import TemporaryDirectory

from setuptools.command.bdist_wheel import bdist_wheel


class FreshWheel(bdist_wheel):
    def run(self):
        if self.skip_build:
            raise ValueError("A current-source wheel requires a fresh build; --skip-build is unsupported")
        with TemporaryDirectory(prefix="temsim-wheel-") as temporary:
            staging = Path(temporary).resolve()
            build = self.reinitialize_command("build", reinit_subcommands=True)
            build.build_base = str(staging / "build")
            # Discard cached/configured output paths, never source inputs.
            # These are finalised together so every build subcommand consumes
            # the same newly allocated library and temporary directories.
            for name in ("build_purelib", "build_platlib", "build_lib",
                         "build_scripts", "build_temp"):
                setattr(build, name, None)
            build.ensure_finalized()
            build_py = self.reinitialize_command("build_py")
            build_py.build_lib = build.build_lib
            self.bdist_dir = str(staging / "wheel")
            super().run()
