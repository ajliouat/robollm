"""Load the byte-identical v1 simulator without importing today's environment.

Only the frozen module's import of ``envs.object_spawner`` is redirected to its
frozen dependency. No global ``envs`` import or module binding is replaced.
"""
from __future__ import annotations

import builtins
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

ROOT = Path(__file__).parent
REVISION = "f102fe10dc30bf2aa18cc16ed0dc43594051abbd"
EXPECTED_HASHES = {
    "multi_object_env.py": "79d2cc7ffd9e0a6900871e136d59a30e4ba4f83a03e9909d4ef886d3dc167142",
    "object_spawner.py": "893c948b024de4f80c6fb72935547b15fdedfdf51d7327bd25379f5ce12e1ba4",
    "assets/tabletop_scene.xml": "04cd5a1a7c5473017dffa8421d76b5c1318fa62d610cd2b555a0820dc978aa88",
}


def manifest() -> dict:
    result = json.loads((ROOT / "manifest.json").read_text())
    if result["source_revision"] != REVISION or set(result["files"]) != set(EXPECTED_HASHES):
        raise RuntimeError("Frozen baseline manifest does not identify the original source")
    for name, expected in EXPECTED_HASHES.items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected or result["files"][name]["sha256"] != expected:
            raise RuntimeError(f"Frozen baseline hash mismatch: {name}")
    return result


def _module(name: str, path: Path, import_function=None) -> ModuleType:
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = __name__
    module.__dict__["__builtins__"] = dict(vars(builtins))
    if import_function is not None:
        module.__dict__["__builtins__"]["__import__"] = import_function
    sys.modules[name] = module  # dataclasses resolves its defining module here.
    try:
        exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def environment_class():
    manifest()
    name = __name__ + "._environment"
    if name not in sys.modules:
        spawner = _module(__name__ + "._spawner", ROOT / "object_spawner.py")

        def frozen_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "envs.object_spawner" and level == 0:
                return spawner
            return builtins.__import__(name, globals, locals, fromlist, level)

        _module(name, ROOT / "multi_object_env.py", frozen_import)
    return sys.modules[name].MultiObjectEnv


def create_environment():
    return environment_class()(n_objects=3, render_mode=None, max_episode_steps=200)
