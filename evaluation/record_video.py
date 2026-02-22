"""
Video/GIF recorder for demo episodes.

Captures RGB frames from MuJoCo overhead camera and saves as
animated GIF using PIL.  No ffmpeg dependency required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _try_import_pil():
    """Import PIL, return None if unavailable."""
    try:
        from PIL import Image
        return Image
    except ImportError:
        return None


def record_episode(
    env: Any,
    policy_fn: Any | None = None,
    seed: int = 42,
    max_steps: int = 200,
) -> tuple[list[np.ndarray], dict]:
    """Record one episode, returning frames and final info.

    Args:
        env: Gymnasium env with render_mode="rgb_array".
        policy_fn: Callable(obs, info) → action. None = random.
        seed: Reset seed.
        max_steps: Safety limit.

    Returns:
        (frames, info): list of RGB numpy arrays, final info dict.
    """
    frames: list[np.ndarray] = []
    obs, info = env.reset(seed=seed)

    # Capture initial frame
    frame = env.render()
    if frame is not None:
        frames.append(frame.copy() if isinstance(frame, np.ndarray) else frame)

    for step in range(max_steps):
        if policy_fn is not None:
            if callable(policy_fn):
                action = policy_fn(obs, info)
            elif hasattr(policy_fn, "act"):
                action = policy_fn.act(info)
            else:
                action = env.action_space.sample()
        else:
            action = env.action_space.sample()

        obs, reward, term, trunc, info = env.step(action)

        frame = env.render()
        if frame is not None:
            frames.append(frame.copy() if isinstance(frame, np.ndarray) else frame)

        if term or trunc:
            break

    return frames, info


def save_gif(
    frames: list[np.ndarray],
    path: str | Path,
    fps: int = 10,
    loop: int = 0,
) -> bool:
    """Save frames as animated GIF using PIL.

    Args:
        frames: List of RGB numpy arrays (H, W, 3).
        path: Output file path.
        fps: Frames per second.
        loop: 0 = infinite loop.

    Returns:
        True if saved successfully, False if PIL unavailable.
    """
    Image = _try_import_pil()
    if Image is None:
        print("PIL not available — cannot save GIF")
        return False

    if not frames:
        print("No frames to save")
        return False

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pil_frames = [Image.fromarray(f.astype(np.uint8)) for f in frames]
    duration = int(1000 / fps)

    pil_frames[0].save(
        str(path),
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=loop,
        optimize=True,
    )
    return True


def save_frames_as_png(
    frames: list[np.ndarray],
    output_dir: str | Path,
    prefix: str = "frame",
) -> int:
    """Save individual frames as PNG files (fallback if no PIL)."""
    Image = _try_import_pil()
    if Image is None:
        return 0

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(frames):
        img = Image.fromarray(frame.astype(np.uint8))
        img.save(str(output_dir / f"{prefix}_{i:04d}.png"))

    return len(frames)


def record_demos(
    output_dir: str | Path = "videos",
    n_episodes: int = 1,
    seed: int = 42,
) -> dict[str, Any]:
    """Record demo GIFs for multiple task types.

    Creates GIFs for:
    - L1 PickPlace (scripted)
    - MoveTo (scripted)
    - L4 Sort (random baseline)

    Returns summary dict with paths and success info.
    """
    from envs.pick_place import PickPlaceEnv
    from envs.move_to import MoveToEnv
    from envs.sort import SortEnv
    from policies.scripted import ScriptedPickPlace, ScriptedMoveTo

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"demos": []}

    # Demo 1: PickPlace with scripted policy
    env = PickPlaceEnv(render_mode="rgb_array", image_size=256)
    scripted = ScriptedPickPlace()
    frames, info = record_episode(env, scripted, seed=seed)
    gif_path = output_dir / "demo_pick_place.gif"
    saved = save_gif(frames, gif_path, fps=10)
    summary["demos"].append({
        "task": "L1-PickPlace",
        "policy": "scripted",
        "frames": len(frames),
        "success": info.get("success", False),
        "gif": str(gif_path) if saved else None,
    })
    env.close()

    # Demo 2: MoveTo with scripted policy
    env = MoveToEnv(render_mode="rgb_array", image_size=256)
    scripted_mt = ScriptedMoveTo()
    frames, info = record_episode(env, scripted_mt, seed=seed)
    gif_path = output_dir / "demo_move_to.gif"
    saved = save_gif(frames, gif_path, fps=10)
    summary["demos"].append({
        "task": "MoveTo",
        "policy": "scripted",
        "frames": len(frames),
        "success": info.get("success", False),
        "gif": str(gif_path) if saved else None,
    })
    env.close()

    # Demo 3: Sort random baseline
    env = SortEnv(render_mode="rgb_array", image_size=256)
    frames, info = record_episode(env, None, seed=seed)
    gif_path = output_dir / "demo_sort_random.gif"
    saved = save_gif(frames, gif_path, fps=10)
    summary["demos"].append({
        "task": "L4-Sort",
        "policy": "random",
        "frames": len(frames),
        "success": info.get("success", False),
        "gif": str(gif_path) if saved else None,
    })
    env.close()

    return summary


if __name__ == "__main__":
    summary = record_demos()
    import json
    print(json.dumps(summary, indent=2))
