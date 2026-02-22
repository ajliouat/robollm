"""RoboLLM training — RL training scripts and configs."""

__all__ = ["ReplayBuffer", "train", "TrainConfig"]


def __getattr__(name: str):
    if name == "ReplayBuffer":
        from training.replay_buffer import ReplayBuffer
        return ReplayBuffer
    if name in ("train", "TrainConfig"):
        from training.train import train, TrainConfig
        return {"train": train, "TrainConfig": TrainConfig}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
