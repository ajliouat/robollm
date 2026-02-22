"""RoboLLM policies — SAC/PPO implementations."""

__all__ = ["Actor", "Critic", "TwinCritic", "SACAgent", "SACConfig"]


def __getattr__(name: str):
    if name in ("Actor", "Critic", "TwinCritic"):
        from policies.networks import Actor, Critic, TwinCritic
        return {"Actor": Actor, "Critic": Critic, "TwinCritic": TwinCritic}[name]
    if name in ("SACAgent", "SACConfig"):
        from policies.sac import SACAgent, SACConfig
        return {"SACAgent": SACAgent, "SACConfig": SACConfig}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
