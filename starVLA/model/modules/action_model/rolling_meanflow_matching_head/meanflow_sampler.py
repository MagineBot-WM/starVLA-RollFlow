"""Stateless, whole-horizon MeanFlow decoding (noise at 0, actions at 1)."""

import torch


@torch.no_grad()
def sample_mean_flow(model, noise, n_steps=4, *, context=None, clip_velocity=0.0, instantaneous=False, **model_kwargs):
    """Integrate average velocities over uniform intervals; return [B,H,A].

    All action tokens share the same source/terminal time at each forward.
    No rolling cache, CPU trajectory copies, or model parameters are created.
    ``model`` follows RollFlow's (x, s, t, context, **kwargs) contract.
    ``instantaneous=True`` uses diagonal velocities model(x,s,s) with Euler
    updates; otherwise queries interval-average velocities model(x,s,t).
    """
    if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps <= 0:
        raise ValueError("n_steps must be a positive integer")
    if noise.ndim != 3 or not noise.is_floating_point():
        raise ValueError("noise must be floating-point [B,H,A]")
    if not 0 <= clip_velocity < float("inf"):
        raise ValueError("clip_velocity must be finite and non-negative")
    x = noise
    for i in range(n_steps):
        s = torch.full((*x.shape[:2], 1), i / n_steps, device=x.device, dtype=torch.float32)
        t = torch.full_like(s, (i + 1) / n_steps)
        velocity = model(x, s, s if instantaneous else t, context, **model_kwargs)
        if velocity.shape != x.shape:
            raise ValueError("velocity must match noise shape")
        if clip_velocity > 0:
            velocity = clip_velocity * torch.tanh(velocity / clip_velocity)
        x = (x.float() + (t - s) * velocity.float()).to(noise.dtype)
    return x
