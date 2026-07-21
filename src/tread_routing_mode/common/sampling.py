import torch


@torch.inference_mode()
def sample_flow(model, noise, labels, *, steps, cfg_scale, save_every=1,
                solver="euler", routing_schedule):
    """Integrate linear-path v-prediction from t=1 to t=0."""
    solver = solver.lower()
    if solver not in {"euler", "heun"}:
        raise ValueError("solver must be 'euler' or 'heun'")
    z = noise.clone()
    dt = 1.0 / steps
    states = [{"step": steps, "t": 1.0, "latent": z.detach().cpu()}]

    def velocity(current, time_value, routing_active, route_ids):
        t = torch.full((current.shape[0],), time_value, device=current.device,
                       dtype=current.dtype)
        conditional = model(
            current, t, labels, route=routing_active, route_ids=route_ids)
        if cfg_scale == 1.0:
            return conditional
        unconditional = model(
            current, t, labels, unconditional=True, route=routing_active,
            route_ids=route_ids)
        return unconditional + cfg_scale * (conditional - unconditional)

    for index in range(steps, 0, -1):
        routing_active, route_ids = routing_schedule.routing_for_step(index, z.device)
        current_velocity = velocity(
            z, index / steps, routing_active, route_ids)
        if solver == "euler":
            z = z - dt * current_velocity
        else:
            predictor = z - dt * current_velocity
            next_velocity = velocity(
                predictor, (index - 1) / steps, routing_active, route_ids)
            z = z - 0.5 * dt * (current_velocity + next_velocity)
        if (steps - index + 1) % save_every == 0 or index == 1:
            states.append({"step": index - 1, "t": (index - 1) / steps,
                           "latent": z.detach().cpu()})
    return z, states
