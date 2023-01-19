import optax
import jax.numpy as jnp
import jax


def optimize_amplitudes(
    threshold, min_steps, max_steps, params, params_phase, optimizer, sample_fn, grad_fn,
    beta=0.9, key=None, min_value=None
):
    opt_state = optimizer.init(params)
    inert_log_energies = []
    step = 0
    min_value = min_value or 0

    while True:
        if key is None:
            samples, weights = sample_fn(params)
            grad, aux = grad_fn(params=params, params_phase=params_phase, samples=samples, weights=weights)
        else:
            key, keynow = jax.random.split(key)
            samples, acceptance_rate = sample_fn(params, keynow)
            grad, aux = grad_fn(params=params, params_phase=params_phase, samples=samples)
            print(f"acceptance rate: {acceptance_rate:.2f}")

        value = aux['value'] - min_value

        print(f"{value:.3g}")

        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)

        yield params, value

        step += 1
        if not inert_log_energies:
            inert_log_energies.append(jnp.log(value))
        else:
            inert_log_energies.append(
                beta * inert_log_energies[-1] +
                (1 - beta) * jnp.log(value)
            )

        if len(inert_log_energies) > min_steps and (
            inert_log_energies[0] - inert_log_energies[-1] > threshold
        ):
            return
        if step >= max_steps:
            print(f"max steps ({max_steps}) reached")
            return


def optimize_phases(
    threshold, min_steps, max_steps, params, params_phase, optimizer, sample_fn, grad_fn,
    beta=0.9, key=None, min_value=None
):
    opt_state = optimizer.init(params_phase)
    inert_log_energies = []
    step = 0
    min_value = min_value or 0

    if key is None:
        samples, weights = sample_fn(params)

    while True:
        if key is None:
            grad, aux = grad_fn(params=params, params_phase=params_phase, samples=samples, weights=weights)
        else:
            key, keynow = jax.random.split(key)
            samples, acceptance_rate = sample_fn(params, keynow)
            grad, aux = grad_fn(params=params, params_phase=params_phase, samples=samples)
            print(f"acceptance rate: {acceptance_rate:.2f}")

        value = aux['value'] - min_value
        print(f"{value:.3g}")

        updates, opt_state = optimizer.update(grad, opt_state, params_phase)
        params_phase = optax.apply_updates(params_phase, updates)

        yield params_phase, value

        step += 1
        if not inert_log_energies:
            inert_log_energies.append(jnp.log(value))
        else:
            inert_log_energies.append(
                beta * inert_log_energies[-1] +
                (1 - beta) * jnp.log(value)
            )

        if len(inert_log_energies) > min_steps and (
            abs(inert_log_energies[-1] - inert_log_energies[-2]) < threshold
        ):
            return
        if step >= max_steps:
            print(f"max steps ({max_steps}) reached")
            return
