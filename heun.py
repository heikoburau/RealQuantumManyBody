import numpy as np


def estimate_next_step(f, t, y, dt, k1=None, num_steps=1):
    if k1 is None:
        k1 = f(t, y)
    k2 = f(t + dt, y + k1 * dt)

    y_next = y + dt / 2 * (k1 + k2)
    if num_steps > 1:
        return estimate_next_step(f, t + dt, y_next, dt, k1=k2, num_steps=num_steps - 1)
    return y_next


def solve(f, t_span, y_0, initial_dt=1e-5, max_dt=0.05, min_dt=1e-5, tol=1e-5):
    begin, end = t_span
    t = begin
    dt = initial_dt
    y = +y_0
    t_list = [t]
    y_list = [y]

    assert begin < end
    assert initial_dt < max_dt

    while t < (end - 1e-8):
        payload = dict(datastep=True)

        assert np.all(np.isfinite(y))

        f_t = f(t, y, payload=payload)

        assert np.all(np.isfinite(f_t))

        step_approved = False
        while not step_approved:
            y_next = estimate_next_step(f, t, y, dt, f_t, 1)
            y_next_prime = estimate_next_step(f, t, y, dt / 2, f_t, 2)

            delta_vec = (y_next - y_next_prime) / 6

            if "S_matrix" in payload:
                S_matrix = payload["S_matrix"]

                delta = np.sqrt((delta_vec.conj() @ S_matrix @ delta_vec).real) / len(y)
            else:
                delta = np.linalg.norm(delta_vec) / len(y)

            dt_prime = dt * (tol / delta)**(1 / 3)

            dt_prime = min(dt_prime, max_dt)
            dt_prime = max(dt_prime, min_dt)

            if dt_prime > 0.999 * dt and dt_prime < 10 * dt:
                step_approved = True

            dt = dt_prime

        dt = min(dt, end - t)

        y = estimate_next_step(f, t, y, dt, f_t, 1)
        y_list.append(y)

        t += dt
        t_list.append(t)

    return np.array(t_list), np.array(y_list)
