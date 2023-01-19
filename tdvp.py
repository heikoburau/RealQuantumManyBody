from .heun import solve
from .jax_extension import vmean
import jax.numpy as jnp
import jax


def flatten_params(params):
    return jnp.concatenate([
        jnp.ravel(leaf)
        for leaf in jax.tree_leaves(params)
    ])


def unflatten_params(flat_params, params_template):
    leaves_template, treedef = jax.tree_flatten(params_template)

    offsets = jnp.cumsum(jnp.array([0] + [l.size if hasattr(l, "size") else 1 for l in leaves_template]))

    leaves = [
        flat_params[i:j].reshape(l_template.shape) if hasattr(l_template, "shape") else type(l_template)(flat_params[i:j])
        for i, j, l_template in zip(offsets[:-1], offsets[1:], leaves_template)
    ]

    return jax.tree_unflatten(treedef, leaves)


def get_S_matrix(v, params, samples, v_phase=None, params_phase=None, batch_size=128, weights=None, which='both'):
    assert which in ('amplitude', 'phase', 'both')

    def on_sample(conf):
        logpsi = v.apply(params, conf)

        if which in ('amplitude', 'both'):
            grad_amplitude = jax.grad(
                lambda params: v.apply(params, conf),
                holomorphic=jnp.iscomplexobj(logpsi)
            )(params)
        if which in ('phase', 'both'):
            grad_phase = jax.grad(
                lambda params: v_phase.apply(params, conf),
            )(params_phase)
            grad_phase = jax.tree_map(lambda x: 1j * x, grad_phase)

        if which == 'amplitude':
            grad = grad_amplitude
        elif which == 'phase':
            grad = grad_phase
        else:
            grad = dict(
                amplitude=grad_amplitude,
                phase=grad_phase
            )

        O_k = flatten_params(grad)

        # r_tanh = 1 / jnp.tanh(logpsi)
        r_tanh = 1

        return dict(
            norm=1.0,
            O_k=O_k * r_tanh,
            O_kk=jnp.outer(O_k.conj(), O_k) * r_tanh**2
        )

    final_state = vmean(on_sample, batch_size=batch_size)(samples, weights=weights)

    norm, O_k, O_kk = [final_state[k] for k in ['norm', 'O_k', 'O_kk']]

    O_k /= norm
    O_kk /= norm

    return O_kk - jnp.outer(O_k.conj(), O_k)


def signal_to_noise_ratio(num_samples, var_energy, sigma2_k, rho_k):
        return jnp.sqrt(
            num_samples / (
                1 + sigma2_k / abs(rho_k)**2 * var_energy
            )
        )


def Sinv_dot_F(S, F, diag_shift=0.0, energy_variance=None, lambda_snr=3, num_samples=None):
    S_regularized = S + diag_shift * jnp.eye(S.shape[0])

    u, sigma2_k, vh = jnp.linalg.svd(S_regularized)
    rho_k = vh @ F

    if energy_variance is not None:
        snr = signal_to_noise_ratio(num_samples, energy_variance, sigma2_k, rho_k)
        params_dot_tilde = rho_k / sigma2_k / (1 + (lambda_snr / snr)**6)
    else:
        params_dot_tilde = rho_k / sigma2_k

    return vh.T.conj() @ params_dot_tilde


class TDVP:
    def __init__(
        self, sample_fn, S_matrix_fn, grad_fn, mc=True,
        tol=1e-5, max_dt=0.05, min_dt=1e-3, reg=1e-5, lambda_snr=3
    ):
        self.sample_fn = sample_fn
        self.S_matrix_fn = S_matrix_fn
        self.grad_fn = grad_fn
        self.mc = mc
        self.tol = tol
        self.max_dt = max_dt
        self.min_dt = min_dt
        self.reg = reg
        self.lambda_snr = lambda_snr

    def derivative(self, t, params, payload):

        params_tree = unflatten_params(params, self.params_orig)

        if self.mc:
            self.key, keynow = jax.random.split(self.key)
            samples, acceptance_rate = self.sample_fn(params_tree, keynow)

            S = self.S_matrix_fn(params_tree, samples)
            grad, aux = self.grad_fn(params_tree, samples)
        else:
            samples, weights = self.sample_fn(params_tree)
            S = self.S_matrix_fn(params_tree, samples, weights)
            grad, aux = self.grad_fn(params_tree, samples, weights)

        # S += (self.reg * jnp.amax(S.diagonal())) * jnp.eye(S.shape[0])
        S += self.reg * jnp.eye(S.shape[0])
        F = -flatten_params(grad)

        # ??? u is never used! ??? (u == v for hermitian matrices)
        u, sigma2_k, vh = jnp.linalg.svd(S)
        rho_k = vh @ F

        if self.mc:
            snr = signal_to_noise_ratio(len(samples), aux['variance'], sigma2_k, rho_k)
            params_dot_tilde = rho_k / sigma2_k / (1 + (self.lambda_snr / snr)**6)
        else:
            params_dot_tilde = rho_k / sigma2_k

        params_dot = vh.T.conj() @ params_dot_tilde

        if payload is not None:
            payload["S_matrix"] = S

            if payload["datastep"]:
                R = S @ params_dot - F
                self.r2_list.append(
                    1 + (params_dot @ R - F @ params_dot) / aux['variance']
                )

            print(f"{t:.3g}", flush=True)
            if self.mc:
                print(f"acceptance_rate = {acceptance_rate:.2f}")

        return params_dot

    def __call__(self, t_span, params, key=None):
        self.params_orig = params
        self.key = key
        self.r2_list = []

        def rhs(t, params, payload=None):
            return self.derivative(t, params, payload)

        self.t_list, self.y_list = solve(
            rhs,
            t_span,
            y_0=flatten_params(params),
            max_dt=self.max_dt,
            min_dt=self.min_dt,
            tol=self.tol
        )

        self.y_list = [
            unflatten_params(y, self.params_orig)
            for y in self.y_list
        ]

        return self.y_list[-1]
