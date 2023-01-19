from .jax_extension import vmean
from jax import vmap
import jax.numpy as jnp
import jax


def get_gradient_conj(conf, which, v, params, v_phase, params_phase, logpsi):
    if which in ('amplitude', 'both'):
        grad_amplitude = jax.grad(
            lambda params: v.apply(params, conf),
            holomorphic=jnp.iscomplexobj(logpsi)
        )(params)
        if jnp.iscomplexobj(logpsi):
            grad_amplitude = jax.tree_map(lambda x: x.conj(), grad_amplitude)
    if which in ('phase', 'both'):
        grad_phase = jax.grad(
            lambda params: v_phase.apply(params, conf),
        )(params_phase)
        grad_phase = jax.tree_map(lambda x: -1j * jnp.pi * x, grad_phase)

    if which == 'amplitude':
        grad = grad_amplitude
    elif which == 'phase':
        grad = grad_phase
    else:
        grad = dict(
            amplitude=grad_amplitude,
            phase=grad_phase
        )

    return grad


class OperatorFunctionality:
    def value(
        self, v, params, samples, v_phase=None, params_phase=None, weights=None, real_valued=True, batch_size=128
    ):
        def on_sample(conf):
            logpsi = v.apply(params, conf)

            if v_phase is None:
                local_energy = self.local_energy(v, params, conf, logpsi)
            else:
                local_energy = self.local_energy_with_phase(v, params, conf, logpsi, v_phase, params_phase)

            return dict(
                local_energy=local_energy,
                local_energy_sq=abs(local_energy)**2
            )

        final_state = vmean(on_sample, batch_size=batch_size)(samples, weights=weights)

        local_energy, local_energy_sq = (final_state[k] for k in [
            'local_energy', 'local_energy_sq'
        ])

        aux = dict(
            variance=local_energy_sq - abs(local_energy)**2
        )

        if real_valued:
            local_energy = local_energy.real

        return local_energy, aux

    def gradient(
        self, v, params, samples, v_phase=None, params_phase=None, weights=None, which='amplitude', real_valued=True, batch_size=128
    ):
        assert which in ('amplitude', 'phase', 'both')

        def on_sample(conf):
            logpsi = v.apply(params, conf)

            if v_phase is None:
                local_energy = self.local_energy(v, params, conf, logpsi)
            else:
                local_energy = self.local_energy_with_phase(v, params, conf, logpsi, v_phase, params_phase)

            grad = get_gradient_conj(conf, which, v, params, v_phase, params_phase, logpsi)

            tree_mul = lambda factor, x: jax.tree_map(lambda x: factor * x, x)

            return dict(
                grad=grad,
                grad_F=tree_mul(local_energy, grad),
                local_energy=local_energy,
                local_energy_sq=abs(local_energy)**2
            )

        final_state = vmean(on_sample, batch_size=batch_size)(samples, weights=weights)

        grad, grad_F, local_energy, local_energy_sq = (final_state[k] for k in [
            'grad', 'grad_F', 'local_energy', 'local_energy_sq'
        ])

        aux = dict(
            value=local_energy.real,
            variance=local_energy_sq - abs(local_energy)**2
        )

        result = jax.tree_multimap(
            lambda grad_F, grad: grad_F - grad * local_energy,
            grad_F, grad
        )
        if real_valued:
            result = jax.tree_map(lambda x: x.real, result)

        return result, aux

    def gradient_sq(
        self,
        v, params, samples, v_phase=None, params_phase=None,
        weights=None, which='amplitude', real_valued=True, batch_size=128
    ):
        def on_sample(conf):
            logpsi = v.apply(params, conf)
            if v_phase is not None:
                phase = v_phase.apply(params_phase, conf)

            conf_prime_list = self.conf_prime_list(conf)

            if v_phase is not None:
                inner_sum = self.matrix_elements(conf) * vmap(
                    lambda conf_p: jnp.exp(
                        v.apply(params, conf_p) - logpsi +
                        1j * jnp.pi * (v_phase.apply(params_phase, conf_p) - phase)
                    )
                )(conf_prime_list)
            else:
                inner_sum = self.matrix_elements(conf) * vmap(
                    lambda conf_p: jnp.exp(v.apply(params, conf_p) - logpsi)
                )(conf_prime_list)

            local_energy = jnp.sum(inner_sum)

            grad_p = vmap(
                lambda conf: get_gradient_conj(conf, which, v, params, v_phase, params_phase, logpsi)
            )(conf_prime_list)

            grad_dot_expval = jax.tree_map(
                lambda grad_p: 2.0 * local_energy * jnp.sum(
                    vmap(
                        lambda a, b: a * b
                    )(grad_p, inner_sum.conj()),
                    axis=0
                ),
                grad_p
            )

            grad = get_gradient_conj(conf, which, v, params, v_phase, params_phase, logpsi)

            return dict(
                grad_dot_expval=grad_dot_expval,
                grad=grad,
                local_energy_sq=abs(local_energy)**2
            )

        final_state = vmean(on_sample, batch_size=batch_size)(samples, weights=weights)

        grad_dot_expval, grad, local_energy_sq = (final_state[k] for k in [
            'grad_dot_expval', 'grad', 'local_energy_sq'
        ])

        aux = dict(
            value=local_energy_sq,
        )

        result = jax.tree_multimap(
            lambda grad_dot_expval, grad: grad_dot_expval - local_energy_sq * grad,
            grad_dot_expval,
            grad
        )

        if real_valued:
            result = jax.tree_map(lambda x: x.real, result)

        return result, aux


class OperatorFunctionalityCont:
    def value(self, samples, batch_size=128):
        def on_sample(conf):
            r, rp = conf[:, 0, :], conf[:, 1, :]

            return self.matrix_element(r, rp)

        return vmean(on_sample, batch_size=batch_size)(samples)

    def gradient(self, v, params, samples, batch_size=128):
        def on_sample(conf):
            r, rp = conf[:, 0, :], conf[:, 1, :]

            mat_el = self.matrix_element(r, rp)

            grad_fn = jax.grad(lambda params, conf: v.apply(params, conf))

            grad_r = grad_fn(params, r)
            grad_rp = grad_fn(params, rp)

            grad = jax.tree_multimap(lambda a, b: 0.5 * (a + b), grad_r, grad_rp)

            tree_mul = lambda factor, x: jax.tree_map(lambda x: factor * x, x)

            return dict(
                grad=grad,
                grad_F=tree_mul(mat_el, grad),
                local_energy=mat_el
            )

        final_state = vmean(on_sample, batch_size=batch_size)(samples)

        grad, grad_F, local_energy = (final_state[k] for k in [
            'grad', 'grad_F', 'local_energy'
        ])

        aux = dict(
            value=local_energy
        )

        result = jax.tree_multimap(
            lambda grad_F, grad: grad_F - grad * local_energy,
            grad_F, grad
        )

        return result, aux


class OperatorFunctionalityBi:
    def value(self, samples, batch_size=128):
        def on_sample(bi_conf):
            r, rp = bi_conf

            return self.matrix_element(r, rp)

        return vmean(on_sample, batch_size=batch_size)(samples)

    def gradient(self, rho, params, samples, batch_size=128):
        def on_sample(bi_conf):
            r, rp = bi_conf

            mat_el = self.matrix_element(r, rp)

            grad_fn = jax.grad(lambda params, bi_conf: rho.apply(params, bi_conf))

            grad = grad_fn(params, bi_conf)

            tree_mul = lambda factor, x: jax.tree_map(lambda x: factor * x, x)

            return dict(
                grad=grad,
                grad_F=tree_mul(mat_el, grad),
                local_energy=mat_el
            )

        final_state = vmean(on_sample, batch_size=batch_size)(samples)

        grad, grad_F, local_energy = (final_state[k] for k in [
            'grad', 'grad_F', 'local_energy'
        ])

        aux = dict(
            value=local_energy
        )

        result = jax.tree_multimap(
            lambda grad_F, grad: grad_F - grad * local_energy,
            grad_F, grad
        )

        return result, aux
