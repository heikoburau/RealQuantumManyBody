from jax import grad
import jax.numpy as jnp


class DensityMatrix:
    def __init__(self, v, num_spins):
        self.v = v
        self.num_spins = num_spins

        self.grad_log_probability = grad(
            lambda params, x: self.log_probability(params, x),
            1
        )

    def log_probability(self, params, conf):
        N = conf.shape[-1] // 2
        conf = conf.reshape((3, 2, N))
        r, r_p = conf[:, 0, :], conf[:, 1, :]
        return (
            self.v.apply(params, r) +
            self.v.apply(params, r_p) +
            jnp.sum(jnp.log(jnp.abs(1 + jnp.sum(r * r_p, axis=0))))
        )
