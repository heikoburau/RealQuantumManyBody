from .OperatorFunctionality import OperatorFunctionality
from jax import vmap
import jax.numpy as jnp


class Operator(OperatorFunctionality):
    def __init__(self, J, h):
        self.J = J
        self.h = h

    def local_energy(self, v, params, conf, v_val):
        N = len(conf)

        s = 2 * conf.astype(jnp.float64) - 1

        # zz-term
        result = self.J * jnp.sum(
            vmap(
                lambda i: s[i] * s[(i + 1) % N]
            )(jnp.arange(N))
        )

        # x-term
        result += self.h * (jnp.sum(
            vmap(
                lambda i: jnp.exp(
                    v.apply(params, conf.at[i].set(1 - conf[i]))
                )
            )(jnp.arange(N))
        ) / jnp.exp(v_val)).real

        return result
