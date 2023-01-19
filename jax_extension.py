import jax
from jax import vmap
import jax.numpy as jnp
from functools import wraps


def vmean(fn, batch_size=128):
    @wraps(fn)
    def wrapper(data, weights=None):
        n = data.shape[0]

        batch_size_ = min(batch_size, n)
        assert n % batch_size_ == 0
        data = data.reshape((n // batch_size_, batch_size_,) + data.shape[1:])

        if weights is None:
            v_fn_b = vmap(lambda b, x: jax.tree_multimap(lambda x, b: x + b, fn(x), b))

            initial_state = vmap(fn)(data[0])

            final_state = jax.lax.fori_loop(
                1,
                n // batch_size_,
                lambda i, val: v_fn_b(val, data[i]),
                initial_state
            )

            return jax.tree_map(lambda a: jnp.sum(a, axis=0) / n, final_state)
        else:
            weights = weights.reshape((n // batch_size_, batch_size_))

            v_fn_a = vmap(lambda a, x: jax.tree_map(lambda x: a * x, fn(x)))
            v_fn_a_b = vmap(lambda a, b, x: jax.tree_multimap(lambda x, b: a * x + b, fn(x), b))

            initial_state = v_fn_a(weights[0], data[0])

            final_state = jax.lax.fori_loop(
                1,
                n // batch_size_,
                lambda i, val: v_fn_a_b(weights[i], val, data[i]),
                initial_state
            )

            return jax.tree_map(lambda a: jnp.sum(a, axis=0), final_state)

    return wrapper
