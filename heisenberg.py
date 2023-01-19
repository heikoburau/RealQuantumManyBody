import jax.numpy as jnp


tetraeder = 3**0.5 * jnp.array([
    [0, 0, 1],
    [(8 / 9)**0.5, 0, -1 / 3],
    [-(2 / 9)**0.5, (2 / 3)**0.5, -1 / 3],
    [-(2 / 9)**0.5, -(2 / 3)**0.5, -1 / 3]
])

tetraeder_2 = 3**0.5 * jnp.array([
    [0, 0, 1],
    [(8 / 9)**0.5, 0, -1 / 3],
    [-(2 / 9)**0.5, (2 / 3)**0.5, -1 / 3],
    [-(2 / 9)**0.5, -(2 / 3)**0.5, -1 / 3],
    [0, 0, 0]
])

tetraeder_3 = jnp.array([
    jnp.array([0, 0, 1]),
    jnp.array([(8 / 9)**0.5, 0, -1 / 3]),
    jnp.array([-(2 / 9)**0.5, (2 / 3)**0.5, -1 / 3]),
    jnp.array([-(2 / 9)**0.5, -(2 / 3)**0.5, -1 / 3]),
    -jnp.array([0, 0, 1]),
    -jnp.array([(8 / 9)**0.5, 0, -1 / 3]),
    -jnp.array([-(2 / 9)**0.5, (2 / 3)**0.5, -1 / 3]),
    -jnp.array([-(2 / 9)**0.5, -(2 / 3)**0.5, -1 / 3]),
    [0, 0, 0]
])

heisenberg_tensor_fn = lambda sign: jnp.array([
    [
        [
            [
                sign * (
                    (tetraeder[i] + tetraeder[ip]) @
                    (tetraeder[j] + tetraeder[jp])
                ) + (
                    (tetraeder[ip] @ tetraeder[j]) *
                    (tetraeder[jp] @ tetraeder[i])
                ) - (
                    (tetraeder[i] @ tetraeder[j]) *
                    (tetraeder[ip] @ tetraeder[jp])
                )
                for jp in range(4)
            ]
            for ip in range(4)
        ]
        for j in range(4)
    ]
    for i in range(4)
]) / 16

heisenberg_tensor_signed = jnp.array([
    heisenberg_tensor_fn(1),
    heisenberg_tensor_fn(-1)
])


bi_tetraeder = jnp.concatenate(
    [tetraeder, -tetraeder],
    axis=0
) / 3**0.5

bi_dot_product = jnp.array([
    [
        tetraeder_3[i] @ tetraeder_3[j]
        for j in range(9)
    ]
    for i in range(9)
])
bi_dot_product_log = jnp.log(1 + bi_dot_product)

heisenberg_bi_tensor = jnp.array([
    [
        [
            [
                (
                    (bi_tetraeder[i] + bi_tetraeder[ip]) @
                    (bi_tetraeder[j] + bi_tetraeder[jp])
                ) + (
                    (bi_tetraeder[ip] @ bi_tetraeder[j]) *
                    (bi_tetraeder[jp] @ bi_tetraeder[i])
                ) - (
                    (bi_tetraeder[i] @ bi_tetraeder[j]) *
                    (bi_tetraeder[ip] @ bi_tetraeder[jp])
                )
                for jp in range(8)
            ]
            for ip in range(8)
        ]
        for j in range(8)
    ]
    for i in range(8)
]) / 16
