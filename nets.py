from .heisenberg import bi_dot_product, tetraeder_2, bi_dot_product_log
from jax.nn.initializers import lecun_normal
import flax.linen as nn
import jax
from jax import vmap, grad, ops
import jax.numpy as jnp
from typing import Sequence
from dataclasses import field
from functools import partial
from itertools import product
# from functools import cached_property


def init_diag_couplings(key, N, alpha):
    result = jnp.zeros((N // 2 + 1, 3))

    for i in range(3):
        result = ops.index_update(result, (0, i), alpha)

    return result


tetraeder = jnp.array([
    [0, 0, 1],
    [(8 / 9)**0.5, 0, -1 / 3],
    [-(2 / 9)**0.5, (2 / 3)**0.5, -1 / 3],
    [-(2 / 9)**0.5, -(2 / 3)**0.5, -1 / 3]
])
triple_product = jnp.array([
    [
        [
            jnp.cross(a, b) @ c
            for c in tetraeder
        ]
        for b in tetraeder
    ]
    for a in tetraeder
])
triple_product /= jnp.amax(triple_product)


class vCN(nn.Module):
    pol: Sequence[float] = field(default_factory=lambda: [3.0, 0.0, 0.0])
    alpha: float = 3.0

    @nn.compact
    def __call__(self, x):
        N = x.shape[-2]

        polarization = self.param('polarization', lambda key: jnp.array(self.pol))

        diag_couplings = self.param('diag_couplings', init_diag_couplings, N, self.alpha)

        x2_diag = vmap(
            lambda i: jnp.sum(x * jnp.roll(x, i, axis=-2), axis=-2)
        )(jnp.arange(N // 2 + 1))

        off_diag_couplings = self.param('off_diag_couplings', lambda key: jnp.zeros((N // 2, 3)))

        x_1 = jnp.roll(x, 1, axis=-1)

        x2_off_diag = vmap(
            lambda i: jnp.sum(x * jnp.roll(x_1, i + 1, axis=-2), axis=-2)
        )(jnp.arange(N // 2))

        return (
            jnp.sum(polarization * jnp.sum(x, axis=-2)) +
            -0.5 * (
                jnp.sum(diag_couplings * x2_diag) +
                jnp.sum(off_diag_couplings * x2_off_diag)
            )
        )

    @property
    def gradient(self):
        return grad(lambda params, x: self.apply(params, x))


class psiCNN(nn.Module):
    features: Sequence[int] = (5,)
    kernel_size: int = 5

    @nn.compact
    def __call__(self, x):
        conv = partial(
            nn.Conv,
            kernel_size=(self.kernel_size,),
            padding=[(0, 0)],
            # use_bias=False,
            kernel_init=lambda key, shape: lecun_normal(dtype=jnp.complex128)(key, shape),
            bias_init=lambda key, shape: jax.random.normal(key, shape, dtype=jnp.complex128),
            dtype=jnp.complex128
        )

        y = 2 * x.astype(jnp.complex128) - 1
        y = y[:, jnp.newaxis]

        def f(y):
            for feat in self.features:
                y = jnp.pad(y, ((0, self.kernel_size - 1), (0, 0)), 'wrap')
                y = conv(features=feat)(y)

                # y = y**2 / 2 - y**4 / 12 + y**6 / 45
                y = jax.nn.elu(y)
                # y = (y + 2) / (y - 2)
                # y = jnp.arcsinh(y)

            return jnp.mean(y)

        return jnp.mean(vmap(f)(jnp.array((
            y,
            -y,
            y[::-1, :],
            -y[::-1, :]
        ))))


class psiCNN_signed(nn.Module):
    features: Sequence[int] = (5,)
    kernel_size: int = 5

    @nn.compact
    def __call__(self, x):
        logpsi = lambda x: psiCNN(features=self.features, kernel_size=self.kernel_size)(x)

        return jnp.log(
            jnp.exp(logpsi(x)) -
            jnp.exp(logpsi(x))
        )

        # a = logpsi(x)
        # b = logpsi(x) + 1j * jnp.pi
        # return jax.lax.cond(
        #     abs(a.real) > abs(b.real),
        #     lambda _: a,
        #     lambda _: b,
        #     None
        # )


class vCNN_Z2(nn.Module):
    # pol: Sequence[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    pol: jnp.ndarray = jnp.array([1.0, 1.0, 1.0, 1.0])

    features: int = 5
    kernel_size: Sequence[int] = (5,)

    num_layers: int = 2

    @nn.compact
    def __call__(self, x, train=False):
        # vcn = vCN(pol=self.pol, alpha=self.alpha)

        conv = partial(
            nn.Conv,
            features=self.features,
            kernel_size=self.kernel_size,
            padding=[(0, 0)]
        )

        x = jax.nn.one_hot(x, 4)
        y = x

        # print(y.shape)

        # y = y.T

        for i in range(self.num_layers):

            y = jnp.pad(y, ((0, self.kernel_size[0] - 1), (0, 0)), 'wrap')
            y = conv()(y)

            y = jax.nn.elu(y)

        y = jnp.sum(y)

        polarization = self.param("polarization", lambda key: self.pol)

        return y + jnp.sum(polarization * x)
        # return jnp.sum(polarization * x)


class vSU2_Dense(nn.Module):
    features: Sequence[int] = (5,)
    noise: jnp.float64 = 1.0
    max_distance: int = 8
    center_coupling: float = 0.0

    @nn.compact
    def __call__(self, x):
        N = x.shape[0]

        pair_ids = jnp.array([
            (i, j)
            for i in range(N) for j in range(i)
            if abs(i - j) < self.max_distance
        ])
        Heisenberg_couplings = 2.0 * vmap(
            lambda ids: x[ids[0]] == x[ids[1]]
        )(pair_ids) - 1.0

        triple_ids = jnp.array([
            (i, j, k)
            for i in range(N) for j in range(i) for k in range(j)
            if (
                abs(i - j) < self.max_distance and
                abs(i - k) < self.max_distance and
                abs(j - k) < self.max_distance
            )
        ])
        triple_products = vmap(
            lambda ids: triple_product[x[ids[0]], x[ids[1]], x[ids[2]]]
        )(triple_ids)
        y = jnp.concatenate([Heisenberg_couplings, triple_products])

        for feat in self.features:
            y = nn.Dense(
                features=feat,
                kernel_init=lambda key, shape: self.noise * lecun_normal()(key, shape),
                dtype=jnp.float64
            )(y)
            y = jax.nn.elu(y)
            # y = 4.0 * jax.nn.sigmoid(y)

            # der Vorfaktor beschleunigt die Konvergenz
            # y = 0.5 * jax.nn.relu6(y)

            # y = nn.relu(y)
            # y = jnp.log(jnp.cosh(y))

        return (
            nn.Dense(features=1, use_bias=False, dtype=jnp.float64)(y)[0] +
            self.param("center_coupling_param", lambda key: self.center_coupling) * (
                2.0 * (x[N // 2 - 1] == x[N // 2]) - 1.0
            )
        )


class vSU2_Dense2(nn.Module):
    features: Sequence[int] = (5,)
    noise: jnp.float64 = 1.0

    @nn.compact
    def __call__(self, x):
        N = x.shape[0]

        pair_ids = jnp.array([
            (i, i + 1)
            for i in range(N - 1)
        ])
        Heisenberg_couplings = 2.0 * vmap(
            lambda ids: x[ids[0]] == x[ids[1]]
        )(pair_ids) - 1.0

        triple_ids = jnp.array([
            (i - 1, i, i + 1)
            for i in range(1, N - 1)
        ])
        triple_products = vmap(
            lambda ids: triple_product[x[ids[0]], x[ids[1]], x[ids[2]]]
        )(triple_ids)
        y = jnp.concatenate([Heisenberg_couplings, triple_products])

        for feat in self.features:
            y = nn.Dense(
                features=feat,
                kernel_init=lambda key, shape: self.noise * lecun_normal()(key, shape),
                dtype=jnp.float64
            )(y)
            y = jax.nn.elu(y)
            # y = 4.0 * jax.nn.sigmoid(y)

            # der Vorfaktor beschleunigt die Konvergenz
            # y = 0.5 * jax.nn.relu6(y)

            # y = nn.relu(y)
            # y = jnp.log(jnp.cosh(y))

        return nn.Dense(
            features=1,
            use_bias=False,
            dtype=jnp.float64
        )(y)[0]


class vSU2_Dense_CNN(nn.Module):
    L: int
    features: Sequence[int] = (5,)
    noise: jnp.float64 = 1.0
    kernel_size: int = 8

    @nn.compact
    def __call__(self, x):
        L = self.L
        x = x.reshape((L, L))

        y = vmap(lambda j: jnp.concatenate([
            2.0 * (x[:-1, j] == x[1:, j]) - 1.0,
            2.0 * (x[:, j] == x[:, (j + 1) % L]) - 1.0,
            2.0 * (x[:-1, j] == x[1:, (j + 1) % L]) - 1.0,
            vmap(
                lambda i: triple_product[x[i - 1, j], x[i, j], x[i + 1, j]]
            )(jnp.arange(1, L - 1)),
        ]))(jnp.arange(L))

        y *= self.noise

        kernel_size = self.kernel_size

        # insert batch dimension
        y = y[jnp.newaxis, ...]

        for feat in self.features:
            y = jnp.pad(
                y,
                pad_width=[(0, 0)] + [((kernel_size - 1) // 2, kernel_size // 2)] + [(0, 0)],
                mode='wrap'
            )

            y = nn.Conv(
                features=feat,
                kernel_size=[kernel_size],
                padding=[(0, 0)],
                dtype=jnp.float64
            )(y)

            y = jax.nn.elu(y)

        y = jnp.mean(y, axis=1).ravel()

        return nn.Dense(
            features=1,
            use_bias=False,
            dtype=jnp.float64
        )(y).ravel()[0]


class vSU2_Dense_sym(vSU2_Dense):
    @nn.compact
    def __call__(self, x):
        v = vSU2_Dense(self.features, self.noise, self.max_distance, self.center_coupling)

        return (v(x) + v(x[::-1])) / 2


class vExact(nn.Module):
    @nn.compact
    def __call__(self, x):
        N = len(x)

        vec = self.param("vec", lambda key: jax.random.normal(key, (4**N,)))

        idx = x @ 4**jnp.arange(N)

        return vec[idx]


class vSU2_Exact(nn.Module):
    @nn.compact
    def __call__(self, x):
        # here, only the dot product is used

        N = x.shape[0]

        y = vmap(
            lambda x_i: x_i == x
        )(x)
        y = jnp.concatenate([
            y[i, :i] for i in range(N)
        ])
        y = y.ravel()

        M = len(y)

        vec = self.param("vec", lambda key: jax.random.normal(key, (2**M,)))

        idx = y @ 2**jnp.arange(M)

        return vec[idx]


class vSU2(nn.Module):
    features: int = 5
    kernel_size: Sequence[int] = (5,)

    reg2: float = 1.0
    reg: float = 1.0
    reg_length: float = 0.25

    @nn.compact
    def __call__(self, x):
        N = x.shape[1]

        regu2 = self.param('regu2', lambda key: self.reg2)
        regu = self.param('regu', lambda key: self.reg)
        regu_length = self.param('regu_length', lambda key: self.reg_length)

        def f(z):
            for feat in self.features:
                z = nn.Dense(feat, use_bias=False)(z)
                z = jax.nn.elu(z)

            return jnp.sum(z)

        x2 = jnp.sum(x * x, axis=0)

        return jnp.mean(
            vmap(
                lambda i: f(jnp.sum(
                    jnp.roll(x, -i, axis=1) * x[:, i][:, jnp.newaxis],
                    axis=0
                ))
            )(jnp.arange(N)),
            axis=0
        ) - regu2 * jnp.sum(x2) - regu * jnp.sum(jnp.exp(x2 / (2.0 * regu_length**2)))


class vSU2_CNN_1D_ref(nn.Module):
    num_layers: int = 1
    features: int = 3
    kernel_size: int = 5
    noise: float = 0.01
    final_weight: float = 1.0
    final_bias: float = 0.1
    momentum: float = 0.5

    @nn.compact
    def __call__(self, x, collect_stats=False):

        N = x.shape[0]

        # input_kernel = self.param(
        #     'input_kernel',
        #     lambda key: self.noise * jax.random.normal(key, (self.features, N - 1))
        # )
        input_kernel = self.param(
            'input_kernel',
            lambda key: self.noise * jax.random.normal(key, (self.features, N - 2, N - 2))
        )

        # norm = partial(
        #     nn.BatchNorm,
        #     use_running_average=not collect_stats,
        #     momentum=self.momentum,
        #     axis_name="batch",
        #     dtype=jnp.float64
        # )

        # y = vmap(
        #     lambda i: jnp.sum(
        #         input_kernel * (
        #             2 * (x[i] == jnp.roll(x, -i)[1:]) - 1
        #             # (x[i] == jnp.roll(x, -i))
        #         ),
        #         axis=1
        #     )
        # )(jnp.arange(N))

        # jk_ids = jnp.array([(j, k) for j in range(N) for k in range(j)])

        y = vmap(
            lambda i: jnp.sum(
                vmap(
                    lambda j: vmap(
                        lambda k: (
                            (
                                input_kernel[:, j, k] -
                                input_kernel[:, k, j]
                            ) * triple_product[
                                x[i],
                                x[(i + 1 + j) % N],
                                x[(i + 1 + k) % N]
                            ]
                        )
                    )(jnp.arange(N - 2))
                )(jnp.arange(N - 2)),
                axis=(0, 1)
            )
        )(jnp.arange(N))

        # y's shape: (sites, features)

        # y += self.param("bias", lambda key: 0.0)

        # y = y[:, jnp.newaxis]

        # p1 = self.param("p1", lambda key: 0.0)
        # p2 = self.param("p2", lambda key: 0.0)
        # p3 = self.param("p3", lambda key: 0.5)
        # p4 = self.param("p4", lambda key: 0.0)

        for l in range(self.num_layers):
            # axes -> (N, in, out)
            kernel = self.param(
                f'kernel_{l}',
                lambda key: lecun_normal()(key, (self.kernel_size, y.shape[1], self.features))
            )

            y = vmap(
                lambda i: jnp.sum(
                    kernel * jnp.roll(y, i, axis=0)[:self.kernel_size, :, jnp.newaxis],
                    axis=(0, 1)
                )
            )(jnp.arange(N))
            y += self.param(f"bias_{l}", lambda key: jnp.zeros((1, y.shape[1])))

            y = jax.nn.elu(y)

            # if l < self.num_layers - 1:
            #     y = norm()(y)

            # p1 = self.param(f"p1_{l}", lambda key: 0.01 * jax.random.normal(key, y.shape))
            # p2 = self.param(f"p2_{l}", lambda key: 0.01 * jax.random.normal(key, y.shape))
            # p3 = self.param(f"p3_{l}", lambda key: 0.5 * jax.random.normal(key, y.shape))
            # p4 = self.param(f"p4_{l}", lambda key: 0.01 * jax.random.normal(key, y.shape))

            # y = p1 * y + p2 * y**2 + p3 * y**3 + p4 * y**4

        y = jnp.mean(y)

        # y = jnp.mean(y, axis=0)
        # return nn.Dense(
        #     features=1,
        #     dtype=jnp.float64
        # )(y)[0]

        return 0.1 + y


class vPyrochlore(nn.Module):
    L: int
    features: Sequence[int] = (8,)
    kernel_size: int = None

    @nn.compact
    def __call__(self, linear_x):
        L = self.L
        x = linear_x.reshape((L, L, L, 4))

        def on_bond(a, b):
            return tetraeder_2[x[a]] @ tetraeder_2[x[b]]
            # return 2.0 * (x[a] == x[b]) - 1.0

        def on_unit_cell(cell):
            cell_1 = (cell[0] - 1 + L) % L, cell[1], cell[2]
            cell_2 = cell[0], (cell[1] - 1 + L) % L, cell[2]
            cell_3 = cell[0], cell[1], (cell[2] - 1 + L) % L

            return jnp.array([
                on_bond((*cell, 0), (*cell, 1)),
                on_bond((*cell, 0), (*cell, 2)),
                on_bond((*cell, 0), (*cell, 3)),
                on_bond((*cell, 0), (*cell_1, 1)),
                on_bond((*cell, 0), (*cell_2, 2)),
                on_bond((*cell, 0), (*cell_3, 3)),
                on_bond((*cell, 1), (*cell, 2)),
                on_bond((*cell, 2), (*cell, 3)),
                on_bond((*cell, 3), (*cell, 1)),
                on_bond((*cell_1, 1), (*cell_2, 2)),
                on_bond((*cell_2, 2), (*cell_3, 3)),
                on_bond((*cell_3, 3), (*cell_1, 1))
            ])

        y = vmap(
            lambda a: vmap(
                lambda b: vmap(
                    lambda c: on_unit_cell((a, b, c))
                )(jnp.arange(self.L))
            )(jnp.arange(self.L))
        )(jnp.arange(self.L))

        kernel_size = self.kernel_size or self.L

        # insert batch dimension
        y = y[jnp.newaxis, ...]

        for feat in self.features:
            y = jnp.pad(
                y,
                pad_width=[(0, 0)] + [((kernel_size - 1) // 2, kernel_size // 2)] * 3 + [(0, 0)],
                mode='wrap'
            )

            y = nn.Conv(
                features=feat,
                kernel_size=[kernel_size] * 3,
                padding=[(0, 0)] * 3,
                dtype=jnp.float64
            )(y)

            y = jax.nn.elu(y)

        y = jnp.mean(y, axis=(1, 2, 3)).ravel()

        return nn.Dense(
            features=1,
            use_bias=False,
            dtype=jnp.float64
        )(y).ravel()[0]


class vPyrochlore_sym(vPyrochlore):
    @nn.compact
    def __call__(self, linear_x):
        L = self.L
        x = linear_x.reshape((L, L, L, 4))

        x_inv = vmap(
            lambda a: vmap(
                lambda b: vmap(
                    lambda c: jnp.array([
                        x[(-a + L) % L, (-b + L) % L, (-c + L) % L, 0],
                        x[-a - 1 + L, (-b + L) % L, (-c + L) % L, 1],
                        x[(-a + L) % L, -b - 1 + L, (-c + L) % L, 2],
                        x[(-a + L) % L, (-b + L) % L, -c - 1 + L, 3]
                    ])
                )(jnp.arange(L))
            )(jnp.arange(L))
        )(jnp.arange(L))

        x_sym = jnp.array([
            x, x_inv
        ])

        C_3 = lambda x: vmap(
            lambda a: vmap(
                lambda b: vmap(
                    lambda c: jnp.array([
                        x[c, a, b, 0],
                        x[c, a, b, 3],
                        x[c, a, b, 1],
                        x[c, a, b, 2]
                    ])
                )(jnp.arange(L))
            )(jnp.arange(L))
        )(jnp.arange(L))
        C_3 = vmap(C_3)

        x_1 = x_sym
        x_2 = C_3(x_1)
        x_3 = C_3(x_2)

        x_sym = jnp.array([x_1, x_2, x_3])
        x_sym = x_sym.reshape((2 * 3, L, L, L, 4))

        v = vPyrochlore(L, self.features, self.kernel_size)
        return jnp.mean(
            vmap(lambda x: v(x.ravel()))(x_sym),
            axis=0
        )


class vPyrochlore_cont(nn.Module):
    L: int
    features: Sequence[int] = (8,)
    kernel_size: int = None

    @nn.compact
    def __call__(self, linear_x):
        L = self.L
        x = linear_x.reshape((3, L, L, L, 4))

        def on_bond(a, b):
            return (
                x[(0, *a)] * x[(0, *b)] +
                x[(1, *a)] * x[(1, *b)] +
                x[(2, *a)] * x[(2, *b)]
            )
            # print(result.shape)
            # return result

        def on_unit_cell(cell):
            cell_1 = (cell[0] - 1 + L) % L, cell[1], cell[2]
            cell_2 = cell[0], (cell[1] - 1 + L) % L, cell[2]
            cell_3 = cell[0], cell[1], (cell[2] - 1 + L) % L

            return jnp.array([
                on_bond((*cell, 0), (*cell, 1)),
                on_bond((*cell, 0), (*cell, 2)),
                on_bond((*cell, 0), (*cell, 3)),
                on_bond((*cell, 0), (*cell_1, 1)),
                on_bond((*cell, 0), (*cell_2, 2)),
                on_bond((*cell, 0), (*cell_3, 3)),
                on_bond((*cell, 1), (*cell, 2)),
                on_bond((*cell, 2), (*cell, 3)),
                on_bond((*cell, 3), (*cell, 1)),
                on_bond((*cell_1, 1), (*cell_2, 2)),
                on_bond((*cell_2, 2), (*cell_3, 3)),
                on_bond((*cell_3, 3), (*cell_1, 1))
            ])

        y = vmap(
            lambda a: vmap(
                lambda b: vmap(
                    lambda c: on_unit_cell((a, b, c))
                )(jnp.arange(self.L))
            )(jnp.arange(self.L))
        )(jnp.arange(self.L))

        kernel_size = self.kernel_size or self.L

        # insert batch dimension
        y = y[jnp.newaxis, ...]

        for feat in self.features:
            y = jnp.pad(
                y,
                pad_width=[(0, 0)] + [((kernel_size - 1) // 2, kernel_size // 2)] * 3 + [(0, 0)],
                mode='wrap'
            )

            y = nn.Conv(
                features=feat,
                kernel_size=[kernel_size] * 3,
                padding=[(0, 0)] * 3,
                dtype=jnp.float64
            )(y)

            y = jax.nn.elu(y)

        y = jnp.mean(y, axis=(1, 2, 3)).ravel()

        return nn.Dense(
            features=1,
            use_bias=False,
            dtype=jnp.float64
        )(y).ravel()[0]


class vPyrochloreBi(nn.Module):
    L: int
    features: Sequence[int] = (8,)
    kernel_size: int = None

    @nn.compact
    def __call__(self, linear_x):
        L = self.L
        x = linear_x.reshape((L, L, L, 4))

        def on_bond(a, b):
            return bi_dot_product[x[a], x[b]]

        def on_unit_cell(cell):
            cell_1 = (cell[0] - 1 + L) % L, cell[1], cell[2]
            cell_2 = cell[0], (cell[1] - 1 + L) % L, cell[2]
            cell_3 = cell[0], cell[1], (cell[2] - 1 + L) % L

            return jnp.array([
                on_bond((*cell, 0), (*cell, 1)),
                on_bond((*cell, 0), (*cell, 2)),
                on_bond((*cell, 0), (*cell, 3)),
                on_bond((*cell, 0), (*cell_1, 1)),
                on_bond((*cell, 0), (*cell_2, 2)),
                on_bond((*cell, 0), (*cell_3, 3)),
                on_bond((*cell, 1), (*cell, 2)),
                on_bond((*cell, 2), (*cell, 3)),
                on_bond((*cell, 3), (*cell, 1)),
                on_bond((*cell_1, 1), (*cell_2, 2)),
                on_bond((*cell_2, 2), (*cell_3, 3)),
                on_bond((*cell_3, 3), (*cell_1, 1))
            ])

        y = vmap(
            lambda a: vmap(
                lambda b: vmap(
                    lambda c: on_unit_cell((a, b, c))
                )(jnp.arange(self.L))
            )(jnp.arange(self.L))
        )(jnp.arange(self.L))

        kernel_size = self.kernel_size or self.L

        # insert batch dimension
        y = y[jnp.newaxis, ...]

        for feat in self.features:
            y = jnp.pad(
                y,
                pad_width=[(0, 0)] + [((kernel_size - 1) // 2, kernel_size // 2)] * 3 + [(0, 0)],
                mode='wrap'
            )

            y = nn.Conv(
                features=feat,
                kernel_size=[kernel_size] * 3,
                padding=[(0, 0)] * 3,
                dtype=jnp.float64
            )(y)

            y = jax.nn.elu(y)

        y = jnp.mean(y, axis=(1, 2, 3)).ravel()

        return nn.Dense(
            features=1,
            use_bias=False,
            dtype=jnp.float64
        )(y).ravel()[0]


class vSU2_CNN_dot(nn.Module):
    features: Sequence[int] = (3, 3)
    kernel_size: int = 5
    noise: float = 0.01
    total_bias: float = 0.1
    take_mean: bool = True

    @nn.compact
    def __call__(self, x):

        N = x.shape[0]

        input_kernel = self.param(
            'input_kernel',
            lambda key: self.noise * jax.random.normal(key, (self.features[0], self.kernel_size))
        )

        y = vmap(
            lambda i: jnp.sum(
                input_kernel * (
                    2 * (x[i] == jnp.roll(x, -i)[1:1 + self.kernel_size]) - 1
                ),
                axis=1
            )
        )(jnp.arange(N))

        # y's shape: (sites, features)

        # if skipped, it acts like an svd
        # y = jax.nn.swish(y)

        # y = z + 0.4 * z**2

        for l, feat in enumerate(self.features[1:]):
            # axes -> (N, in, out)
            kernel = self.param(
                f'kernel_{l}',
                lambda key: lecun_normal()(key, (self.kernel_size, y.shape[1], feat))
            )

            y = vmap(
                lambda i: jnp.sum(
                    kernel * jnp.roll(y, i, axis=0)[:self.kernel_size, :, jnp.newaxis],
                    axis=(0, 1)
                )
            )(jnp.arange(N))
            y += self.param(f"bias_{l}", lambda key: jnp.zeros((1, y.shape[1])))

            y = jax.nn.elu(y)
            # y = z + 0.4 * z**2

        # return jnp.mean(y) + self.param("tot_bias", lambda key: self.total_bias)
        if self.take_mean:
            y = jnp.mean(y, axis=0)
        return nn.Dense(
            features=1,
            bias_init=lambda key, shape: self.total_bias,
            dtype=jnp.float64
        )(y).ravel()[0]
        # M = y.shape[0]

        # return jnp.mean(y[:M // 2]) - jnp.mean(y[M // 2:]) + self.param("tot_bias", lambda key: self.total_bias)


class vSU2_sym(vSU2_CNN_dot):
    @nn.compact
    def __call__(self, x):
        v = vSU2_CNN_dot(self.features, self.kernel_size, self.noise, self.total_bias, self.take_mean)

        return (v(x) + v(x[::-1])) / 2


class vSU2_sym2(nn.Module):
    features: Sequence[int] = (3, 3)
    kernel_size: int = 5
    noise: float = 0.01
    total_bias: float = 0.0
    take_mean: bool = True

    @nn.compact
    def __call__(self, x):

        N = x.shape[0]

        input_kernel = self.param(
            'input_kernel',
            lambda key: self.noise * jax.random.normal(key, (self.features[0], self.kernel_size))
        )

        def sym_dot_products(x, i):
            y = 2 * (x[i] == jnp.roll(x, -i)) - 1
            return (
                y[1: 1 + self.kernel_size] +
                y[::-1][:self.kernel_size]
            )

        y = vmap(
            lambda i: jnp.sum(
                input_kernel * sym_dot_products(x, i),
                axis=1
            )
        )(jnp.arange(N))
        # y = vmap(
        #     lambda i: sym_dot_products(x, i).astype(jnp.float64)
        # )(jnp.arange(N))

        # y's shape: (sites, features)
        # if skipped, it acts like an svd
        y = jax.nn.elu(y)

        for l, feat in enumerate(self.features[1:]):
            # axes -> (N, in, out)
            kernel = self.param(
                f'kernel_{l}',
                lambda key: lecun_normal()(key, (self.kernel_size, y.shape[1], feat))
            )

            y = vmap(
                lambda i: jnp.sum(
                    kernel * jnp.roll(y, i, axis=0)[:self.kernel_size, :, jnp.newaxis],
                    axis=(0, 1)
                )
            )(jnp.arange(N))
            y += self.param(f"bias_{l}", lambda key: jnp.zeros((1, y.shape[1])))

            y = jax.nn.elu(y)
            # y = z + 0.4 * z**2

        # return jnp.mean(y) + self.param("tot_bias", lambda key: self.total_bias)
        if self.take_mean:
            y = jnp.mean(y, axis=0)
        return nn.Dense(
            features=1,
            bias_init=lambda key, shape: self.total_bias,
            dtype=jnp.float64
        )(y).ravel()[0]


# class vSU2_sign_sym(vSU2_sym):
#     take_mean: bool = False

#     @nn.compact
#     def __call__(self, x):
#         v = vSU2_sym(self.features, self.kernel_size, self.noise, self.total_bias, self.take_mean)

#         return jnp.prod(jnp.tanh(v(x)))


class vSU2_CNN_triple(nn.Module):
    features: Sequence[int] = (3, 3)
    kernel_size: int = 5
    noise: float = 0.01
    total_bias: float = 0.1

    @nn.compact
    def __call__(self, x):

        N = x.shape[0]

        input_kernel = self.param(
            'input_kernel',
            lambda key: self.noise * jax.random.normal(
                key,
                (self.features[0], self.kernel_size, self.kernel_size)
            )
        )

        y = vmap(
            lambda i: jnp.sum(
                vmap(
                    lambda j: vmap(
                        lambda k: (
                            (
                                input_kernel[:, j, k] -
                                input_kernel[:, k, j]
                            ) * triple_product[
                                x[i],
                                x[(i + 1 + j) % N],
                                x[(i + 1 + k) % N]
                            ]
                        )
                    )(jnp.arange(self.kernel_size))
                )(jnp.arange(self.kernel_size)),
                axis=(0, 1)
            )
        )(jnp.arange(N))

        # y's shape: (sites, features)
        y = jax.nn.softplus(y)

        for l, feat in enumerate(self.features[1:]):
            # axes -> (N, in, out)
            kernel = self.param(
                f'kernel_{l}',
                lambda key: lecun_normal()(key, (self.kernel_size, y.shape[1], feat))
            )

            y = vmap(
                lambda i: jnp.sum(
                    kernel * jnp.roll(y, i, axis=0)[:self.kernel_size, :, jnp.newaxis],
                    axis=(0, 1)
                )
            )(jnp.arange(N))
            y += self.param(f"bias_{l}", lambda key: jnp.zeros((1, y.shape[1])))

            y = jax.nn.softplus(y)

        y = jnp.mean(y, axis=0)
        return nn.Dense(
            features=1,
            bias_init=lambda key, shape: self.total_bias,
            dtype=jnp.float64
        )(y).ravel()[0]


class vSU2_CNN_both(nn.Module):
    features: Sequence[int] = (3, 3)
    kernel_size: int = 5
    noise: float = 0.01
    total_bias: float = 0.1

    @nn.compact
    def __call__(self, x):
        return vSU2_CNN_dot(
            features=self.features,
            kernel_size=self.kernel_size,
            noise=self.noise,
            total_bias=self.total_bias
        )(x) + vSU2_CNN_triple(
            features=self.features,
            kernel_size=self.kernel_size,
            noise=self.noise,
            total_bias=self.total_bias
        )(x)


class vSU2_CNN_full(nn.Module):
    features: Sequence[int] = (3, 3)
    kernel_size: int = 5
    noise: float = 0.01
    # total_bias: float = 0.1

    @nn.compact
    def __call__(self, x):

        N = x.shape[0]

        input_kernel_dot = self.param(
            'input_kernel_dot',
            lambda key: self.noise * jax.random.normal(key, (self.features[0], self.kernel_size))
        )

        y_dot = vmap(
            lambda i: jnp.sum(
                input_kernel_dot * (
                    2 * (x[i] == jnp.roll(x, -i)[1:1 + self.kernel_size]) - 1
                ),
                axis=1
            )
        )(jnp.arange(N))

        input_kernel_triple = self.param(
            'input_kernel_triple',
            lambda key: self.noise * jax.random.normal(
                key,
                (self.features[0], self.kernel_size, self.kernel_size)
            )
        )

        y_triple = vmap(
            lambda i: jnp.sum(
                vmap(
                    lambda j: vmap(
                        lambda k: (
                            (
                                input_kernel_triple[:, j, k] -
                                input_kernel_triple[:, k, j]
                            ) * triple_product[
                                x[i],
                                x[(i + 1 + j) % N],
                                x[(i + 1 + k) % N]
                            ]
                        )
                    )(jnp.arange(self.kernel_size))
                )(jnp.arange(self.kernel_size)),
                axis=(0, 1)
            )
        )(jnp.arange(N))

        # y's shape: (sites, features)
        y = jnp.concatenate([y_dot, y_triple], axis=1)

        y = jax.nn.elu(y)

        for l, feat in enumerate(self.features[1:]):
            # axes -> (N, in, out)
            kernel = self.param(
                f'kernel_{l}',
                lambda key: lecun_normal()(key, (self.kernel_size, y.shape[1], feat))
            )

            y = vmap(
                lambda i: jnp.sum(
                    kernel * jnp.roll(y, i, axis=0)[:self.kernel_size, :, jnp.newaxis],
                    axis=(0, 1)
                )
            )(jnp.arange(N))
            y += self.param(f"bias_{l}", lambda key: jnp.zeros((1, y.shape[1])))

            y = jax.nn.elu(y)

        y = jnp.mean(y, axis=0)

        # M = y.shape[0]

        # return jnp.mean(y[:M // 2]) - jnp.mean(y[M // 2:]) + self.param("tot_bias", lambda key: self.total_bias)

        return nn.Dense(
            features=1,
            use_bias=False,
            dtype=jnp.float64
        )(y).ravel()[0]


class vSU2_CNN_2D(nn.Module):
    L: int
    num_layers: int = 1
    features: int = 3
    kernel_size: int = (5, 5)

    reg2: float = 1.0
    reg: float = 1.0
    reg_length: float = 1.0

    @nn.compact
    def __call__(self, x):

        N = x.shape[1]

        input_kernel = self.param('input_kernel', lambda key: 0.0 * jax.random.normal(key, (self.L, self.L)))

        def f(x):
            y = vmap(
                lambda i: vmap(
                    lambda j: jnp.sum(
                        input_kernel * jnp.sum(
                            x[:, i, j][:, jnp.newaxis, jnp.newaxis] * jnp.roll(jnp.roll(x, -i, axis=1), -j, axis=2),
                            axis=0
                        ),
                    )
                )(jnp.arange(self.L))
            )(jnp.arange(self.L))

            y = y[:, :, jnp.newaxis]

            for l in range(self.num_layers):
                # axes -> (L, L, in, out)
                kernel = self.param(
                    f'kernel_{l}',
                    lambda key: 0.2 * lecun_normal()(key, (self.kernel_size[0], self.kernel_size[1], y.shape[-1], self.features))
                )

                y = vmap(
                    lambda i: vmap(
                        lambda j: jnp.sum(
                            kernel * jnp.roll(
                                jnp.roll(y, i, axis=0),
                                j,
                                axis=1
                            )[:self.kernel_size[0], :self.kernel_size[1], :, jnp.newaxis],
                            axis=(0, 1, 2)
                        )
                    )(jnp.arange(self.L))
                )(jnp.arange(self.L))

                y = jax.nn.elu(y)

            return jnp.sum(y)

        y = x.reshape((3, self.L, self.L))

        y_T = jnp.swapaxes(y, 1, 2)

        y = jnp.mean(vmap(f)(jnp.concatenate(
            (
                y[jnp.newaxis, :, :, :],
                y[jnp.newaxis, :, :, ::-1],
                y[jnp.newaxis, :, ::-1, :],
                y[jnp.newaxis, :, ::-1, ::-1],
                y_T[jnp.newaxis, :, :, :],
                y_T[jnp.newaxis, :, :, ::-1],
                y_T[jnp.newaxis, :, ::-1, :],
                y_T[jnp.newaxis, :, ::-1, ::-1],
            ),
            axis=0
        )))

        regu2 = self.param('regu2', lambda key: self.reg2)
        regu = self.param('regu', lambda key: self.reg)
        regu_length = self.param('regu_length', lambda key: self.reg_length)

        x2 = jnp.sum(x * x, axis=0)

        return y - regu2 * jnp.sum(x2) - regu * jnp.sum(jnp.exp(x2 / (2.0 * regu_length**2)))
        # return y - self.reg2 * jnp.sum(x2) - self.reg * jnp.sum(jnp.exp(x2 / (2.0 * self.reg_length**2)))


class vSU2_cl(nn.Module):
    reg: float = 1.0

    reg2: float = 1.0
    reg3: float = 4.0

    @nn.compact
    def __call__(self, x):
        N = x.shape[1]

        cov = self.param('cov', lambda key: jnp.zeros(N).at[0].set(self.reg))

        A = -(cov @ jnp.sum(
            vmap(
                lambda i: jnp.sum(
                    jnp.roll(x, -i, axis=1) * x[:, i][:, jnp.newaxis],
                    axis=0
                )
            )(jnp.arange(N)),
            axis=0
        )).flatten()[0]

        x2 = jnp.sum(x**2, axis=0)

        B = -jnp.sum(jax.nn.relu(self.reg2 * (jnp.exp(self.reg2 * (x2 - 1.0)) - 1.0)))

        return A + B


cell_ids = jnp.array(list(product(range(2), repeat=3)))
all_unit_cell_confs = jnp.array(list(product(range(4), repeat=4)))
all_unit_cell_vecs = vmap(
    vmap(
        lambda i: tetraeder[i]
    )
)(all_unit_cell_confs)


# class RNNCell(nn.Module):
#     L: int
#     dim_h: int

#     @partial(
#         nn.transforms.scan,
#         variable_broadcast='params',
#         split_rngs={'params': False}
#     )
#     @nn.compact
#     def __call__(self, full_state, n):
#         L = self.L
#         dim_h = self.dim_h

#         i, j, k = cell_ids[n]
#         key, hidden_state, visible_state, logpsi, linear_conf = full_state

#         W_h = self.param("W_h", lambda key_: 0.05 * jax.random.normal(key_, (dim_h, dim_h)))
#         W_v = self.param("W_v", lambda key_: 0.1 * jax.random.normal(key_, (dim_h, 256)))
#         b_h = self.param("b_h", lambda key_: 0.1 * jax.random.normal(key_, (dim_h,)))

#         next_local_hidden_state = W_h @ (
#             hidden_state[(i - 1 + L) % L, j, k] +
#             hidden_state[i, (j - 1 + L) % L, k] +
#             hidden_state[i, j, (k - 1 + L) % L]
#         ) + W_v @ (
#             visible_state[(i - 1 + L) % L, j, k] +
#             visible_state[i, (j - 1 + L) % L, k] +
#             visible_state[i, j, (k - 1 + L) % L]
#         ) + b_h
#         next_local_hidden_state = jax.nn.elu(next_local_hidden_state)
#         hidden_state = hidden_state.at[i, j, k].set(next_local_hidden_state)

#         W_s = self.param("W_s", lambda key_: 0.1 * jax.random.normal(key_, (256, dim_h)))
#         b_s = self.param("b_s", lambda key_: 0.1 * jax.random.normal(key_, (256,)))

#         if linear_conf is None:
#             key, keynow = jax.random.split(key)
#             c = jax.random.categorical(keynow, W_s @ next_local_hidden_state + b_s)
#         else:
#             c = linear_conf[n]
#             logpsi += 0.5 * nn.log_softmax(W_s @ next_local_hidden_state + b_s)[c]

#         visible_state = visible_state.at[i, j, k].set(jax.nn.one_hot(c, 256))

#         return (key, hidden_state, visible_state, logpsi, linear_conf), c


class RNNCell(nn.Module):
    L: int
    dim_h: int

    @partial(
        nn.transforms.scan,
        variable_broadcast='params',
        split_rngs={'params': False}
    )
    @nn.compact
    def __call__(self, full_state, n):
        L = self.L
        dim_h = self.dim_h

        i, j, k = cell_ids[n]
        key, hidden_state, visible_state, logpsi, linear_conf = full_state

        W_h = self.param("W_h", lambda key_: 0.05 * jax.random.normal(key_, (3, dim_h, dim_h)))
        W_v = self.param("W_v", lambda key_: 0.1 * jax.random.normal(key_, (3, dim_h, 4)))
        b_h = self.param("b_h", lambda key_: 0.05 * jax.random.normal(key_, (3, dim_h,)))
        b_v = self.param("b_v", lambda key_: 0.05 * jax.random.normal(key_, (3, dim_h,)))

        def on_prev_state(state, W, b=None):
            prev_states = jnp.array([
                state[(i - 1 + L) % L, j, k],
                state[i, (j - 1 + L) % L, k],
                state[i, j, (k - 1 + L) % L],
            ])

            prev_states_length = jnp.sqrt(jnp.sum(prev_states**2, axis=-1))
            prev_states_dir = prev_states / prev_states_length[..., jnp.newaxis]

            x = prev_states_length
            x = vmap(
                lambda W, x: W @ x
            )(W, x)

            if b is not None:
                x += b

            # return x

            x = jax.nn.elu(x)

            # shape = (4, 3)
            return jnp.sum(x[..., jnp.newaxis] * prev_states_dir, axis=0)

        local_hidden_state = (
            on_prev_state(hidden_state, W_h, b_h) +
            on_prev_state(visible_state, W_v, b_v)
        )

        hidden_state = hidden_state.at[i, j, k].set(local_hidden_state)

        x = vmap(
            vmap(
                lambda basis_vec: vmap(
                    lambda local_hs: local_hs @ basis_vec
                )(local_hidden_state)
            )
        )(all_unit_cell_vecs)

        x = x.reshape((256, dim_h * 4))

        W_s = self.param("W_s", lambda key_: 0.5 * jax.random.normal(key_, (256, dim_h * 4)))

        y = jnp.sum(W_s * x, axis=1)

        if linear_conf is None:
            key, keynow = jax.random.split(key)
            c = jax.random.categorical(keynow, y)
        else:
            c = linear_conf[n]
            logpsi += 0.5 * nn.log_softmax(y)[c]

        visible_state = visible_state.at[i, j, k].set(all_unit_cell_vecs[c])

        return (key, hidden_state, visible_state, logpsi, linear_conf), c


class RNNCell_CNN(nn.Module):
    L: int
    features: Sequence[int] = (8,)
    kernel_size: int = None

    @partial(
        nn.transforms.scan,
        variable_broadcast='params',
        split_rngs={'params': False}
    )
    @nn.compact
    def __call__(self, full_state, n):
        L = self.L
        key, logpsi, linear_conf = full_state

        v_cnn = vPyrochlore(L, self.features, self.kernel_size)

        x = vmap(
            lambda c: v_cnn(linear_conf.at[n].set(c))
        )(jnp.arange(4))

        if key is None:
            logpsi += 0.5 * nn.log_softmax(x)[linear_conf[n]]
        else:
            key, keynow = jax.random.split(key)
            linear_conf = linear_conf.at[n].set(jax.random.categorical(
                keynow, x
            ))

        return (key, logpsi, linear_conf), None


class vRNN_CNN(nn.Module):
    L: int
    features: Sequence[int] = (8,)
    kernel_size: int = None

    @nn.compact
    def __call__(self, x):
        L = self.L
        x = x.ravel()

        if len(x) == 2:
            key = x
            logpsi = None
            linear_conf = -jnp.ones(L**3 * 4, dtype=jnp.int8)
        else:
            key = None
            logpsi = 0.0
            linear_conf = x

        full_state, _ = RNNCell_CNN(L, self.features, self.kernel_size)(
            (key, logpsi, linear_conf),
            jnp.arange(L**3 * 4)
        )
        _, logpsi, linear_conf = full_state

        if key is not None:
            return linear_conf
        else:
            return logpsi


class RNNCell_CNN_Bi(nn.Module):
    L: int
    features: Sequence[int] = (8,)
    kernel_size: int = None

    @partial(
        nn.transforms.scan,
        variable_broadcast='params',
        split_rngs={'params': False}
    )
    @nn.compact
    def __call__(self, full_state, n):
        L = self.L
        key, conf, other_conf, log_prob = full_state

        v_cnn = vPyrochloreBi(L, self.features, self.kernel_size)

        x = vmap(
            lambda c: v_cnn(conf.at[n].set(c))
        )(jnp.arange(8))

        if other_conf is not None:
            x += vmap(
                lambda c: bi_dot_product_log[c, other_conf[n]]
            )(jnp.arange(8))

        if key is None:
            log_prob += nn.log_softmax(x)[conf[n]]
        else:
            key, keynow = jax.random.split(key)
            conf = conf.at[n].set(jax.random.categorical(
                keynow, x
            ))

        return (key, conf, other_conf, log_prob), None


class rhoRNN_CNN_Bi(nn.Module):
    L: int
    features: Sequence[int] = (8,)
    kernel_size: int = None

    @nn.compact
    def __call__(self, x):
        L = self.L

        rnn_cell = RNNCell_CNN_Bi(L, self.features, self.kernel_size)

        if x.dtype == jnp.uint32:
            key = x
            conf = -jnp.ones(L**3 * 4, dtype=jnp.int8)
            other_conf = None
            log_prob = None
        else:
            key = None
            conf = x[0]
            other_conf = None
            log_prob = 0.0

        full_state, _ = rnn_cell(
            (key, conf, other_conf, log_prob),
            jnp.arange(L**3 * 4)
        )
        key, conf_1, _, log_prob = full_state

        if log_prob is None:
            conf = -jnp.ones(L**3 * 4, dtype=jnp.int8)
        else:
            conf = x[1]
        other_conf = conf_1

        full_state, _ = rnn_cell(
            (key, conf, other_conf, log_prob),
            jnp.arange(L**3 * 4)
        )
        _, conf_2, _, log_prob = full_state

        if log_prob is None:
            return jnp.array([
                conf_1,
                conf_2
            ])
        else:
            return log_prob


# class vRNN(nn.Module):
#     L: int
#     dim_h: int

#     @nn.compact
#     def __call__(self, x):
#         L = self.L
#         dim_h = self.dim_h

#         if len(x.shape) == 4:
#             linear_conf = x.reshape((L**3, 4))
#             linear_conf = vmap(
#                 lambda c: (
#                     4**0 * c[3] +
#                     4**1 * c[2] +
#                     4**2 * c[1] +
#                     4**3 * c[0]
#                 )
#             )(linear_conf)
#             key = None
#         else:
#             linear_conf = None
#             key = x

#         hidden_state = jnp.zeros((L, L, L, dim_h))
#         visible_state = jnp.zeros((L, L, L, 4**4))

#         full_state, conf = RNNCell(L, dim_h)(
#             (key, hidden_state, visible_state, 0.0, linear_conf),
#             jnp.arange(L**3)
#         )

#         if linear_conf is None:
#             result = vmap(
#                 lambda c: all_unit_cell_confs[c]
#             )(conf.ravel())

#             return result.reshape((L, L, L, 4))
#         else:
#             return full_state[-2]


class vRNN(nn.Module):
    L: int
    dim_h: int
    key: jnp.ndarray = jax.random.PRNGKey(0)

    @nn.compact
    def __call__(self, x):
        L = self.L
        dim_h = self.dim_h

        if len(x.shape) == 4:
            linear_conf = x.reshape((L**3, 4))
            linear_conf = vmap(
                lambda c: (
                    4**0 * c[3] +
                    4**1 * c[2] +
                    4**2 * c[1] +
                    4**3 * c[0]
                )
            )(linear_conf)
            key = None
        else:
            linear_conf = None
            key = x

        hidden_state = 1e-7 * jax.random.normal(self.key, (L, L, L, dim_h, 3))
        visible_state = 1e-7 * jax.random.normal(self.key, (L, L, L, 4, 3))

        full_state, conf = RNNCell(L, dim_h)(
            (key, hidden_state, visible_state, 0.0, linear_conf),
            jnp.arange(L**3)
        )

        if linear_conf is None:
            result = vmap(
                lambda c: all_unit_cell_confs[c]
            )(conf.ravel())

            return result.reshape((L, L, L, 4))
        else:
            return full_state[-2]
