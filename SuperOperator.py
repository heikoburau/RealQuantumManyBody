from QuantumExpression import PauliExpression
from .OperatorFunctionality import OperatorFunctionality
from jax import vmap
import jax.numpy as jnp
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, Sequence


U = 3**0.5 * jnp.array([
    [0, 0, 1],
    [(8 / 9)**0.5, 0, -1 / 3],
    [-(2 / 9)**0.5, (2 / 3)**0.5, -1 / 3],
    [-(2 / 9)**0.5, -(2 / 3)**0.5, -1 / 3]
])
U = jnp.concatenate((jnp.ones((4, 1)), -U), axis=1) / 2
U_full = jnp.kron(U, U)


@dataclass
class SuperOperator(OperatorFunctionality):
    sites_ij: Sequence[Tuple[int, int]]
    matrices_ij: Sequence[jnp.ndarray]

    @staticmethod
    def from_expr(expr, part):
        matrices_dict_ij = defaultdict(lambda: jnp.zeros((16, 16)))

        for term in expr:
            sites = []
            compact_term = 1
            for i, (site, op) in enumerate(term.pauli_string):
                sites.append(site)

                compact_term *= PauliExpression(i, op)

            matrix = term.coefficient.real * compact_term.sparse_matrix(len(sites), "paulis").todense()

            if part == "real":
                matrix = matrix.real
            if part == "imag":
                matrix = matrix.imag

            matrix = U_full @ matrix @ U_full.T
            matrices_dict_ij[tuple(sorted(sites))] += matrix

        sites_ij = jnp.array(sorted(list(matrices_dict_ij)), dtype=jnp.int16)
        matrices_ij = jnp.array([matrices_dict_ij[tuple(ij)] for ij in sites_ij])

        return SuperOperator(sites_ij, matrices_ij)

    @property
    def matrix(self):
        N = jnp.amax(self.sites_ij) + 1

        result = jnp.zeros((4**N, 4**N))

        for (i, j), mat in zip(self.sites_ij, self.matrices_ij):
            T = jnp.zeros((16, 4**N))

            for k in range(16):
                T = T.at[k, (k // 4) * 4**i + (k % 4) * 4**j].set(1.0)

            result += T.T @ mat @ T

        return result

    def conf_prime_list(self, conf):
        N = conf.shape[0]
        b = vmap(
            lambda sites: vmap(
                lambda col: conf.at[sites[0]].set(col // 4).at[sites[1]].set(col % 4)
            )(jnp.arange(16))
        )(self.sites_ij)

        return b.reshape((16 * len(self.sites_ij), N))

    def matrix_elements(self, conf):
        b = vmap(
            lambda sites, matrix: vmap(
                lambda col: matrix[(conf[sites[0]] * 4) + conf[sites[1]], col]
            )(jnp.arange(16))
        )(self.sites_ij, self.matrices_ij)

        return b.reshape(16 * len(self.sites_ij))

    def local_energy(self, v, params, conf, v_val):
        v_p_list = vmap(lambda conf: v.apply(params, conf))(self.conf_prime_list(conf))

        return self.matrix_elements(conf) @ jnp.exp(v_p_list - v_val)

    def local_energy_with_phase(self, v, params, conf, v_val, v_phase, params_phase):
        conf_prime_list = self.conf_prime_list(conf)

        v_p_list = vmap(lambda conf: v.apply(params, conf))(conf_prime_list)
        phase_p_list = vmap(lambda conf: v_phase.apply(params_phase, conf))(conf_prime_list)

        phase = v_phase.apply(params_phase, conf)

        return self.matrix_elements(conf) @ jnp.exp(v_p_list - v_val + 1j * jnp.pi * (phase_p_list - phase))
