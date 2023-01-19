from .heisenberg import (
    heisenberg_tensor_signed,
    heisenberg_bi_tensor
)
from .OperatorFunctionality import (
    OperatorFunctionality,
    OperatorFunctionalityCont,
    OperatorFunctionalityBi
)
from dataclasses import dataclass
from jax import vmap
import jax.numpy as jnp


@dataclass
class PyrochloreLattice(OperatorFunctionality):
    L: int

    def local_energy(self, v, params, linear_conf, v_val):
        return self.local_energy_with_phase(v, params, linear_conf, v_val)

    def local_energy_with_phase(self, v, params, linear_conf, v_val, v_phase=None, params_phase=None):
        conf = linear_conf.reshape((self.L, self.L, self.L, 4))

        if v_phase is not None:
            phase = v_phase.apply(params_phase, conf)

        def on_matrix_element(a, b, ip, jp, sign):
            mat_el = heisenberg_tensor_signed[
                sign,
                conf[a],
                conf[b],
                ip,
                jp
            ]
            confp = conf.at[a].set(ip).at[b].set(jp)
            if v_phase is None:
                return mat_el * jnp.exp(
                    v.apply(params, confp) - v_val
                )
            else:
                return mat_el * jnp.exp(
                    v.apply(params, confp) - v_val +
                    1j * jnp.pi * (v_phase.apply(params_phase, confp) - phase)
                )

        def on_bond(a, b, sign):
            return jnp.sum(
                vmap(
                    lambda ip: vmap(
                        lambda jp: on_matrix_element(a, b, ip, jp, sign)
                    )(jnp.arange(4))
                )(jnp.arange(4))
            )

        def on_unit_cell(cell):
            L = self.L
            cell_1 = (cell[0] - 1 + L) % L, cell[1], cell[2]
            cell_2 = cell[0], (cell[1] - 1 + L) % L, cell[2]
            cell_3 = cell[0], cell[1], (cell[2] - 1 + L) % L

            return (
                on_bond((*cell, 0), (*cell, 1), 1) +
                on_bond((*cell, 0), (*cell, 2), 1) +
                on_bond((*cell, 0), (*cell, 3), 0) +
                on_bond((*cell, 0), (*cell_1, 1), 1) +
                on_bond((*cell, 0), (*cell_2, 2), 1) +
                on_bond((*cell, 0), (*cell_3, 3), 0) +
                on_bond((*cell, 1), (*cell, 2), 0) +
                on_bond((*cell, 2), (*cell, 3), 1) +
                on_bond((*cell, 3), (*cell, 1), 1) +
                on_bond((*cell_1, 1), (*cell_2, 2), 0) +
                on_bond((*cell_2, 2), (*cell_3, 3), 1) +
                on_bond((*cell_3, 3), (*cell_1, 1), 1)
            )

        return jnp.mean(
            vmap(
                lambda x: vmap(
                    lambda y: vmap(
                        lambda z: on_unit_cell((x, y, z))
                    )(jnp.arange(self.L))
                )(jnp.arange(self.L))
            )(jnp.arange(self.L))
        ) / 4 / 4


@dataclass
class PyrochloreLatticeCont(OperatorFunctionalityCont):
    L: int

    def matrix_element(self, linear_r, linear_rp):
        L = self.L
        r = jnp.moveaxis(linear_r.reshape((3, L, L, L, 4)), 0, -1)
        rp = jnp.moveaxis(linear_rp.reshape((3, L, L, L, 4)), 0, -1)

        def on_bond(i, j):
            a = r[i]
            b = r[j]
            ap = rp[i]
            bp = rp[j]

            return (
                (a + ap) @ (b + bp) +
                (ap @ b) * (bp @ a) -
                (a @ b) * (ap @ bp)
            ) / (
                (1 + (a @ ap)) *
                (1 + (b @ bp))
            ) / 16

        def on_unit_cell(cell):
            cell_1 = (cell[0] - 1 + L) % L, cell[1], cell[2]
            cell_2 = cell[0], (cell[1] - 1 + L) % L, cell[2]
            cell_3 = cell[0], cell[1], (cell[2] - 1 + L) % L

            return (
                on_bond((*cell, 0), (*cell, 1)) +
                on_bond((*cell, 0), (*cell, 2)) +
                on_bond((*cell, 0), (*cell, 3)) +
                on_bond((*cell, 0), (*cell_1, 1)) +
                on_bond((*cell, 0), (*cell_2, 2)) +
                on_bond((*cell, 0), (*cell_3, 3)) +
                on_bond((*cell, 1), (*cell, 2)) +
                on_bond((*cell, 2), (*cell, 3)) +
                on_bond((*cell, 3), (*cell, 1)) +
                on_bond((*cell_1, 1), (*cell_2, 2)) +
                on_bond((*cell_2, 2), (*cell_3, 3)) +
                on_bond((*cell_3, 3), (*cell_1, 1))
            )

        return jnp.mean(
            vmap(
                lambda x: vmap(
                    lambda y: vmap(
                        lambda z: on_unit_cell((x, y, z))
                    )(jnp.arange(L))
                )(jnp.arange(L))
            )(jnp.arange(L))
        ) / 4 / 4


@dataclass
class PyrochloreLatticeBi(OperatorFunctionalityBi):
    L: int

    def matrix_element(self, linear_conf, linear_confp):
        L = self.L
        conf = linear_conf.reshape((L, L, L, 4))
        confp = linear_confp.reshape((L, L, L, 4))

        def on_bond(i, j):
            a = conf[i]
            b = conf[j]
            ap = confp[i]
            bp = confp[j]

            return heisenberg_bi_tensor[a, b, ap, bp]

        def on_unit_cell(cell):
            cell_1 = (cell[0] - 1 + L) % L, cell[1], cell[2]
            cell_2 = cell[0], (cell[1] - 1 + L) % L, cell[2]
            cell_3 = cell[0], cell[1], (cell[2] - 1 + L) % L

            return (
                on_bond((*cell, 0), (*cell, 1)) +
                on_bond((*cell, 0), (*cell, 2)) +
                on_bond((*cell, 0), (*cell, 3)) +
                on_bond((*cell, 0), (*cell_1, 1)) +
                on_bond((*cell, 0), (*cell_2, 2)) +
                on_bond((*cell, 0), (*cell_3, 3)) +
                on_bond((*cell, 1), (*cell, 2)) +
                on_bond((*cell, 2), (*cell, 3)) +
                on_bond((*cell, 3), (*cell, 1)) +
                on_bond((*cell_1, 1), (*cell_2, 2)) +
                on_bond((*cell_2, 2), (*cell_3, 3)) +
                on_bond((*cell_3, 3), (*cell_1, 1))
            )

        return jnp.mean(
            vmap(
                lambda x: vmap(
                    lambda y: vmap(
                        lambda z: on_unit_cell((x, y, z))
                    )(jnp.arange(L))
                )(jnp.arange(L))
            )(jnp.arange(L))
        ) / 4 / 4
