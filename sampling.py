from .heisenberg import bi_dot_product_log
import jax
import jax.numpy as jnp
from jax import vmap
from dataclasses import dataclass


@dataclass
class MCMCSampler:
    num_sites: int
    num_markov_chains: int
    num_samples: int
    num_thermalization_steps: int
    local_hilbert_space: int = 4
    num_sweeps: int = 1

    def next_conf(self, key, conf, log_weight, accepted):
        keys = jax.random.split(key, 4)

        N = len(conf)
        pos = jax.random.randint(keys[1], (1,), 0, N)
        shift = jax.random.randint(keys[2], (1,), 1, self.local_hilbert_space)

        proposed_conf = conf.at[pos].set((conf[pos] + shift) % self.local_hilbert_space)
        proposed_log_weight = 2.0 * self.v.apply(self.params, proposed_conf)

        alpha = jnp.exp(proposed_log_weight - log_weight)
        alpha = jax.lax.min(1.0, alpha)

        return jax.lax.cond(
            jax.random.uniform(keys[3]) < alpha,
            lambda _: (keys[0], proposed_conf, proposed_log_weight, accepted + 1),
            lambda _: (keys[0], conf, log_weight, accepted),
            None
        )

    def sweep(self, key, conf, log_weight):
        N = len(conf)

        return jax.lax.fori_loop(
            0,
            N * self.num_sweeps,
            lambda i, state: self.next_conf(*state),
            (key, conf, log_weight, 0)
        )

    def update_step(self, i, full_state):
        key, conf, log_weight, all_confs, acceptance_rate = full_state

        key, conf, log_weight, accepted = self.sweep(key, conf, log_weight)
        accepted /= len(conf) * self.num_sweeps
        acceptance_rate += accepted

        all_confs = all_confs.at[i].set(conf)

        return (key, conf, log_weight, all_confs, acceptance_rate)

    def __call__(self, v, params, key):
        self.v = v
        self.params = params
        all_confs = jnp.empty(
            (
                self.num_markov_chains,
                self.num_samples // self.num_markov_chains,
                self.num_sites
            ),
            dtype=jnp.int8
        )

        key, keynow = jax.random.split(key)

        initial_confs = jax.random.randint(
            keynow,
            (self.num_markov_chains, self.num_sites),
            0, self.local_hilbert_space,
            dtype=jnp.int8
        )

        keys = jax.random.split(key, self.num_markov_chains)

        keys, confs, log_weights = vmap(
            lambda keys, confs: jax.lax.fori_loop(
                0,
                self.num_thermalization_steps,
                lambda i, state: self.sweep(*state)[:3],
                (keys, confs, -20.0)
            )
        )(keys, initial_confs)

        keys, confs, log_weights, all_confs, acceptance_rate = vmap(
            lambda keys, confs, log_weight, all_confs: jax.lax.fori_loop(
                0,
                self.num_samples // self.num_markov_chains,
                self.update_step,
                (keys, confs, log_weight, all_confs, 0.0)
            )
        )(keys, confs, log_weights, all_confs)

        acceptance_rate = jnp.sum(acceptance_rate) / self.num_samples

        return all_confs.reshape(self.num_samples, self.num_sites), acceptance_rate


@dataclass
class MCMCSamplerBi:
    num_sites: int
    num_markov_chains: int
    num_samples: int
    num_thermalization_steps: int
    local_hilbert_space: int = 8
    num_sweeps: int = 1

    def log_probability(self, bi_conf):
        r, rp = bi_conf
        return (
            self.v.apply(self.params, r) +
            self.v.apply(self.params, rp) +
            jnp.sum(vmap(lambda r, rp: bi_dot_product_log[r, rp])(r, rp))
        )

    def next_conf(self, key, bi_conf, log_weight, accepted):
        keys = jax.random.split(key, 5)

        N = bi_conf.shape[1]
        idx = jax.random.randint(keys[1], (1,), 0, 2)
        pos = jax.random.randint(keys[2], (1,), 0, N)
        shift = jax.random.randint(keys[3], (1,), 1, self.local_hilbert_space)

        proposed_bi_conf = bi_conf.at[idx, pos].set((bi_conf[idx, pos] + shift) % self.local_hilbert_space)
        proposed_log_weight = self.log_probability(proposed_bi_conf)

        alpha = jnp.exp(proposed_log_weight - log_weight)
        alpha = jax.lax.min(1.0, alpha)

        return jax.lax.cond(
            jax.random.uniform(keys[4]) < alpha,
            lambda _: (keys[0], proposed_bi_conf, proposed_log_weight, accepted + 1),
            lambda _: (keys[0], bi_conf, log_weight, accepted),
            None
        )

    def sweep(self, key, bi_conf, log_weight):
        N = bi_conf.shape[1]

        return jax.lax.fori_loop(
            0,
            N * self.num_sweeps,
            lambda i, state: self.next_conf(*state),
            (key, bi_conf, log_weight, 0)
        )

    def update_step(self, i, full_state):
        key, bi_conf, log_weight, all_confs, acceptance_rate = full_state

        key, bi_conf, log_weight, accepted = self.sweep(key, bi_conf, log_weight)
        accepted /= bi_conf.shape[1] * self.num_sweeps
        acceptance_rate += accepted

        all_confs = all_confs.at[i].set(bi_conf)

        return (key, bi_conf, log_weight, all_confs, acceptance_rate)

    def __call__(self, v, params, key):
        self.v = v
        self.params = params
        all_confs = jnp.empty(
            (
                self.num_markov_chains,
                self.num_samples // self.num_markov_chains,
                2,
                self.num_sites
            ),
            dtype=jnp.int8
        )

        key, keynow = jax.random.split(key)

        initial_confs = jax.random.randint(
            keynow,
            (self.num_markov_chains, self.num_sites),
            0, self.local_hilbert_space,
            dtype=jnp.int8
        )
        initial_confs = initial_confs[:, jnp.newaxis, :]
        initial_confs = jnp.repeat(initial_confs, 2, axis=1)

        keys = jax.random.split(key, self.num_markov_chains)

        keys, confs, log_weights = vmap(
            lambda keys, confs: jax.lax.fori_loop(
                0,
                self.num_thermalization_steps,
                lambda i, state: self.sweep(*state)[:3],
                (keys, confs, -20.0)
            )
        )(keys, initial_confs)

        keys, confs, log_weights, all_confs, acceptance_rate = vmap(
            lambda keys, confs, log_weight, all_confs: jax.lax.fori_loop(
                0,
                self.num_samples // self.num_markov_chains,
                self.update_step,
                (keys, confs, log_weight, all_confs, 0.0)
            )
        )(keys, confs, log_weights, all_confs)

        acceptance_rate = jnp.sum(acceptance_rate) / self.num_samples

        return all_confs.reshape(self.num_samples, 2, self.num_sites), acceptance_rate


class ExactSampler:
    def __init__(self, num_sites, local_hilbert_space=4):
        self.N = num_sites
        self.local_hilbert_space = local_hilbert_space
        self.num_samples = local_hilbert_space**num_sites

    def conf_s(self, s):
        return (s // self.local_hilbert_space**jnp.arange(self.N)) % self.local_hilbert_space

    def __call__(self, v, params):
        confs = vmap(self.conf_s)(jnp.arange(self.num_samples))

        v_vec = jnp.exp(vmap(lambda conf: v.apply(params, conf))(confs))
        v_vec /= jnp.linalg.norm(v_vec)

        return confs, abs(v_vec)**2


def normalize_conf(conf):
    # shape: (3, N)

    return conf / jnp.sqrt(
        conf[0]**2 + conf[1]**2 + conf[2]**2
    )


@dataclass
class MALASampler:
    num_sites: int
    num_markov_chains: int
    num_samples: int
    num_thermalization_steps: int
    # sigma_sq: float

    def next_conf(self, key, conf, accepted):
        keys = jax.random.split(key, 3)

        grad_conf = lambda conf: self.rho.grad_log_probability(self.params, conf)
        drift = lambda conf: 0.5 * self.sigma_sq * vmap(
            lambda grad_conf: self.Lambda @ grad_conf
        )(grad_conf(conf))

        noise = jax.random.normal(
            keys[1],
            shape=conf.shape
        )
        noise = jnp.sqrt(self.sigma_sq) * vmap(
            lambda noise: self.L @ noise
        )(noise)

        proposed_conf = conf + drift(conf) + noise
        proposed_conf = normalize_conf(proposed_conf)

        log_p = lambda conf: self.rho.log_probability(self.params, conf)
        log_q = lambda y, x: -1 / (2 * self.sigma_sq) * jnp.sum(vmap(
            lambda vec: vec @ self.Lambda_inv @ vec
        )(y - (x + drift(x)))).flatten()[0]

        alpha = jnp.exp(
            log_p(proposed_conf) + log_q(conf, proposed_conf) -
            log_p(conf) - log_q(proposed_conf, conf)
        )
        alpha = jax.lax.min(1.0, alpha)

        return jax.lax.cond(
            jax.random.uniform(keys[2]) < alpha,
            lambda _: (keys[0], proposed_conf, accepted + 1),
            lambda _: (keys[0], conf, accepted),
            None
        )

    def sweep(self, key, conf):
        N = conf.shape[-1] // 2

        return jax.lax.fori_loop(
            0,
            N,
            lambda i, state: self.next_conf(*state),
            (key, conf, 0)
        )

    def update_step(self, i, full_state):
        key, conf, all_confs, acceptance_rate = full_state

        key, conf, accepted = self.sweep(key, conf)
        accepted /= conf.shape[-1] // 2
        acceptance_rate += accepted

        all_confs = all_confs.at[i].set(conf)

        return key, conf, all_confs, acceptance_rate

    def __call__(self, rho, params, key, sigma_sq=1e-2):
        self.rho = rho
        self.params = params
        self.sigma_sq = sigma_sq
        all_confs = jnp.empty(
            (
                self.num_markov_chains,
                self.num_samples // self.num_markov_chains,
                3,
                2 * self.num_sites
            ),
        )

        self.Lambda = jnp.eye(2 * self.num_sites) / 5
        self.Lambda_inv = jnp.linalg.inv(self.Lambda)
        self.L = jnp.linalg.cholesky(self.Lambda)

        key, keynow = jax.random.split(key)
        initial_confs = jax.random.normal(
            keynow,
            shape=(self.num_markov_chains, 3, 2 * self.num_sites)
        )
        initial_confs = vmap(normalize_conf)(initial_confs)

        keys = jax.random.split(key, self.num_markov_chains)

        keys, confs = vmap(
            lambda keys, confs: jax.lax.fori_loop(
                0,
                self.num_thermalization_steps,
                lambda i, state: self.sweep(*state)[:2],
                (keys, confs)
            )
        )(keys, initial_confs)

        keys, confs, all_confs, acceptance_rate = vmap(
            lambda keys, confs, all_confs: jax.lax.fori_loop(
                0,
                self.num_samples // self.num_markov_chains,
                self.update_step,
                (keys, confs, all_confs, 0.0)
            )
        )(keys, confs, all_confs)

        acceptance_rate = jnp.sum(acceptance_rate) / self.num_samples

        return all_confs.reshape(self.num_samples, 3, 2, self.num_sites), acceptance_rate
