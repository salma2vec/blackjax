import functools
import math

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
from absl.testing import absltest, parameterized
from jax.flatten_util import ravel_pytree

import blackjax
from blackjax.vi.schrodinger_follmer import as_top_level_api as schrodinger_follmer
from tests.fixtures import BlackJAXTest


class SchrodingerFollmerTest(BlackJAXTest):
    @chex.all_variants(with_pmap=True)
    @parameterized.named_parameters(
        ("scalar", ()),
        ("vector", (6,)),
        ("matrix", (2, 3)),
        ("matrix_matching_sample_count", (4, 3)),
        ("tensor", (2, 4, 3)),
    )
    def test_step_estimator_reduces_sample_axis(self, shape):
        position = jnp.linspace(-0.2, 0.3, math.prod(shape)).reshape(shape)
        logdensity = lambda x: -0.5 * jnp.sum(((x - 0.4) / 1.3) ** 2)
        algorithm = blackjax.schrodinger_follmer(logdensity, 4, 4)
        state = algorithm.init(position)._replace(position=position, time=0.25)
        key = self.next_key()
        next_state, info = self.variant(algorithm.step)(key, state)

        # Evaluate the weighted Monte Carlo estimator in flat coordinates.
        drift_key, sde_key = jax.random.split(key)
        flat_position = position.reshape(-1)
        noise = jax.random.normal(drift_key, (4, flat_position.size))
        scale = jnp.sqrt(1 - state.time)
        perturbed = flat_position + scale * noise
        log_weights = -0.5 * jnp.sum(((perturbed - 0.4) / 1.3) ** 2, axis=1)
        log_weights += 0.5 * jnp.sum(perturbed**2, axis=1)
        weights = jax.nn.softmax(log_weights)
        expected_drift = jnp.sum(weights[:, None] * noise, axis=0) / scale
        expected_position = flat_position + 0.25 * expected_drift
        expected_position += 0.5 * jax.random.normal(sde_key, flat_position.shape)

        chex.assert_trees_all_close(
            info.drift, expected_drift.reshape(shape), rtol=1e-5, atol=1e-5
        )
        chex.assert_trees_all_close(
            next_state.position, expected_position.reshape(shape), rtol=1e-5, atol=1e-5
        )
        chex.assert_trees_all_close(next_state.time, 0.5)

    @chex.all_variants(with_pmap=True)
    def test_pytree_samples_match_flat_coordinates(self):
        position = {"bias": jnp.zeros(2), "weights": jnp.zeros((4, 3))}
        flat_position, _ = ravel_pytree(position)
        flat_logdensity = lambda x: -0.5 * jnp.sum((x - 0.4) ** 2)
        logdensity = lambda x: flat_logdensity(ravel_pytree(x)[0])
        algorithm = blackjax.schrodinger_follmer(logdensity, 3, 4)
        flat_algorithm = blackjax.schrodinger_follmer(flat_logdensity, 3, 4)
        state = algorithm.init(position)
        flat_state = flat_algorithm.init(flat_position)
        key = self.next_key()
        samples = self.variant(lambda k, s: algorithm.sample(k, s, 5))(key, state)
        expected = flat_algorithm.sample(key, flat_state, 5)
        flat_samples = jax.vmap(lambda x: ravel_pytree(x)[0])(samples.position)
        chex.assert_trees_all_close(
            flat_samples, expected.position, rtol=1e-5, atol=1e-5
        )
        chex.assert_trees_all_close(samples.time, expected.time)

    @chex.all_variants(with_pmap=True)
    def test_recover_posterior(self):
        """Simple Normal mean test"""

        ndim = 2

        rng_key_observed, rng_key_init = jax.random.split(self.next_key(), 2)
        true_mu = jnp.arange(ndim, dtype=float)
        true_cov = jnp.array([[1.0, 0.5], [0.5, 2.0]])
        true_prec = jnp.linalg.inv(true_cov)

        def logp_posterior_conjugate_normal_model(
            observed, prior_mu, prior_prec, true_prec
        ):
            n = observed.shape[0]
            posterior_cov = jnp.linalg.inv(prior_prec + n * true_prec)
            posterior_mu = (
                posterior_cov
                @ (
                    prior_prec @ prior_mu[:, None]
                    + n * true_prec @ observed.mean(0)[:, None]
                )
            )[:, 0]
            return posterior_mu

        def logp_unnormalized_posterior(x, observed, prior_mu, prior_prec, true_cov):
            logp = 0.0
            logp += stats.multivariate_normal.logpdf(x, prior_mu, prior_prec)
            logp += stats.multivariate_normal.logpdf(observed, x, true_cov).sum()
            return logp

        prior_mu = jnp.zeros(ndim)
        prior_prec = jnp.eye(ndim)

        # Simulate the data
        observed = jax.random.multivariate_normal(
            rng_key_observed, true_mu, true_cov, shape=(25,)
        )

        logp_model = functools.partial(
            logp_unnormalized_posterior,
            observed=observed,
            prior_mu=prior_mu,
            prior_prec=prior_prec,
            true_cov=true_cov,
        )

        initial_position = jnp.zeros((ndim,))
        posterior_mu = logp_posterior_conjugate_normal_model(
            observed, prior_mu, prior_prec, true_prec
        )

        # n_steps=100 SDE steps with n_inner_samples=100 drift Monte Carlo
        # samples — previously (50, 25), which gave a variational bias of order
        # ~0.13 on the second posterior component for an unlucky `BlackJAXTest`
        # date-rotated seed (e.g. 2026-05-20 → seed=20260520). Doubling each
        # parameter cuts the worst-seed max-diff from 0.131 → 0.060, giving a
        # comfortable safety margin under `atol=0.1` across 20 sampled daily
        # seeds for ~0.25s of extra wall time.
        schrodinger_follmer_algo = schrodinger_follmer(logp_model, 100, 100)

        initial_state = schrodinger_follmer_algo.init(initial_position)
        schrodinger_follmer_algo_sample = self.variant(
            lambda k, s: schrodinger_follmer_algo.sample(k, s, 100)
        )
        sampled_states = schrodinger_follmer_algo_sample(rng_key_init, initial_state)
        sampled_position = sampled_states.position
        chex.assert_trees_all_close(
            sampled_position.mean(0), posterior_mu, rtol=1e-2, atol=1e-1
        )

        # make sure basic interface is independently covered
        _ = schrodinger_follmer_algo.step(rng_key_init, initial_state)


if __name__ == "__main__":
    absltest.main()
