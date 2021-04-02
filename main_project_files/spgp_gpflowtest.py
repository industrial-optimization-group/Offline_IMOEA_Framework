from dataclasses import dataclass

import numpy as np
import pytest
import tensorflow as tf

import GPflow.gpflow as gpflow
from GPflow.gpflow.config import default_float
from GPflow.gpflow.utilities import to_default_float


@dataclass(frozen=True)
class Datum:
    rng: np.random.RandomState = np.random.RandomState(0)
    X: np.ndarray = rng.randn(100, 2)
    Y: np.ndarray = rng.randn(100, 1)
    Z: np.ndarray = rng.randn(10, 2)
    Xs: np.ndarray = rng.randn(10, 2)
    lik = gpflow.likelihoods.Gaussian()
    kernel = gpflow.kernels.Matern32()


def test_sgpr_qu():
    rng = Datum().rng
    X = to_default_float(rng.randn(100, 2))
    Z = to_default_float(rng.randn(20, 2))
    Y = to_default_float(np.sin(X @ np.array([[-1.4], [0.5]])) + 0.5 * rng.randn(len(X), 1))
    model = gpflow.models.SGPR(
        (X, Y), kernel=gpflow.kernels.SquaredExponential(), inducing_variable=Z
    )

    gpflow.optimizers.Scipy().minimize(model.training_loss, variables=model.trainable_variables)

    qu_mean, qu_cov = model.compute_qu()
    f_at_Z_mean, f_at_Z_cov = model.predict_f(model.inducing_variable.Z, full_cov=True)

    np.testing.assert_allclose(qu_mean, f_at_Z_mean, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(tf.reshape(qu_cov, (1, 20, 20)), f_at_Z_cov, rtol=1e-5, atol=1e-5)