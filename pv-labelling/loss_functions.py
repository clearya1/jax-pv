import numpy as np
from matplotlib import pyplot as plt
import pickle
from functools import partial
import jax
import jax.numpy as jnp
from jax import config
import optax
from jax import jit, vmap

config.update("jax_enable_x64", True)

import jax_cfd.base as cfd

plt.rcParams.update({
    "text.usetex": True
})

from typing import Callable, Tuple, List, Union
Array = Union[np.ndarray, jnp.ndarray]

import sys
sys.path.append("../jax-pv/")

import helper
from utils import _xy_shift
from time_forward_map import State, advance_velocity_module

@partial(jax.vmap, in_axes=(None, 0, 0, 0,))
def fourier_gaussian_2pi(grid: cfd.grids.Grid, cov: float, mu: Array, gamma):
    """ cov = sigma^2 """

    k_mesh = [2. * jnp.pi * x for x in grid.rfft_mesh()]
    k_abs_sq = k_mesh[0]**2 + k_mesh[1]**2
    
    prefactor = 1. * 0.25 / jnp.pi**2

    shift_factor = jnp.exp(-1.j * (k_mesh[0]*mu[0] + k_mesh[1]*mu[1]))

    gaussian_factor = jnp.exp(-1. * k_abs_sq * cov * 0.5)

    return prefactor * shift_factor * gaussian_factor * grid.shape[0] * grid.shape[1] * gamma

@partial(jax.vmap, in_axes=(None, 0, 0, 0,))
def fourier_gaussian(grid: cfd.grids.Grid, cov: float, mu: Array, gamma):
    """ cov = sigma^2 """

    k_mesh = [2. * jnp.pi * x for x in grid.rfft_mesh()]
    k_abs_sq = k_mesh[0]**2 + k_mesh[1]**2

    L = grid.domain[0][1]
    
    prefactor = 1. / L**2

    shift_factor = jnp.exp(-1.j * (k_mesh[0]*mu[0] + k_mesh[1]*mu[1]))

    gaussian_factor = jnp.exp(-1. * k_abs_sq * cov * 0.5)

    return prefactor * shift_factor * gaussian_factor * grid.shape[0] * grid.shape[1] * gamma

@partial(jit, static_argnums = (3,))
def periodic_gaussian_smear(state: Array, gammas: Array, covs: float, grid: cfd.grids.Grid) -> Array:
    """
      Gaussian smearing of the point vortices' circulation over a periodic grid.
      Returns: The Gaussian smearing of the vortices on a periodic domain.
    """
    n = len(gammas)
    
    mus = jnp.dstack((state[:n], state[n:])).reshape((n, 2))  
    gaussians_fft = fourier_gaussian(grid, covs, mus, gammas)
    gaussians = jnp.fft.irfftn(gaussians_fft, axes=(1,2))

    return jnp.sum(gaussians, axis=0)

def loss_match_pv_upo(x_0: Array, gamma_init: Array, period: float, upo_snapshots, grid: cfd.grids.Grid, forward_map, covs):
    
    samples = len(upo_snapshots)
    state_i = State(steps = 0,
                    T = period / float(samples),
                    x_old = x_0,
                    x_new = x_0,
                    gammas = gamma_init,
                    avg_observable = 0.)

    # initial loss contribution
    gaussian = periodic_gaussian_smear(state_i.x_new, gamma_init, covs, grid)
    loss = jnp.linalg.norm(upo_snapshots[0] - gaussian)**2. / jnp.linalg.norm(upo_snapshots[0])**2

    # add up loss throughout period of UPO
    for i in range(1, samples):
        
        state_i = forward_map(state_i)
        x_i = state_i.x_new
        
        state_i = State(steps = 0,
                    T = period / float(samples),
                    x_old = x_i,
                    x_new = x_i,
                    gammas = gamma_init,
                    avg_observable = 0.)
        
        gaussian = periodic_gaussian_smear(x_i, gamma_init, covs, grid)

        loss += jnp.linalg.norm(upo_snapshots[i] - gaussian)**2. / jnp.linalg.norm(upo_snapshots[i])**2

    return loss

def loss_min_total_circulation(gammas: Array):

    return np.sum(gammas) ** 2

def loss_upo_gaussian(x_0: Array, gamma_init: Array, period: float, shifts: Array, grid: cfd.grids.Grid, forward_map_nobc):
  # Need to use the same areas to allow for vortices to switch (let area = 0.1)!

  cov_same = jnp.ones_like(gamma_init) * 0.1
  
  state_0 = State(steps = 0,
                  T = period,
                  x_old = x_0,
                  x_new = x_0,
                  gammas = gamma_init,
                  avg_observable = 0.)

  gaussian_0 = periodic_gaussian_smear(state_0.x_new, gamma_init, cov_same, grid)
  
  state_T = forward_map_nobc(state_0)

  x_T_s = _xy_shift(state_T.x_new, shifts)

  gaussian_T_s = periodic_gaussian_smear(x_T_s, gamma_init, cov_same, grid)

  return jnp.linalg.norm(gaussian_T_s - gaussian_0)**2 / jnp.linalg.norm(gaussian_0)**2

def x_permute(x: Array, permutation: Array):
  """ Return the permutated state """

  n = len(permutation)
  permutation = jnp.concatenate((permutation, permutation+n))
  return x[permutation]

def loss_upo_vortices(x_0: Array, gamma_init: Array, period: float, shifts: Array, forward_map, permutation):
    """ Term which seeks to match the vortices at the end of the UPO period with the initial vortices,
        but with a permutation of the vortices. And a term which matches up the circulation of any permuted vortices."""
    state_0 = State(steps = 0,
                    T = period,
                    x_old = x_0,
                    x_new = x_0,
                    gammas = gamma_init,
                    avg_observable = 0.)
    
    state_T = forward_map(state_0)

    x_T_s = _xy_shift(state_T.x_new, shifts)
    x_T_s_p = x_permute(x_T_s, permutation)

    return jnp.linalg.norm(x_T_s_p - x_0)**2 / jnp.linalg.norm(x_0)**2


