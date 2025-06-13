import numpy as np
from matplotlib import pyplot as plt
import pickle
from functools import partial
import jax
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

import jax_cfd.base as cfd
import jax_cfd.spectral as spectral

from typing import Union, Callable
Array = Union[np.ndarray, jnp.ndarray]

import sys
sys.path.append("../UPOs/")

import vortex_analysis as va
import opt_newt_jaxcfd.newton.newton as nt
import opt_newt_jaxcfd.newton.newton_spectral as nt_sp
import opt_newt_jaxcfd.interact_jaxcfd.time_forward_map_spectral as tfm
import opt_newt_jaxcfd.interact_jaxcfd.interact_spectral as insp

from scipy import optimize

def simulate_period(omega: Array,
                    period: float,
                    N_samples: int,
                    dt_stable: float,
                    Re: float,
                    grid: cfd.grids.Grid) -> Array:
  
  T_per_sample = period / N_samples

  temp = max(int(T_per_sample / dt_stable), 1)

  dt_exact = T_per_sample / temp

  while dt_exact > dt_stable:        # make sure that we always meet CFL condition
    dt_exact /= 2.

  Nt_per_sample = round(T_per_sample / dt_exact)    # need round here or else risking sending .999 -> 0 rather than 1

  forward_map = tfm.generate_time_forward_map(
        dt_exact,
        Nt_per_sample,
        grid,
        1./Re
  )
  trajectory = np.zeros((N_samples,)+omega.shape)
  omega_hat = jnp.fft.rfftn(omega)
  for i in range(N_samples):
    omega_hat = forward_map(omega_hat)
    trajectory[i] = jnp.fft.irfftn(omega_hat)

  return jnp.array(trajectory)

def roll_and_shift(snapshots: Array, index: int, shift: float, grid: cfd.grids.Grid):
  """ 
      Roll the snapshots so that the starting snapshot is the index-th snapshot
      Apply the shift to the snapshots that got rolled so that the rolled period is dynamical
  """

  snapshots = np.array(snapshots)
  vmapped_shift = jax.vmap(insp.x_shift, in_axes=(0, None, None))
  snapshots[:index] = vmapped_shift(snapshots[:index], grid, -shift)
  snapshots = np.roll(snapshots, -index, axis=0)

  return snapshots

def periodic_euclidean_norm(vort1: Array, vort2: Array, L):
    """ Compute the euclidean distance between two vortices, allowing for periodic bc """

    total = 0
    for a, b in zip(vort1, vort2):
        delta = abs(b - a)
        if delta > L - delta:
            delta = L - delta
        total += delta ** 2
    return total ** 0.5

def best_permutation(state1: Array, state2: Array, Lx):
    """ Find the best permutation of state2 that minimises the periodic euclidean distance with state1.
        Both states are assumed to be in the form [x1, x2, ..., xn, y1, y2, ..., yn].
        The permutation is returned as a list of indices.
        We use the Hungarian method to solve the linear assignment problem.
    """

    n = int(len(state1)/2)
    permutation = np.zeros(n, dtype=int)
    cost_matrix = np.zeros((n,n), dtype=float)
    for i in range(n):
        vort1 = [state1[i], state1[i+n]]
        for j in range(n):
            vort2 = [state2[j], state2[j+n]]
            cost_matrix[i,j] = periodic_euclidean_norm(vort1, vort2, Lx)

    # Hungarian method for linear assignment problem
    _, best_assignment = optimize.linear_sum_assignment(cost_matrix)
    return best_assignment