import numpy as np
import jax
from jax import jit
import jax.numpy as jnp
from functools import partial
from typing import Union

import VorticesMotion as vm
import velocity_transforms as vt
import utils

Array = Union[np.ndarray, jnp.ndarray]

from jax import config
config.update("jax_enable_x64", True)

def velocities_for_jacobian(params: Array, gammas: Array, indices: Array, polar_vel: float) -> Array:
  """
    Returns the induced velocity in the corotating frame for every vortex in the system
  """

  def _f(carry, x):
    #a, b = jnp.split(carry, 2)
    u, v = vm._induced_velocity(carry, gammas, x)
    return carry, jnp.array([u, v])
  
  _, k = jax.lax.scan(_f, params, indices)     # loops over each vortex and computes its induced velocity
  velocities = jnp.transpose(k)                      # need to transpose the np.stacked velocities
  
  a, b = jnp.split(params, 2)
  #_, polar_vel = cartesian_to_polar(a,b,velocities[0], velocities[1])  #should already be centered on c.o.m
  
  # transform velocities to co rotating frame
  velocities = velocities.at[0].set(velocities[0] + polar_vel * b)
  velocities = velocities.at[1].set(velocities[1] - polar_vel * a)
  
  return jnp.concatenate((velocities[0], velocities[1]))
  
def stability_eigs(params: Array) -> [Array, Array]:
  """ Computes the eigenvalues and eigenvectors of an equilibrium's stability matrix """
  
  n = int(len(params)*0.5)
  ind = utils.indices(n)
  
  polar_vel = vt.mean_angular_velocity(params, jnp.ones(n), ind, n)
  
  jacobian = jax.jacfwd(velocities_for_jacobian)(params, jnp.ones(n), ind, polar_vel)
  
  eigenvalues, eigenvectors = jnp.linalg.eig(jacobian)
  
  return eigenvalues * polar_vel, eigenvectors

def stability_directions(params: Array) -> [Array, Array, Array]:
  """ Computes the stable and unstable directions of an equilibrium, tolerance is set to 10^{-5} """
  
  eigenvalues, eigenvectors = stability_eigs(params)
  n = int(len(params)*0.5)
  
  bool_if_stable = jnp.where(jnp.real(eigenvalues) < -1e-5, np.full(2*n, True), np.full(2*n, False))
  bool_if_unstable = jnp.where(jnp.real(eigenvalues) > 1e-5, np.full(2*n, True), np.full(2*n, False))
  bool_if_zero = jnp.where(jnp.abs(eigenvalues) < 1e-5, np.full(2*n, True), np.full(2*n, False))
  
  stable_eigenvectors = jnp.transpose(eigenvectors[:, bool_if_stable])
  unstable_eigenvectors = jnp.transpose(eigenvectors[:, bool_if_unstable])
  zero_eigenvectors = jnp.transpose(eigenvectors[:, bool_if_zero])
  
  return eigenvalues, stable_eigenvectors, unstable_eigenvectors, zero_eigenvectors

