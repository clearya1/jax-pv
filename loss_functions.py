import numpy as np
import jax
from jax import jit
import jax.numpy as jnp
from functools import partial
from typing import Callable, Tuple, List, Union
import utils
import VorticesMotion as vm
import jax.scipy.stats as stats

Array = Union[np.ndarray, jnp.ndarray]

from jax.config import config
config.update("jax_enable_x64", True)

# ----- Grid based Gaussian observable --------

@jit
def gaussian_smear(state: Array, grid: Array, cov: float = 1.) -> Array:
  """
      Gaussian smearing of the point vortices' circulation over a 64 x 64 grid.
      Returns: The Gaussian smearing of the vortices
  """
  
  n = int(len(state)*0.5)

  def _f(x, y):
    return stats.multivariate_normal.pdf(grid, mean=jnp.array([x, y]), cov=cov)
    
  vmapped_f = jax.vmap(_f)
  
  return jnp.sum(vmapped_f(state[:n], state[n:]), axis=0)
 
# ----------- Helper functions for REQ search --------

@jit
def relative_distances(state: Array, indices: Array) -> Array:
  """ Returns the n(n-1)/2 relative distances in the x and y components in the following order of combinations given by indices: (1,2), (1,3), ..., (1, n), (2, 3), ..., (n-1, n). Distance of second entry relative to the first entry. """
  
  n = int(len(state)*0.5)
  def _f(x):
    return state[x[1]] - state[x[0]], state[x[1]+n] - state[x[0]+n]
    
  vmapped_f = jax.vmap(_f)
  
  rel_x, rel_y = vmapped_f(indices)
  return jnp.sqrt( rel_x**2 + rel_y**2 )
  
def sigmoid(x: float, delta: float) -> float:
  """ Compute the sigmoid function """
  
  return 1. / (1. + jnp.exp(-x/delta))
  
def delta_approx(x: float, delta: float) -> float:
  """ Compute the delta function approximation """
  
  return jnp.exp(-(x/delta)**2)
  
  
def mexican_hat(x: float, delta: float) -> float:
  """ Compute the mexican hat function """
  
  return (1 - (x/delta)**2) * jnp.exp(-(x/delta)**2)
  
@jit
def squared_loss(state: jnp.array, gammas: Array, rel_indices: Array, init_r_distances: Array) -> float:
  """ Return the mean of the squared difference between each `state` vortex's relative distance in from each other and the initial relative distances """
  
  # get relative distances between vortices
  r_distances = relative_distances(state, rel_indices)
  
  loss = jnp.mean((r_distances - init_r_distances)**2.)
  
  return loss
  
@jit
def squared_loss_normalised(state: jnp.array, gammas: Array, rel_indices: Array, init_r_distances: Array) -> float:
  """ Return the sum of the squared difference between each `state` vortex's relative distance from each other and the initial relative distances, normalised by sum of initial relative distances """
  
  # get relative distances between vortices
  r_distances = relative_distances(state, rel_indices)
  loss = jnp.sum((r_distances - init_r_distances)**2.) / jnp.sum((init_r_distances)**2.)
  
  return loss
  
# ---------------- REQ Loss Functions -----------------

@partial(jit, static_argnums=(2))
def convergence_check_normalised(init_pos: jnp.array, gammas: jnp.array, Nt: int) -> float:
  """ Return loss from difference in relative distances over the trajectory """

  system = vm.Vortices(init_pos, gammas)
  
  init_r_distances = relative_distances(init_pos, system.rel_indices)
  
  # sum up the loss over the trajectory
  loss = 0

  system.rk2_finalN(0.01, Nt)
  loss += squared_loss_normalised(system.state, system.gammas, system.rel_indices, init_r_distances)
  
  system.rk2_finalN(0.01, Nt)
  loss += squared_loss_normalised(system.state, system.gammas, system.rel_indices, init_r_distances)

  return loss
  
@partial(jit, static_argnums=(2))
def random_search(scaled_params: jnp.array, gammas: jnp.array, Nt: int, omega_const: float, indices: Array, alpha: float) -> float:
  """ Minimise change in relative distances and minimise F """
    
  system = vm.Vortices(scaled_params, gammas)
  
  init_r_distances = relative_distances(scaled_params, system.rel_indices)
  
  # sum up the loss over the trajectory
  loss = alpha * system.return_F(omega_const)

  system.rk2_finalN(0.01, Nt)
  loss += (1. - alpha) * squared_loss(system.state, system.gammas, system.rel_indices, init_r_distances)
  
  system.rk2_finalN(0.01, Nt)
  loss += (1. - alpha) * squared_loss(system.state, system.gammas, system.rel_indices, init_r_distances)

  return loss
  
@partial(jit, static_argnums=(2))
def targetted_search(scaled_params: jnp.array, gammas: jnp.array, Nt: int, omega_const: float, F_targ: float, indices: Array, delta: float) -> float:
  """ Minimise change in relative distances and search for F about F_targ """
  
  system = vm.Vortices(scaled_params, gammas)
  
  init_r_distances = relative_distances(scaled_params, system.rel_indices)

  loss = 100. * sigmoid(system.return_F(omega_const) - F_targ, delta)
  
  # other options for targetted search
  #loss = delta_approx(system.return_F(omega_const) - F_targ, delta)
  #loss = mexican_hat(system.return_F(omega_const) - F_targ, delta)
  
  # sum up the loss over the trajectory
  system.rk2_finalN(0.01, Nt)
  loss += squared_loss(system.state, system.gammas, system.rel_indices, init_r_distances)

  return loss

# ------------- Connections Loss Function -------------

@partial(jit, static_argnums=(3))
def homoconnect(params: jnp.array, eq: Array, gammas: jnp.array, Nt: int, unstable_eigenvectors: Array, eps: float, eq_final: Array) -> float:
  """ Return loss equal to Euclidean distance between final eq and final state on trajectory"""
  
  # I don't think I need to pullback as the optimiser should take care of this itself
  # c.o.m is a fixed reference point throughout the trajectory.
  
  n = int(len(eq)/2)
  theta_symm = params[1]
  cs = params[0]
  
  state = eq + eps * jnp.real(jnp.dot(cs, unstable_eigenvectors)) / jnp.linalg.norm(jnp.real(jnp.dot(cs, unstable_eigenvectors)))
  
  system = vm.Vortices(state, gammas)
  
  # make sure trajectory is on same H surface as equilibrium
  H = vm._return_H(eq_final,gammas,n)
  H_traj = system.return_H()
  delta = utils.compute_H_scaling(H_traj, H, n)
  
  # simulate Nt timesteps
  system.rk2_finalN(0.001, Nt)
  
  # rotate final state on trajectory by theta
  rotated_state = utils._rotate(system.state, theta_symm)

  # contribution to loss from final position
  final_eucl_dist = (rotated_state[:n] - delta * eq_final[:n])**2. + (rotated_state[n:] - delta * eq_final[n:])**2.
  loss = jnp.mean(final_eucl_dist)

  return loss
