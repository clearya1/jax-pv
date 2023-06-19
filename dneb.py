import numpy as np
import jax
from jax import jit
import jax.numpy as jnp

import pickle
from functools import partial
from datetime import datetime
import optax

import matplotlib.pyplot as plt

from typing import Callable, Tuple, List, Union, Sequence
Array = Union[np.ndarray, jnp.ndarray]

import VorticesMotion as vm
import velocity_transforms as vt
import loss_functions as lf
import utils
import hungarian as hun

from os.path import exists

from jax.config import config
config.update("jax_enable_x64", True)

# ------------------ ROTATING SO OVERLAPPING OPTIMISATION ------------------------
  
def gaussian_overlap(theta: float, eq1: Array, grid: Array, smear2: Array, cov: float) -> float:
  """
      Compute the L2 difference (overlap) between the Gaussian smearing of eq1 and eq2, where eq1 is rotated by angle theta, and smear2 is Gaussian smearing of eq2 over the same grid
  """
  
  eq1 = utils._rotate(eq1, theta)
  
  smear1 = lf.gaussian_smear(eq1, grid, cov)
  
  return jnp.mean(jnp.square(smear2 - smear1))
  
# compiling optimiser steps
@jit
def gradfn(theta, eq1, grid, smear2, cov):
  return jax.value_and_grad(jax.remat(gaussian_overlap))(theta, eq1, grid, smear2, cov)
  
@partial(jit, static_argnums=(2))
def updatefn(theta, opt_state, optimizer, eq1, grid, smear2, cov):

  loss, grads = gradfn(theta, eq1, grid, smear2, cov)
  
  updates, opt_state = optimizer.update(grads, opt_state)

  theta = optax.apply_updates(theta, updates)
  
  return theta, opt_state, loss
    

def rotate_so_overlapping(eq1, eq2, grid, cov=1., opt_N=10):
  """
      Returns the equilibria eq1 and eq2 so that they overlap as much as possible.
      Using the Gaussian smearing loss function.
  """
  
  start_learning_rate = 1e-2
  optimizer = optax.adam(start_learning_rate)
  
  theta = 0.
  opt_state = optimizer.init(theta)
  
  smear2 = lf.gaussian_smear(eq2, grid, cov)
  
  for i in range(opt_N):
    theta, opt_state, loss = updatefn(theta, opt_state, optimizer, eq1, grid, smear2, cov)
    
  return utils._rotate(eq1, theta), eq2, loss
  
def rotate_so_overlapping_hungarian(eq1, eq2):
  """
      Discretises the symmetry reducing angle, and uses the Hungarian algorithm to overlap as much as possible.
  """
  
  # discretise the angle
  N_angles = 100
  thetas = np.linspace(0, 2.*jnp.pi, N_angles, endpoint = False)
  losses = np.zeros_like(thetas)
  
  for i, theta in enumerate(thetas):
    losses[i] = hun.hungarian_distance(eq1, utils._rotate(eq2, theta))
  
  best_angle_ind = np.argmin(losses)
  best_angle = thetas[best_angle_ind]
  eq2 = utils._rotate(eq2, best_angle)
  
  return eq1, eq2

  
# ------------------------------ DNEB METHOD CODE -------------------------------

def interpolate_between_minima(eq1: Array, eq2: Array, N: int) -> Sequence[Array]:
  """
      Use Hungarian Method to match up vortices in eq1 to closest vortices in eq2.
      Linearlly interpolate between these assignments with N states.
      Return N+2 states, 0-th index = eq1, N+1-th index = eq2, N interpolations in between.
  """
  
  n = int(0.5*len(eq1))
  
  # reorder eq1 so that indices of closest vortices in eq1 and eq2 match up
  eq2, _ = hun.hungarian_sorting(eq2, eq1)
  
  full_chain = np.zeros((N+2, 2*n))
  
  for i in range(2*n):
    full_chain[:,i] = np.linspace(eq1[i], eq2[i], N+2)    # this includes endpoints
    
  return full_chain
  
def V_true(interpolation: Array, omega: float) -> float:
  """
      Compute the true potential of the ensemble. We are in a rotating cylinder with fixed rotation omega.
  """
  n = int(len(interpolation[0])*0.5)
  ind = utils.indices(n)
  gammas = jnp.ones(n)

  def _f(x):
    F = vm._return_F(x, gammas, n, omega)
    return F

  # map computation of free energy over 0-th axis of input
  vmapped_f = jax.vmap(_f)
  
  Fs = vmapped_f(interpolation)

  V = jnp.sum(Fs)
  
  return V, Fs
  
def V_true_for_grad(interpolation: Array, omega: float) -> float:
  V, _ = V_true(interpolation, omega)
  return V
  
def V_spring(interpolation: Array, grid: Array, eq1: Array, eq2: Array, k_spr: float) -> float:
  """
      Compute the spring potential of the ensemble, with spring force constant k_spr
      Using Gaussian smearing distance metric on the grid.
  """
  n = int(len(eq1)*0.5)
  gammas = jnp.ones(n)
  
  upper_copy = jnp.vstack((interpolation, eq2))
  lower_copy = jnp.vstack((eq1, interpolation))
  
  def _f(a, b):
    _, _, loss = rotate_so_overlapping(a, b, grid)
    return loss

  # map computation of distance over both 0-th axis of inputs (lower and upper copies)
  vmapped_f = jax.vmap(_f)

  distances = vmapped_f(lower_copy, upper_copy)
  V = jnp.sum(distances)
  
  return 0.5 * k_spr * V

  
def tangent_to_pathway_at_index(full_chain: Array, energies: Array, index: int) -> Array:
  """
      Compute the tangent to the pathway at index, using the correct formula depending on whether it is a local optimum or not.
  """
  if (energies[index+1] > energies[index] and energies[index-1] < energies[index]):

    tangent = (full_chain[index+1] - full_chain[index]) / jnp.linalg.norm(full_chain[index+1] - full_chain[index])

  elif (energies[index+1] < energies[index] and energies[index-1] > energies[index]):

    tangent = -1. * (full_chain[index-1] - full_chain[index]) / jnp.linalg.norm(full_chain[index-1] - full_chain[index])

  else:

    tangent = (full_chain[index+1] - full_chain[index-1]) / jnp.linalg.norm(full_chain[index+1] - full_chain[index-1])
    
  return tangent


def tangent_to_pathway(full_chain: Array, energies: Array) -> Array:
  """
      Compute the tangent to the full pathway.
  """
  
  N_energies = len(energies)
  n = full_chain.shape[1]
  tangents = np.zeros((N_energies-2, n))
  for i in range(1, N_energies-1):
    tangents[i-1] = tangent_to_pathway_at_index(full_chain, energies, i)
  
  return tangents
  
  

def decompose_gradient(gradient: Array, tangent: Array) -> [Array, Array]:
  """
      Return the components of the gradient parallel and perpendicular to the tangent vector.
  """
  
  parallel = jnp.dot(gradient, tangent) * tangent
  
  perpendicular = gradient - parallel
  
  return parallel, perpendicular
  
def neb(gradient_true: Array, gradient_spring: Array, tangent: Array) -> Array:
  """
      Nudges the gradient for the NEB method. Returns the nudged gradient.
  """
  
  vmapped_decompose_gradient = jax.vmap(decompose_gradient, (0,0), (0,0))
  
  _, true_perp = vmapped_decompose_gradient(gradient_true, tangent)
  
  spring_parallel, _ = vmapped_decompose_gradient(gradient_spring, tangent)
  
  return true_perp + spring_parallel
  
def dneb(gradient_true: Array, gradient_spring: Array, tangent: Array) -> Array:
  """
      Doubly nudges the gradient for the DNEB method. Returns the doubly nudged gradient.
  """
  
  vmapped_decompose_gradient = jax.vmap(decompose_gradient, (0,0), (0,0))
  
  _, true_perp = vmapped_decompose_gradient(gradient_true, tangent)
  
  spring_parallel, spring_perp = vmapped_decompose_gradient(gradient_spring, tangent)
  
  spring_perp_parallel, _ = vmapped_decompose_gradient(spring_perp, true_perp / jnp.linalg.norm(true_perp))
  
  return true_perp + spring_parallel + spring_perp_parallel
  
@jit
def gradfn_V_true(interpolation: Array, omega: float):
  return jax.value_and_grad(V_true_for_grad)(interpolation, omega)
  
@jit
def gradfn_V_spring(interpolation: Array, grid: Array, eq1: Array, eq2: Array, k_spr: float):
  return jax.value_and_grad(V_spring)(interpolation, grid, eq1, eq2, k_spr)
  
  
# don't jit or partial jit here because of unvectorised tangents calculation
def updatefn_NEB(interpolation: Array, opt_state, optimizer, grid: Array, eq1: Array, eq2: Array, omega: float, k_spr: float = 1.):
  """
      Updates the interpolation. Note that interpolation only refers to the intermediate steps. Full_chain refers to the whole chain, including the minima fixed at the end points.
  """
  
  n = int(len(eq1)*0.5)
  
  full_chain = jnp.vstack((eq1,interpolation,eq2))
  
  true_loss, true_grads = gradfn_V_true(interpolation, omega)              # gradient of true potential
  
  spring_loss, spring_grads = gradfn_V_spring(interpolation, grid, eq1, eq2, k_spr)      # gradient of spring potential

  _, energies = V_true(full_chain, omega)          # energies of interpolation
  tangents = tangent_to_pathway(full_chain, energies)   # tangents at interpolation states
  neb_grad = neb(true_grads, spring_grads, tangents)

  updates, opt_state = optimizer.update(neb_grad, opt_state)

  interpolation = optax.apply_updates(interpolation, updates)

  return interpolation, opt_state, true_loss + spring_loss, neb_grad
  
# don't jit or partial jit here because of unvectorised tangents calculation
def updatefn_DNEB(interpolation: Array, opt_state, optimizer, grid: Array, eq1: Array, eq2: Array, omega: float, k_spr: float = 1.):
  """
      Updates the interpolation. Note that interpolation only refers to the intermediate steps. Full_chain refers to the whole chain, including the minima fixed at the end points.
  """
  
  n = int(len(eq1)*0.5)
  
  full_chain = jnp.vstack((eq1,interpolation,eq2))
  
  true_loss, true_grads = gradfn_V_true(interpolation, omega)              # gradient of true potential
  
  spring_loss, spring_grads = gradfn_V_spring(interpolation, grid, eq1, eq2, k_spr)      # gradient of spring potential

  _, energies = V_true(full_chain, omega)          # energies of interpolation
  tangents = tangent_to_pathway(full_chain, energies)   # tangents at interpolation states
  dneb_grad = dneb(true_grads, spring_grads, tangents)

  updates, opt_state = optimizer.update(dneb_grad, opt_state)

  interpolation = optax.apply_updates(interpolation, updates)

  return interpolation, opt_state, true_loss + spring_loss, dneb_grad
  
def main(n, eq1, eq2, omega, gammas):
  """ Computes the energy minimising pathway between the two relative equilibria eq1 and eq2, with circulations gammas and n number of vortices """

  print(datetime.now())
  
  ind = utils.indices(n)
  
  # optimiser parameters
  N_opt = 100
  print_cycle = N_opt
  start_learning_rate = 5e-1
  optimizer = optax.adagrad(start_learning_rate)
  
  N_interpolate = 100
  k_spr = 1.

  # create the grid to smear the states

  x_min = jnp.min(jnp.array([eq1, eq2]))
  x_max = jnp.max(jnp.array([eq1, eq2]))
  x_min_abs = np.abs(x_min)
  x_max_abs = np.abs(x_max)
  x_min -= 0.25 * x_min_abs
  x_max += 0.25 * x_max_abs

  grid_size = 64
  x = jnp.linspace(x_min,x_max,grid_size)
  y = x
  X,Y = jnp.meshgrid(x,y)
  grid = np.empty(X.shape + (2,))
  grid[:, :, 0] = X; grid[:, :, 1] = Y

  # smear the states with an appropriately chosen covariance matrix

  cov = 0.01*((x_max - x_min))**2/float(n)

  # scaling them here so they have the same omega !
  eq1 = vt.scale_to_omega(eq1, gammas, omega, n, ind)
  eq2 = vt.scale_to_omega(eq2, gammas, omega, n, ind)
  
  smear2 = lf.gaussian_smear(eq2, grid, cov)
  
  # ----- always check the smearing is good ---------------
#  plt.pcolormesh(grid[:,:,0], grid[:,:,1], smear2)
#  plt.show()
  
  # rotating them here so they overlap as much as possible
  eq1, eq2 = rotate_so_overlapping_hungarian(eq1, eq2)

  # generate initial sequence of states
  full_chain = interpolate_between_minima(eq1, eq2, N_interpolate)
  interpolation = full_chain[1:-1]
  eq1 = full_chain[0]      # DON'T FORGET TO DO THIS LINE AS INTERPOLATION REARRANGES VORTEX INDICES
  eq2 = full_chain[-1]
  
  opt_state = optimizer.init(interpolation)
  
  interpolation, opt_state, loss, grads = updatefn_NEB(interpolation, opt_state, optimizer, grid, eq1, eq2, omega, k_spr)
  print(f'Optimiser Step 0: {loss = }')

  for j in range(1,N_opt+1):
  
    interpolation, opt_state, loss, grads = updatefn_NEB(interpolation, opt_state, optimizer, grid, eq1, eq2, omega, k_spr)
      
    if j%10==0:
      print(f'Optimiser Step {j}: {loss = }')
        
    if np.all(np.abs(grads) < 1e-5):
      print(f'Optimiser Step {j}: {loss = }')
      print('------------------------------------------------------------------')
      break
  
  return interpolation, eq1, eq2


if __name__ == "__main__":
  
  n = 10
  
  # ------------- read in all equilibria ----------------
  
  eq_in_file = 'data/n'+str(n)

  with open(eq_in_file, 'rb') as f:
    full_data = pickle.load(f)
    
  equilibria = full_data['params']

  gammas = jnp.ones(n)
  ind = utils.indices(n)

  delta_f = full_data['delta_f']
    
  order_by_delta_f = True
    
  if order_by_delta_f:
    sorted_ind = np.argsort(delta_f)
    equilibria = equilibria[sorted_ind]
    delta_f = delta_f[sorted_ind]
  
  omega = vt.mean_angular_velocity(equilibria[0], gammas, ind, n)
  
  index1 = 0
  index2 = 1
  eq1 = equilibria[index1]
  eq2 = equilibria[index2]
  main(n, eq1, eq2, omega, gammas)
