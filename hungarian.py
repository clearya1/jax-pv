import numpy as np
import jax
from jax import jit
import jax.numpy as jnp
import scipy
from typing import Union

Array = Union[np.ndarray, jnp.ndarray]

from jax.config import config
config.update("jax_enable_x64", True)

def hungarian_distance(state: Array, target_state: Array) -> float:
  """ Computes the minimum distance between state points and target_state points using the Hungarian algorithm"""
  
  n = int(len(state)/2)
  state = jnp.c_[state[:n], state[n:]]
  target = jnp.c_[target_state[:n], target_state[n:]]
  
  # compute distance between each pair of vortex positions
  cost_matrix = scipy.spatial.distance.cdist(target, state)
  
  # Hungarian method for linear assignment problem
  best_inverse_assignment, best_assignment = scipy.optimize.linear_sum_assignment(cost_matrix)

  return cost_matrix[best_inverse_assignment, best_assignment].sum()
  
def hungarian_sorting(state: Array, target_state: Array) -> [Array, Array]:
  """ Performs the Hungarian algorithm to compute linear assignment between state points and target_state points. Returns the permuted state and the optimal permutation operator."""
  
  n = int(len(state)/2)
  xpos, ypos = jnp.split(state, 2)
  state = jnp.c_[state[:n], state[n:]]
  target = jnp.c_[target_state[:n], target_state[n:]]
  
  # compute distance between each pair of vortex positions
  cost_matrix = scipy.spatial.distance.cdist(target, state)
  
  # Hungarian method for linear assignment problem
  best_inverse_assignment, best_assignment = scipy.optimize.linear_sum_assignment(cost_matrix)
  
  #print(best_assignment)
  #print('-------', cost_matrix[best_inverse_assignment, best_assignment].sum())
  return jnp.concatenate((xpos[best_assignment], ypos[best_assignment])), best_assignment
