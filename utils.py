import numpy as np
import jax
from jax import jit
import jax.numpy as jnp
from typing import Callable, Tuple, List, Union

Array = Union[np.ndarray, jnp.ndarray]

from jax import config
config.update("jax_enable_x64", True)

@jit
def _rotate(state: Array, theta: float) -> Array:
  """ Helper function to rotate vortices by theta """
  
  rotation_matrix = jnp.array([[jnp.cos(theta), -1.* jnp.sin(theta)],
                              [jnp.sin(theta), jnp.cos(theta)    ]])
  n = int(len(state)/2)
  positions = jnp.c_[state[:n], state[n:]]

  # function to rotate each vortex position by theta
  def _f(x):
    return rotation_matrix @ x
  vmapped_f = jax.vmap(_f)
  
  slice_positions = vmapped_f(positions)
  return jnp.transpose(slice_positions).flatten()
  
@jit
def _pullback(state: Array, index: int) -> Array:
  """ Helper function to perform pullback so that the vortex at index `index` has x = 0 """
  
  n = int(len(state)/2)
  theta = jnp.arctan2(state[index], state[n+index])
  
  statehat = _rotate(state, theta)
  
  return statehat
  
def polygon(n: int, radius: float=1, rotation: float=0) -> Array:
  """ Creates an n-gon of specified radius and returns the x and y co-ordinates of the points """

  one_segment = np.pi * 2 / n

  state = np.zeros(2*n)

  for i in range(n):
    state[i] = np.sin(one_segment * i + rotation) * radius
    state[n+i] = np.cos(one_segment * i + rotation) * radius

  return state
  
def n_gon_angular_vel(n: int, gamma: float, radius: float) -> float:
  """ Computes the analytic angular velocity for an n-gon """
  
  return (n - 1) * gamma / (4.0 * jnp.pi * radius**2)
  
@jit
def compute_H_scaling(H_target: float, H_current: float, n: int) -> float:
  """ Returns the required scaling for the positions so that an `n` vortex system with Hamiltonian = `H_current` equals the target one `H_target` """
  
  return jnp.exp( 4. * jnp.pi * (H_current - H_target) / (n * (n - 1.)) )
  
@jit
def centre_on_com(state: Array, gammas: Array) -> Array:
  """ Centres the vortices on the centre of vorticity of the system """
  
  n = int(len(state)/2)
  x_com = jnp.average(state[:n], weights=gammas)
  y_com = jnp.average(state[n:], weights=gammas)
  
  state = state.at[:n].add(-1*x_com)
  state = state.at[n:].add(-1*y_com)
  
  return state
  
def indices(n: int) -> Array:
  """ Returns the array of indices needed for jitting the total_angular_velocity function """
  
  indices = jnp.zeros((n, n, 2), dtype=int)
  for i in range(0,n):
    for j in range(0,n):
      indices = indices.at[i,j].set([i,j])

  indices2 = jnp.zeros((n, n-1, 2), dtype=int)

  for i in range(n):
    indices2 = indices2.at[i].set(jnp.delete(indices[i], i, axis=0))
    
  return indices2

  
def rel_indices(n: int) -> Array:
  """ Returns the n(n-1)/2 combinations: (1,2), (1,3), ..., (1, n), (2, 3), ..., (n-1, n). """
  
  return np.transpose(np.triu_indices(n, 1))
  
def _xy_shift(state: Array, xy: Array) -> Array:
  """ Applies a shift in xy = (x,y) direction """
  
  n = int(len(state)/2)

  shift_vector = jnp.zeros_like(state)
  shift_vector = shift_vector.at[:n].set(xy[0])
  shift_vector = shift_vector.at[n:].set(xy[1])
  
  return state + shift_vector
  
@jit
def centre_on_com_periodic(state: Array, gammas: Array, L) -> Array:
  """ Centres the vortices on the centre of vorticity of the system """
  
  n = int(len(state)/2)
  
#  x_com = jnp.dot(state[:n], gammas)
#  y_com = jnp.dot(state[n:], gammas)
  x_com = jnp.average(state[:n], weights=jnp.abs(gammas))
  y_com = jnp.average(state[n:], weights=jnp.abs(gammas))
  
  state = state.at[:n].add(-1*x_com + L/2)
  state = state.at[n:].add(-1*y_com + L/2)
  
  return state % L
  
