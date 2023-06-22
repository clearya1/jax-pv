import numpy as np
import typing
import jax
from jax import jit
import jax.numpy as jnp
from functools import partial
from typing import Callable, Tuple, List, Union
import utils
import loss_functions as lf

Array = Union[np.ndarray, jnp.ndarray]

from jax.config import config
config.update("jax_enable_x64", True)

# Helper Functions to JIT up the class

@jit
def _r_squared(state: Array, i: int, j: int) -> float:
  """ Returns the squared distance between the vortices at index i and index j """

  n = int(len(state)/2)
  return (state[i] - state[j])**2 + (state[n+i] - state[n+j])**2


@partial(jit, static_argnums=4)
def _induced_velocity(state: Array, gammas: Array, indices: Array, L: float, m=2):
  """ Calculates the total induced velocity for a vortex due to all the other vortices in a periodic domain """

  n = int(len(state)/2)
  
  temps = jnp.zeros(2, dtype=jnp.float32)
  
  mlist_others = jnp.arange(-1.*m, m+1, 1)
  mlist_same = jnp.delete(mlist_others, m, axis=0)
  
  # start with the other vortices in the domain and their ghost images
  
  # x are the indices we scan over for one single vortex
  # carry are variables to which we iteratively add velocities
  def _f_others(carry, x):

    index = x[0]
    k = x[1]
    
    dx = state[index] - state[k]
    dy = state[index+n] - state[k+n]
    
    # 1 corresponds to summing over the x direction to get the cot
    # 2 corresponds to summing over the y direction to get the cot

    arg_real1 = (2.0 * jnp.pi * dx / L)
    arg_real2 = (-2.0 * jnp.pi * dy / L)
    
    # sum over the ghost strips (m in mlist_others)
    def _f_inner(carry, m):
      
      arg_imag1 = 2.0 * jnp.pi * (dy / L + m)
      arg_imag2 = 2.0 * jnp.pi * (dx / L + m)

      denom1 = jnp.cosh(arg_imag1) - jnp.cos(arg_real1)
      denom2 = jnp.cosh(arg_imag2) - jnp.cos(arg_real2)

      # take the y direction from 1 (i.e. summed over x first)
      # take the x direction from 2 (i.e. summed over y first)

      y_cot = jnp.sin(arg_real1) / denom1
      x_cot = jnp.sin(arg_real2) / denom2
    
      carry = carry.at[0].add( gammas[k] * x_cot )
      carry = carry.at[1].add( gammas[k] * y_cot )
      
      return carry, None
        
    carry, _ = jax.lax.scan( _f_inner, carry, mlist_others )
    
    return carry, None
    
  uv_others, _ = jax.lax.scan( _f_others, temps, indices)
  
  
  # Now do the ghost vortices of the vortex we are dealing with
  index = indices[0][0]
  dx = 0.
  dy = 0.
  arg_real1 = 0.
  arg_real2 = 0.
    
  # sum over ghost strips (m in mlist_same)
  def _f_same(carry, m):
  
    arg_imag1 = 2.0 * jnp.pi * (dy / L + m)
    arg_imag2 = 2.0 * jnp.pi * (dx / L + m)

    denom1 = jnp.cosh(arg_imag1) - jnp.cos(arg_real1)
    denom2 = jnp.cosh(arg_imag2) - jnp.cos(arg_real2)

    # take the y direction from 1 (i.e. summed over x first)
    # take the x direction from 2 (i.e. summed over y first)

    y_cot = jnp.sin(arg_real1) / denom1
    x_cot = jnp.sin(arg_real2) / denom2
  
    carry = carry.at[0].add( gammas[index] * x_cot )
    carry = carry.at[1].add( gammas[index] * y_cot )
    
    return carry, None
    
  uv_final, _ = jax.lax.scan(_f_same, uv_others, mlist_same)
  
  return uv_final / (2 * L)
  

@jit
def _return_H(state: Array, gammas: Array, indices: Array, L: float):
  """ Returns the Hamiltonian of the system """
  
  carry = lf.periodic_relative_distances_xy(state, indices, L)
  
  def _f(carry, x):
    H_temp = -1. * gammas[x[0]] * gammas[x[1]] * jnp.log((carry[0]**2.+carry[1]**2.)**(0.5))
    return carry, H_temp
  
  _, H_list = jax.lax.scan(_f, carry, indices)

  return np.sum(H_list) / (4 * jnp.pi)
  
@partial(jit, static_argnums=4)
def _every_induced_velocity(state: Array, gammas: Array, indices: Array, L: float, m=2):
  """ Returns the induced velocity for every vortex in the system """
  
  vmapped_induced_velocity = jax.vmap(_induced_velocity, (None, None, 0, None, None))
  uv_final = vmapped_induced_velocity(state, gammas, indices, L, m)
  return uv_final.T.flatten()

@partial(jit, static_argnums=5)
def _rk2_step(state: Array, gammas: Array, indices: Array, dt: float, L: float, m=2):
  """ Performs a single RK2 step """

  k1 = _every_induced_velocity(state, gammas, indices, L)

  # k2
  state = state + k1 * dt
  
  k2 = _every_induced_velocity(state, gammas, indices, L)

  # update positions
  state = state + (k2 - k1) * dt * 0.5
  
  # enforce periodic boundary conditions
  state = (state % L + L) % L

  return state
  
@partial(jit, static_argnums=(6))
def _rk2_finalT(state: Array, gammas: Array, indices: Array, N: int, dt: float, L: float, m=2):

  """ Helper function to perform N RK2 steps """
  
  # function to perform rk2 step
  def _f(i, val):
    x = _rk2_step(val, gammas, indices, dt, L)
    return x
    
  final_positions = jax.lax.fori_loop(0, N, _f, state)
  
  return final_positions
  
@partial(jit, static_argnums=(3,6))
def _rk2_finalN(state: Array, gammas: Array, indices: Array, N: int, dt: float, L: float, m=2):

  """ Helper function to perform N RK2 steps """
  
  # function to perform rk2 step
  def _f(carry, x):
    out = _rk2_step(carry, gammas, indices, dt, L, m)
    return out, None
  
  final_positions, _ = jax.lax.scan(_f, state, None, length=N)
  
  return final_positions
  
class PeriodicVortices:
  """
    Class containing the system of vortices in the periodic domain of length LxL.
    
    Inputs:
      state : 2n dimensional vector with the first half containing the x coordinates of the vortices and the second half the y coordinates.
      gammas: n dimensional vector equal to the circulations of the vortices.
      L     : Length of 2d square domain
      m     : number of periodic strips to sum over, default = 2.
  """
  def __init__(self, state: Array, gammas: Array, L: float, m=2):

      self.n = int(len(state)/2)
      
      self.state = jnp.array(state)
      self.gammas = jnp.array(gammas)
      self.indices = utils.indices(self.n)
      self.rel_indices = utils.rel_indices(self.n)
      
      self.L = L                           # size of domain
      self.m = m                           # number of truncations

      self.enforce_boundary()              # enforce boundary conditions at the start to make sure all in domain

  def return_H(self):
    """ Updates the class variable containing the Hamiltonian"""

    return _return_H(self.state,self.gammas,self.rel_indices,self.L)

  def r_squared(self, i: int, j: int):
    """ Returns the squared distance between the vortices at index i and index j """

    return _r_squared(self.state, i, j)

  def induced_velocity(self, indices: Array):
    """ Calculates the total induced velocity for a vortex due to all the other vortices """

    return _induced_velocity(self.state, self.gammas, indices, self.L, self.m)
    
  def every_induced_velocity(self, indices: Array):
    """ Returns the induced velocity for every vortex in the system """
    
    return _every_induced_velocity(self.state, self.gammas, indices, self.L, self.m)
    
  def enforce_boundary(self):
    """ Enforces periodic boundary conditions, does nothing if we have an unbounded domain """
      
    self.state = (self.state % self.L + self.L) % self.L
        

# ----------------------------- Time Integator ---------------------------------

  def rk2_finalT(self, dt: float, Tf: float):
    """ 2nd order Runge-Kutta method full integrator due to velocities induced by all other vortices """

    eps = 0.001
    N = jnp.int32(jnp.floor_divide(Tf - eps, dt))
    
    final_dt = Tf - N * dt

    self.state = _rk2_finalT(self.state, self.gammas, self.indices, N, dt, self.L, self.m)
    self.state = _rk2_step(self.state, self.gammas, self.indices, final_dt, self.L, self.m)
    
  def rk2_finalN(self, dt: float, N: int):
    """ 2nd order Runge-Kutta method full integrator due to velocities induced by all other vortices """

    self.state = _rk2_finalN(self.state, self.gammas, self.indices, N, dt, self.L, self.m)

