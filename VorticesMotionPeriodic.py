import numpy as np
import typing
import jax
from jax import jit
import jax.numpy as jnp
from functools import partial
from typing import Callable, Tuple, List, Union
import utils

Array = Union[np.ndarray, jnp.ndarray]

from jax import config
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
  
  temps = jnp.zeros(2, dtype=jnp.float64)
  
  mlist_others = jnp.arange(-1.*m, m+1, 1)
  
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
    arg_real2 = (2.0 * jnp.pi * dy / L)
    
    # sum over the ghost strips (m in mlist_others)
    def _f_inner(carry, m):
      
      arg_imag1 = 2.0 * jnp.pi * (dy / L + m)
      arg_imag2 = 2.0 * jnp.pi * (dx / L + m)

      denom1 = jnp.cosh(arg_imag1) - jnp.cos(arg_real1)
      denom2 = jnp.cosh(arg_imag2) - jnp.cos(arg_real2)

      # take the y direction from 1 (i.e. summed over x first)
      # take the x direction from 2 (i.e. summed over y first)

      y_cot = jnp.sin(arg_real1) / denom1
      x_cot = -1. * jnp.sin(arg_real2) / denom2
    
      carry = carry.at[0].add( gammas[k] * x_cot )
      carry = carry.at[1].add( gammas[k] * y_cot )
      
      return carry, None
        
    carry, _ = jax.lax.scan( _f_inner, carry, mlist_others )
    
    return carry, None
    
  uv_others, _ = jax.lax.scan( _f_others, temps, indices)
  
  return uv_others / (2 * L)
  
def velocity_field(grid: Array, state: Array, gammas: Array, L: float, m=2) -> Array:
  """ Compute the velocity field for a state of vortices """
  
  n = int(len(state)*0.5)
  Nx, Ny = grid[0].shape
  inds = utils.indices(n+1)[-1]
  
  vel_field_x = np.zeros((Nx, Ny))
  vel_field_y = np.zeros((Nx, Ny))
  
  gammas_temp = np.zeros(n)
  gammas_temp[:n] = gammas
  
  for i in range(Nx):
    for j in range(Ny):
      temp_state = np.zeros(2*n + 2)
      temp_state[:n] = state[:n]
      temp_state[n] = grid[0][i,j]
      temp_state[n+1:-1] = state[n:]
      temp_state[-1] = grid[1][i,j]
      
      vel_field_x[i,j], vel_field_y[i,j] = _induced_velocity(temp_state, gammas_temp, inds, L, m)
  
  def _f(x, y, g):
    r_sq = (grid[0] - x)**2. + (grid[1]- y)**2.
    vel_field_x = -1.0 * g * (grid[1] - y) / r_sq
    vel_field_y = g * (grid[0] - x) / r_sq
    return vel_field_x, vel_field_y
    
  vmapped_f = jax.vmap(_f, (0,0,0))
  vel_fields_x, vel_fields_y = vmapped_f(state[:n], state[n:], gammas)
  vel_field_x = jnp.sum(vel_fields_x, axis=0)
  vel_field_y = jnp.sum(vel_fields_y, axis=0)
  return vel_field_x / (2 * jnp.pi), vel_field_y / (2 * jnp.pi)
  

@partial(jit, static_argnums=(2,4,))
def _return_H(state: Array, gammas: Array, n: int, L: float, m: int = 2):
  """ Returns the Hamiltonian of the system - taken from Weiss and McWilliams 1991 with change of coords to LxL domain"""
  
  indices1, indices2 = jnp.triu_indices(n, k=1)
  
  def h(dx, dy):
    mlist = jnp.arange(-1.*m, m+1, 1)
    
    def h_per_m(dx, dy, m):
      return jnp.log( jnp.cosh(dx - 2.*jnp.pi*m) - jnp.cos(dy)) - jnp.log( jnp.cosh( 2.*jnp.pi*m ) )
    
    vmapped_h_per_m = jax.vmap(h_per_m, in_axes=(None, None, 0))
    
    return jnp.sum( vmapped_h_per_m(dx, dy, mlist)  ) - dx**2 / (2. * jnp.pi)
    
  def _H(i, j):
    
    dx = 2.*jnp.pi*(state[i] - state[j]) / L
    dy = 2.*jnp.pi*(state[i+n] - state[j+n]) / L
    
    return h(dx, dy) * gammas[i] * gammas[j]
    
  vmapped_H = jax.vmap(_H)

  return -1. * jnp.sum( vmapped_H( indices1, indices2 ) ) / (2 * 2. * jnp.pi)
  
def _compute_uv(state: Array, gammas: Array, n: int, L: float, m: int = 2):
  """ Compute the average velocity of each of the vortices. For a REQ, this will be the velocity of the translating reference frame in which the REQ becomes an EQ. """
  
  inds = utils.indices(n)
  velocities = _every_induced_velocity(state, gammas, inds, L, m)
  u = jnp.mean(velocities[:n])
  v = jnp.mean(velocities[n:])
  uv = jnp.array([u, v])
  
  return uv

@partial(jit, static_argnums=(3,5,))
def _return_H_translating_frame(state: Array, gammas: Array, uv: Array, n: int, L: float, m: int = 2):
  """ Returns the Hamiltonian of the system in a frame of reference translating with velocity (u,v) - taken from Weiss and McWilliams 1991 with change of coords to LxL domain"""
  
  H_stationary = _return_H(state, gammas, n, L, m)
  
  H_translate = jnp.sum(gammas * state[:n] * uv[1]) - jnp.sum(gammas * state[n:] * uv[0])
  
  return H_stationary + H_translate #* 2. * jnp.pi / L
  
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
  
@partial(jit, static_argnums=5)
def _rk2_step_nobc(state: Array, gammas: Array, indices: Array, dt: float, L: float, m=2):
  """ Performs a single RK2 step """

  k1 = _every_induced_velocity(state, gammas, indices, L)

  # k2
  state = state + k1 * dt
  
  k2 = _every_induced_velocity(state, gammas, indices, L)

  # update positions
  state = state + (k2 - k1) * dt * 0.5
  
  # don't enforce periodic boundary conditions

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
  
@partial(jit, static_argnums=(6))
def _rk2_finalT_nobc(state: Array, gammas: Array, indices: Array, N: int, dt: float, L: float, m=2):

  """ Helper function to perform N RK2 steps """
  
  # function to perform rk2 step
  def _f(i, val):
    x = _rk2_step_nobc(val, gammas, indices, dt, L)
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
  
@partial(jit, static_argnums=(3,6))
def _rk2_finalN_nobc(state: Array, gammas: Array, indices: Array, N: int, dt: float, L: float, m=2):

  """ Helper function to perform N RK2 steps """
  
  # function to perform rk2 step
  def _f(carry, x):
    out = _rk2_step_nobc(carry, gammas, indices, dt, L, m)
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

    return _return_H(self.state,self.gammas,self.n,self.L,self.m)

  def r_squared(self, i: int, j: int):
    """ Returns the squared distance between the vortices at index i and index j """

    return _r_squared(self.state, i, j)

  def induced_velocity(self, indices: Array):
    """ Calculates the total induced velocity for a vortex due to all the other vortices """

    return _induced_velocity(self.state, self.gammas, indices, self.L, self.m)
    
  def every_induced_velocity(self):
    """ Returns the induced velocity for every vortex in the system """
    
    return _every_induced_velocity(self.state, self.gammas, self.indices, self.L, self.m)
    
  def enforce_boundary(self):
    """ Enforces periodic boundary conditions, does nothing if we have an unbounded domain """
      
    self.state = (self.state % self.L + self.L) % self.L
        

# ----------------------------- Time Integator ---------------------------------

  def rk2_finalT(self, dt: float, Tf: float):
    """ 2nd order Runge-Kutta method full integrator due to velocities induced by all other vortices """

    eps = 0.001
    N = jnp.int64(jnp.floor_divide(Tf - eps, dt))
    
    final_dt = Tf - N * dt

    self.state = _rk2_finalT(self.state, self.gammas, self.indices, N, dt, self.L, self.m)
    self.state = _rk2_step(self.state, self.gammas, self.indices, final_dt, self.L, self.m)
    
  def rk2_finalN(self, dt: float, N: int):
    """ 2nd order Runge-Kutta method full integrator due to velocities induced by all other vortices """

    self.state = _rk2_finalN(self.state, self.gammas, self.indices, N, dt, self.L, self.m)

