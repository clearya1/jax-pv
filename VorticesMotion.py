import numpy as np
import jax
from jax import jit
import jax.numpy as jnp
from functools import partial
from typing import Callable, Tuple, List, Union
import utils
import velocity_transforms as vt

Array = Union[np.ndarray, jnp.ndarray]

from jax.config import config
config.update("jax_enable_x64", True)

@jit
def _r_squared(state: Array, i: int, j: int) -> float:
  """ Returns the squared distance between the vortices at index i and index j """

  n = int(len(state)/2)
  return (state[i] - state[j])**2 + (state[n+i] - state[n+j])**2

@jit
def _induced_velocity(state: Array, gammas: Array, indices: Array) -> Array:
  """ Calculates the total induced velocity for a vortex due to all the other vortices """
  
  n = int(len(state)/2)
  
  def _f(inds):

    index = inds[0]
    j = inds[1]
    r_sq = _r_squared(state, index, j)
    # changed sign convention to match with C&Z
    u = -1.0 * gammas[j] * (state[n+index] - state[n+j]) / r_sq
    v = gammas[j] * (state[index] - state[j]) / r_sq
    
    return jnp.array([u, v])
    
  vmapped_f = jax.vmap(_f)
  uv_all = vmapped_f(indices)
  uv_final = jnp.sum(uv_all, axis=0)
  return uv_final / (2 * jnp.pi)
  
  
def velocity_field(grid: Array, state: Array, gammas: Array) -> Array:
  """ Compute the velocity field for a state of vortices """
  
  n = int(len(state)*0.5)
  
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

@partial(jit, static_argnums=2)
def _return_H(state: Array, gammas: Array, n: int) -> float:
  """ Returns the Hamiltonian of the system"""
  
  indices1, indices2 = jnp.triu_indices(n, k=1)
  
  def _f(i, j):
    return gammas[i] * gammas[j] * jnp.log(_r_squared(state,i,j))
    
  vmapped_f = jax.vmap(_f)
  
  return jnp.sum(vmapped_f(indices1, indices2)) / (-4. * jnp.pi)

@partial(jit, static_argnums=3)
def _return_delta_f_omegafixed(state: Array, gammas: Array, indices: Array, n: int, omega: float) -> float:
  """ Returns the difference between the free energy of the system and the continuum model free energy, with the rate of rotation set to omega"""
  
  indices1, indices2 = jnp.triu_indices(n, k=1)
  
  def _f(i, j):
    return jnp.log(omega * _r_squared(state,i,j))
    
  vmapped_f = jax.vmap(_f)
  
  return -1.*jnp.sum(vmapped_f(indices1, indices2))
  
  
  
@partial(jit, static_argnums=3)
def _return_delta_f(state: Array, gammas: Array, indices: Array, n: int) -> float:
  """ Returns the difference between the free energy of the system and the continuum model free energy"""
  
  omega = vt.mean_angular_velocity(state, gammas, indices, n)
  omega = 2. * jnp.pi * omega # non-dimensionalise!

  return _return_delta_f_omegafixed(state, gammas, indices, n, omega)
  
  
@partial(jit, static_argnums=3)
def _return_full_delta_f(state: Array, gammas: Array, indices: Array, n: int) -> float:
  """ Returns the difference between the free energy of the system and the continuum model free energy"""
  
  constant_term = -0.25*n**2 + 0.5*n**2*jnp.log(n) + n * (0.748752485 - 0.5)

  return _return_delta_f(state, gammas, indices, n) + constant_term
  
@partial(jit, static_argnums=3)
def _return_full_delta_f_omegafixed(state: Array, gammas: Array, indices: Array, n: int, omega: float) -> float:
  """ Returns the difference between the free energy of the system and the continuum model free energy, with the rate of rotation set to omega"""
  
  constant_term = -0.25*n**2 + 0.5*n**2*jnp.log(n) + n * (0.748752485 - 0.5)

  return _return_delta_f_omegafixed(state, gammas, indices, n, omega) + constant_term

@partial(jit, static_argnums=2)
def _return_L(state: Array, gammas: Array, n: int) -> float:
  """ Returns the angular momentum of the system """

  def _f(x,y,g):
    return g * (1. - (x**2. + y**2.))
    
  vmapped_f = jax.vmap(_f)
    
  return 0.5 * jnp.sum(vmapped_f(state[:n], state[n:], gammas))
  
@partial(jit, static_argnums=2)
def _return_F(state: Array, gammas: Array, n: int, omega: float) -> float:
  """ Returns the free energy of the system (dimensional) """

  return _return_H(state,gammas,n) - omega * _return_L(state,gammas,n)
  
@jit
def _every_induced_velocity(state: Array, gammas: Array, indices: Array) -> Array:
  """ Returns the induced velocity for every vortex in the system """
  
  vmapped_induced_velocity = jax.vmap(_induced_velocity, (None, None, 0))
  uv_final = vmapped_induced_velocity(state, gammas, indices)
  return uv_final.T.flatten()

  
@jit
def _rk2_step(state: Array, gammas: Array, indices: Array, dt: float) -> Array:
  """ Performs a single RK2 step """
  
  k1 = _every_induced_velocity(state, gammas, indices)

  # k2
  state = state + k1 * dt
  
  k2 = _every_induced_velocity(state, gammas, indices)

  # update positions
  state = state + (k2 - k1) * dt * 0.5

  return state
  
@jit
def _rk2_finalT(state: Array, gammas: Array, indices: Array, N: int, dt: float) -> Array:

  """ Helper function to perform N RK2 steps """
  
  # function to perform rk2 step
  def _f(i, val):
    x = _rk2_step(val, gammas, indices, dt)
    return x
    
  final_positions = jax.lax.fori_loop(0, N, _f, state)
  
  return final_positions
  
@partial(jit, static_argnums=(3))
def _rk2_finalN(state: Array, gammas: Array, indices: Array, N: int, dt: float) -> Array:

  """ Helper function to perform N RK2 steps """
  
  # function to perform rk2 step
  def _f(carry, x):
    out = _rk2_step(carry, gammas, indices, dt)
    return out, None
  
  final_positions, _ = jax.lax.scan(_f, state, None, length=N)
  
  return final_positions
  
  
  
class Vortices:
  """
    Class containing the system of vortices in the unbounded domain.
    
    Inputs:
      state : 2n dimensional vector with the first half containing the x coordinates of the vortices and the second half the y coordinates.
      gammas: n dimensional vector equal to the circulations of the vortices.
  """
  def __init__(self, state: Array, gammas: Array):

    self.n = int(len(state)/2)
    
    self.state = jnp.array(state)
    self.gammas = jnp.array(gammas)
    self.indices = utils.indices(self.n)
    self.rel_indices = utils.rel_indices(self.n)

  def return_H(self) -> float:
    """ Compute the dimensional Hamiltonian"""

    return _return_H(self.state,self.gammas,self.n)
    
  def return_L(self) -> float:
    """ Compute the dimensional angular momentum"""
    
    return _return_L(self.state,self.gammas,self.n)
    
  def return_F(self, omega: float) -> float:
    """ Compute the dimensional free energy"""
    
    return _return_F(self.state,self.gammas,self.n,omega)

  def r_squared(self, i: int, j: int) -> float:
    """ Returns the squared distance between the vortices at index i and index j """

    return _r_squared(self.state, i, j)

  def induced_velocity(self, indices: Array) -> Array:
    """ Calculates the total induced velocity for a vortex due to all the other vortices """

    return _induced_velocity(self.state, self.gammas, indices)
    
  def every_induced_velocity(self, indices: Array) -> Array:
    """ Returns the induced velocity for every vortex in the system """
    
    return _every_induced_velocity(self.state, self.gammas, indices)
        

# ----------------------------- Time Integator ---------------------------------
    
  def rk2_finalT(self, dt: float, Tf: float):
    """ 2nd order Runge-Kutta method full integrator due to velocities induced by all other vortices """

    eps = 0.001
    N = jnp.int64(jnp.floor_divide(Tf - eps, dt))
    
    final_dt = Tf - N * dt
    
    self.state = _rk2_finalT(self.state, self.gammas, self.indices, N, dt)
    self.state = _rk2_step(self.state, self.gammas, self.indices, final_dt)
  
  def rk2_finalN(self, dt: float, N: int):
    """ 2nd order Runge-Kutta method full integrator due to velocities induced by all other vortices """

    self.state = _rk2_finalN(self.state, self.gammas, self.indices, N, dt)
