import numpy as np
import jax
from jax import jit
import jax.numpy as jnp
from functools import partial
from typing import Callable, Tuple, List, Union
import utils

Array = Union[np.ndarray, jnp.ndarray]

from jax.config import config
config.update("jax_enable_x64", True)

# Helper Functions to JIT up the class

@jit
def _r_squared(state: Array, i: int, j: int) -> float:
  """ Returns the squared distance between the vortices at index i and index j """

  n = int(len(state)/2)
  return (state[i] - state[j])**2 + (state[n+i] - state[n+j])**2

@jit
def _induced_velocity(state: Array, gammas: Array, indices: Array, omega: float) -> Array:
  """ Calculates the total induced velocity for a vortex due to all the other vortices
      Matching the units in Campbell&Ziff (1979)
      Changed the sign of velocity here - should do the same for other classes """

  temps = jnp.zeros(2)
  n = int(len(state)/2)

  # x are the indices we scan over for one single vortex
  # carry are variables to which we iteratively add velocities
  def _f(carry, x):

    index = x[0]
    j = x[1]
    r_sq = _r_squared(state, index, j)
    
    # changed sign convention to match with C&Z
    # real vortices in the domain
    carry = carry.at[0].add( -1.0 * gammas[j] * (state[n+index] - state[n+j]) / r_sq )
    carry = carry.at[1].add( gammas[j] * (state[index] - state[j]) / r_sq )
    
    # ghost vortices outside of the domain
    ghost_rsq = state[j]**2 + state[n+j]**2
    vort_to_ghost_rsq = (state[index] - state[j]/ghost_rsq)**2 + (state[n+index] - state[n+j]/ghost_rsq)**2
    
    ghost_x_temp = gammas[j] * (state[n+index] - state[n+j]/ghost_rsq) / vort_to_ghost_rsq
    ghost_y_temp = -1.0 * gammas[j] * (state[index] - state[j]/ghost_rsq) / vort_to_ghost_rsq
    
    ghost_x_temp = jnp.where(ghost_rsq == 0., 0.0, ghost_x_temp)   # add nothing for ghost vortex at infinity
    ghost_y_temp = jnp.where(ghost_rsq == 0., 0.0, ghost_y_temp)   # add nothing for ghost vortex at infinity
    
    carry = carry.at[0].add( ghost_x_temp )
    carry = carry.at[1].add( ghost_y_temp )
    
    return carry, None
    
    
  uv_final, _ = jax.lax.scan(_f, temps, indices)    # looping here over all the other vortices for a single vortex
  
  # self-ghost vortex
  
  try:      # this is to allow for a single vortex in the cylinder
    index = indices.flatten()[0]
  except:
    index = 0
    
  ghost_rsq = state[index]**2 + state[n+index]**2
  vort_to_ghost_rsq = (state[index] - state[index]/ghost_rsq)**2 + (state[n+index] - state[n+index]/ghost_rsq)**2
  
  ghost_x_temp = gammas[index] * (state[n+index] - state[n+index]/ghost_rsq) / vort_to_ghost_rsq
  ghost_y_temp = -1.0 * gammas[index] * (state[index] - state[index]/ghost_rsq) / vort_to_ghost_rsq
  
  ghost_x_temp = jnp.where(ghost_rsq == 0., 0.0, ghost_x_temp)   # add nothing for ghost vortex at infinity
  ghost_y_temp = jnp.where(ghost_rsq == 0., 0.0, ghost_y_temp)   # add nothing for ghost vortex at infinity
  
  uv_final = uv_final.at[0].add( ghost_x_temp )
  uv_final = uv_final.at[1].add( ghost_y_temp )
  
  # constant angular rotation of cylinder, omega
  uv_final = uv_final.at[0].add( omega * state[n+index] )
  uv_final = uv_final.at[1].add( -1. * omega * state[index] )

  return uv_final / (2. * jnp.pi)

@partial(jit, static_argnums=2)
def _return_H(state: Array, gammas: Array, n: int) -> float:
  """ Returns the Hamiltonian of the system - matching the units in Campbell&Ziff (1979)"""

  H_temp = 0.0
  for i in range(n):
    # self ghost vortex
    H_temp += gammas[i] * gammas[i] * jnp.log( (1 - state[i]**2. - state[n+i]**2.) )
    for j in range(i+1,n):
      # real vortices
      H_temp -= gammas[i] * gammas[j] * jnp.log( _r_squared(state,i,j) )
      # ghost vortices (not including self-ghost)
      H_temp += gammas[i] * gammas[j] * jnp.log( (1 - state[i]*state[j] - state[n+i]*state[n+j])**2. + (state[i]*state[n+j] - state[j]*state[n+i])**2. )

  return H_temp / (4. * jnp.pi)
  
@partial(jit, static_argnums=2)
def _return_L(state: Array, gammas: Array, n: int) -> float:
  """ Returns the angular momentum of the system. """

  L_temp = 0.0
  for i in range(n):
    L_temp += gammas[i] * (1. - (state[i]**2. + state[n+i]**2.))

  return L_temp * 0.5
  
@partial(jit, static_argnums=2)
def _return_F(state: Array, gammas: Array, n: int, omega: float) -> float:
  """ Returns the angular momentum of the system """

  return _return_H(state,gammas,n) - omega * _return_L(state,gammas,n)
  
@jit
def _every_induced_velocity(state: Array, gammas: Array, indices: Array, omega: float) -> Array:
  """ Returns the induced velocity for every vortex in the system """

  def _f(carry, x):
    u = _induced_velocity(carry, gammas, x, omega)
    return carry, u
  
  _, k = jax.lax.scan(_f, state, indices)    # loops over each vortex and computes its induced velocity
  return jnp.transpose(k).flatten()       # need to transpose the np.stacked velocities

@jit
def _rk2_step(state: Array, gammas: Array, indices: Array, dt: float, omega: float) -> Array:
  """ Performs a single RK2 step """
  
  k1 = _every_induced_velocity(state, gammas, indices, omega)

  # k2
  state = state + k1 * dt
  
  k2 = _every_induced_velocity(state, gammas, indices, omega)

  # update positions
  state = state + (k2 - k1) * dt * 0.5

  return state

@jit
def _rk2_finalT(state: Array, gammas: Array, indices: Array, N: int, dt: float, omega: float) -> Array:

  """ Helper function to perform N RK2 steps """
  
  # function to perform rk2 step
  def _f(i, val):
    x = _rk2_step(val, gammas, indices, dt, omega)
    return x
    
  final_positions = jax.lax.fori_loop(0, N, _f, state)
  
  return final_positions
  
@partial(jit, static_argnums=(3))
def _rk2_finalN(state: Array, gammas: Array, indices: Array, N: int, dt: float, omega: float) -> Array:

  """ Helper function to perform N RK2 steps """
  
  # function to perform rk2 step
  def _f(carry, x):
    out = _rk2_step(carry, gammas, indices, dt, omega)
    return out, None
  
  final_positions, _ = jax.lax.scan(_f, state, None, length=N)
  
  return final_positions
  
  
class VorticesDisc:
  """
    Class containing the system of vortices in the rotating, disc-bounded domain.
    
    Inputs:
      state : 2n dimensional vector with the first half containing the x coordinates of the vortices and the second half the y coordinates.
      gammas: n dimensional vector equal to the circulations of the vortices.
      omega: non-dimensionalised angular velocity of the domain.
  """
  def __init__(self, state: Array, gammas: Array, omega: float=0.):

    self.n = int(len(state)/2)
    
    self.state = jnp.array(state)
    self.gammas = jnp.array(gammas)
    self.indices = utils.indices(self.n)
    self.rel_indices = utils.rel_indices(self.n)
    self.omega = omega

  def return_H(self) -> float:
    """ Compute the dimensional Hamiltonian"""

    return _return_H(self.state,self.gammas,self.n)
    
  def return_L(self) -> float:
    """ Compute the dimemsional angular momentum"""
    
    return _return_L(self.state,self.gammas,self.n)
    
  def return_F(self, omega: float) -> float:
    """ Compute the dimensional free energy"""
    
    return _return_F(self.state,self.gammas,self.n,omega)

  def r_squared(self, i: int, j: int) -> float:
    """ Returns the squared distance between the vortices at index i and index j """

    return _r_squared(self.state, i, j)

  def induced_velocity(self, index: int) -> Array:
    """ Calculates the total induced velocity for a vortex due to all the other vortices """

    return _induced_velocity(self.state, self.gammas, index, self.omega)
    
  def every_induced_velocity(self, indices) -> Array:
    """ Returns the induced velocity for every vortex in the system """
    
    return _every_induced_velocity(self.state, self.gammas, indices, self.omega)
        

# ----------------------------- Time Integator ---------------------------------

  def rk2_finalT(self, dt: float, Tf: float):
  #def rk2(self, dt: float, N: int):
    """ 2nd order Runge-Kutta method full integrator due to velocities induced by all other vortices """

    eps = 0.001
    N = jnp.int64(jnp.floor_divide(Tf - eps, dt))
    
    final_dt = Tf - N * dt
    
    self.state = _rk2_finalT(self.state, self.gammas, self.indices, N, dt, self.omega)
    self.state = _rk2_step(self.state, self.gammas, self.indices, final_dt, self.omega)
    
  def rk2_finalN(self, dt: float, N: int):
    """ 2nd order Runge-Kutta method full integrator due to velocities induced by all other vortices """

    self.state = _rk2_finalN(self.state, self.gammas, self.indices, N, dt, self.omega)

