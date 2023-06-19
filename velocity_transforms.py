import numpy as np
import jax
from jax import jit
import jax.numpy as jnp
from functools import partial
from typing import Callable, Tuple, List, Union
import utils
import VorticesMotion as vm

Array = Union[np.ndarray, jnp.ndarray]

from jax.config import config
config.update("jax_enable_x64", True)

@jit
def velocity_cartesian_to_polar(x, y, u, v, x_ref=None, y_ref=None):
  """ Returns the velocity in polar co-ordinates given the velocity and position in Cartesian co-ordinates. `x_ref` and `y_ref` should be set to zeros in general, when the system is centred on its centre of vorticity """

  if x_ref == None and y_ref == None:
    n = len(x)
    x_ref = jnp.zeros(n)
    y_ref = jnp.zeros(n)

  r = (x-x_ref)**2 + (y-y_ref)**2

  radial = ((x-x_ref) * u + (y-y_ref) * v) / jnp.sqrt(r)
  angular = ((x-x_ref) * v - (y-y_ref) * u) / r

  return radial, angular


@partial(jit, static_argnums=3)
def mean_angular_velocity(params: Array, gammas: Array, indices: Array, n: int):
  """ Compute the mean angular velocity (excluding any vortices at the centre of vorticity) of state """
  
  params = utils.centre_on_com(params, gammas)  # first centre on centre of vorticity
  
  uvs = vm._every_induced_velocity(params, gammas, indices)
  _, ang_vels = velocity_cartesian_to_polar(params[:n], params[n:], uvs[:n], uvs[n:])
  
  bool_mask = jnp.where(params[:n]**2 + params[n:]**2 > 1e-3, jnp.ones(n), jnp.zeros(n))
  mean_ang_vel = jnp.mean(ang_vels, where=bool_mask)
  
  return mean_ang_vel
  
def scale_to_omega(params, gammas, omega, n, indices):
  """ Scales the vortex positions (params) so that the system is rotating at rate omega """
  
  mean_ang_vel = mean_angular_velocity(params, gammas, indices, n)

  delta = jnp.sqrt(jnp.abs(mean_ang_vel) / omega)   # scaling for each vortex
  
  return params * delta




  

