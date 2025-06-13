import numpy as np
from matplotlib import pyplot as plt
import pickle
from functools import partial
import jax
import jax.numpy as jnp
from jax import config
import optax
from jax import jit, vmap

config.update("jax_enable_x64", True)

plt.rcParams.update({
    "text.usetex": True
})

from typing import Callable, Tuple, List, Union
Array = Union[np.ndarray, jnp.ndarray]

import sys
sys.path.append("../jax-pv/")

import vortex_analysis as va
import helper

from collections import namedtuple
import equinox

# first create the time forward map function
State = namedtuple("State", "steps T x_old x_new gammas avg_observable")

def advance_velocity_module(step_fun, dt, obs_fn, max_steps=None):
  """Returns a function that takes State(time=t0) and returns a State(t0+T). step_fn (and obs_fn) takes xys and gammas"""

  def cond_fun(state):
    """When this returns true, continue time stepping `state`."""
    return dt * state.steps < state.T

  def body_fun(state): 
    """Increment `state` by one time step."""
    v_update = step_fun(state.x_new, state.gammas)
    observable = obs_fn(v_update, state.gammas)
    obs_avg = (state.avg_observable * state.steps + observable) / (state.steps + 1)
    return State(steps=state.steps + 1,
                 T = state.T,
                 x_old=state.x_new,
                 x_new=v_update,
                 gammas=state.gammas,
                 avg_observable=obs_avg)

  # Define a diffrax loop function, assigning every arg except init_val
  bounded_while_fun = partial(
      equinox.internal.while_loop,
      cond_fun=cond_fun,
      body_fun=body_fun,
      max_steps=max_steps,
      kind="bounded",
      base=64)

  def interpolate_fun(state):
    """Interpolate between x_old and x_new to get v at time=T."""
    time_old = (state.steps - 1) * dt
    time_new = state.steps * dt
    delta_t = time_new - time_old
    step_fraction = (state.T - time_old) / delta_t

    delta_x = state.x_new - state.x_old
    x_T = state.x_old + delta_x * step_fraction
      
    return State(
        steps = state.steps + step_fraction,
        T = state.T,
        x_old = state.x_old,
        x_new = x_T,
        gammas=state.gammas,
        avg_observable = state.avg_observable)
  
  def time_advance_state_fun(state):
    state_final = bounded_while_fun(init_val=state)
    return interpolate_fun(state_final)

  return jax.jit(time_advance_state_fun)