""" Arnoldi for stability of periodic solutions """ 
from typing import Callable, Tuple, List, Union

import jax
import jax.numpy as jnp
from jax.numpy.linalg import norm
Array = jnp.ndarray

from functools import partial
from jax import config
config.update("jax_enable_x64", True)

from newton_upo import poGuess, x_shift_all, x_permute
from VorticesMotionPeriodic import PeriodicVortices
  
from scipy.linalg import eig

def _monodromy_state(
    rpo: poGuess,
    forward_map: Callable[[Array], Array],
    L = 2.*jnp.pi,
  ) -> Array:
  
  state = rpo.state_out
  shifts = rpo.shifts_out
  gammas = rpo.gammas
  permutation = rpo.permutation
  
  n = len(gammas)
  
  def perm_shifted_forward_map(state: Array) -> Array:
  
    state_and_gammas = jnp.concatenate((state, gammas))
    state_and_gammas_T = forward_map(state_and_gammas)
    state_T = state_and_gammas_T[:2*n]
    shifted_state_T = x_shift_all(state_T, shifts, L)
    permuted_shifted_state_T = x_permute(shifted_state_T, permutation)
    return permuted_shifted_state_T
    
  grad_perm_shifted_forward_map = jax.jacfwd(perm_shifted_forward_map)
  
  return grad_perm_shifted_forward_map(state)

def _monodromy_state_and_gammas(
    rpo: poGuess,
    forward_map: Callable[[Array], Array],
    L = 2.*jnp.pi,
  ) -> Array:
  
  state = rpo.state_out
  shifts = rpo.shifts_out
  gammas = rpo.gammas
  permutation = rpo.permutation
  
  n = len(gammas)
  
  state_and_gammas = jnp.concatenate((state, gammas))
  
  def perm_shifted_forward_map(state_and_gammas: Array) -> Array:
  
    state_and_gammas_T = forward_map(state_and_gammas)
    shifted_state_T = x_shift_all(state_and_gammas_T[:2*n], shifts, L)
    permuted_shifted_state_T = x_permute(shifted_state_T, permutation)
    permuted_gammas = state_and_gammas[2*n:][permutation]
    permuted_shifted_state_and_gammas_T = jnp.concatenate((permuted_shifted_state_T, permuted_gammas))
    return permuted_shifted_state_and_gammas_T
    
  grad_perm_shifted_forward_map = jax.jacfwd(perm_shifted_forward_map)
  
  return grad_perm_shifted_forward_map(state_and_gammas)
  
def compute_rpo_stability(
    rpo: poGuess,
    dt_stable: float,
    L = 2.*jnp.pi,
) -> Tuple[Array]:

  dt_exact = rpo.T_out / int(rpo.T_out / dt_stable)
  Nt = int(rpo.T_out / dt_exact)
  n = len(rpo.gammas)

  def forward_map(state_and_gammas):
    state = state_and_gammas[:2*n]
    gammas = state_and_gammas[2*n:]
    system = PeriodicVortices(state, gammas, L, 4)

    system.rk2_finalN(dt_exact, Nt)

    return jnp.concatenate((system.state, system.gammas))

  monodromy = _monodromy_state_and_gammas(rpo, forward_map, L)

  floquet_mults, floquet_eigenv = eig(monodromy)
  return floquet_mults, floquet_eigenv


from scipy.sparse.linalg import eigs as arp_eigs
from scipy.sparse.linalg import LinearOperator

def _timestep_monodromy(
    X: Array, # perturbation on state_and_gammas
    rpo: poGuess,
    forward_map: Callable[[Array], Array],
    L = 2.*jnp.pi,
    eps_fd: float=1e-7
) -> Array:
  state = rpo.state_out
  shifts = rpo.shifts_out
  gammas = rpo.gammas
  permutation = rpo.permutation

  n = len(gammas)

  state_and_gammas = jnp.concatenate((state, gammas))

  # TODO replace finite difference jacobian estimate with jax
  eps_new = eps_fd * norm(state_and_gammas) / norm(X)

  state_and_gammas_T = forward_map(state_and_gammas + eps_new * X)
  shifted_state_T = x_shift_all(state_and_gammas_T[:2*n], shifts, L)

  permuted_shifted_state_T = x_permute(shifted_state_T, permutation)

  # ==== pretty sure we need to permute the gammas ====
  permuted_gammas = state_and_gammas_T[2*n:][permutation]

  permuted_shifted_state_and_gammas_T = jnp.concatenate((permuted_shifted_state_T, permuted_gammas))

  AX = (1./eps_new) * (permuted_shifted_state_and_gammas_T - state_and_gammas)

  return AX
    

def compute_rpo_stability_sparse(
    rpo: poGuess,
    dt_stable: float,
    L = 2.*jnp.pi,
    N_eig: int=50,
    eps_fd: float=1e-7
) -> Tuple[Array]:

  dt_exact = rpo.T_out / int(rpo.T_out / dt_stable)
  Nt = int(rpo.T_out / dt_exact)
  n = len(rpo.gammas)
  
  def forward_map(state_and_gammas):
    state = state_and_gammas[:2*n]
    gammas = state_and_gammas[2*n:]
    system = PeriodicVortices(state, gammas, L, 4)
    
    system.rk2_finalN(dt_exact, Nt)
    
    return jnp.concatenate((system.state, system.gammas))
  
  Ndof = 3*n
  
  timestepper = partial(_timestep_monodromy, rpo=rpo, forward_map=forward_map, L=L, eps_fd=eps_fd)
  AX = LinearOperator(shape=(Ndof, Ndof), matvec=timestepper, dtype='float64')

  seed = 42
  key = jax.random.PRNGKey(seed)
  pert = jax.random.uniform(key, (3*n,), minval=-1., maxval=1.)
  
  starting_vec = jnp.concatenate((rpo.state_out, rpo.gammas)) + 0.01 * pert
  
  floquet_mults, floquet_eigenv = arp_eigs(AX, k=N_eig, v0=starting_vec)
  return floquet_mults, floquet_eigenv
