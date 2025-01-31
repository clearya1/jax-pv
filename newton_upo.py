import logging
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

import jax_cfd.base as cfd
import jax_cfd.spectral as spectral

plt.rcParams.update({
    "text.usetex": True
})

from typing import Callable, Tuple, List, Union
Array = Union[np.ndarray, jnp.ndarray]

import sys
sys.path.append("../UPOs/")

sys.path.append("../jax-pv/")
sys.path.append("../TurbulenceVortices/")
from VorticesMotionPeriodic import PeriodicVortices

import helper
import opt_newt_jaxcfd.newton.newton as nt
import opt_newt_jaxcfd.newton.newton_spectral as nt_sp
import opt_newt_jaxcfd.interact_jaxcfd.time_forward_map as tfm
import opt_newt_jaxcfd.interact_jaxcfd.interact_spectral as insp
from opt_newt_jaxcfd.interact_jaxcfd.downsample_spectral import downsample_rft_traj

from VorticesMotionPeriodic import _rk2_finalT, _every_induced_velocity, _rk2_step, _rk2_finalT_nobc
from utils import indices, _rotate, _xy_shift
import scipy.linalg as la
import arnoldi as ar
from copy import copy

class poGuess:
  def __init__(
      self, 
      state: Array,
      gammas: Array,
      T: float,
      shifts: Array,
      permutation: Array=None,
      guess_loss: float=None
  ):
    self.state_init = state
    self.gammas = gammas
    self.T_init = T
    self.shifts_init = shifts
    self.permutation = permutation
    self.guess_loss = guess_loss

  def record_outcome(
      self,
      state_out: Array,
      T_out: Array,
      shifts_out: Array,
      newton_residual_history: List[float],
      converged: bool
  ):
    self.state_out = state_out
    self.T_out = T_out
    self.shifts_out = shifts_out
    self.newt_resi_history = newton_residual_history
    self.converged = converged

# define the time forward map for the Newton method

class rpoNewtonSolver:
  def __init__(
      self,
      dt_stable: float=1e-4,
      eps_newt: float=1e-10, 
      eps_gm: float=1e-3,
      nmax_newt: int=100, 
      nmax_hook: int=10,
      nmax_gm: int=100, 
      Delta_rel: float=0.1
  ):
    """ Delta_start * norm(x) will be the size of the Hookstep constraint. Renormalise """  

    self.eps_newt = eps_newt # Newton step convergence
    self.eps_gm = eps_gm # GMRES Krylov convergence
    self.nmax_newt = nmax_newt # max newton iterations
    self.nmax_hook = nmax_hook
    self.nmax_gm = nmax_gm # max steps for GMRES

    self.dt_stable = dt_stable
    self.Delta_rel = Delta_rel

    self.L = 2. * jnp.pi    # square domain length
    self.m = 4              # number of ghost domains to sum over

  def _initialise_guess(
      self, 
      po_guess: poGuess
  ):
    self.x_guess = po_guess.state_init
    self.T_guess = po_guess.T_init
    self.a_guess = po_guess.shifts_init
    self.gammas = po_guess.gammas

    self.permutation = po_guess.permutation

    self.T_current = 0. # keep track for re-jitting time forward map

    self.Delta_start = self.Delta_rel * la.norm(self.x_guess)

    self.original_shape = self.x_guess.shape
    self.n = int(self.x_guess.size/2)
    self.Ndof = self.x_guess.size
    self.ind = indices(self.n)

    self._update_F()
    self._update_dx_dt()

  def rhs_equations(self, x):
    """ Compute f(x) = dx / dt """

    return _every_induced_velocity(x, self.gammas, self.ind, self.L, self.m)

  def rk2_finalT(self, state: Array, N: int, dt: float):
    """ 2nd order Runge-Kutta method full integrator due to velocities induced by all other vortices """

    state = _rk2_finalT_nobc(state, self.gammas, self.ind, N, dt, self.L, self.m)

    return state

  def _jit_tfm(
      self
  ):
    """ Create time forward map minimal number of times """
    dt_exact = self.T_current / int(self.T_current / self.dt_stable)
    Nt = int(self.T_current / dt_exact)
    self.current_tfm = partial(self.rk2_finalT, N=Nt, dt=dt_exact)

  def _timestep_DNS(
      self, 
      x_0: Array,
      T_march: float
  ) -> Array:
    if T_march != self.T_current:
      self.T_current = T_march
      self._jit_tfm() 
  
    x_T = self.current_tfm(x_0)
    return x_T 

  def norm_field(
    self,
    field: Array
  ) -> float:
    return la.norm(field.reshape((-1,)))
  
  def iterate(
      self, 
      po_guess: poGuess
  ) -> poGuess:
    self._initialise_guess(po_guess)

    nmax_hook_taken = 0   # allow 3 nmax_hook violations

    newt_res = la.norm(self.F)
    res_history = [newt_res / self.norm_field(self.x_guess)]
    logging.info(f"Starting Newton residual: {la.norm(self.F) / self.norm_field(self.x_guess)}")
    logging.info(f"Starting shift guess: {self.a_guess}")
    logging.info(f"Starting T guess: {self.T_guess}")
    newt_count = 0
    converged = 1
    while la.norm(self.F) / la.norm(self.x_guess) > self.eps_newt:
      kr_basis, gm_res, _ = ar.gmres(self._timestep_A, -self.F, self.eps_gm, self.nmax_gm)
      dx, _ = ar.hookstep(kr_basis, 2*self.Delta_start)

      # save local before updating as will be taking modulus due to periodic b.c.
      u_local = copy(self.x_guess)
      
      self.x_guess += dx[:self.Ndof]
      self.a_guess += dx[self.Ndof:-1]
      self.T_guess += dx[-1]

      # respect periodic b.c.
      self.x_guess = (self.x_guess % self.L + self.L) % self.L

      self._update_F()
      self._update_dx_dt()

      newt_new = la.norm(self.F)

      # (more) hooksteps if reqd
      Delta = self.Delta_start
      hook_count = 1
      logging.info(f"old res: {newt_res}, new_res: {newt_new}")
      if newt_new > newt_res:
        #u_local = self.x_guess - dx[:self.Ndof]
        a_local = self.a_guess - dx[self.Ndof:-1]
        T_local = self.T_guess - dx[-1]
        logging.info("Starting Hookstep... ")
        while newt_new > newt_res and hook_count < self.nmax_hook + 2:
          dx, _ = ar.hookstep(kr_basis, Delta)
          self.x_guess = u_local +  dx[:self.Ndof]
          self.a_guess = a_local + dx[self.Ndof:-1]
          self.T_guess = T_local + dx[-1]

          # respect periodic b.c.
          self.x_guess = (self.x_guess % self.L + self.L) % self.L

          self._update_F()
          self._update_dx_dt()

          newt_new = la.norm(self.F)
          Delta /= 2.
          hook_count += 1
        logging.info(f"# hooksteps: {hook_count}")
      logging.info(f"Current Newton residual:  {la.norm(self.F) / la.norm(self.x_guess)}")
      logging.info(f"shift guess:  {self.a_guess}")
      logging.info(f"T guess: {self.T_guess}")
      newt_res = newt_new
      res_history.append(newt_res / la.norm(self.x_guess))
      newt_count += 1
      
      if newt_count > self.nmax_newt: 
        logging.info("Newton count exceeded limit. Ending guess.")
        converged = 0
        break
      if (hook_count > self.nmax_hook):
        nmax_hook_taken += 1
        if (nmax_hook_taken > 3):
          logging.info("Hook steps exceeded limit 3 times. Ending guess.")
          break
      if self.T_guess < 0.:
        logging.info("Negative period invalid. Ending guess.")
        break
    po_guess.record_outcome(self.x_guess, self.T_guess, self.a_guess, res_history, converged)
    return po_guess
      
  def _timestep_A(
      self, 
      eta_w_T: Array      # this is the array we are acting the Jacobian on
  ) -> Array:
  
    _, Aeta = jax.jvp(self.F_for_jacobian_x, (self.x_guess,), (eta_w_T[:self.Ndof],))
      
    shift_partial_action = jnp.concatenate((np.ones(self.n)*eta_w_T[self.Ndof], np.ones(self.n)*eta_w_T[self.Ndof+1]))
      
    Aeta += shift_partial_action
    Aeta += self.dxTp_dT.reshape((-1,)) * eta_w_T[-1]

    Aeta_w_x = np.append(Aeta, np.sum(eta_w_T[:self.n]) )            # x symm gauge fixing
    Aeta_w_x = np.append(Aeta_w_x, np.sum(eta_w_T[self.n:self.Ndof]) )        # y symm gauge fixing
    Aeta_w_x_T = np.append(Aeta_w_x, np.dot(self.dx0_dt, eta_w_T[:self.Ndof]) )        # y symm gauge fixing

    return Aeta_w_x_T
    
  def F_for_jacobian_x(
      self, x
  ):
    x_T = self._timestep_DNS(x, self.T_guess)

    x_T_p = x_permute(x_T, self.permutation)

    # no need to shift as shift is additive
    return x_T_p - x
    
  def _update_F(
      self
  ):
    self.x_T = self._timestep_DNS(self.x_guess, self.T_guess)
    shifted_x_T = _xy_shift(self.x_T, self.a_guess)
    self.shifted_permuted_x_T = x_permute(shifted_x_T, self.permutation)
    
    self.F = np.append(self.shifted_permuted_x_T - self.x_guess, [0., 0., 0.]) # 2 zeros for symmetries (x and y direction), 1 zero for time

  def _update_dx_dt(
      self
  ):
    self.dx0_dt = self.rhs_equations(self.x_guess) 
    self.dxTp_dT = self.rhs_equations(self.shifted_permuted_x_T)
    
# ----------------- Periodic Lattice Solver ---------------

class latticeGuess:
  def __init__(
      self,
      state: Array,
      gammas: Array,
      T: float,
      shifts: Array,
      L: float,
      guess_loss: float=None
  ):
    self.state_init = state
    self.gammas = gammas
    self.T = T
    self.shifts_init = shifts
    self.guess_loss = guess_loss
    self.L = L

  def record_outcome(
      self,
      state_out: Array,
      shifts_out: Array,
      newton_residual_history: List[float],
      converged: bool
  ):
    self.state_out = state_out
    self.shifts_out = shifts_out
    self.newt_resi_history = newton_residual_history
    self.converged = converged

class latticeNewtonSolver:
  def __init__(
      self,
      dt_stable: float=1e-4,
      eps_newt: float=1e-10,
      eps_gm: float=1e-3,
      nmax_newt: int=100,
      nmax_hook: int=10,
      nmax_gm: int=100,
      Delta_rel: float=0.1
  ):
    """ Delta_start * norm(x) will be the size of the Hookstep constraint. Renormalise """

    self.eps_newt = eps_newt # Newton step convergence
    self.eps_gm = eps_gm # GMRES Krylov convergence
    self.nmax_newt = nmax_newt # max newton iterations
    self.nmax_hook = nmax_hook
    self.nmax_gm = nmax_gm # max steps for GMRES

    self.dt_stable = dt_stable
    self.Delta_rel = Delta_rel

    self.m = 4              # number of ghost domains to sum over

  def _initialise_guess(
      self,
      lattice_guess: latticeGuess
  ):
    self.x_guess = lattice_guess.state_init
    self.T = lattice_guess.T
    self.a_guess = lattice_guess.shifts_init
    self.gammas = lattice_guess.gammas
    self.L = lattice_guess.L             # square domain length

    self.Delta_start = self.Delta_rel * la.norm(self.x_guess)

    self.original_shape = self.x_guess.shape
    self.n = int(self.x_guess.size/2)
    self.Ndof = self.x_guess.size
    self.ind = indices(self.n)
    
    dt_exact = self.T / int(self.T / self.dt_stable)
    Nt = int(self.T / dt_exact)
    self.tfm = partial(self.rk2_finalT_nobc, N=Nt, dt=dt_exact)

    self._update_F()

  def rk2_finalT_nobc(self, state: Array, N: int, dt: float):
    """ 2nd order Runge-Kutta method full integrator due to velocities induced by all other vortices """

    state = _rk2_finalT_nobc(state, self.gammas, self.ind, N, dt, self.L, self.m)

    return state

  def _timestep_DNS(
      self,
      x_0: Array
  ) -> Array:
  
    x_T = self.tfm(x_0)
    return x_T

  def norm_field(
    self,
    field: Array
  ) -> float:
    return la.norm(field.reshape((-1,)))
  
  def iterate(
      self,
      lattice_guess: latticeGuess
  ) -> poGuess:
    self._initialise_guess(lattice_guess)

    nmax_hook_taken = 0   # allow 3 nmax_hook violations

    newt_res = la.norm(self.F)
    res_history = [newt_res / self.norm_field(self.x_guess)]
    logging.info(f"Starting Newton residual: {la.norm(self.F) / self.norm_field(self.x_guess)}")
    logging.info(f"Starting shift guess: {self.a_guess}")
    newt_count = 0
    converged = 1
    while la.norm(self.F) / la.norm(self.x_guess) > self.eps_newt:
      kr_basis, gm_res, _ = ar.gmres(self._timestep_A, -self.F, self.eps_gm, self.nmax_gm)
      dx, _ = ar.hookstep(kr_basis, 2*self.Delta_start)

      # save local before updating as will be taking modulus due to periodic b.c.
      u_local = copy(self.x_guess)
      
      self.x_guess += dx[:self.Ndof]
      self.a_guess += dx[self.Ndof:]

      # respect periodic b.c.
      self.x_guess = (self.x_guess % self.L + self.L) % self.L

      self._update_F()

      newt_new = la.norm(self.F)

      # (more) hooksteps if reqd
      Delta = self.Delta_start
      hook_count = 1
      logging.info(f"old res: {newt_res}, new_res: {newt_new}")
      if newt_new > newt_res:
        a_local = self.a_guess - dx[self.Ndof:]
        logging.info("Starting Hookstep... ")
        while newt_new > newt_res and hook_count < self.nmax_hook + 2:
          dx, _ = ar.hookstep(kr_basis, Delta)
          self.x_guess = u_local +  dx[:self.Ndof]
          self.a_guess = a_local + dx[self.Ndof:]

          # respect periodic b.c.
          self.x_guess = (self.x_guess % self.L + self.L) % self.L

          self._update_F()

          newt_new = la.norm(self.F)
          Delta /= 2.
          hook_count += 1
        logging.info(f"# hooksteps: {hook_count}")
      logging.info(f"Current Newton residual:  {la.norm(self.F) / la.norm(self.x_guess)}")
      logging.info(f"shift guess:  {self.a_guess}")
      newt_res = newt_new
      res_history.append(newt_res / la.norm(self.x_guess))
      newt_count += 1
      
      if newt_count > self.nmax_newt:
        logging.info("Newton count exceeded limit. Ending guess.")
        converged = 0
        break
      if (hook_count > self.nmax_hook):
        nmax_hook_taken += 1
        if (nmax_hook_taken > 3):
          logging.info("Hook steps exceeded limit 3 times. Ending guess.")
          converged = 0
          break
    lattice_guess.record_outcome(self.x_guess, self.a_guess, res_history, converged)
    return lattice_guess
      
  def _timestep_A(
      self,
      eta_w: Array      # this is the array we are acting the Jacobian on
  ) -> Array:
  
    _, Aeta = jax.jvp(self.F_for_jacobian_x, (self.x_guess,), (eta_w[:self.Ndof],))
      
    shift_partial_action = jnp.concatenate((jnp.ones(self.n)*eta_w[self.Ndof], jnp.ones(self.n)*eta_w[self.Ndof+1]))
      
    Aeta += shift_partial_action

    Aeta_w_x = np.append(Aeta, jnp.sum(eta_w[:self.n]) )            # x symm gauge fixing
    Aeta_w_x = np.append(Aeta_w_x, jnp.sum(eta_w[self.n:self.Ndof]) )        # y symm gauge fixing

    return Aeta_w_x
    
  def F_for_jacobian_x(
      self, x
  ):
    x_T = self._timestep_DNS(x)

    # need to shift as shift is additive, also not taking modulo anymore
    return x_T - x
    
  def _update_F(
      self
  ):
    x_T = self._timestep_DNS(self.x_guess)
    # no longer taking modulo
    #shifted_x_T = x_shift_all(x_T, self.a_guess, self.L)
    shifted_x_T = _xy_shift(x_T, self.a_guess)
    
    self.F = np.append(shifted_x_T - self.x_guess, [0., 0.]) # 2 zeros for symmetries (x and y direction)

# ------ Utility things ---------

def x_shift_all(
  x: Array,
  shifts: Array,
  L: float=2.*jnp.pi,
) -> Array:
  """ Shift 1 rotational symmetry. Note that cov is conserved if the total circulation is nonzero.
      Also need to respect periodic boundary conditions.
  """

  x =  _xy_shift(x, shifts)
  x = (x%L + L)%L
  
  return x

def x_permute(x: Array, permutation: Array):
  """ Return the permutated state """

  n = len(permutation)
  permutation = jnp.concatenate((permutation, permutation+n))
  return x[permutation]

def trim_ad_output(state, gammas, thresh=0.05):
  """ Remove the vortices which have circulation smaller than thresh """
  inds = np.where(np.abs(gammas) < thresh)[0]
  logging.info(f"Trimming {len(inds)} vortices with circulation less than {thresh}")
  state_trimmed = np.delete(state, inds+len(gammas))
  state_trimmed = np.delete(state_trimmed, inds)
  
  gammas_trimmed = np.delete(gammas, inds)
  gammas_trimmed = np.array(gammas_trimmed) - np.mean(gammas_trimmed)
  return state_trimmed, gammas_trimmed

def to_cycles(perm):
  """ take a permutation list of indices and extract all cycles """
  pi = {i: perm[i] for i in range(len(perm))}
  cycles = []
  while pi:
    elem0 = next(iter(pi)) # arbitrary starting element
    this_elem = pi[elem0]
    next_item = pi[this_elem]
    cycle = []
    while True:
      cycle.append(this_elem)
      del pi[this_elem]
      this_elem = next_item
      if next_item in pi:
        next_item = pi[next_item]
      else:
        break
    cycles.append(cycle)
  return cycles

def respect_permutation_symmetry(state, gammas, period, shifts, Lx=2.*jnp.pi ):
  """ Check if any vortices have permuted, ensure that circulations are equal for any permuted vortices. Return modified circulations and the permutation. """
  # check if any vortices have permuted
  system = PeriodicVortices(state, gammas, Lx, 4)
  dt = 0.001
  N_steps = int(period / dt)
  system.rk2_finalN(dt, N_steps)
  state_T_shifted = _xy_shift(system.state, shifts)
  
  permutation = helper.best_permutation(state, state_T_shifted, Lx)
  cycles = to_cycles(permutation)

  for cycle in cycles:
    if len(cycle) > 1:
      gammas[cycle] = np.mean(gammas[cycle])

  gammas = np.array(gammas) - np.mean(gammas)

  return gammas, permutation
