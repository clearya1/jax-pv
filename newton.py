from typing import Callable, Tuple, List, Union

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as la
from utils import _rotate, centre_on_com

Array = Union[np.ndarray, jnp.ndarray]

import arnoldi as ar

class relEqGuess:
  def __init__(
      self, 
      x: Array,
      T: float,
      shift: float,
      guess_loss: float=None
  ):
    self.x_init = x
    self.T = T
    self.shift_init = shift
    self.guess_loss = guess_loss

  def record_outcome(
      self,
      x_out: Array,
      shift_out: float,
      newton_residual_history: List[float],
      converged: bool
  ):
    self.x_out = x_out
    self.shift_out = shift_out
    self.newt_resi_history = newton_residual_history
    self.converged = converged


class newtonSolver:
  def __init__(
      self, 
      forward_map: Callable[[Array, float], Array],
      dt_stable: float=1e-3,
      eps_newt: float=1e-10, 
      eps_gm: float=1e-3,
      nmax_newt: int=100, 
      nmax_hook: int=10,
      nmax_gm: int=100, 
      Delta_rel: float=0.1
  ):
    """ Delta_start * norm(x) will be the size of the Hookstep constraint. Renormalise """
    self.forward_map = forward_map    

    self.eps_newt = eps_newt # Newton step convergence
    self.eps_gm = eps_gm # GMRES Krylov convergence
    self.nmax_newt = nmax_newt # max newton iterations
    self.nmax_hook = nmax_hook
    self.nmax_gm = nmax_gm # max steps for GMRES

    self.dt_stable = dt_stable
    self.Delta_rel = Delta_rel

  def _initialise_guess(
      self, 
      releq_guess: relEqGuess
  ):
    self.x_guess = releq_guess.x_init
    self.T = releq_guess.T
    self.a_guess = releq_guess.shift_init

    self.Delta_start = self.Delta_rel * la.norm(self.x_guess)
    print('Starting Delta = ', self.Delta_start)

    self.original_shape = self.x_guess.shape
    self.n = int(self.x_guess.size/2)
    self.Ndof = self.x_guess.size

    self._update_F()

  def _timestep_DNS(
      self, 
      x0: Array,
      T_march: float
  ) -> Array:
  
    xT = self.forward_map(x0, T_march)
    
    return xT
  
  def iterate(
      self, 
      releq_guess: relEqGuess
  ) -> relEqGuess:
    self._initialise_guess(releq_guess)

    newt_res = la.norm(self.F)
    res_history = []
    newt_count = 0
    converged = 1
    while la.norm(self.F) / la.norm(self.x_guess) > self.eps_newt:
      kr_basis, gm_res, _ = ar.gmres(self._timestep_A, -self.F, self.eps_gm, self.nmax_gm)
      dx, _ = ar.hookstep(kr_basis, 2*self.Delta_start)
      
      self.x_guess += dx[:self.Ndof]
      self.a_guess += dx[-1]

      self._update_F()

      newt_new = la.norm(self.F)

      # (more) hooksteps if reqd
      Delta = self.Delta_start
      hook_count = 1
      print("old res: ", newt_res, "new_res: ", newt_new)
      if newt_new > newt_res:
        u_local = self.x_guess - dx[:self.Ndof]
        a_local = self.a_guess - dx[-1]
        print("Starting Hookstep... ")
        while newt_new > newt_res and hook_count < self.nmax_hook + 2:
          dx, _ = ar.hookstep(kr_basis, Delta)
          self.x_guess = u_local +  dx[:self.Ndof]
          self.a_guess = a_local + dx[-1]

          self._update_F()

          newt_new = la.norm(self.F)
          Delta /= 2.
          hook_count += 1
        print("# hooksteps:", hook_count)
      print("Current Newton residual: ", la.norm(self.F) / 
           la.norm(self.x_guess))
      print("shift guess: ", self.a_guess)
      newt_res = newt_new
      res_history.append(newt_res / la.norm(self.x_guess))
      newt_count += 1
      
      if newt_count > self.nmax_newt: 
        print("Newton count exceeded limit. Ending guess.")
        converged = 0
        break
      if hook_count > self.nmax_hook:
        print("Hook steps exceeded limit. Ending guess.")
        converged = 0
        break
    self.x_guess = centre_on_com(self.x_guess, np.ones(self.n))    # centre the guess back on its com
    releq_guess.record_outcome(self.x_guess, self.a_guess, res_history, converged)
    return releq_guess
      
  def _timestep_A(
      self, 
      eta_w: Array      # this is the array we are acting the Jacobian on
  ) -> Array:
  
    _, Aeta = jax.jvp(self.F_for_jacobian_x, (self.x_guess,), (eta_w[:self.Ndof],))

    x_T = self._timestep_DNS(self.x_guess, self.T)
    shift_partial_action = jnp.concatenate((-1.*x_T[self.n:], x_T[:self.n])) * eta_w[-1]
  
    Aeta += shift_partial_action

    Aeta_w_x = np.append(Aeta, np.dot(eta_w[self.n:-1],self.x_guess[:self.n]) - np.dot(self.x_guess[self.n:],eta_w[:self.n]) )    # theta deltas

    return Aeta_w_x
    
  def F_for_jacobian_x(
      self, x
  ):
    x_T = self._timestep_DNS(x, self.T)
    shifted_x_T = x_shift_all(x_T, self.a_guess)
    
    return shifted_x_T - x
    
  def F_for_jacobian_shift(
      self, shift
  ):
    x_T = self._timestep_DNS(self.x_guess, self.T)
    shifted_x_T = x_shift_all(x_T, shift)
    
    return shifted_x_T - self.x_guess
    
    

  def _update_F(
      self
  ):
    self.x_T = self._timestep_DNS(self.x_guess, self.T)
    shifted_x_T = x_shift_all(self.x_T, self.a_guess)
    
    self.F = np.append(shifted_x_T - self.x_guess, [0.]) # 1 zero for rotational shift row

def x_shift_all(
  x: Array,
  shift: float
) -> Array:
  """ Shift 1 rotational symmetry. Note that cov is conserved if the total circulation is nonzero """
  
  x =  _rotate(x, shift)
  
  return x
