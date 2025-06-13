import logging
logging.basicConfig(filename=f'log.out',
                      filemode='a',
                      format='%(asctime)s: %(message)s',
                      datefmt='%H:%M:%S',
                      level=logging.INFO)

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
from VorticesMotionPeriodic import PeriodicVortices
from newton_upo import rpoNewtonSolver, poGuess, x_permute, trim_ad_output, respect_permutation_symmetry

import vortex_analysis as va
import helper
import opt_newt_jaxcfd.newton.newton as nt
import opt_newt_jaxcfd.newton.newton_spectral as nt_sp
import opt_newt_jaxcfd.interact_jaxcfd.time_forward_map as tfm
import opt_newt_jaxcfd.interact_jaxcfd.interact_spectral as insp
from opt_newt_jaxcfd.interact_jaxcfd.downsample_spectral import downsample_rft_traj

from VorticesMotionPeriodic import _rk2_finalT, _every_induced_velocity, _rk2_step
from utils import indices, _rotate, _xy_shift
import scipy.linalg as la
import arnoldi as ar
from copy import copy

if __name__=="__main__":

  folder_in = "./ad_out_dynamical/"
  folder_out = "./newt_out_dynamical/"
  solns = np.arange(int(sys.argv[1]), int(sys.argv[2]), 1, dtype=int)

  # set up the UPO parameters
  Re = 100.
  Nx = 128
  Ny = 128
  Lx = 2 * jnp.pi
  Ly = 2 * jnp.pi
  grid = cfd.grids.Grid((Nx, Ny), domain=((0, Lx), (0, Ly)))
  max_velocity = 5. # estimate (not prescribed)
  dt_stable = cfd.equations.stable_time_step(max_velocity, 0.5, 1./Re, grid)
  trim_thresh = 0.05

  N_mode_offset = int(sys.argv[3])  # extract N_mode + N_mode_offset vortices
  N_mode_offset_p_or_n = "p" if N_mode_offset >= 0 else "n"

  for soln in solns:

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    logging.info(f"========= Solution {soln} in [ {solns[0]}, {solns[-1]} ] ==========")

    file_name_in = folder_in+f"pv_ad_out_soln_{soln}_dyn_sample_{N_mode_offset_p_or_n}_{abs(N_mode_offset)}.pkl"
    with open(file_name_in, "rb") as f:
      ad_out = pickle.load(f)
      state = ad_out["state"]
      gammas = ad_out["gammas"]
      period = ad_out["period"]
      shifts = ad_out["shifts"]

    # trim any excess vortices which have negligible circulation
    state, gammas = trim_ad_output(state, gammas, trim_thresh)

    # check if any vortices have permuted
    gamma_tampered, permutation = respect_permutation_symmetry(state, gammas, period, shifts)

    logging.info(f"Permutation = {permutation}")
    logging.info(f"{gammas = }, {gamma_tampered = }")
    po_guess = poGuess(copy(state), copy(gamma_tampered), copy(period), copy(shifts), permutation=permutation)

    # set up newton solver
    newton_solver = rpoNewtonSolver(nmax_hook=20)

    # iterate newton
    po_updated = newton_solver.iterate(po_guess)

    file_name_out = folder_out+f"pv_newt_out_soln_{soln}_dyn_sample_{N_mode_offset_p_or_n}_{abs(N_mode_offset)}.pkl"
    with open(file_name_out, "wb") as f:
        save = {
          "state" : po_updated.state_out,
          "gammas": po_updated.gammas,
          "period": po_updated.T_out,
          "shifts": po_updated.shifts_out,
          "res_history"  : po_updated.newt_resi_history,
          "trim_thresh" : trim_thresh,
          "permutation" : permutation,
          "N_mode_offset" : N_mode_offset,
        }
        pickle.dump(save, f)

