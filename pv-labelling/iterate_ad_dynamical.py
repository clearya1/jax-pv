import logging
logging.basicConfig(filename=f'log.out',
                      filemode='a',
                      format='%(asctime)s: %(message)s',
                      datefmt='%H:%M:%S',
                      level=logging.INFO)

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
from functools import partial
import jax
import jax.numpy as jnp
from jax import config
import optax
from jax import jit, vmap
from scipy import stats

config.update("jax_enable_x64", True)

import jax_cfd.base as cfd
import jax_cfd.spectral as spectral

plt.rcParams.update({
    "text.usetex": True
})

from typing import Callable, Tuple, List, Union
Array = Union[np.ndarray, jnp.ndarray]

import sys

from VorticesMotionPeriodic import PeriodicVortices

import vortex_analysis as va
import helper
import loss_functions as lf
from loss_functions import periodic_gaussian_smear
import extract_dynamical as ed

from VorticesMotionPeriodic import _rk2_finalT, _every_induced_velocity, _rk2_step, _rk2_step_nobc
from time_forward_map import advance_velocity_module, State
from utils import indices, _xy_shift
import glob

from matplotlib import rc

rc('font', family='serif', size=9)
#rc('figure', dpi='1000')
rc('text', usetex=True)
plt.rcParams.update({
    'text.latex.preamble': r'\usepackage{amsfonts}',
    'lines.linewidth' : 0.4,
})

if __name__ == "__main__":

  folder = "./Re100_128/"
  folder_out = "./ad_out_dynamical/"
  solns = np.arange(int(sys.argv[1]), int(sys.argv[2]), 1 ,dtype=int)

  # set up the UPO parameters
  Re = 100.

  # set up the vortex extraction parameters
  thresh_rms = 2.0   # rms vorticity factor threshold
  thresh_area = 0.1
  samples = 50       # number of snapshots per period of UPO
  sim_thres = 0.1    # threshold for extracting vortex trajectories
  N_mode_offset = int(sys.argv[3])  # extract N_mode + N_mode_offset vortices
  N_mode_offset_p_or_n = "p" if N_mode_offset >= 0 else "n"

  # set up the optimisation parameters
  start_learning_rate = 1e-2      # initial learning rate for Adam optimiser
  opt_N = 1000                    # number of optimiser steps
  kappa = 0.5                      # initial kappa
  anneal = True                  # whether to anneal or not

  # ------- define the functions to update the initial random guess ---------
  @partial(jit, static_argnums=(4,5,))
  def gradfn_match(state_init, gamma_init, period, upo_snapshots, grid, forward_map, sigma2):
    return jax.value_and_grad(lf.loss_match_pv_upo, argnums=(0,1))(state_init, gamma_init, period, upo_snapshots, grid, forward_map, sigma2)
    
  @partial(jit, static_argnums=(4,5,))
  def gradfn_upo(state_init, gamma_init, period, shifts, forward_map, grid):
    return jax.value_and_grad(lf.loss_upo_gaussian, argnums=(0,1,3,))(state_init, gamma_init, period, shifts, grid, forward_map)
    
  @partial(jit, static_argnums=(5,6,7,10,))
  def updatefn_match_area_upo(state, gammas, period, shifts, upo_snapshots, grid, forward_map, forward_map_nobc, sigma2, opt_state, optimizer, kappa = 0.5):
    # anneal between the matching loss and the upo loss. For matching loss: train state, gammas. For upo loss: train state, gammas, shifts
    # set kappa equal to 0.5 for equal weighting of the two losses.

    #params_match = [state, gammas]
    loss_match, grads_match = gradfn_match(state, gammas, period, upo_snapshots, grid, forward_map, sigma2)
    grads_match = list(grads_match)
    grads_match = [x * kappa for x in grads_match]

    #params_upo = [state, gammas, shifts]
    loss_upo, grads_upo = gradfn_upo(state, gammas, period, shifts, forward_map_nobc, grid)
    grads_upo = list(grads_upo)

    grads_upo = [x * (1. - kappa) for x in grads_upo]

    params = [state, gammas, shifts]
    # combine all the gradients
    grads_upo[0] += grads_match[0]
    grads_upo[1] += grads_match[1]
    
    updates, opt_state = optimizer.update(grads_upo, opt_state)

    params = optax.apply_updates(params, updates)

    return params, opt_state, loss_match, loss_upo
    
  # -------------------------------------------------------

  for soln in solns:

    logging.info(soln)

    # read in the solution
    file_name = folder+f"soln_array_Re100_{soln}.npy"
    file_meta_name = folder+f"soln_meta_Re100_{soln}.npy"

    logging.info(f"{file_name = }")
    with open(file_name, "rb") as f:
        omega_rft = jnp.load(f)
    with open(file_meta_name, "rb") as f:
        meta = jnp.load(f)
        period_upo = meta[0]
        shift_upo = meta[1]

    omega = jnp.fft.irfftn(omega_rft)

    Nx = len(omega)
    Ny = Nx
    Lx = 2 * jnp.pi
    Ly = 2 * jnp.pi
    grid = cfd.grids.Grid((Nx, Ny), domain=((0, Lx), (0, Ly)))
    max_velocity = 5. # estimate (not prescribed)
    dt_stable = cfd.equations.stable_time_step(max_velocity, 0.5, 1./Re, grid)

    # extract the vortices from trajectory of RPO
    vortices = np.zeros((samples, Nx, Ny))
    N_vortices = np.arange(samples, dtype=int)

    trajectory = helper.simulate_period(omega, period_upo, samples, dt_stable, Re, grid)

    areas = []
    circulations = []
    centres_of_vorts = []
    filtered_fluc_trajectory, filtered_trajectory = va.compute_filtered_vort_fluctuations(trajectory.reshape((trajectory.shape[0], -1)), Nx, Ny, thresh_rms)
    filtered_trajectory = filtered_trajectory.reshape((-1, Nx, Ny))

    for i, filtered_snapshot in enumerate(filtered_trajectory):

        filtered_snapshot = va.vortex_remove_small_area(filtered_snapshot, grid.step[0], grid.step[1], thresh_area)
        vortices[i], N_vortices[i], area, circulation = va.vortex_count_area_circulation(filtered_snapshot, grid.step[0], grid.step[1])
        _, _, centres_of_vort = va.vortex_centre_of_vorticity(filtered_snapshot)
        areas.append(area)
        circulations.append(circulation)
        centres_of_vorts.append(centres_of_vort)

    # dynamical initialisation of a point vortex model for the UPO

    # switch to physical coordinates
    centres_of_vorts = [[np.array(x)/Nx*Lx for x in xs] for xs in centres_of_vorts]
    # compute average number of vortices
    Nv_mode = stats.mode(N_vortices)[0] + N_mode_offset

    # first connect the vortices into trajectories
    trajectories, time_stamps = ed.track_vortex_trajectories(centres_of_vorts, circulations, areas, shift_upo, sim_thres)
    longest_trajectories = [trajectories[i] for i in range(Nv_mode)]
    longest_time_stamps = [time_stamps[i] for i in range(Nv_mode)]

    x_init, y_init, best_snapshot = ed.positions_for_initialisation(longest_trajectories, longest_time_stamps)
    gamma_init, area_init = ed.circulations_and_areas_for_initialisation(longest_trajectories)
    sigma2 = area_init * 0.1
    state_init = np.concatenate((x_init, y_init))

    # save the extraction for this RPO
    with open(f"./extract_out_dynamical/initial_extraction_{soln}_dyn_sample_{N_mode_offset_p_or_n}_{abs(N_mode_offset)}.pkl", "wb") as f:
      save = {
        "state" : state_init,
        "gammas": gamma_init,
        "period": period_upo,
        "shifts": [shift_upo, 0.],
        "sigma2": sigma2,
        "thresh_rms" : thresh_rms,
        "thresh_area": thresh_area,
        "samples": samples,
        "sim_thres": sim_thres,
        "start_learning_rate": start_learning_rate,
        "opt_N": opt_N,
        "starting_snapshot" : best_snapshot,
        "N_mode_offset"     : N_mode_offset,
      }
      pickle.dump(save, f)

    rolled_trajectory = helper.roll_and_shift(trajectory, best_snapshot, shift_upo, grid)
    
    # define the forward map step function
    Nv = len(gamma_init)
    inds = indices(Nv)
    dt = 0.001

    step_fn = partial(_rk2_step, indices=inds, dt=dt, L = Lx, m=4)
    step_fn_nobc = partial(_rk2_step_nobc, indices=inds, dt=dt, L = Lx, m=4)
    forward_map = advance_velocity_module(step_fn, dt, lambda x, y: 0., max_steps = int(10.*period_upo/dt))
    forward_map_nobc = advance_velocity_module(step_fn_nobc, dt, lambda x, y: 0., max_steps = int(10.*period_upo/dt))

    # set up the optimiser

    param_labels = ["state", "gammas", "shifts"]
    optimizer = optax.multi_transform(
        {
          'state': optax.adam(start_learning_rate),
          'gammas': optax.adam(start_learning_rate),
          'shifts': optax.adam(start_learning_rate),
        },
        param_labels)

    shifts_both = jnp.array([shift_upo, 0.])
    opt_state = optimizer.init([state_init, gamma_init, shifts_both])
    state_both = state_init

    gammas_both = np.array(gamma_init) - np.mean(gamma_init)

    # run the optimiser

    for i in range(opt_N):
        ad_out, opt_state, loss_match, loss_upo = updatefn_match_area_upo(state_both, gammas_both, period_upo, shifts_both, rolled_trajectory, grid, forward_map, forward_map_nobc, sigma2, opt_state, optimizer, kappa = kappa)
        state_both, gammas_both, shifts_both = ad_out
        # project back onto net zero circulation
        gammas_both = jnp.array(gammas_both) - jnp.mean(gammas_both)
        # enforce periodic b.c. on state
        state_both = (jnp.array(state_both)%Lx + Lx)%Lx
        if i%100 == 0:
            logging.info(f"{i}, {loss_match = }, {loss_upo = }")

        if loss_upo < 1e-5:
            logging.info("AD Converged to 1e-5, therefore breaking! :)")
            break

        if anneal:
            kappa -= 1./float(opt_N)

    logging.info(f"Final loss: {loss_match = }, {loss_upo = }")
    # save the optimised parameters
    file_name_out = folder_out+f"pv_ad_out_soln_{soln}_dyn_sample_{N_mode_offset_p_or_n}_{abs(N_mode_offset)}.pkl"
    with open(file_name_out, "wb") as f:
        save = {
            "state" : state_both,
            "gammas": gammas_both,
            "period": period_upo,
            "shifts": shifts_both,
            "loss_match"  : loss_match,
            "loss_upo"  : loss_upo,
            "kappa"     : kappa,
            "anneal"    : anneal,
            "sigma2"    : sigma2,
            "thresh_rms" : thresh_rms,
            "thresh_area": thresh_area,
            "samples": samples,
            "sim_thres": sim_thres,
            "start_learning_rate": start_learning_rate,
            "opt_N": opt_N,
            "starting_snapshot" : best_snapshot,
            "N_mode_offset"     : N_mode_offset,
        }
        pickle.dump(save, f)













