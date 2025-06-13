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
from copy import copy

import helper
from copy import copy

def similarity_measure_of_vortices(X, Y, L):
    """ Input:
            X : (x,y,circ,area) of vortex X at snapshot i
            Y : (x,y,circ,area) of vortex Y at snapshot i+1
            L : size of domain
        Returns:
            Similarity metric between X and Y
    """
    cov_dist_sq = helper.periodic_euclidean_norm(X[:2], Y[:2], L)**2

    return np.sqrt(cov_dist_sq + jnp.linalg.norm(Y[2:] - X[2:])**2) / jnp.linalg.norm(np.array([L, L, X[2], X[3]]))

def shift_reduced_similarity_measure_of_vortices(X, Y, L, shift):
    """ Input:
            X : (x,y,circ,area) of vortex at final snapshot 
            Y : (x,y,circ,area) of vortex at initial snapshot
            L : size of domain
            shift : UPO shift required if we are computing distances between snapshot N and snapshot 0
        Returns:
            Similarity metric between X and Y
        Use this function to see if we should connect trajectories up between the final and initial snapshot.
        For this reason, we must reduce the shift symmetry of the UPO
    """
    
    # transposed shift due to transposed internal representation of centres of vorticity
    cov_dist_sq = helper.periodic_euclidean_norm(X[:2], Y[:2]+np.array([0., shift]), L)**2

    return np.sqrt(cov_dist_sq + jnp.linalg.norm(Y[2:] - X[2:])**2) / jnp.linalg.norm(np.array([L, L, X[2], X[3]]))


def track_vortex_trajectories(centres_of_vorticity, circulations, areas, shift_upo, tol=1e-1, Nx = 128, Lx = 2.*jnp.pi): 
    """ 
        Takes in all the centres of vorticity, circulation and areas of the vortices extracted from each upo snapshot, and the upo shift, and tolerance.
        These are typically, but not necessarily, sorted by the magnitudes of the circulation.
        Create a vector X = (x,y,Gamma,area) for each vortex at each snapshot.
        Consider vortices in sequential snapshots as part of the same trajectory if || X_2 - X_1 ||/|| X_1 || < tol.
        When following trajectories, we don't use the same vortex twice.
        Returns: 
            1. all the trajectories in the form of the [[X1, X2, ....], [...], ...].shape = (N_traj, ?, 4), sorted by length
            2. their time indices in shape (N_traj, ?)
        
    """

    # Compute the vector X for each vortex
    samples = len(circulations)
    #samples_ind = np.arange(samples)
    Xs = []
    for i in range(samples):
        X_temp = []
        for (x,y), circ, area in zip(centres_of_vorticity[i], circulations[i], areas[i]):
            X_temp.append([x,y, circ, area])
        Xs.append(X_temp)

    # Cycle through all the vortices, deleting them if they are included in a trajectory
    trajectories = []
    time_stamps = []
    for i in range(samples):

        snapshot = copy(Xs[i])
        # start a trajectory with each vortex remaining in Xs, and then delete that vortex from Xs
        for Xv in snapshot:
            trajectory = [Xv]
            time_stamp = [i]
            Xs[i].remove(Xv)
            
            # ======== go forward in time =========
            # start j at i+1 and go all the way to the end of the UPO --- no point in going to i-1 as we deleted all vortices at 0
            for j in range(i+1, samples):
                trajectory_ended = True
                # now we cycle through all the vortices at index j
                for Yv in copy(Xs[j]):
                    similarity = similarity_measure_of_vortices(np.array(Xv), np.array(Yv), Lx)
                    # if vortex is sufficiently similar to previous vortex, append and delete this vortex from Xs, and go to next snapshot
                    if similarity < tol:
                        trajectory.append(Yv)
                        time_stamp.append(j)
                        Xs[j].remove(Yv)
                        Xv = Yv
                        trajectory_ended = False
                        break
                    else:
                        continue
                # if none in snapshot are similar, then the trajectory is over
                if trajectory_ended:
                    break
                # =========================================
            
            trajectories.append(trajectory)
            time_stamps.append(time_stamp)
            
    # ======== check if we should connect any trajectories across the period of the UPO ==========

    connected_trajectories = []
    connected_time_stamps = []
    trajectories_to_connect = []

    N_traj = len(time_stamps)
    # search over each snapshot
    for i in range(N_traj):
            
        # check if final timestamp in trajectory is final snapshot (and first timestamp isn't the first snapshot)
        if (time_stamps[i][-1] == samples - 1) and (time_stamps[i][0] != 0):
            # try find a match between vortex and another vortex in a different snapshot
            for j in range(N_traj):
                # don't connect vortex to a vortex in the same snapshot
                if i == j:
                    continue
                else:
                    # if this vortex has an initial time stamp equal to 0, then check shift reduced similarity
                    if (time_stamps[j][-1] != samples - 1) and (time_stamps[j][0] == 0):
                        similarity = shift_reduced_similarity_measure_of_vortices(np.array(trajectories[i][-1]), np.array(trajectories[j][0]), Lx, shift_upo )
                        if (similarity < tol) and (set(time_stamps[j]).isdisjoint(time_stamps[i])):
                            # only have one breakpoint so don't need to worry about stitching multiple trajectories together
                            trajectories_to_connect.append((i,j))      
                            break
                        else:
                            continue
                            
    logging.info(f"Connecting {len(trajectories_to_connect)} Trajectories")

    all_traj_inds = np.arange(N_traj)
    for (i,j) in trajectories_to_connect:
        connected_trajectories.append(trajectories[i] + trajectories[j])
        connected_time_stamps.append(time_stamps[i] + time_stamps[j])
        all_traj_inds = np.delete(all_traj_inds, all_traj_inds==i)
        all_traj_inds = np.delete(all_traj_inds, all_traj_inds==j)
    for i in all_traj_inds:
        connected_trajectories.append(trajectories[i])
        connected_time_stamps.append(time_stamps[i])
                            
    # sort the trajectories by sum of |circulation| along trajectory
    circ_sort_inds = np.argsort([np.sum(np.abs(x)[:,2]) for x in connected_trajectories])[::-1]
    connected_trajectories = [connected_trajectories[i] for i in circ_sort_inds]
    connected_time_stamps = [connected_time_stamps[i] for i in circ_sort_inds]
            
    return connected_trajectories, connected_time_stamps
    
    
def track_vortex_trajectories_decaying(centres_of_vorticity, circulations, areas, tol=1e-1, Nx = 128, Lx = 2.*jnp.pi):
    """
        Takes in all the centres of vorticity, circulation and areas of the vortices extracted from a decaying trajectory, and tolerance.
        These are typically, but not necessarily, sorted by the magnitudes of the circulation.
        Create a vector X = (x,y,Gamma,area) for each vortex at each snapshot.
        Consider vortices in sequential snapshots as part of the same trajectory if || X_2 - X_1 ||/|| X_1 || < tol.
        When following trajectories, we don't use the same vortex twice.
        Returns:
            1. all the trajectories in the form of the [[X1, X2, ....], [...], ...].shape = (N_traj, ?, 4), sorted by length
            2. their time indices in shape (N_traj, ?)
        
    """

    # Compute the vector X for each vortex
    samples = len(circulations)
    #samples_ind = np.arange(samples)
    Xs = []
    for i in range(samples):
        X_temp = []
        for (x,y), circ, area in zip(centres_of_vorticity[i], circulations[i], areas[i]):
            X_temp.append([x,y, circ, area])
        Xs.append(X_temp)

    # Cycle through all the vortices, deleting them if they are included in a trajectory
    trajectories = []
    time_stamps = []
    # start at the final snapshot and go backwards
    for i in range(samples-1, -1, -1):

        snapshot = copy(Xs[i])
        # start a trajectory with each vortex remaining in Xs, and then delete that vortex from Xs
        for Xv in snapshot:
            trajectory = [Xv]
            time_stamp = [i]
            Xs[i].remove(Xv)
            
            # ======== go backwards in time =========
            # start j at i-1 and go all the way to the start of the decaying trajectory
            for j in range(i-1, -1, -1):
                trajectory_ended = True
                # now we cycle through all the vortices at index j
                for Yv in copy(Xs[j]):
                    similarity = similarity_measure_of_vortices(np.array(Xv), np.array(Yv), Lx)
                    # if vortex is sufficiently similar to previous vortex, append and delete this vortex from Xs, and go to next snapshot
                    if similarity < tol:
                        trajectory.append(Yv)
                        time_stamp.append(j)
                        Xs[j].remove(Yv)
                        Xv = Yv
                        trajectory_ended = False
                        break
                    else:
                        continue
                # if none in snapshot are similar, then the trajectory is over
                if trajectory_ended:
                    break
                # =========================================
            
            trajectories.append(trajectory)
            time_stamps.append(time_stamp)
                            
    # sort the trajectories by lengths of the vortex trajectories
    len_sort_inds = np.argsort([len(x) for x in trajectories])[::-1]
    trajectories = [trajectories[i] for i in len_sort_inds]
    time_stamps = [time_stamps[i] for i in len_sort_inds]

    trajectories = [traj[::-1] for traj in trajectories]
    time_stamps = [time_stamp[::-1] for time_stamp in time_stamps]
            
    return trajectories, time_stamps

# now code to average areas and circulations over each of the trajectories

def time_averaged_circ_area(trajectory): 
    """ Takes in vector X = (x,y,circulations,areas) along a trajectory and returns time averaged circ and area"""

    trajectory = np.array(trajectory)
    av_circulation = np.mean(trajectory[:,2])
    av_area = np.mean(trajectory[:,3])

    return av_circulation, av_area

def circulations_and_areas_for_initialisation(trajectories):
    """ Takes in the trajectories of all the vortices and returns the time averaged circulations and areas for the initialisation 
        of the optimisation.
    """

    circs = []
    areas = []
    for trajectory in trajectories:
        circ, area = time_averaged_circ_area(trajectory)
        circs.append(circ)
        areas.append(area)

    return np.array(circs), np.array(areas)

def positions_for_initialisation(trajectories, time_stamps):
    """ Inputs:
            the trajectories of the Nv_mode longest vortices
            the time stamps of the Nv_mode longest trajectories
        Returns 
            the initialisation of the vortex positions
            the time stamp of the initial snapshot
        
        If there exists a snapshot where all the vortices exist, then choose the vortex positions in that snapshots as the initialisation.
        If not, then choose the snapshot where the most amount of vortices exist, and compute the average positions of the remaining vortices.
    """

    Nv = len(time_stamps)
    # need to keep track of which vortices are in each time stamp, so create vortex label list with same shape as time_stamps
    # also keep track of the indices of each time stamp
    vortex_labels = []
    time_stamp_inds = []
    for i in range(Nv):
        vortex_label = np.ones(len(time_stamps[i]))*i
        time_stamp_ind = np.arange(len(time_stamps[i]))
        vortex_labels.append(vortex_label)
        time_stamp_inds.append(time_stamp_ind)
        
    time_stamps_flat = np.array([ x for xs in time_stamps for x in xs ])
    time_stamp_inds_flat = np.array([ x for xs in time_stamp_inds for x in xs ])
    
    assert len(time_stamps_flat) == len(time_stamp_inds_flat)

    # creates an array of indices, sorted by unique element
    idx_sort = np.argsort(time_stamps_flat, kind="mergesort")
    
    # sorts arrays so all unique elements are together 
    sorted_time_stamps_flat = time_stamps_flat[idx_sort]

    # extracts the index of each first repeated element
    unique_time_stamps, unique_time_stamps_idx_start, time_stamps_count = np.unique(sorted_time_stamps_flat, return_index=True, return_counts=True)

    # splits the sorting indices into separate arrays based on the cutoffs between unique values
    res = np.split(idx_sort, unique_time_stamps_idx_start[1:])
    
    # if there is a snapshot with Nv vortices, return vortex positions and that time stamp
    if Nv in time_stamps_count:
        # extract the first snapshot with Nv vortices
        time_stamp_with_Nv_vorts = np.asarray(time_stamps_count == Nv).nonzero()[0][0]
        # snapshot index of each trajectory
        time_stamp_inds_per_trajectory = res[time_stamp_with_Nv_vorts]
        index_per_trajectory = time_stamp_inds_flat[time_stamp_inds_per_trajectory]

        x_init = np.zeros(Nv)
        y_init = np.zeros(Nv)
        t_init = time_stamps[0][index_per_trajectory[0]]
        for i, (idx, trajectory) in enumerate(zip(index_per_trajectory, trajectories)):
            x_init[i] = trajectory[idx][1]
            y_init[i] = trajectory[idx][0]

        for i in range(1, Nv):
          assert t_init == time_stamps[i][index_per_trajectory[i]]
        
        return x_init, y_init, t_init

    # if there is not a snapshot with Nv vortices, find the next best snapshot and compute the centres of vorticity of the missing vortices
    # next best snapshot is defined by the snapshot with the largest sum of the trajectory lengths
    else:
        # compute the sum of the trajectories' |cumulative circulation| which exist in each snapshot
        time_stamp_count_dict = {}
        for traj_time_stamps, traj in zip(time_stamps, trajectories):
            for t in traj_time_stamps:
                if t in time_stamp_count_dict:
                    time_stamp_count_dict[t] += abs(np.sum(np.array(traj)[:,2]))
                else:
                    time_stamp_count_dict[t] = abs(np.sum(np.array(traj)[:,2]))
        

        # find the snapshot which maximises this
        best_snapshot = max(time_stamp_count_dict, key=time_stamp_count_dict.get)

        # figure out which vortices do not exist in this snapshot and compute their centre of vorticity
        x_init = np.zeros(Nv)
        y_init = np.zeros(Nv)
        for i, traj_time_stamps in enumerate(time_stamps):
            if best_snapshot in traj_time_stamps:
                ind_along_traj = np.where(np.array(traj_time_stamps,dtype=int) == best_snapshot)[0][0]
                y_init[i] = trajectories[i][ind_along_traj][0]
                x_init[i] = trajectories[i][ind_along_traj][1]
            else:
                y_init[i] = np.mean(np.array(trajectories[i])[:,0])
                x_init[i] = np.mean(np.array(trajectories[i])[:,1])

        # put everything together and return
        
        return x_init, y_init, best_snapshot
