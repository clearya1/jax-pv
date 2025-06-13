import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

def compute_time_average(vort_1darray: np.ndarray):
  """ Batch of 1d vorticity fields """
  return np.mean(vort_1darray, axis=0)

def compute_time_and_x_avg(vort_all_array: np.ndarray, Nx: int, Ny: int):
  """ Batch of 1d vorticiy fields """
  vort_images = vort_all_array.reshape((vort_all_array.shape[0], Nx, Ny))
  vort_x_avg = np.concatenate([np.mean(vort_images, axis=1, keepdims=True)] * Nx, axis=1)  # computes the average along x axis, and then replaces each point in the field with this average
  return compute_time_average(vort_x_avg.reshape((vort_all_array.shape[0], -1)))
  
# work flow is:
# 1) filtered_trajectory = compute_filtered_vort_fluctuations(trajectory)
# 2) for filtered_field in filtered_trajectory: labelled_vortices, n_vortices = extract_vortices(filtered_field)

def compute_filtered_vort_fluctuations(vort_all_array: np.ndarray,   # input shape = (-1, Nx * Ny)
                                       Nx: int, 
                                       Ny: int, 
                                       thresh_rms: float) -> np.ndarray:
  """ Extracts vortices as fluctuations about the x and time average vorticity """
  vort_time_x_average = compute_time_and_x_avg(vort_all_array, Nx, Ny)
  vort_fluc = vort_all_array - vort_time_x_average[np.newaxis, :]
  vort_rms = np.mean(vort_fluc ** 2) ** 0.5
  vort_max = np.max(np.abs(vort_all_array))
  
  vort_fluc_filtered = np.zeros_like(vort_fluc)
  vort_filtered = np.zeros_like(vort_fluc)
  condition_pl = vort_fluc > thresh_rms * vort_rms
  condition_mn = vort_fluc < -thresh_rms * vort_rms
  vort_fluc_filtered[condition_pl] = vort_fluc[condition_pl]
  vort_fluc_filtered[condition_mn] = vort_fluc[condition_mn]
  vort_filtered[condition_pl] = vort_all_array[condition_pl]
  vort_filtered[condition_mn] = vort_all_array[condition_mn]
  return vort_fluc_filtered, vort_filtered                                              # output shape = (-1, Nx * Ny)

def compute_filtered_vort(vort_all_array: np.ndarray,   # input shape = (-1, Nx * Ny)
                          Nx: int, 
                          Ny: int, 
                          thresh_rms: float) -> np.ndarray:
  """ Extracts vortices as true vorticity given some threshold of the rms vorticity, without subtracting the time and x average"""
  vort_rms = np.mean(vort_all_array ** 2) ** 0.5
  vort_max = np.max(np.abs(vort_all_array))
  
  vort_filtered = np.zeros_like(vort_all_array)
  condition_pl = vort_all_array > thresh_rms * vort_rms
  condition_mn = vort_all_array < -thresh_rms * vort_rms
  vort_filtered[condition_pl] = vort_all_array[condition_pl]
  vort_filtered[condition_mn] = vort_all_array[condition_mn]
  return vort_filtered                                              # output shape = (-1, Nx * Ny)

def extract_vortices(filtered_vort_snapshot: np.ndarray):
  """ Individual images """
  if filtered_vort_snapshot.ndim != 2:
    raise ValueError(f"Expecting 2D array, received {filtered_vort_snapshot.shape = }")
  Nx, Ny = filtered_vort_snapshot.shape
  
  # extracts individual vortices from filtered vortex fields (ie output of compute_filtered_vort_fluctuations)
  label_array, _ = ndi.label(filtered_vort_snapshot)
  
  # dealing with periodic boundary conditions here
  for i in range(Nx):
    if label_array[i,0] > 0 and label_array[i,-1] > 0:
      label_array[label_array == label_array[i, -1]] = label_array[i, 0]

  for j in range(Ny):
    if label_array[0,j] > 0 and label_array[-1,j] > 0:
      label_array[label_array == label_array[-1, j]] = label_array[0, j]
      
  num_unique_labels = len(np.unique(label_array)) - 1
  return label_array, num_unique_labels

def vortex_count_area_circulation(filtered_vort_snapshot: np.ndarray,
                                  dx: float,
                                  dy: float):
  """ Individual images """
  labels, _ = extract_vortices(filtered_vort_snapshot)
  unique_labels = np.unique(labels)[1:]

  areas = []
  circulations = []
  for label in unique_labels:
    vortex = filtered_vort_snapshot[labels == label]
    vortex_area = vortex.size * dx * dy
    circulation = dx * dy * np.sum(vortex)
    areas.append(vortex_area)
    circulations.append(circulation)
  return labels, len(areas), areas, circulations

def vortex_remove_small_area(filtered_vort_snapshot: np.ndarray,
                             dx: float,
                             dy: float,
                             area_thresh: float):
  labels, _, areas, _ = vortex_count_area_circulation(filtered_vort_snapshot, dx, dy)
  unique_labels = np.unique(labels)[1:]
  for label, area in zip(unique_labels,areas):
    if area < area_thresh * np.max(areas):
      filtered_vort_snapshot[labels == label] *= 0
  return filtered_vort_snapshot


def vortex_centre_of_vorticity(filtered_vort_snapshot: np.ndarray):

  labels, _ = extract_vortices(filtered_vort_snapshot)
  unique_labels = np.unique(labels)[1:]

  Nx = len(filtered_vort_snapshot[0])
  Ny = len(filtered_vort_snapshot[:,0])

  centres_of_vort  = []
  for label in unique_labels:
    vortex = np.where(labels==label, filtered_vort_snapshot, 0.)
    X_vortex, Y_vortex = np.meshgrid(np.arange(Nx), np.arange(Ny))     # need to recreate for each vortex

    # dealing with periodic boundary conditions here
    if np.any(vortex[0,:] != 0) and np.any(vortex[-1,:] != 0):       # need to impose periodic bc
      try:
        possible_slices = np.where(~vortex.any(axis=1))[0]             # the indices of the rows which are all equal to 0
        jump_index = possible_slices[-1]          # choose the last one so we modify the smallest block possible
        Y_vortex[jump_index:, :] -= Ny                # shift whole domain after halfplane separation to negative (need this as vortices are curved)
      except:                                     # if the vortex spans the whole domain, then do nothing
        pass
    if np.any(vortex[:,0] != 0) and np.any(vortex[:,-1] != 0):       # need to impose periodic bc
      try:
        possible_slices = np.where(~vortex.any(axis=0))[0]             # the indices of the rows which are all equal to 0
        jump_index = possible_slices[-1]        # choose the last one so we modify the smallest block possible
        X_vortex[:, jump_index:] -= Nx              # shift whole domain after halfplane separation to negative
      except:                                     # if the vortex spans the whole domain, then do nothing
        pass  
    x_cov = np.average(X_vortex, weights=np.abs(vortex))
    y_cov = np.average(Y_vortex, weights=np.abs(vortex))

    # in case the cov is returned as a negative number
    x_cov = (x_cov%Nx + Nx)%Nx
    y_cov = (y_cov%Ny + Ny)%Ny
    centres_of_vort.append((x_cov, y_cov))

  return labels, len(centres_of_vort), centres_of_vort
    

