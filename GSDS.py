import numpy as np
import math
import cvxpy as cp
from SCA import SCA

#------------------------------------------------------------
def ObjectiveFunction(beamforming_vec, channel_matrix, devices_selection_array, noise_variance, P_zero, clients_num_sample):
  # This function returns the value for the objective function we aim to minimize 
  term1= (4/(np.sum(clients_num_sample)**2))*(((1- devices_selection_array)@clients_num_sample)**2)
  if np.sum(devices_selection_array) != 0:
    term2= (noise_variance/P_zero)/((devices_selection_array@clients_num_sample)**2)
    fh= channel_matrix@np.conjugate(beamforming_vec)
    term3 = (clients_num_sample**2)*devices_selection_array/(np.abs(fh)**2)
    term4= np.max(term3)
    return term1+ term2*term4
  else:
    return term1

#----------------------------------------------------------
def UpdateSpaceBasis(space, vec):
  # This function updates the basis set for a subspace (space) by adding the orthogonal component of a given vector to the set of basis.
  new_space = []
  q = 1*vec
  for i in range(len(space)):
    new_space.append(space[i])
    rij = vec@ np.conj(space[i])
    q = q - (rij*space[i])/(space[i]@np.conj(space[i]))

  if (np.linalg.norm(q) > 1e-10):
    new_space.append(q)

  return new_space
#------------------------------------------------------------
def VectorProjection(space, vec):
  projection = np.zeros(len(vec))+1j*np.zeros(len(vec))
  for i in range(len(space)):
    projection += ((vec @np.conj(space[i]))/(space[i]@np.conj(space[i])))*space[i]

  return projection
#------------------------------------------------------------
def MaxProjectionIndex(space, channel_matrix, device_selection_vector):
  # This function finds the index of the device that has the maximum channle projection norm on a space with given basis among the unchosen devices.
  projected_values= np.zeros(len(channel_matrix))
  for i in range(len(channel_matrix)):
    if device_selection_vector[i]==0:
      projected_vector= VectorProjection(space, channel_matrix[i])
      projected_values[i]= np.linalg.norm(projected_vector)
  max_indx= np.argmax(projected_values)

  return max_indx
#-------------------------------------------------------------
def GSDS(scaled_channel_matrix, channel_matrix, noise_variance, P_zero, num_of_antennas, num_of_clients, clients_num_sample):
  device_selection_vector= np.zeros(len(channel_matrix))
  channel_norms=[]
  Set_of_basis=[]
  information_matrix= np.zeros((num_of_clients, num_of_clients+1)) # Stores the device selection array together with corresponding objective function value.
  for i in range(len(channel_matrix)):
    channel_norms.append(np.linalg.norm(channel_matrix[i]))

  max_indx = np.argmax(channel_norms)
  device_selection_vector[max_indx] = 1
  Set_of_basis.append(channel_matrix[max_indx])

  f_bench, _ = SCA(scaled_channel_matrix, device_selection_vector, num_of_antennas, num_of_clients, 0, True)
  information_matrix[0, 0:num_of_clients] = 1*device_selection_vector
  information_matrix[0, num_of_clients] = ObjectiveFunction(f_bench, channel_matrix, device_selection_vector, noise_variance, P_zero, clients_num_sample)

  for j in range(num_of_clients-1):
    indx= MaxProjectionIndex(Set_of_basis, channel_matrix, device_selection_vector)
    device_selection_vector[indx] = 1
    f_bench, _ = SCA(scaled_channel_matrix, device_selection_vector, num_of_antennas, num_of_clients, 0, True) 
    information_matrix[j+1, 0:num_of_clients] = 1*device_selection_vector
    information_matrix[j+1, num_of_clients] = ObjectiveFunction(f_bench, channel_matrix, device_selection_vector, noise_variance, P_zero, clients_num_sample)

    Set_of_basis = UpdateSpaceBasis(Set_of_basis , channel_matrix[indx]) # Updates the set of basis for the subspace formed by the channle vectors of the selected devices by adding the channel vector of the newly selected device

  
  indx= np.argmin(information_matrix[:, num_of_clients])
  GSDS_device_selection = information_matrix[indx, 0:num_of_clients]
  f_GSDS,_ = SCA(scaled_channel_matrix, GSDS_device_selection, num_of_antennas, num_of_clients, 0, True)

  return f_GSDS, GSDS_device_selection
#-------------------------------------------------------------------




