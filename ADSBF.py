import numpy as np
from SCA import SCA

#------------------------------------------------------------------
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
#-------------------------------------------------------------------
def DeviceSelectionFunc(beamforming_vec, channel_matrix, noise_variance, P_zero, clients_num_sample, num_of_clients):
  # This function implements Algorithm 2 (Optimal Device Selection Given Beamforming)
  fh= channel_matrix@np.conjugate(beamforming_vec)
  stored_array= (clients_num_sample**2)/(np.abs(fh)**2)
  index= np.argsort(stored_array)
  devices_selection_array= np.ones(num_of_clients)
  emp_array= np.zeros(num_of_clients)
  for i in range(num_of_clients):
    emp_array[num_of_clients-1-i]= ObjectiveFunction(beamforming_vec, channel_matrix, devices_selection_array, noise_variance, P_zero, clients_num_sample)
    devices_selection_array[index[num_of_clients-1-i]]=0

  new_index= np.argsort(emp_array)
  optimal_numberof_devices=  new_index[0]+1
  devices_selection_array= np.ones(num_of_clients)

  for i in range(num_of_clients- optimal_numberof_devices):
    devices_selection_array[index[num_of_clients-1-i]]=0

  return devices_selection_array
#---------------------------------------------------------------------
def ADSBF(num_of_iteration, channel_matrix, scaled_channel_matrix, noise_variance, P_zero, clients_num_sample, num_of_clients, num_of_antennas):
  # This function implements Algorithm 3 (ADSBF)
  opt_devices_selection_array= np.ones(num_of_clients)
  beamforming_vec,_ = SCA(scaled_channel_matrix, opt_devices_selection_array, num_of_antennas, num_of_clients, 0, True) ### beamforming optimization
  new_val = ObjectiveFunction(beamforming_vec, channel_matrix, opt_devices_selection_array, noise_variance, P_zero, clients_num_sample)
  print(new_val)
  for i in range(num_of_iteration):
    print('--------------','Iteration'+str(i+1),': ----------------')
    old_val = 1*new_val
    opt_devices_selection_array = DeviceSelectionFunc(beamforming_vec, channel_matrix, noise_variance, P_zero, clients_num_sample, num_of_clients) ### device selection optimization
    beamforming_vec, _ = SCA(scaled_channel_matrix, opt_devices_selection_array, num_of_antennas, num_of_clients, beamforming_vec, False) ### beamforming optimization using SCA method with initial point from the previous iteration
    new_val = ObjectiveFunction(beamforming_vec, channel_matrix, opt_devices_selection_array, noise_variance, P_zero, clients_num_sample)
    print('Objective function value is: ', new_val)

    if abs(new_val - old_val) < 0.001:
      print('Early stopping happend')
      break

  return beamforming_vec, opt_devices_selection_array
#-----------------------------------------------------------------------------