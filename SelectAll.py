import numpy as np
from SCA import SCA

#--------------------------------------------------------------
def SelectAll(scaled_channel_matrix, num_of_antennas, num_of_clients):
  # This function implements the Select-All approach
  devices_selection_array = np.ones(num_of_clients)
  f_beam, _ =  SCA(scaled_channel_matrix, devices_selection_array, num_of_antennas, num_of_clients, 0, True)

  return f_beam, devices_selection_array

#----------------------------------------------------------------



