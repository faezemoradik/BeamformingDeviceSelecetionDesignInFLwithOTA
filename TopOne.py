import numpy as np

#-------------------------------------------------------------
def StrongestChannel(channel_matrix):
  # This function selects the device with strongest channel condition and returns the corresponding device selection array
  device_selection_array= np.zeros(len(channel_matrix))
  channel_norms= np.zeros(len(channel_matrix))
  for i in range(len(channel_matrix)):
    channel_norms[i]= np.linalg.norm(channel_matrix[i], 2)

  max_indx = np.argmax(channel_norms)
  device_selection_array[max_indx] = 1

  return device_selection_array, max_indx

#--------------------------------------------------------------
def TopOne(channel_matrix):
  # This function implements the top-one approach
  devices_selection_array, indx = StrongestChannel(channel_matrix)
  f_beam = channel_matrix[indx]/ np.linalg.norm(channel_matrix[indx])

  return f_beam, devices_selection_array

#----------------------------------------------------------------
