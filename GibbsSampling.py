import numpy as np
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

#-------------------------------------------------------------------
def GibbsSampling(scaled_channel_matrix, channel_matrix, noise_variance, P_zero, num_iteration, clients_num_sample, num_of_clients, num_of_antennas, ro=0.9):
  # This function implements the Gibbs Sampling method 
  x_init= np.ones(num_of_clients)
  x= 1*x_init
  beta =1 

  for j in range(num_iteration):
    print('--------------','Iteration'+str(j+1),': ----------------')
    F_set= [] 
    optimal_values = [] 
    F_set.append(1*x)
    beam_vector, _ = SCA(scaled_channel_matrix, x, num_of_antennas, num_of_clients, 0, True)
    optimal_values.append( ObjectiveFunction(beam_vector, channel_matrix, x, noise_variance, P_zero, clients_num_sample))
    for i in range(num_of_clients):
      x_tilda= 1*x
      x_tilda[i]= 1 - x[i] 
      if (np.sum(x_tilda)!=0):
        F_set.append(x_tilda)
        beam_vector, _ = SCA(scaled_channel_matrix, x_tilda, num_of_antennas, num_of_clients, 0, True)
        optimal_values.append(ObjectiveFunction(beam_vector, channel_matrix, x_tilda, noise_variance, P_zero, clients_num_sample))

    optimal_values = np.array(optimal_values)

    scaled_values= -1*(optimal_values/beta)
    max_value = np.max(scaled_values)
    scaled_values -= max_value

    exp_array = np.exp(scaled_values)
    probability_array = exp_array/np.sum(exp_array)

    index= np.random.choice(np.arange(len(optimal_values)), 1, p= probability_array)
    x= 1* F_set[index[0]]
    print('device_selection:', x, '| objective function:' ,optimal_values[index[0]])
    beta= ro*beta

  f_beam, _ = SCA(scaled_channel_matrix, x, num_of_antennas, num_of_clients, 0, True)

  return f_beam, x
#--------------------------------------------------------------------------------------------------------------