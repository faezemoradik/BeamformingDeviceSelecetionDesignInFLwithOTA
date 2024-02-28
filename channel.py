import numpy as np


#--------------------------------------------------------------------
def clients_distance(r_min , r_max, num_of_clients):
  r = np.random.uniform(low= r_min, high= r_max, size= num_of_clients) 

  return r ### clients disances from the PS
#-------------------------------------------------------------------
def Channel_Condition(r_min, r_max, num_of_clients, num_of_antennas, clients_num_sample):
  clients_distances = clients_distance(r_min, r_max, num_of_clients)
  channel_matrix= np.zeros((num_of_clients, num_of_antennas))+ 1j*np.zeros((num_of_clients, num_of_antennas))
  scaled_channel_matrix= np.zeros((num_of_clients, num_of_antennas))+ 1j*np.zeros((num_of_clients, num_of_antennas))

  c1= 3.2*((np.log10(11.75*1))**2)-4.97
  c2= (44.9-6.55*np.log10(30))*np.log10(clients_distances/1e3)
  path_loss_db= 46.3+33.9*np.log10(2000)-13.82*np.log10(30)-c1+c2
  path_loss= np.power(10, path_loss_db/10)
  
  for i in range(num_of_clients):
    channel_matrix[i] = np.sqrt(1/path_loss[i])*(1*np.random.normal(0, 1/np.sqrt(2), num_of_antennas)+1j*np.random.normal(0, 1/np.sqrt(2) , num_of_antennas))
    scaled_channel_matrix[i] = 1e7*channel_matrix[i]/clients_num_sample[i]

  scaled_channel_matrix= scaled_channel_matrix.T
  
  return channel_matrix, scaled_channel_matrix
#-------------------------------------------------------------------

