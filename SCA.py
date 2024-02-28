import numpy as np
import math
import cvxpy as cp

#------------------------------------------------------------------------------------
def Randomization(F_SDR, num_random_samples, devices_selection_array, scaled_channel_matrix, num_of_antennas, num_of_clients):
  # This function applies the randmoization to the solution found by the SDR.
  rank= np.linalg.matrix_rank(F_SDR)
  #print( 'Ranke of the matrix is: ', rank )
  eigen_values, eigen_vectors = np.linalg.eig(F_SDR)
  #print(eigen_values)
  eigen_values= np.clip(np.real(eigen_values), 0, None)
  idx = np.argsort(eigen_values)
  eigen_values= eigen_values[idx]
  eigen_vectors= eigen_vectors[:,idx]
  if (rank == 1):
    return eigen_vectors[:, num_of_antennas-1]

  else:
    sigma= np.diag(eigen_values)
    sigma_sqrt= np.sqrt(sigma)
    randomization= np.random.normal(0, 1, (num_of_antennas, num_random_samples ))
    Candidates = (eigen_vectors@sigma_sqrt)@ randomization
    for b in range(num_random_samples):
      for i in range(num_of_clients):
        if devices_selection_array[i]==1:
          h_conj= np.conjugate(scaled_channel_matrix[:,i])
          d= np.outer(scaled_channel_matrix[:,i], h_conj)
          constraint= np.conjugate(Candidates[:,b])@d@Candidates[:,b]
          if(constraint < 1):
            Candidates[:,b] = Candidates[:,b]/np.sqrt(constraint)

    candidates_norm= np.zeros(num_random_samples)
    for n in range(num_random_samples):
      candidates_norm[n]= np.linalg.norm(Candidates[:,n])

    can_idx = np.argsort(candidates_norm)
    candidates_norm= candidates_norm[can_idx]
    Candidates= Candidates[:,can_idx]


  return Candidates[:,0], candidates_norm[0]**2 
#---------------------------------------------------------------------------------------
def SDR(scaled_channel_matrix, devices_selection_array, num_of_antennas, num_of_clients):
  # This function computes the Semi-Definite Relaxation (SDR) solution for the single-group downlink multicast beamforming problem.
  F1= cp.Variable((num_of_antennas, num_of_antennas), hermitian = True)
  objective_function = (cp.trace(F1))/1
  constraints=[ F1 >> 0]
  for i in range(num_of_clients):
    if devices_selection_array[i] ==1:
      h_conj= np.conjugate(scaled_channel_matrix[:,i])
      d= np.outer(scaled_channel_matrix[:,i], h_conj)
      constraints.append(cp.real(cp.trace(d @ F1)) >= 1)

  prob = cp.Problem(cp.Minimize(objective_function), constraints)
  prob.solve()
  #print('Problem Status is: ',prob.status ,'|', 'Optimal value: ', prob.value ) #'\n','Optimal F_SDR: ', F_SDR.value)
  #------ Apply Randomization ------
  f_prime_sdr, loss= Randomization(F1.value, 1000 , devices_selection_array, scaled_channel_matrix, num_of_antennas, num_of_clients)
  f_sdr=  f_prime_sdr/ np.linalg.norm(f_prime_sdr,2)

  return f_sdr, f_prime_sdr

#----------------------------------------------------------------------------------------------

def SCA(scaled_channel_matrix, devices_selection_array, num_of_antennas, num_of_clients, initial_point, self_initialization=True):
  # This function computes the SCA solution for the single-group downlink multicast beamforming problem.
  if self_initialization:
    f_sdr, f_prime_sdr= SDR(scaled_channel_matrix , devices_selection_array, num_of_antennas, num_of_clients)
    z_init= 1*f_prime_sdr
  else: 
    z_init= 1* initial_point

  z= 1*z_init
  num_iterations= 10
  for k in range(num_iterations):
    
    x= cp.Variable(2*num_of_antennas)
    objective_func= cp.norm(x,2)
    constraints=[]
    for i in range(num_of_clients):
      if devices_selection_array[i] ==1:
        hh= np.outer(scaled_channel_matrix[:,i], np.conjugate(scaled_channel_matrix[:,i]))
        real_p= np.real(hh@z)
        imag_p= np.imag(hh@z)
        v= np.concatenate((real_p, imag_p))
        #print(v)
        z_conj= np.conjugate(z)
        gamma1= np.absolute(z_conj@scaled_channel_matrix[:,i])
        gamma2 = 1+ math.pow(gamma1,2)
        constraints.append( x@ v >= gamma2/2)

    prob = cp.Problem(cp.Minimize(objective_func), constraints)
    prob.solve(solver= cp.CVXOPT)
    #print(prob.status,'|', prob.value)
    if (abs((prob.value)**2- (np.linalg.norm(z,2))**2) < 1e-2):
      break

    z= x.value[0:num_of_antennas] + 1j* x.value[num_of_antennas:2*num_of_antennas]
  
  f_prime_sca= x.value[0:num_of_antennas] + 1j* x.value[num_of_antennas:2*num_of_antennas]
  f_sca= f_prime_sca/ np.linalg.norm(f_prime_sca,2)

  return f_sca, 1e7*f_prime_sca
#-----------------------------------------------------------------------------------------
