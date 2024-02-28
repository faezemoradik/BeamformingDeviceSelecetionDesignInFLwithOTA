import numpy as np
import cvxpy as cp
import copy

#------------------------------------------------------------------
def Initial_point_prob_two(cost, channel_matrix, gamma):
  size= channel_matrix.shape
  K=size[0]
  N=size[1]
  M_var= cp.Variable((N,N), hermitian =True)
  constraints = [cp.real(cp.trace(M_var))-1 >= 0]
  constraints += [M_var >> 0]
  constraints += [cp.real(cp.trace(M_var))-gamma[k]*cp.real(np.conj(channel_matrix[k]).T@M_var@channel_matrix[k])<=0 for k in range(K)]
  prob = cp.Problem(cp.Minimize(cost),constraints)
  prob.solve()

  return M_var.value, prob.status

#----------------------------------------------------------------------
def Initial_point_prob_one(cost, N, K, channel_matrix, gamma):
  flag = True

  M_var= cp.Variable((N,N), hermitian = True)
  x_var = cp.Variable(K, nonneg = True)


  constraints = [cp.real(cp.trace(M_var))-1 >= 0]
  constraints += [M_var >> 0]
  constraints+= [cp.real(cp.trace(M_var))-gamma[k]*cp.real(np.conj(channel_matrix[k]).T@M_var@channel_matrix[k])-x_var[k]<=0 for k in range(K)]

  prob = cp.Problem(cp.Minimize(cost), constraints)

  prob.solve()

  if prob.status == 'infaesible':
    #print('Problem is infeasible')
    flag = False

  return flag , M_var.value, x_var.value

#----------------------------------------------------------------------
def feasibility_DC( initial_point, channel_matrix, gamma, maxiter):
  size= channel_matrix.shape
  K=size[0]
  N=size[1]
  M_var= cp.Variable((N,N), hermitian =True)
  M_partial= cp.Parameter((N,N), hermitian =True)


  M = 1*initial_point
  obj0 = Problem_two_objective(M)

  constraints = [cp.real(cp.trace(M_var))-1 >= 0]
  constraints += [M_var >> 0]
  constraints += [cp.real(cp.trace(M_var))-gamma[k]*cp.real(np.conj(channel_matrix[k]).T@M_var@channel_matrix[k])<=0 for k in range(K)]
  cost= (cp.real(cp.trace((np.eye(N)-M_partial.H)@M_var)))*1

  prob = cp.Problem(cp.Minimize(cost),constraints)

  for iter in range(maxiter):
    _,V = np.linalg.eigh(M)
    u= V[:,N-1]
    M_partial.value = copy.deepcopy(np.outer(u,u.conj()))

    prob.solve()
    M= copy.deepcopy(M_var.value)


    err= obj0-Problem_two_objective(M)
    obj0 = Problem_two_objective(M)

    if err<1e-8:
      break

  u,s,_= np.linalg.svd(M,compute_uv=True,hermitian=True)
  m= u[:,0]
  feasibility= sum(s[1:])<1e-6

  if feasibility:
    for i in range(K):
      flag= np.linalg.norm(m)**2/np.linalg.norm(m.conj()@channel_matrix[i])**2 <= gamma[i]

      if not flag:
        feasibility= False


  return m, feasibility

#----------------------------------------------------------------------

def Problem_one_objective(x , M , k):
  obj= np.linalg.norm(x,1)
  x_abs = np.abs(x)
  indx = np.argsort(-1*x_abs)
  obj -= np.sum(x_abs[indx[0:k]])
  s = np.linalg.svd(M, compute_uv= False, hermitian= True)
  obj +=  sum(s[1:])

  return obj

#----------------------------------------------------------------------
def Problem_two_objective(M):
  s = np.linalg.svd(M, compute_uv= False, hermitian= True)
  obj =  sum(s[1:])

  return obj

#----------------------------------------------------------------------
def device_selection_DC(N, K, channel_matrix, gamma, maxiter):
  M_var= cp.Variable((N,N), hermitian = True)
  x_var = cp.Variable(K, nonneg = True)
  x_partial= cp.Parameter(K)
  M_partial= cp.Parameter((N,N), hermitian = True)

  constraints = [cp.real(cp.trace(M_var))-1 >= 0]
  constraints += [M_var >> 0]

  constraints+= [cp.real(cp.trace(M_var))-gamma[k]*cp.real(np.conj(channel_matrix[k]).T@M_var@channel_matrix[k])-x_var[k]<=0 for k in range(K)]
  cost= (cp.norm(x_var,1)-x_partial.H@x_var + cp.real(cp.trace((np.eye(N)-M_partial.H)@M_var)))*1

  prob = cp.Problem(cp.Minimize(cost), constraints)

  for c in range(K+1):

    print('Outer Round is:', c)
    flag, M, x = Initial_point_prob_one(1, N, K, channel_matrix, gamma)

    if flag== True:
      obj0 = Problem_one_objective(x , M , c)

      for iter in range(maxiter):
        print('Inner round is:', iter)
        x_abs = np.abs(x)
        x_p = np.zeros([K,])
        ind= np.argsort(-x_abs)

        x_p[ind[0:c]]= copy.deepcopy(np.sign(x[ind[0:c]]))

        x_partial.value = copy.deepcopy(x_p)
        _,V = np.linalg.eigh(M)
        u=V[:,N-1]

        M_partial.value = copy.deepcopy(np.outer(u,u.conj()))

        prob.solve()

        x= copy.deepcopy(x_var.value)
        M= copy.deepcopy(M_var.value)

        err= obj0- Problem_one_objective(x , M , c)
        obj0 = Problem_one_objective(x , M , c)

        if err<1e-8:
          break


      if obj0 < 1e-6 :
        break


  ind= np.argsort(x)

  for i in np.arange(K):
    print('Feasibilty round is:', i)

    active_device_num= K-i
    active_device= np.asarray(ind[0:active_device_num])

    initial_point, prob_status = Initial_point_prob_two( 1, channel_matrix[active_device,:], gamma[active_device])

    if prob_status == 'optimal':
      m, feasibility = feasibility_DC(initial_point, channel_matrix[active_device,:], gamma[active_device], maxiter)
      if feasibility:
        break
      else:
        m= None
        active_device = None

    else:
      m= None
      active_device = None

  return m, active_device

#----------------------------------------------------------------------
def DC_NORIS(num_of_antennas, num_of_clients, channel_matrix, gamma , maxiter):
  print('gamma is: ', gamma)

  device_selection = np.zeros(num_of_clients)

  m, active_device = device_selection_DC( num_of_antennas, num_of_clients, channel_matrix, gamma, maxiter)

  if m is not None:
    print('case 1')
    beam_forming= 1*m
    device_selection[active_device]=1

  else:
    print('case 2')
    beam_forming= channel_matrix[0]/np.linalg.norm(channel_matrix[0])
    device_selection[0] =1

  return beam_forming, device_selection

#----------------------------------------------------------------------
