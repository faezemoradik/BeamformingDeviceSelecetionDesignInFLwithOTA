import numpy as np
import torch
import pickle
import os
import argparse
from channel import Channel_Condition
from data import MNIST_PROCESS, CIFAR10_PROCESS
from NonIdealFL import NonIdealFedSGD


#----------------------------------------------------
def get_parameter():
  parser=argparse.ArgumentParser()
  parser.add_argument("-myseed",default=0,type=int,help="seed")
  parser.add_argument("-dataset",default='MNIST',type=str,help="dataset name")
  parser.add_argument("-method",default='GSDS',type=str,help="method name")
  args= parser.parse_args()
  return args

#------------------------------------------------------------
def main():
  if torch.cuda.is_available():
    print("CUDA is available")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
  else:
    print("CUDA is not available")

  args = get_parameter()
  dataset= args.dataset
  method= args.method
  myseed = args.myseed
  print('method: ', method)
  print('dataset: ', dataset)
  print('myseed: ', myseed)

  ############ Parameters #########################
  P_zero = 1e-3 ### Max allowed transmit power of devices 
  num_of_antennas= 16 ### number of antennas at parameter server
  num_of_clients = int(200)### number of clients
  r_max =100
  r_min =10

  if dataset== 'MNIST':
    x_train_dict, y_train_dict , train_loader, test_loader, clients_num_sample = MNIST_PROCESS(num_of_clients)
    model_name = 'LogisticRegression'
    batch_size= 270
    num_epoch= 100
    noise_variance= 1e-3*np.power(10, -20/10) 
    learning_rate = 0.05
  elif dataset== 'CIFAR10':
    x_train_dict, y_train_dict , train_loader, test_loader, clients_num_sample = CIFAR10_PROCESS(num_of_clients)
    model_name = 'ResNet'
    batch_size= 50
    num_epoch= 400
    noise_variance= 1e-3*np.power(10, -50/10) 
    learning_rate = 0.01


  performance = dict()

  np.random.seed(myseed)
  torch.random.manual_seed(myseed)

  channel_matrix, scaled_channel_matrix = Channel_Condition(r_min, r_max, num_of_clients, num_of_antennas, clients_num_sample)
  train_acc_list, test_acc_list, train_loss_list, test_loss_list, transmit_powers, run_time, devices_selection_array = NonIdealFedSGD(x_train_dict, y_train_dict, train_loader, 
                                                                                                                                      test_loader, model_name, num_epoch, num_of_clients, 
                                                                                                                                      clients_num_sample, num_of_antennas, learning_rate, 
                                                                                                                                      channel_matrix, scaled_channel_matrix, P_zero, noise_variance,
                                                                                                                                      batch_size, method)
  

  performance[str(myseed)]= dict()
  performance[str(myseed)]['train_acc'] = train_acc_list
  performance[str(myseed)]['test_acc'] = test_acc_list
  performance[str(myseed)]['train_loss'] = train_loss_list
  performance[str(myseed)]['test_loss'] = test_loss_list
  performance[str(myseed)]['transmit_power'] = transmit_powers
  performance[str(myseed)]['run_time'] = run_time
  performance[str(myseed)]['selected_devices_num']= np.sum(devices_selection_array)


  with open(method+dataset+'seed'+str(myseed), 'wb') as f:
    pickle.dump(performance, f)

#-------------------------------------------------------------------------
main()
