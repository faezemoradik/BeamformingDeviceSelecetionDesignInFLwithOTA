import numpy as np
import pandas as pd
import torch
import math
from torchvision import datasets
import torchvision.transforms as transforms


#---------------------------------------------------------------------------
def split_and_shuffle_labels(y_data, seed, amount):
  y_data= pd.DataFrame(y_data, columns=["labels"])
  y_data["i"]= np.arange(len(y_data))##index of data
  label_dict = dict()
  for i in range(10):
    var_name="label" + str(i)
    label_info= y_data[y_data["labels"]==i]
    np.random.seed(seed)
    label_info= np.random.permutation(label_info)
    label_info= label_info[0:amount]
    label_info= pd.DataFrame(label_info, columns=["labels","i"])
    label_dict.update({var_name: label_info })
  return label_dict

#---------------------------------------------------------------------------
def get_iid_subsamples_indices(label_dict, num_of_clients, clients_num_sample):
  clients_dict= dict()
  indx=0
  for i in range(num_of_clients):
    client_name="client"+str(i)
    dumb= pd.DataFrame()
    for j in range(10):
      label_name=str("label")+str(j)
      a=label_dict[label_name][indx: indx+ int(clients_num_sample[i]/10)]
      dumb= pd.concat([dumb,a], axis=0)

    dumb.reset_index(drop=True, inplace=True)    
    clients_dict.update({client_name: dumb}) 
    indx += int(clients_num_sample[i]/10)

  return clients_dict

#---------------------------------------------------------------------------
def get_Non_iid_subsamples_indices(label_dict, num_of_clients, clients_num_sample):
  new_label_data_frame= pd.DataFrame()
  for j in range(10):
    label_name= str("label")+str(j)
    a= label_dict[label_name]
    new_label_data_frame= pd.concat([new_label_data_frame, a])

  new_label_data_frame.reset_index(drop=True, inplace=True) 

  clients_dict= dict()
  indx=0
  for i in range(num_of_clients):
    client_name= "client"+str(i)
    clients_dict.update({client_name: new_label_data_frame[indx: indx+ int(clients_num_sample[i])]}) 
    indx += int(clients_num_sample[i])

  return clients_dict

#---------------------------------------------------------------------------
def create_train_data_dict_for_all_clients(clients_dict, x_data, y_data):
  
  device = 'cuda:0' if torch.cuda.is_available() else 'cpu' 
  x_data_dict= dict()
  y_data_dict= dict()
    
  for i in range(len(clients_dict)):
    client_name= "client"+str(i)
        
    indices= np.sort(np.array(clients_dict[client_name]["i"]))
        
    x_info= x_data[indices,:]
    
    x_data_dict.update({client_name : x_info.to(device)})
        
    y_info= y_data[indices]
    y_data_dict.update({client_name: y_info.to(device)})
        
  return x_data_dict, y_data_dict

#----------------------------------------------------------------------
def MNIST_PROCESS(num_of_clients):
  transformation = transforms.Compose([transforms.ToTensor(), torch.flatten])
  train_data = datasets.MNIST(root='./Data/MNIST', train=True, download=True, transform=transformation )
  test_data = datasets.MNIST(root='./Data/MNIST', train=False, download=True, transform=transformation )

  train_loader= torch.utils.data.DataLoader(train_data, batch_size= 60000) 
  test_loader= torch.utils.data.DataLoader(test_data, batch_size= 100) 
  train_amount= 5421 ### MNIST

  for j, (batch_data, batch_labels) in enumerate(train_loader):
    x_train = batch_data
    y_train= batch_labels

  train_loader= torch.utils.data.DataLoader(train_data, batch_size= 100) 

  clients_num_sample= [10*int(math.floor(train_amount/num_of_clients))]* num_of_clients  ### number of traning data points at each client
  clients_num_sample= np.array(clients_num_sample)
  print(clients_num_sample)

  #----------------------- Train data is distributed among clients: ----------------------------
  label_dict_train= split_and_shuffle_labels(y_data=y_train, seed=1, amount=train_amount)

  clients_dict_train= get_iid_subsamples_indices(label_dict_train, num_of_clients, clients_num_sample)
  #clients_dict_train= get_Non_iid_subsamples_indices(label_dict_train, num_of_clients, clients_num_sample)

  x_train_dict, y_train_dict = create_train_data_dict_for_all_clients(clients_dict= clients_dict_train, x_data= x_train, y_data= y_train)

  return x_train_dict, y_train_dict, train_loader, test_loader, clients_num_sample

#---------------------------------------------------------------------------
def CIFAR10_PROCESS(num_of_clients):
  transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
  # Only the data is normalaized we do not need to augment the test data
  transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

  # choose the training and test datasets
  train_data = datasets.CIFAR10('./Data/CIFAR10', train=True, download=True, transform=transform_train)
  test_data = datasets.CIFAR10('./Data/CIFAR10', train=False, download=True, transform=transform_test)

  train_loader= torch.utils.data.DataLoader(train_data, batch_size= 50000) 
  test_loader= torch.utils.data.DataLoader(test_data, batch_size= 100) 

  for j, (batch_data, batch_labels) in enumerate(train_loader):
    x_train = batch_data
    y_train= batch_labels

  train_loader= torch.utils.data.DataLoader(train_data, batch_size= 100) 

  train_amount= 5000 ### CIFAR10
  clients_num_sample= [10*int(math.floor(train_amount/num_of_clients))]* num_of_clients  ### number of traning data points in each client
  clients_num_sample= np.array(clients_num_sample)
  print(clients_num_sample)

  #---------------- Train data is distributed among clients: ----------

  label_dict_train= split_and_shuffle_labels(y_data=y_train, seed=1, amount=train_amount)

  clients_dict_train= get_iid_subsamples_indices(label_dict_train, num_of_clients, clients_num_sample)
  #clients_dict_train= get_Non_iid_subsamples_indices(label_dict_train, num_of_clients, clients_num_sample)

  x_train_dict, y_train_dict = create_train_data_dict_for_all_clients(clients_dict= clients_dict_train, x_data= x_train, y_data= y_train)

  return x_train_dict, y_train_dict, train_loader, test_loader, clients_num_sample




