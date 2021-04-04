# WS server example

import asyncio
import websockets

# Federated Learning Model in PyTorch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from DPMechanisms import gaussian_noise_weight,gaussian_noise, gaussian_noise_ls, clip_grad


class MLP(nn.Module):
    """A simple implementation of Deep Neural Network model"""
    def __init__(self, num_feature, output_size):
        super(MLP, self).__init__()
        self.hidden = 300
        self.model = nn.Sequential(
            nn.Linear(num_feature, self.hidden),
            #nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            #nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(self.hidden, output_size))
    def forward(self, x):
        return self.model(x)

import numpy as np
import copy
import random
import csv
import pandas
import torch
def load_cnn_virus(data_path):
    '''
    args:
        data_path:the path of csv dataset.
    function:
        1.load the csv to array
        2.shuffle the array
        3.split the array to feature and label
        4.min_max normalize the feature
    return:
        a list like: [(features,labels)]
    '''

    dataframe = pandas.read_csv(data_path)

    array = dataframe.values
    random.shuffle(array) # random the dataset

    features = array[:,0:300]

    labels = array[:,300] 
    #labels = (labels <14)
    from sklearn import preprocessing

    min_max_scaler = preprocessing.MinMaxScaler()
    features = min_max_scaler.fit_transform(features)
    features = torch.FloatTensor(features)


    print("data have load!!!!,features shape is ",features.shape) 


    non_iid = []

    non_iid.append((features,labels))
    return non_iid


class FLClient(nn.Module):
    """ Client of Federated Learning framework.
        1. Receive global model from server
        2. Perform local training (compute gradients)
        3. Return local model (gradients) to server
    """
    def __init__(self, model, output_size, data_path, lr, E, batch_size,  epsilon,delta, device=None):
    #def __init__(self, model, output_size, data, lr, E, batch_size, clip, sigma, device=None):
        """
        :param model: ML model's training process should be implemented
        :param data: (tuple) dataset, all data in client side is used as training data
        :param lr: learning rate
        :param E: epoch of local update
        """
        super(FLClient, self).__init__()
        self.device = device
        self.BATCH_SIZE = batch_size
        self.data = load_cnn_virus(data_path)
        self.data = self.data[0]
        torch_dataset = TensorDataset(torch.tensor(self.data[0]),
                                      torch.tensor(self.data[1]))
        self.data_size = len(torch_dataset)
        self.data_loader = DataLoader(
            dataset=torch_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=True
        )
        self.noise = None
        self.lr = lr
        self.E = E
        #self.sigma = sigma
        self.epsilon = epsilon
        self.delta = delta

        #print("data[0].shape[1]",data[0].shape[1])
        self.model = model(self.data[0].shape[1], output_size).to(self.device)

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)



    def update(self):
        """local model update"""
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        #optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
        optimizer = self.optimizer
        for e in range(self.E):
            for batch_x, batch_y in self.data_loader:
                #print("batch_x",batch_x)
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_y = self.model(batch_x)
                #print("pred_y",pred_y)
                loss = criterion(pred_y, batch_y.long()) / len(self.data_loader)
                loss.backward()
            # bound l2 sensitivity (gradient clipping)

                optimizer.step()
                optimizer.zero_grad()
        '''
        # Add Gaussian noise
        # 1. compute l2-sensitivity by Client Based DP-FedAVG Alg.
        # 2. add noise
        sensitivity = 2 * self.lr * self.clip / self.data_size + (self.E - 1) * 2 * self.lr * self.clip
        new_param = copy.deepcopy(self.model.state_dict())
        for name in new_param:
            new_param[name] = torch.zeros(new_param[name].shape).to(self.device)
            new_param[name] += 1.0 * self.model.state_dict()[name]
            #new_param[name] += gaussian_noise_weight(self.model.state_dict()[name].shape, sensitivity,self.epsilon, self.delta, device=self.device)
        self.model.load_state_dict(copy.deepcopy(new_param))
        '''


async def fit(websocket, path):
    print("here is fit function")
    '''
    args:
        path: tag the uri,we can use it in the future
        websocket:used in receiving and sending the model
    function:
        1.receive the broadcast model which is binary 
        2.write the binary to pt file
        3.load the pt file and update the client model
        4.transfer the updated model to pt file 
        5.transfer the pt file to binary
        6.send the binary model 
    '''
    # receive the broadcast model which is binary 
    f = open("./broadcast_recv.pt", "wb")
    print("waitting the broadcast model ......")
    broadcast_recv = await websocket.recv()
    f.write(broadcast_recv)
    f.close
    print("received the broadcast model and start updating!!!")

    #name = int.from_bytes(name, byteorder='little', signed=True)
    broadcast_recv = torch.jit.load("./broadcast_recv.pt")
    broadcast_recv_par = broadcast_recv.model.state_dict()
    
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in broadcast_recv_par.items():
        name ="model."+k # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
        new_state_dict[name] = v #新字典的key值对应的value为一一对应的值。 

    flclient.model.load_state_dict(copy.deepcopy(new_state_dict))
    '''
    params = list(flclient.model.named_parameters())
    
    for i in params:
        print(i)
    '''
    #model_par = list(broadcast_recv.named_parameters())
    #flclient.model = broadcast_recv
    flclient.update()
    '''
    params = list(flclient.model.named_parameters())
    
    for i in params:
        print(i)
    '''
    # send the updated model 
    updated_model_send = torch.jit.trace(flclient.model, torch.zeros([1, 1, 300], dtype=torch.float),check_trace=False)
    torch.jit.save(updated_model_send,"updated_model_send.pt")
    f = open("updated_model_send.pt", "rb") 
    updated_model_send = f.read()
    await websocket.send(updated_model_send)
    f.close
    import os
    os.remove("updated_model_send.pt") 

flclient = FLClient(model = MLP,
                    output_size = 15,
                    data_path = "0_3_300.csv",
                    lr = 0.0001,
                    E = 1,
                    batch_size = 256,
                    epsilon = 50,
                    delta = 0.00001,
                    device = "cpu")
print("hahaha")
start_server = websockets.serve(fit, "0.0.0.0", 28092)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
