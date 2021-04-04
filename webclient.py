# WS client example

import asyncio
import websockets
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from DPMechanisms import gaussian_noise_weight,gaussian_noise, gaussian_noise_ls, clip_grad

import numpy as np
import copy
import random

use_cuda = torch.cuda.is_available()   
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
device ="cpu"
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



class FLServer(nn.Module):
    """ Server of Federated Learning
        1. Receive model (or gradients) from clients
        2. Aggregate local models (or gradients)
        3. Compute global model, broadcast global model to clients
    """
    def __init__(self, fl_par):
        super(FLServer, self).__init__()

        self.device = fl_par['device']
        self.C = fl_par['C']  # (float) C in [0, 1]
        self.clip = fl_par['clip']
        self.data_path = fl_par['data_path']
        self.data_target = (load_cnn_virus(self.data_path))
        self.data = []
        self.target = []
        for sample in self.data_target:
            self.data += [torch.tensor(sample[0]).to(self.device)]  # test set
            self.target += [torch.tensor(sample[1]).to(self.device)]  # target label
        self.input_size = int(self.data[0].shape[1])
        print("self.input_size",self.input_size)
        self.trace_model = None
        self.global_model = fl_par['model'](self.input_size, fl_par['output_size']).to(self.device)
        #self.global_model = torch.jit.trace(self.global_model, torch.zeros([1, 1, 300], dtype=torch.float),check_trace=False)
        #self.global_model.save("1.pt")
        #self.global_model =  torch.jit.load("1.pt")
    def aggregated(self, models):
        """FedAvg"""
        models_par = [client_model.state_dict() for client_model in models]
        new_par = copy.deepcopy(models_par[0])

        for name in new_par:
            #print(name)
            new_par[name] = torch.zeros(new_par[name].shape).to(self.device)
        for idx, par in enumerate(models_par):
            print("len(models_par)0",len(models_par))
            w = 1/len(models_par)
            for name in new_par:
                #print(name)
                # new_par[name] += par[name] * (self.weight[idxs_users[idx]] / np.sum(self.weight[idxs_users]))
                new_par[name] += par[name] * (w / self.C)
                #print(new_par[name])
        '''
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        
        for k, v in new_par.items():
            name ="model."+k # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
            new_state_dict[name] = v #新字典的key值对应的value为一一对应的值。 
        '''
        self.global_model.load_state_dict(new_par)
        return self.global_model.state_dict().copy()


    def test_acc_femnist(self):
        self.global_model.eval()
        correct = 0
        tot_sample = 0
        for i in range(len(self.data)):
            t_pred_y = self.global_model(self.data[i])
            _, predicted = torch.max(t_pred_y, 1)
            predicted= predicted
            print("predicted",predicted)

            print("self.target[i]",self.target[i])
            correct += (predicted == self.target[i]).sum().item()
            tot_sample += self.target[i].size(0)
        acc = correct / tot_sample
        return acc
    async def train_config(self,uri):
        '''
            args:
                uri:IP+port,for example:"ws://localhost:8780"
                trace_model:the binary flow of torch.jit.trace's pt file

            return:
                torch's model
        '''
        async with websockets.connect(uri,timeout=None,max_size=None,ping_timeout=None) as websocket:
            await websocket.send(self.trace_model)

            model_updated = await websocket.recv()
            with open("./model_update_recv.pt", "wb") as f2:
                f2.write(model_updated)

            model = torch.jit.load("./model_update_recv.pt")

            params = list(model.named_parameters())
            '''
            for i in params:
                print(i)
            '''
            return model
            

import numpy as np
import copy
import random
import random
import csv
import pandas
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


    print("features",features.shape) 


    non_iid = []

    non_iid.append((features,labels))
    return non_iid






import websocket
import websockets
"""
async def train_config(ws,trace_model):
    '''
        args:
            ws:websocket object

        return:
            torch's model
    '''
    await ws.send(traced_model)

    model_updated = await ws.recv()
    with open("./model_update_recv.pt", "wb") as f2:
        f2.write(model_updated)

    model = torch.jit.load("./model_update_recv.pt")

    return model
"""

fl_param = {
    'output_size': 15,
    'model': MLP,
    'E': 1,
    'C': 1,
    'epsilon':0.5,
    'delta':0.00001,
    'clip': 4,
    'data_path':'test_300_0.2per.csv',
    'batch_size': 128,
    'device': device
}
fl_entity = FLServer(fl_param).to(device)


async def hello(aggregated_round):
    '''
        args:
            aggregated_round: the global aggregated round
        function:
            for aggregated_round:
                1.connect the server, serlizing the global model to binary
                2.asynchronous connecting, sending the binary model to clients and fitting
                3.waitting to receive the clients model
                4.aggregate
                5.test the acc

    '''

    for i in range(0,aggregated_round):
        uris=[
        "ws://106.12.19.48:28092",
        #"ws://localhost:8760"
        ]
        #ws_list=[]
        '''
        for i in uris:
            ws = websocket.WebSocket()

            ws = websockets.connect(i)
            ws_list.append(ws)
        '''   
        #ws_list = [websockets.connect(uri) as  for uri in uris]
        #print("connected!!!")
        #broadcast the model
        broadcast_send = fl_entity.global_model
        '''
        params = list(broadcast_send.named_parameters())
    
        for i in params:
            print(i)
        '''
        broadcast_send = torch.jit.trace(broadcast_send, torch.zeros([1, 1, 300], dtype=torch.float),check_trace=False)
        torch.jit.save(broadcast_send,"broadcast_send.pt")#model to file
        with open("broadcast_send.pt", "rb") as f: #file to binary
            fl_entity.trace_model = f.read() #binary
            #broadcasting......
            #print(len(uris[0]))
            #fl_entity.train_config("ws://106.12.19.48:28092")
            print("waitting the updated model....")
            results = await asyncio.gather(*[fl_entity.train_config(uri)for uri in uris])
            print("received the updated model and starting aggregating")

            #aggregating ......
            fl_entity.aggregated(results)

            #testing.....
            acc = fl_entity.test_acc_femnist()
            print("round:{0}, acc:{1}".format(i,acc))
        import os
        os.remove("broadcast_send.pt") 

asyncio.get_event_loop().run_until_complete(hello(aggregated_round=200))
