import copy
import math
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client


class clientAPPLE(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.loss = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.drlr = args.dr_learning_rate
        self.num_clients = args.num_clients
        self.lamda = 1
        self.mu = args.mu
        self.L = int(args.L * args.global_rounds)
        self.learning_rate = self.learning_rate * self.num_clients

        self.model_cs = []

        self.ps = [1/args.num_clients for _ in range(self.num_clients)]
        self.p0 = None
        self.model_c = copy.deepcopy(self.model)

    def train(self, R):
        trainloader = self.load_train_data()
        
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                self.aggregate_parameters()

                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()

                for param_c, param in zip(self.model_cs[self.id].parameters(), self.model.parameters()):
                    param_c.data = param_c - self.learning_rate * param.grad * self.ps[self.id]

                for cid in range(self.num_clients):
                    cnt = 0
                    p_grad = 0
                    for param_c, param in zip(self.model_cs[cid].parameters(), self.model.parameters()):
                        p_grad += torch.mean(param.grad * param_c).item()
                        cnt += 1
                    p_grad = p_grad / cnt
                    p_grad = p_grad + self.lamda * self.mu * (self.ps[cid] - self.p0[cid])
                    self.ps[cid] = self.ps[cid] - self.drlr * p_grad

        if R < self.L:
            self.lamda = (math.cos(R * math.pi / self.L) + 1) / 2
        else:
            self.lamda = 0

        self.model_c = copy.deepcopy(self.model)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        
    def set_models(self, model_cs):
        self.model_cs = model_cs

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def aggregate_parameters(self):
        assert (len(self.model_cs) > 0)

        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
            
        for w, client_model in zip(self.ps, self.model_cs):
            self.add_parameters(w, client_model)