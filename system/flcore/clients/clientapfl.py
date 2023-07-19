import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client

from sklearn.preprocessing import label_binarize
from sklearn import metrics

class clientAPFL(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.loss = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.alpha = args.alpha
        self.model_per = copy.deepcopy(self.model)
        # self.optimizer_per = torch.optim.SGD(self.model_per.parameters(), lr=self.learning_rate)
        self.optimizer_per = torch.optim.Adam(self.model_per.parameters(), lr=self.learning_rate)

    def train(self):
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
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()

                self.optimizer_per.zero_grad()
                output_per = self.model_per(x)
                loss_per = self.loss(output_per, y)
                loss_per.backward()
                self.optimizer_per.step()

                self.alpha_update()

        for lp, p in zip(self.model_per.parameters(), self.model.parameters()):
            lp.data = (1 - self.alpha) * p + self.alpha * lp

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    # https://github.com/MLOPTPSU/FedTorch/blob/b58da7408d783fd426872b63fbe0c0352c7fa8e4/fedtorch/comms/utils/flow_utils.py#L240
    def alpha_update(self):
        grad_alpha = 0
        for l_params, p_params in zip(self.model.parameters(), self.model_per.parameters()):
            dif = p_params.data - l_params.data
            grad = self.alpha * p_params.grad.data + (1-self.alpha) * l_params.grad.data
            grad_alpha += dif.view(-1).T.dot(grad.view(-1))
        
        grad_alpha += 0.02 * self.alpha
        self.alpha = self.alpha - self.learning_rate * grad_alpha
        self.alpha = np.clip(self.alpha.item(), 0.0, 1.0)

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        self.model_per.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model_per(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                y_true.append(label_binarize(y.detach().cpu().numpy(), classes=np.arange(self.num_classes)))

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc