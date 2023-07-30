import copy
import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data

from torch.utils.data import ConcatDataset

from hessian_eigenthings import compute_hessian_eigenthings
from utils.tent import tent

import torchvision.transforms.functional as TF
from PIL import Image
import cv2

import gpytorch
import math


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        self.model = copy.deepcopy(args.model)
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_steps = args.local_steps

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs["train_slow"]
        self.send_slow = kwargs["send_slow"]
        self.train_time_cost = {"num_rounds": 0, "total_cost": 0.0}
        self.send_time_cost = {"num_rounds": 0, "total_cost": 0.0}

        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma
        self.sample_rate = self.batch_size / self.train_samples

        self.best_model_dict = None

        self.hold_out_id = args.hold_out_id

        self.save_img = args.save_img

        self.pruning = args.pruning
        self.sparsity_ratio = args.sparsity_ratio

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None, ood_eval=False, dataset_ids=[]):
        if batch_size == None:
            batch_size = self.batch_size
        if ood_eval:
            test_data_list = []
            for dataset_id in dataset_ids:
                if dataset_id == self.hold_out_id:
                    train_data = read_client_data(
                        self.dataset, dataset_id, is_train=True
                    )
                    test_data_list.append(train_data)
                test_data = read_client_data(self.dataset, dataset_id, is_train=False)
                test_data_list.append(test_data)
            concat_test_data = ConcatDataset(test_data_list)
            return DataLoader(
                concat_test_data, batch_size, drop_last=False, shuffle=False
            )
        else:
            test_data = read_client_data(self.dataset, self.id, is_train=False)
            return DataLoader(test_data, batch_size, drop_last=False, shuffle=False)

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self, ood_eval=False, dataset_ids=[]):
        if ood_eval:
            testloaderfull = self.load_test_data(
                ood_eval=ood_eval, dataset_ids=dataset_ids
            )
            if self.hold_out_id != 1e8:
                oof_testloaderfull = self.load_test_data(
                    ood_eval=ood_eval, dataset_ids=[self.hold_out_id]
                )
        else:
            testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        print_count = 1

        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average="micro")

        if ood_eval and self.hold_out_id != 1e8:
            oof_test_acc = 0
            oof_test_num = 0
            y_prob = []
            y_true = []

            print_count = 1

            with torch.no_grad():
                for x, y in oof_testloaderfull:
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = self.model(x)

                    oof_test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                    oof_test_num += y.shape[0]

                    y_prob.append(output.detach().cpu().numpy())
                    nc = self.num_classes
                    if self.num_classes == 2:
                        nc += 1
                    lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                    if self.num_classes == 2:
                        lb = lb[:, :2]
                    y_true.append(lb)

            # self.model.cpu()
            # self.save_model(self.model, 'model')

            y_prob = np.concatenate(y_prob, axis=0)
            y_true = np.concatenate(y_true, axis=0)

            oof_auc = metrics.roc_auc_score(y_true, y_prob, average="micro")

            return test_acc, test_num, auc, oof_test_acc, oof_test_num, oof_auc
        else:
            return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        loss = 0

        # Create two directories for two labels
        max_zero_num = 32
        max_one_num = 32
        cur_zero_num = 0
        cur_one_num = 0

        for x, y in trainloader:
            if self.save_img:
                for i, (img, label) in enumerate(zip(x, y)):
                    mean = (0.5, 0.5, 0.5)
                    std = (0.5, 0.5, 0.5)
                    t_mean = torch.FloatTensor(mean).view(3, 1, 1).expand(img.shape)
                    t_std = torch.FloatTensor(std).view(3, 1, 1).expand(img.shape)
                    img = img * t_std + t_mean
                    img = (
                        img.mul_(255)
                        .add_(0.5)
                        .clamp_(0, 255)
                        .permute(1, 2, 0)
                        .type(torch.uint8)
                        .numpy()
                    )
                    img = TF.to_pil_image(img).convert("RGB")
                    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                    if label == 0.0 and cur_zero_num < max_zero_num:
                        cv2.imwrite(
                            f"../tmp/save_img/label0_"
                            + self.dataset
                            + "_client"
                            + str(self.id)
                            + "_index"
                            + str(i)
                            + ".png",
                            img,
                        )
                        cur_zero_num += 1
                    elif label == 1.0 and cur_one_num < max_one_num:
                        cv2.imwrite(
                            f"../tmp/save_img/label1_"
                            + self.dataset
                            + "_client"
                            + str(self.id)
                            + "_index"
                            + str(i)
                            + ".png",
                            img,
                        )
                        cur_one_num += 1

            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            output = self.model(x)
            train_num += y.shape[0]
            loss += self.loss(output, y).item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return loss, train_num

    def compute_hessian_eigen(self):
        trainloader = self.load_train_data()
        compute_model = copy.deepcopy(self.model).cuda()
        # loss = torch.nn.functional.cross_entropy
        loss = torch.nn.CrossEntropyLoss()
        num_eigenthings = 5  # compute num of eigenvalues/eigenvectors
        eigenvals, eigenvecs = compute_hessian_eigenthings(
            compute_model,
            trainloader,
            loss,
            num_eigenthings,
            full_dataset=True,
        )
        return eigenvals, eigenvecs

    def quick_adaptation_eval(self, dataset_ids):
        copy_model = copy.deepcopy(self.model)
        model = tent.configure_model(copy_model)
        params, param_names = tent.collect_params(model)
        optimizer = torch.optim.Adam(params, lr=1e-3)
        tented_model = tent.Tent(model, optimizer)
        testloaderfull = self.load_test_data(ood_eval=True, dataset_ids=dataset_ids)

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
                output = tented_model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        test_auc = metrics.roc_auc_score(y_true, y_prob, average="micro")
        test_acc /= test_num
        return test_acc, test_auc

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y

    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(
            item,
            os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"),
        )

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(
            os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt")
        )

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))

    # only support SNIP currently
    def gen_mask(self, algo="SNIP", sparsity_ratio=0.9, layerwise=False):
        self.mask_state_dict = self.model.state_dict()

        # compute gradient
        trainloader = self.load_train_data()
        for x, y in trainloader:
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            output = self.model(x)
            loss = self.loss(output, y).item() * y.shape[0]
            loss.backward()

        # calculate score
        self.scores = {}
        for name, param in self.model.named_parameters():
            if algo == "SNIP":
                self.scores[name] = torch.clone(param.grad.data).detach().abs_()
            else:
                raise ValueError("Not implemented.")

        # normalize score
        all_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        norm = torch.sum(all_scores)
        for name, _ in self.model.named_parameters():
            self.scores[name].div_(norm)
        
        if layerwise:
            for name, param in self.model.named_parameters():
                score = self.scores[name]
                k = int(sparsity_ratio * score.numel())
                if not k < 1:
                    threshold, _ = torch.kthvalue(torch.flatten(score), k)
                    zero = torch.tensor([0.]).to(self.device)
                    one = torch.tensor([1.]).to(self.device)
                    self.mask_state_dict[name].copy_(torch.where(score <= threshold, zero, one))
        else:
            global_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
            k = int(sparsity_ratio * global_scores.numel())
            if not k < 1:
                threshold, _ = torch.kthvalue(global_scores, k)
                for name, param in self.model.named_parameters():
                    score = self.scores[name] 
                    zero = torch.tensor([0.]).to(self.device)
                    one = torch.tensor([1.]).to(self.device)
                    self.mask_state_dict[name].copy_(torch.where(score <= threshold, zero, one))


    def apply_mask(self):
        for p, m in zip(self.model.parameters(), self.mask_state_dict):
            p.mul_(m)
