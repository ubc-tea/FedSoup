import torch
import time
import copy
import random
import numpy as np
from flcore.clients.clientfomo import clientFomo
from flcore.servers.serverbase import Server
from threading import Thread

import numpy as np


class FedFomo(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(args, clientFomo)

        self.P = torch.diag(torch.ones(self.num_clients, device=self.device))
        self.uploaded_models = [self.global_model]
        self.uploaded_ids = []
        self.M = min(args.M, self.join_clients)

        self.hold_out_id = args.hold_out_id

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

    def train(self):
        for i in range(self.global_rounds + 1):
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            # self.aggregate_parameters()

        print("\nBest global accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))

        print("\nEvaluating Post-Fine-Tuning and OOD Performance...")
        self.evaluate()
        self.evaluate(ood_eval=True)

        # ================= new function =================
        # compute sharpness
        if self.monitor_hessian:
            print("Computing sharpness")
            eigenvals_list = []
            for client in self.select_clients():
                eigenvals, _ = client.compute_hessian_eigen()
                eigenvals_list.append(eigenvals[0])
            print("\nHessian eigenval list: ", eigenvals_list)
            print("\nHessian eigenval mean: ", np.mean(eigenvals_list))

        if self.test_time_adaptation:
            self.tta_eval()

        # save each client model item
        for client in self.select_clients():
            client.save_item(
                item=client.model,
                item_name=self.goal,
                item_path="models/" + self.dataset + "/",
            )

        # ================= new function =================

        # add post fine-tuning
        for client in self.selected_clients:
            print("Client LR: ", client.learning_rate)
            client.batch_size = 128
        for ft_idx in range(15):
            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train) for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            print("\nFine-Tuning Iteration: ", ft_idx)
            self.evaluate()
            self.evaluate(ood_eval=True)

        self.save_results()
        self.save_global_model()

    def send_models(self):
        assert len(self.selected_clients) > 0
        for client in self.selected_clients:
            start_time = time.time()

            if client.send_slow:
                time.sleep(0.1 * np.abs(np.random.rand()))

            if len(self.uploaded_ids) > 0:
                M_ = min(self.M, len(self.uploaded_models))  # if clients dropped
                if client.id > self.hold_out_id:
                    new_uploaded_ids = [
                        i - 1 if i > self.hold_out_id else i for i in self.uploaded_ids
                    ]
                    indices = torch.topk(
                        self.P[client.id - 1][new_uploaded_ids], M_
                    ).indices.tolist()
                else:
                    new_uploaded_ids = [
                        i - 1 if i > self.hold_out_id else i for i in self.uploaded_ids
                    ]
                    indices = torch.topk(
                        self.P[client.id][new_uploaded_ids], M_
                    ).indices.tolist()

                uploaded_ids = []
                uploaded_models = []
                for i in indices:
                    uploaded_ids.append(self.uploaded_ids[i])
                    uploaded_models.append(self.uploaded_models[i])

                client.receive_models(uploaded_ids, uploaded_models)

            client.send_time_cost["num_rounds"] += 1
            client.send_time_cost["total_cost"] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert len(self.selected_clients) > 0

        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.join_clients)
        )

        self.uploaded_ids = []
        self.uploaded_weights = []
        tot_samples = 0
        self.uploaded_models = []
        for client in active_clients:
            client_time_cost = (
                client.train_time_cost["total_cost"]
                / client.train_time_cost["num_rounds"]
                + client.send_time_cost["total_cost"]
                / client.send_time_cost["num_rounds"]
            )
            if client_time_cost <= self.time_threthold:
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                tot_samples += client.train_samples
                self.uploaded_models.append(copy.deepcopy(client.model))
                if client.id > self.hold_out_id:
                    self.P[client.id - 1] += client.weight_vector
                else:
                    self.P[client.id] += client.weight_vector
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
