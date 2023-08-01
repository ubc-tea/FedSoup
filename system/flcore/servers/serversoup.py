import time
from flcore.clients.clientavg import clientAVG
from flcore.clients.clientsoup import clientSoup
from flcore.servers.serverbase import Server
from threading import Thread

import torch.multiprocessing as mp

import random
import numpy as np


class FedSoup(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(args, clientSoup)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()

            self.selected_clients = self.select_clients()
            self.send_models()

            # for client in self.selected_clients:
            #     idx = 0
            #     for name, param in client.model.named_parameters():
            #         idx += 1
            #         if idx == rand_num:
            #             print(client.id)
            #             print(name, param)

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate(ood_eval=True)

            for client in self.selected_clients:
                if (
                    len(client.learning_rate_decay) > 0
                    and i >= client.learning_rate_decay[0]
                ):
                    client.learning_rate *= 0.5
                    print("Current Learning Rate: ", client.learning_rate)
                    client.learning_rate_decay.pop(0)

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train) for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print("-" * 25, "time cost", "-" * 25, self.Budget[-1])

        print("\nBest global accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        print("\nEvaluating Post-Fine-Tuning and OOD Performance...")
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

        print("\nFine-Tuning with the Last Trained Model......")
        # add post fine-tuning
        for client in self.selected_clients:
            print("Client LR: ", client.learning_rate)
            client.batch_size = 128
            client.fixed_soup_flat = True
        for ft_idx in range(15):
            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train) for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            print("\nFine-Tuning Iteration: ", ft_idx)
            self.evaluate(ood_eval=True)

        # print("\nFine-Tuning with the Best Trained Model......")
        # # add post fine-tuning
        # for client in self.selected_clients:
        #     print("Client LR: ", client.learning_rate)
        #     client.batch_size = 128
        #     client.train_round = 0
        #     client.model.load_state_dict(client.best_model_dict)
        #     client.fixed_soup_flat = True
        # for ft_idx in range(15):
        #     threads = [Thread(target=client.train) for client in self.selected_clients]
        #     [t.start() for t in threads]
        #     [t.join() for t in threads]
        #     print("\nFine-Tuning Iteration: ", ft_idx)
        #     self.evaluate(ood_eval=True)

        self.save_results()
        self.save_global_model()
