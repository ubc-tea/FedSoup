from flcore.clients.clientmoon import clientMOON
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data
from threading import Thread
import time

import numpy as np


class MOON(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(args, clientMOON)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def train(self):
        local_acc = []
        self.done = False
        i = 0
        while not self.done:
            # for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            if i % self.eval_gap == 0:
                print("\nEvaluate local model")
                self.evaluate(acc=local_acc)

            if i > 0 and i % 1000 == 0:
                print("\nEvaluate ID, OOD and OOF Performance")
                self.evaluate()
                self.evaluate(ood_eval=True)
                break

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print("-" * 50, self.Budget[-1])

            self.done = self.check_done(
                acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt
            )
            i += 1

        print("\nBest global accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nBest local accuracy.")
        print(max(local_acc))
        print("\nAveraged time per iteration.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

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
