import copy
import torch
from flcore.clients.clientdyn import clientDyn
from flcore.servers.serverbase import Server
from threading import Thread
import time


class FedDyn(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(args, clientDyn)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

        self.alpha = args.alpha

        self.server_state = copy.deepcopy(args.model)
        for param in self.server_state.parameters():
            param.data = torch.zeros_like(param.data)

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
            self.update_server_state()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print("-" * 50, self.Budget[-1])
            print("-" * 50, self.Budget[-1])
            self.done = self.check_done(
                acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt
            )
            self.done = self.check_done(
                acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt
            )

        print("\nBest global accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nBest local accuracy.")
        print(max(local_acc))
        print("\nAveraged time per iteration.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))
        print("\nEvaluating Post-Fine-Tuning and OOD Performance...")
        self.evaluate()
        self.evaluate(ood_eval=True)

        # add post fine-tuning
        for client in self.selected_clients:
            print("Client LR: ", client.learning_rate)
            client.batch_size = 128
        for ft_idx in range(15):
            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            # threads = [Thread(target=client.train) for client in self.selected_clients]
            # [t.join() for t in threads]

            print("\nFine-Tuning Iteration: ", ft_idx)
            self.evaluate()
            self.evaluate(ood_eval=True)

        self.save_results()
        self.save_global_model()

    def add_parameters(self, client_model):
        for server_param, client_param in zip(
            self.global_model.parameters(), client_model.parameters()
        ):
            server_param.data += client_param.data.clone() / self.join_clients

    def aggregate_parameters(self):
        assert len(self.uploaded_models) > 0

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data = torch.zeros_like(param.data)

        for client_model in self.uploaded_models:
            self.add_parameters(client_model)

        for server_param, state_param in zip(
            self.global_model.parameters(), self.server_state.parameters()
        ):
            server_param.data -= (1 / self.alpha) * state_param

    def update_server_state(self):
        assert len(self.uploaded_models) > 0

        model_delta = copy.deepcopy(self.uploaded_models[0])
        for param in model_delta.parameters():
            param.data = torch.zeros_like(param.data)

        for client_model in self.uploaded_models:
            for server_param, client_param, delta_param in zip(
                self.global_model.parameters(),
                client_model.parameters(),
                model_delta.parameters(),
            ):
                delta_param.data += (client_param - server_param) / self.num_clients

        for state_param, delta_param in zip(
            self.server_state.parameters(), model_delta.parameters()
        ):
            state_param.data -= self.alpha * delta_param
