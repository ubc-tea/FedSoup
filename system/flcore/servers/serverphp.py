from flcore.clients.clientphp import clientPHP
from flcore.servers.serverbase import Server
from threading import Thread
import time
import copy


class FedPHP(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(args, clientPHP)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def train(self):
        for i in range(self.global_rounds + 1):
            self.selected_clients = self.select_clients()
            self.send_models(i)

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
            self.aggregate_parameters()

        print("\nBest global accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))

        print("\nEvaluating Post-Fine-Tuning and OOD Performance...")
        self.evaluate(ood_eval=True)

        self.save_results()
        self.save_global_model()

    def send_models(self, R):
        assert len(self.selected_clients) > 0

        for client in self.selected_clients:
            start_time = time.time()

            client.set_parameters(self.global_model, R)

            client.send_time_cost["num_rounds"] += 1
            client.send_time_cost["total_cost"] += 2 * (time.time() - start_time)
