import random
import time
from flcore.clients.clientapple import clientAPPLE
from flcore.servers.serverbase import Server
from threading import Thread


class APPLE(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(args, clientAPPLE)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

        self.uploaded_models = [c.model_c for c in self.clients]

        train_samples = 0
        for client in self.clients:
            train_samples += client.train_samples
        p0 = [client.train_samples / train_samples for client in self.clients]

        for c in self.clients:
            c.p0 = p0

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.clients:
                client.train(i)

            # threads = [Thread(target=client.train)
            #            for client in self.clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()

            self.Budget.append(time.time() - s_t)
            print("-" * 50, self.Budget[-1])

        print("\nBest global accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        print("\nEvaluating Post-Fine-Tuning and OOD Performance...")
        self.evaluate(ood_eval=True)

        self.save_results()
        self.save_global_model()

    def send_models(self):
        assert len(self.clients) > 0

        for client in self.clients:
            start_time = time.time()

            client.set_models(self.uploaded_models)

            client.send_time_cost["num_rounds"] += 1
            client.send_time_cost["total_cost"] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert len(self.selected_clients) > 0

        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.join_clients)
        )

        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            client_time_cost = (
                client.train_time_cost["total_cost"]
                / client.train_time_cost["num_rounds"]
                + client.send_time_cost["total_cost"]
                / client.send_time_cost["num_rounds"]
            )
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model_c)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
