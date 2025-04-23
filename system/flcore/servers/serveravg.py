import time
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
import numpy
import copy
# import yaml
import statistics


class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)
        self.args = args

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.device = args.device


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()


            for client in self.selected_clients:
                client.train()

            self.receive_models()
            self.receive_grads()
           
            self.aggregate_parameters()
            

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")

        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        # if self.num_new_clients > 0:
        #     self.eval_new_clients = True
        #     self.set_new_clients(clientAVG)
        #     print(f"\n-------------Fine tuning round-------------")
        #     print("\nEvaluate new clients")
        #     self.domain_evaluate()
