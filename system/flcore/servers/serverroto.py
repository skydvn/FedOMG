import time
import copy
from flcore.clients.clientroto import clientRoto
from flcore.servers.serverbase import Server
from threading import Thread
import numpy as np
import yaml


class FedRoto(Server):
    def __init__(self, args, times):
        super().__init__(args, times)  # note this

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientRoto)  # set client RotoGrad
        self.update_grad = []  # store model with parameters update

        """
        Define args: 
            - roto_layer_start: the starting point index of the roto 
            - roto_layer_num: number of roto layers
            -> roto_layer_end = start + num        

        Get key of the Roto of the global model
            - roto_layer_num = args.roto_layer_num
            - If you define roto_layer_num last layers = Roto -> get list: 
                Key_dict = {key1, key2, key3}
                    - key1 = "Fully Connected Layers 1"
        """

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []



    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            # only thing update here is the weight for parameter
            """
            Calculate Gradient Trajectory
                - Loops over all clients:
                    - grad_t[u] = w^{t+1}_u - w^{t}_g
            """
            """
            The work is to get the gradient update of the local model from the global model
            So local model is stored in the list self.uploaded_models = [] which elements is client.model
            the connection between the global and the local: 
            """
            self.update_grad = copy.deepcopy(self.uploaded_models)
            # This for copy the list to store all the gradient update value

            for model in self.update_grad:
                for param in model.parameters():
                    param.data.zero_()
            # Set all in place value to zero to store new gradient value

            for grad_model, local_model in zip(self.update_grad, self.uploaded_models):
                for grad_param, local_param in zip(grad_model.parmameters(), local_model.parameters()):
                    grad_param.data = local_param.data - self.global_model.parameters().data

            # self.upload_grad store all client update parameters




            """
                - Loops key in Key_dict:
                    G~_k = self.updated_model[key] * G_k
                        * (Layer-wise operation or Norm over all parameters)
            """

            """
                U_k = G_k / ||G~_k||
                alpha_k = ||G_k||/||G~_k||
                    * (alpha_k -> scalar or vector or matrix)
            """

            """
                Do from 9 -> 17
            """

            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)

            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientRoto)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    """
    Calculate Gradient Trajectory
    This for outside function later 
    """

    def get_gradient(self):
        assert (len(self.upload_weights) > 0)
        self.update_grad = []
        pass
