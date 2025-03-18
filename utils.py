import copy
import torch
from ConsumeQueue import ConsumeQueue
from collections import OrderedDict


def prRed(skk): print("\033[91m {}\033[00m".format(skk))


def prGreen(skk): print("\033[92m {}\033[00m".format(skk))


def fedAvg_run(consume_queue: ConsumeQueue, num_users: int):
    while True:
        # collect all the weights from the clients
        all_weights = []
        for i in range(num_users):
            all_weights.append(consume_queue.consume(f"Client{i}ForFedAvg"))

        # calculate the average weights
        new_weights, weights_cnt = OrderedDict(), OrderedDict()
        for weights in all_weights:
            for key in weights.keys():
                if key not in new_weights.keys():
                    new_weights[key] = copy.deepcopy(weights[key])
                    weights_cnt[key] = 1
                else:
                    new_weights[key] += weights[key]
                    weights_cnt[key] += 1

        for layer in new_weights.keys():
            new_weights[layer] = torch.div(new_weights[layer], weights_cnt[layer])

        for idx in range(num_users):
            consume_queue.produce(f"FedAvgForClient{idx}", new_weights)

        prGreen('FedAvg done')
