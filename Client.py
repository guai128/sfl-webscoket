import asyncio
import random
from collections import OrderedDict

import torch
import tqdm
import websockets
from torch.utils.data import DataLoader
from torch import nn

from data.DataLoader import load_data
from ResNet import all_layers


# 关于客户端向服务端发送消息的逻辑
# 1. 客户端向服务端发送消息，消息包括：层数、 数据维度、 具体维度参数、 激活值、标签（数据按此顺序发送）
# 2. 待发送数据将被打包成一维数组并以字符串的形式发送（用空格分割）
#

class ClientModel(nn.Module):
    max_layers = 4
    min_layers = 2

    def __init__(self, num_layers):
        super(ClientModel, self).__init__()
        # the number of layers must be between 2 and 4
        assert ClientModel.max_layers >= num_layers >= ClientModel.min_layers
        self.layers = [layer() for layer in all_layers[0: num_layers]]
        self.layer = nn.Sequential(*self.layers)

    def layerCount(self):
        return len(self.layers)

    def forward(self, x):
        x = self.layer(x)
        return x


class Client(object):
    available_batch_size = [64, 128, 256]

    def __init__(self, lr, device, dataset_train):
        num_layers = 4  # random.randint(ClientModel.min_layers, ClientModel.max_layers)
        self.model = ClientModel(num_layers).to(device)
        self.device = device
        self.lr = lr
        self.batch_size = random.choice(Client.available_batch_size)
        self.ldr_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.last_client_model_parameters = None
        self.calculate_server_websocket = None  # websocket connection for server
        self.fed_server_websocket = None  # websocket connection for average model parameters

    def reload_model(self, num_layers):
        self.model = ClientModel(num_layers).to(self.device)
        if self.last_client_model_parameters is not None:
            self.load_state_dict(self.last_client_model_parameters)

    def load_state_dict(self, state_dict):
        new_weights = OrderedDict()
        model_parameters = self.model.state_dict()
        for key in model_parameters.keys():
            new_weights[key] = state_dict[key]

        self.model.load_state_dict(new_weights)

    def IncreasedComputationalPressure(self):
        idx = Client.available_batch_size.index(self.batch_size)
        if idx < len(Client.available_batch_size) - 1:
            self.batch_size = Client.available_batch_size[idx + 1]
            self.ldr_train = DataLoader(self.ldr_train.dataset, batch_size=self.batch_size, shuffle=True)
        elif self.model.layerCount() < ClientModel.max_layers:
            self.reload_model(self.model.layerCount() + 1)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def DecreasedComputationalPressure(self):
        idx = Client.available_batch_size.index(self.batch_size)
        if idx > 0:
            self.batch_size = Client.available_batch_size[idx - 1]
            self.ldr_train = DataLoader(self.ldr_train.dataset, batch_size=self.batch_size, shuffle=True)
        elif self.model.layerCount() > ClientModel.min_layers:
            self.reload_model(self.model.layerCount() - 1)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    async def train(self):
        self.model.train()
        for batch_idx, (images, labels) in enumerate(tqdm.tqdm(self.ldr_train)):
            images = images.to(self.device)
            self.optimizer.zero_grad()
            # ---------forward prop-------------
            fx = self.model(images)
            # Sending activations to server and receiving gradients from server
            fx_shape = list(fx.shape)
            send_data = self.model.layerCount().to_bytes(4, byteorder='little')
            send_data += len(fx_shape).to_bytes(4, byteorder='little')
            for val in fx_shape:
                send_data += val.to_bytes(4, byteorder='little')
            send_data += fx.cpu().detach().numpy().tobytes() + labels.numpy().tobytes()
            await self.calculate_server_websocket.send(send_data)
            recv_data = await self.calculate_server_websocket.recv()
            # --------backward prop -------------
            dfx = torch.frombuffer(recv_data, dtype=torch.float32).reshape(fx.shape).to(self.device)
            fx.backward(dfx)
            self.optimizer.step()

        # save the model
        torch.save(self.model.state_dict(), f"model/client_model.pth")

    # async def averageModelParameters(self):
    #     model_parameters = self.model.state_dict()
    #     consume_queue.produce(f"Client{self.id}ForFedAvg", model_parameters)
    #     self.last_client_model_parameters = consume_queue.consumeNonBlock(f"FedAvgForClient{self.id}")
    #     if self.last_client_model_parameters is not None:
    #         # update the model parameters
    #         self.load_state_dict(self.last_client_model_parameters)

    async def responseServerOrder(self):
        order_cnt, order = [0] * 3, -1
        while order is not None:
            order = int(await self.calculate_server_websocket.recv())
            if order is not None:
                order_cnt[order] += 1

        total_order = sum(order_cnt)
        # 奖惩机制
        if order_cnt[1] > total_order * 0.6:
            self.IncreasedComputationalPressure()
        elif order_cnt[2] > total_order * 0.6:
            self.DecreasedComputationalPressure()

    async def client_run(self, sever_path='ws://192.168.31.161:9000', fed_path='ws://localhost:8889'):
        async with websockets.connect(sever_path, max_size=2 ** 60) as self.calculate_server_websocket:
            # async with websockets.connect(fed_path) as self.fed_server_websocket:
            while True:
                await self.train()
                # check if there are any orders from the server
                # await self.responseServerOrder()
                # await self.averageModelParameters()


if __name__ == '__main__':
    lr = 0.001
    device = "cuda"
    dataset_train = load_data("./data/split-data/client0.csv")

    client = Client(lr, device, dataset_train)
    asyncio.get_event_loop().run_until_complete(client.client_run())
