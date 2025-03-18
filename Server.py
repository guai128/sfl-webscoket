import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Thread

import numpy
import torch
import websockets
from torch import nn
from ConsumeQueue import ConsumeQueue
from ResNet import all_layers
from utils import prRed


# non-normal server model should contain 3 to 4 layers
class SeverNonNormal(nn.Module):
    max_layers = 4
    min_layers = 2

    def __init__(self):
        super(SeverNonNormal, self).__init__()
        # the number of layers must be between 2 and 4
        self.layers = [layer() for layer in all_layers[SeverNonNormal.min_layers: SeverNonNormal.max_layers]]
        self.layer = nn.Sequential(*self.layers)

    def forward(self, x, start_layer):
        for layer in self.layers[start_layer:]:
            x = layer(x)

        return x


# normal server model should contain 3 layers：layer5, layer6, layer7
class ServerNormal(nn.Module):
    num_classes = 7

    def __init__(self):
        super(ServerNormal, self).__init__()
        self.layer4 = all_layers[4]()
        self.layer5 = all_layers[5]()
        self.layer6 = all_layers[6](ServerNormal.num_classes)

    def forward(self, x):
        x4 = self.layer4(x)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)
        return x6


class Server:
    def __init__(self, lr, device):
        self.server_normal = ServerNormal().to(device)
        self.server_non_normal = SeverNonNormal().to(device)
        self.lr = lr
        self.device = device
        self.consume_queue = ConsumeQueue()
        self.clients = set()
        self.connections = {}

    @staticmethod
    def websocketToClientID(websocket):
        host, port = websocket.remote_address[0], websocket.remote_address[1]
        return f"{host}:{port}"

    def register_client(self, websocket):
        client_id = self.websocketToClientID(websocket)
        self.clients.add(client_id)
        self.connections[client_id] = websocket
        self.consume_queue.register(f"Client{client_id}ForServer")
        self.consume_queue.register(f"ServerForClient{client_id}")

    def unregister_client(self, websocket):
        client_id = self.websocketToClientID(websocket)
        self.clients.remove(client_id)
        self.connections.pop(client_id)

    def server_run(self):
        optimizer_server_non_normal = torch.optim.Adam(self.server_non_normal.parameters(), lr=self.lr)
        optimizer_server_normal = torch.optim.Adam(self.server_normal.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        def trainOneRound():
            if len(self.clients) == 0:
                return
            all_clients, un_arrive_clients = list(self.clients.copy()), list(self.clients.copy())
            round_start_time = time.time()
            arrive_time = [0. for _ in range(len(un_arrive_clients))]
            last_arrive_time, delay_avg, calculate_time = 0., 0., 0.
            while len(un_arrive_clients) > 0:
                # if the server has been waiting for too long, it will stop waiting
                arrive_client_cnt = len(self.clients) - len(un_arrive_clients)
                if arrive_client_cnt > 0:
                    delay_avg = sum(arrive_time) / arrive_client_cnt
                    tolerance = delay_avg / arrive_client_cnt
                    current_time = time.time() - calculate_time - round_start_time
                    if current_time - last_arrive_time > tolerance:
                        pass  # or break

                visited = set()
                for client in un_arrive_clients:
                    if client not in self.clients:
                        visited.add(client)
                        continue

                    client_data = self.consume_queue.consumeNonBlock(f"Client{client}ForServer")
                    if client_data is None:
                        continue

                    last_arrive_time = time.time() - calculate_time - round_start_time
                    arrive_time[all_clients.index(client)] = last_arrive_time

                    # start to calculate the time
                    calculate_start_time = time.time()
                    # reconstruct all data from bytes
                    layer_count = numpy.frombuffer(client_data[:4], dtype=numpy.int32)[0]
                    dim_count = numpy.frombuffer(client_data[4:8], dtype=numpy.int32)[0]
                    fx_shape = list(numpy.frombuffer(client_data[8: 8 + 4 * dim_count], dtype=numpy.int32))
                    data_cnt = 1
                    for i in range(dim_count):
                        data_cnt *= fx_shape[i]

                    client_fx = (torch.frombuffer(client_data[8 + 4 * dim_count: 8 + 4 * dim_count + 4 * data_cnt],
                                                  dtype=torch.float32)
                                 .reshape(fx_shape).to(self.device).requires_grad_(True))

                    client_y = (torch.frombuffer(client_data[8 + 4 * dim_count + 4 * data_cnt:], dtype=torch.int64)
                                .to(self.device))

                    # start to train the server model
                    self.server_non_normal.train()
                    self.server_normal.train()

                    optimizer_server_non_normal.zero_grad()
                    optimizer_server_normal.zero_grad()

                    server_fx = self.server_non_normal(client_fx, layer_count - 2)
                    server_fx = self.server_normal(server_fx)

                    _, predicted = torch.max(server_fx.data, 1)
                    loss = criterion(server_fx, client_y)

                    loss.backward()

                    optimizer_server_non_normal.step()
                    optimizer_server_normal.step()

                    # send the gradients to the client
                    dfx_client = client_fx.grad.cpu().detach().numpy().tobytes()
                    self.consume_queue.produce(f"ServerForClient{client}", dfx_client)

                    optimizer_server_non_normal.step()
                    optimizer_server_normal.step()
                    visited.add(client)

                    # complete the calculate time
                    # calculate_time += time.time() - calculate_start_time

                un_arrive_clients = list(set(un_arrive_clients) - visited)
            if len(self.clients) > 0:
                print(f"round time: {time.time() - round_start_time}")

            # # 记录用户惩罚和奖励
            # for client in un_arrive_clients:
            #     self.consume_queue.produce(f"ServerForClient{client}Order", 2)
            prRed(f"clients arrive time: {arrive_time}")
            if len(un_arrive_clients) > 0:
                prRed(f"latency: {time.time() - round_start_time} there are {len(un_arrive_clients)} clients not finished")

            # for idx in self.clients:
            #     if idx not in un_arrive_clients:
            #         if arrive_time[idx] < delay_avg * 0.5:
            #             self.consume_queue.produce(f"ServerForClient{idx}Order", 1)
            #         else:
            #             self.consume_queue.produce(f"ServerForClient{idx}Order", 0)

        while not self.isConvergence():
            trainOneRound()
            torch.save(self.server_normal.state_dict(), f"model/server_normal.pth")

    async def send_message_to_client(self):
        while True:
            for client_id in self.clients:
                response_data = self.consume_queue.consumeNonBlock(f"ServerForClient{client_id}")
                if response_data is not None and client_id in self.connections:
                    await self.connections[client_id].send(response_data)

    def isConvergence(self):
        return False

    async def onMessage(self, websocket, path):
        try:
            self.register_client(websocket)
            print(f"Client {self.websocketToClientID(websocket)} connected")
            while True:
                data = await websocket.recv()
                self.consume_queue.produce(f"Client{self.websocketToClientID(websocket)}ForServer", data)
                response_data = await asyncio.to_thread(
                    self.consume_queue.consume, f"ServerForClient{self.websocketToClientID(websocket)}"
                )
                await websocket.send(response_data)

        except websockets.WebSocketException:
            self.unregister_client(websocket)
            print(f"Client {self.websocketToClientID(websocket)} disconnected")

    def mainloop(self, host='192.168.31.161', port=9000):
        calculate_thread = Thread(target=self.server_run)
        calculate_thread.daemon = True
        calculate_thread.start()

        server = websockets.serve(self.onMessage, host, port, max_size=2 ** 60)
        asyncio.get_event_loop().run_until_complete(server)
        asyncio.get_event_loop().run_forever()


if __name__ == '__main__':
    server = Server(lr=0.01, device='cuda')
    server.mainloop()
