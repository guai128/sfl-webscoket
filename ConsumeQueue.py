import queue
import asyncio

class ConsumeQueue:
    def __init__(self, delegation=queue.Queue):
        self.content = {}
        self.delegation = delegation

    def register(self, topic):
        self.content[topic] = self.delegation()

    def unregister(self, topic):
        del self.content[topic]

    def consume(self, topic):
        return self.content[topic].get()

    def consumeNonBlock(self, topic):
        if self.content[topic].empty():
            return None

        return self.content[topic].get()

    def produce(self, topic, data):
        self.content[topic].put(data)

    async def produceAsync(self, topic, data):
        await self.content[topic].put(data)

    async def consumeAsync(self, topic):
        return await self.content[topic].get()
