import heapq

class MyHeap():
    def __init__(self, capacity):
        self.heap = []
        self.capacity = capacity

    def push(self, item):
        heapq.heappush(self.heap, item)
        if len(self.heap) > self.capacity:
            heapq.heappop(self.heap)

    def pop(self):
        heapq.heappop(self.heap)

    def __getitem__(self, item):
        return self.heap[item]

    def __len__(self):
        return len(self.heap)

if __name__=='__main__':
    u = MyHeap(2)
    u.push((1, 'a'))
    u.push((3, 'b'))
    u.push((2, 'c'))
    print(u[0])

