import heapq

class MyHeap(object):
    def __init__(self, capacity, item_length=3):
        self.heap = []
        self.capacity = capacity
        self.item_length = item_length

    def push(self, item):
        assert len(item) == self.item_length
        heapq.heappush(self.heap, item)
        if len(self.heap) > self.capacity:
            heapq.heappop(self.heap)

    def pop(self):
        heapq.heappop(self.heap)

    def __getitem__(self, item):
        return self.heap[item]

    def __len__(self):
        return len(self.heap)

    def as_list(self):
        return [item[self.item_length-1] for item in self.heap]

if __name__=='__main__':
    u = MyHeap(2)
    u.push((1, 'a'))
    u.push((3, 'b'))
    u.push((2, 'c'))
    print(len(u))

