from collections import deque

class MemoryBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)