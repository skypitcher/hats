import heapq

from prelude import *
from typing import List

from sortedcontainers import SortedList


def get_path(data):
    lv = tree_level(data)
    if lv == 0:
        path = "root"
    else:
        symbols = data[-lv:, :].flat
        path = "".join("{:d}".format(int(x)) for x in symbols)
    return path


class TreeNode:
    def __init__(self, idx, omega: List[int], data: np.ndarray, parent=None):
        self.__idx = idx
        self.__omega = omega
        self.__data = data
        self.__path = get_path(data)
        self.__lv = tree_level(data)
        self.__parent = parent
        self.__f = 0
        self.__next_succ_id = 0
        self.__generated_successors: List[TreeNode] = []
        self.__forgotten_successors: List[TreeNode] = []
        self.is_in_memory = False

    def index(self):
        return self.__idx

    def get_path(self):
        return self.__path

    def __repr__(self):
        return "{}(f={:.4f})".format(self.get_path(), self.get_f())

    def get_data(self):
        return self.__data

    def get_parent(self):
        return self.__parent

    def get_lv(self):
        return self.__lv

    def is_leaf(self):
        return len(self.__generated_successors) == 0

    def is_goal(self):
        return is_goal(self.__data)

    def get_f(self):
        return self.__f

    def set_f(self, f):
        self.__f = f

    def __cmp_key(self):
        return self.__f, -self.__lv, self.__idx, -id(self)

    def __eq__(self, other):
        return self.__cmp_key() == other.__cmp_key()

    def __lt__(self, other):
        return self.__cmp_key() < other.__cmp_key()

    def __iter__(self):
        return self

    def __next__(self):
        succ = self.next_successor()
        if succ is None:
            raise StopIteration()
        else:
            return succ

    def next_successor(self):
        if not self.is_completed():
            data = np.copy(self.__data)
            data[-(self.get_lv() + 1)] = self.__omega[self.__next_succ_id]
            succ = TreeNode(self.__next_succ_id, self.__omega, data, self)
            self.__generated_successors.append(succ)
            self.__next_succ_id += 1
            return succ
        elif len(self.__forgotten_successors) > 0:
            succ = min(self.__forgotten_successors)
            self.__generated_successors.append(succ)
            self.__forgotten_successors.remove(succ)
            return succ
        else:
            return None

    def is_completed(self):
        return self.__next_succ_id >= len(self.__omega)

    def get_backup_f(self):
        min_f = np.inf
        if len(self.__generated_successors) > 0:
            min_f = min(min_f, min(self.__generated_successors).get_f())
        if len(self.__forgotten_successors) > 0:
            min_f = min(min_f, min(self.__forgotten_successors).get_f())
        return min_f

    def all_successors_in_memory(self):
        if len(self.__generated_successors) != len(self.__omega):
            return False
        else:
            for succ in self.__generated_successors:
                if not succ.is_in_memory:
                    return False
            return True

    def remmember(self, succ):
        self.__forgotten_successors.append(succ)
        self.__generated_successors.remove(succ)


class PriorityQueue:
    def __init__(self, capacity):
        self.__items: List[TreeNode] = []
        self.max_mem_use = 0
        self.capacity = capacity

    def size(self):
        return len(self.__items)

    def __iter__(self):
        return self.__items.__iter__()

    def __len__(self):
        return self.size()

    def is_empty(self):
        return self.size() == 0

    def is_full(self):
        return self.size() == self.capacity

    def top(self):
        if self.is_empty():
            return None
        else:
            return self.__items[0]

    def push(self, item):
        if self.size() == self.capacity:
            raise MemoryError("Not enough space")

        assert not item.is_in_memory
        heapq.heappush(self.__items, item)
        item.is_in_memory = True
        self.max_mem_use = max(self.max_mem_use, self.size())

    def pop(self):
        if self.is_empty():
            return None
        else:
            item = heapq.heappop(self.__items)
            assert item.is_in_memory
            item.is_in_memory = False
            return item

    def __contains__(self, item):
        return item in self.__items

    def clear(self):
        self.__items.clear()


class AVLTree:
    def __init__(self, capacity):
        self.__container = SortedList()
        self.__capacity = capacity
        self.__max_mem_use = 0

    def max_memory_usage(self):
        return self.__max_mem_use

    def size(self):
        return len(self.__container)

    def is_full(self):
        return self.size() == self.__capacity

    def is_empty(self):
        return self.size() == 0

    def capacity(self):
        return self.__capacity

    def __len__(self):
        return self.size()

    def __contains__(self, item):
        return item in self.__container

    def push(self, item):
        if self.is_full():
            raise MemoryError("No enough space")
        assert not item.is_in_memory
        self.__container.add(item)
        item.is_in_memory = True
        self.__max_mem_use = max(self.__max_mem_use, self.size())

    def min(self):
        return self.__container[0]

    def remove(self, item):
        self.__container.remove(item)
        assert item.is_in_memory
        item.is_in_memory = False

    def pop_min(self):
        item = self.__container.pop(0)
        assert item.is_in_memory
        item.is_in_memory = False
        return item

    def pop_max(self):
        item = self.__container.pop(-1)
        item.is_in_memory = False
        return item

    def pop_max_leaf(self):
        results = []
        while True:
            item = self.pop_max()
            if item.is_leaf():
                node = item
                break
            else:
                results.append(item)
        for item in results:
            self.push(item)
        return node

    def clear(self):
        self.__container.clear()
