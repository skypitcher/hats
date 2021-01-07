from typing import List

from prelude import *

from tree import TreeNode, PriorityQueue, AVLTree


class AStar:
    def __init__(self, heuristic=None):
        self.__queue = PriorityQueue(capacity=np.inf)
        self.nodes_expanded = 0
        self.nodes_generated = 0
        self.__heuristic = heuristic

    def max_mem_usage(self):
        return self.__queue.max_mem_use

    def search(self, y: np.ndarray, R: np.ndarray, omega):
        self.nodes_expanded = 0
        self.nodes_generated = 0

        self.__queue.clear()
        root = TreeNode(-1, omega, np.zeros_like(y), None)
        self.__queue.push(root)

        while True:
            if self.__queue.is_empty():
                raise RuntimeError("Search failed!")

            best = self.__queue.pop()
            if best.is_goal():
                return best.get_data()

            self.nodes_expanded += 1

            for succ in best:
                self.__initilize(y, R, succ)
                self.__queue.push(succ)
                self.nodes_generated += 1

    def __initilize(self, y, R, node):
        if node.get_f() == 0:
            g = tree_g(y, R, node.get_data())
            if node.is_goal():
                h = 0
            else:
                h = self._compute_heuristic(y, R, node.get_data())
            max_f = max(node.get_f(), g + h)
            node.set_f(max_f)

    def _compute_heuristic(self, y, R, psv):
        if self.__heuristic is None:
            return 0
        else:
            return self.__heuristic(y, R, psv)


class SMAStar:
    def __init__(self, capacity, heuristic=None):
        self.__tree = AVLTree(capacity=capacity)
        self.nodes_expanded = 0
        self.nodes_generated = 0
        self.__heuristic = heuristic

    def max_mem_usage(self):
        return self.__tree.max_memory_usage()

    def __call__(self, *args, **kwargs):
        return self.search(*args, omega=[-1, 1])

    def search(self, y: np.ndarray, R: np.ndarray, omega):
        self.nodes_expanded = 0
        self.nodes_generated = 0

        self.__tree.clear()
        root = TreeNode(-1, omega, np.zeros_like(y), None)
        self.__tree.push(root)

        while True:
            if self.__tree.is_empty():
                raise RuntimeError("Search failed!")

            best = self.__tree.min()
            if best.is_goal():
                return best.get_data()

            completed_before = best.is_completed()
            succ = best.next_successor()
            if not completed_before and best.is_completed():
                self.nodes_expanded += 1

            if succ is not None:
                self.__initilize(y, R, succ)

            if best.is_completed():
                self.__adjust(best)

            if self.__tree.is_full():
                self.__take_space()

            if succ is not None:
                self.nodes_generated += 1
                self.__tree.push(succ)

            if best.all_successors_in_memory():
                self.__tree.remove(best)

    def __initilize(self, y, R, node):
        if node.get_f() == 0:
            g = tree_g(y, R, node.get_data())
            if node.is_goal():
                h = 0
            else:
                h = self._compute_heuristic(y, R, node.get_data())
            max_f = max(node.get_f(), g + h)
            node.set_f(max_f)

    def __adjust(self, node: TreeNode):
        recover_list: List[TreeNode] = []

        # use while loop to prevent potential stack overflow in recursion
        while True:
            adjuested = False
            if node is not None and node.is_completed():
                backup_f = node.get_backup_f()
                if backup_f < np.inf and backup_f != node.get_f():
                    if node.is_in_memory:
                        self.__tree.remove(node)
                        recover_list.append(node)
                    node.set_f(backup_f)
                    adjuested = True

            if not adjuested:
                break

            node = node.get_parent()

        for item in recover_list:
            self.__tree.push(item)

    def __take_space(self):
        while self.__tree.is_full():
            worst = self.__tree.pop_max_leaf()
            parent = worst.get_parent()
            parent.remmember(worst)
            if not parent.is_in_memory:
                assert parent not in self.__tree
                self.__tree.push(parent)

    def _compute_heuristic(self, y, R, psv):
        if self.__heuristic is None:
            return 0
        else:
            return self.__heuristic(y, R, psv)
