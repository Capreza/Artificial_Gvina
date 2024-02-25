import numpy as np
from IPython.display import clear_output
import time
from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
from heapdict import heapdict
import random

def get_manheten_distance(state_1, state_2, N):
    row_1 = state_1 // N
    row_2 = state_2 // N
    row_distance = abs(row_2 - row_1)
    col_1 = state_1 % N
    col_2 = state_2 % N
    col_distance = abs(col_2 - col_1)
    return row_distance + col_distance

def get_houristic(state, env):
    current_index = state[0]
    goal_states = env.get_goal_states().copy()
    if not state[1]:
        goal_states += [env.d1]
    if not state[2]:
        goal_states += [env.d2]
    min_goal = np.inf 
    for goal in goal_states:
        distance_goal = get_manheten_distance(state[0], goal[0], env.nrow)
        if distance_goal < min_goal:
            min_goal = distance_goal
    return min_goal 
        
class Node:
    def __init__(self, state, h=None, g=None):
        self.pre_action = None
        self.state = state
        self.father = None
        self.h = h
        self.g = g

        
    def solution(self):
        action_list_from_init = []
        curr_node = self
        while curr_node.father is not None:
            action_list_from_init.insert(0, curr_node.pre_action)
            curr_node = curr_node.father
        return action_list_from_init
            
    def __eq__(self,other):
        return other.state == self.state

    def __hash__(self):
        return hash(self.state)
    
    def f(self, weight=0.5):
        return self.h*weight + self.g *(1-weight)
     
        
class Agent:
    def __init__(self):
        self.env = None

    def animation(self, epochs: int ,state: int, action: List[int], total_cost: int, created_nodes: int) -> None:
        # clear_output(wait=True)
        # print(self.env.render())
        # print(f"Timestep: {epochs}")
        # print(f"Created Nodes: {created_nodes}")
        # print(f"State: {state}")
        # print(f"Action: {action}")
        # print(f"Total Cost: {total_cost}")
        # time.sleep(1)
        pass
    
    def search(self, env: DragonBallEnv):
        raise NotImplementedError
    
    def step_by_solution(self, action_list_from_init, epochs, created_nodes=0):
        state = self.env.get_initial_state()
        cost = 0
        terminated = False
        for action in action_list_from_init:
            state, new_cost, terminated = self.env.step(action)
            cost = cost + new_cost
            self.animation(epochs,state,action,cost, created_nodes)
        return cost
    
    def expand(self, node :Node):
        succs = self.env.succ(node.state) #there might be a problem with remembering what balls i already picked up
        expanded = []
        if not node.state[0] in [state[0] for state in self.env.get_goal_states()]:
            for action, successor in succs.items():
                if successor[0] is not None :
                    new_state_index = successor[0][0]
                    is_d1 = self.env.d1[0] == successor[0][0]
                    is_d2 = self.env.d2[0] == successor[0][0]
                    new_state = (new_state_index, is_d1 or node.state[1], is_d2 or node.state[2])
                    new_successor = (new_state, successor[1], successor[2])
                    expanded.append((new_successor, action))
        return expanded
    

class BFSAgent(Agent):
    def __init__(self):
        Agent.__init__(self)
        
    def search(self, env: DragonBallEnv):
        self.env = env
        self.env.reset()
        epochs = 0
        created_nodes = 0
        actions_to_solutions = []
        state = self.env.get_initial_state()
        open_list = []
        close_list = []
        node = Node(state)
        open_list.append(node)
        while open_list:
            node = open_list.pop(0)
            epochs += 1
            close_list.append(node)
            for succ, action in self.expand(node):
                child = Node(succ[0])
                created_nodes += 1 
                if child not in close_list and child not in open_list:
                    # TODO: track path
                    # see if problem is solved, if so return solution
                    # add to open list 
                    child.father = node
                    child.pre_action = action
                    if self.env.is_final_state(child.state):
                        actions_to_solutions = child.solution()
                        break
                    else:
                        open_list.append(child)
            if actions_to_solutions:
                cost = self.step_by_solution(actions_to_solutions, epochs, created_nodes)
                return (actions_to_solutions, cost, epochs)
        return ([-1,-1,-1], -1, epochs)    

class WeightedAStarAgent(Agent):
    def __init__(self) -> None:
        Agent.__init__(self)

    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        epochs = 0
        created_nodes = 1
        actions_to_solutions = []
        state = self.env.get_initial_state()
        open_heap = heapdict()
        close_dct = {}
        node = Node(state,h=get_houristic(state, env), g=0)
        open_heap[node] = (node.f(h_weight), node.state[0])
        while open_heap:
            node = open_heap.popitem()[0]
            close_dct[node] = (node.f(h_weight), node.state[0])
            if self.env.is_final_state(node.state):
                actions_to_solutions = node.solution()
                break
            epochs += 1
            for succ, action in self.expand(node):
                child = Node(succ[0], h=get_houristic(succ[0], env), g=node.g + succ[1])
                created_nodes += 1 
                child.father = node
                child.pre_action = action
                if child not in close_dct.keys() and child not in open_heap.keys():
                    open_heap[child] = (child.f(h_weight), child.state[0])
                elif child in open_heap.keys():
                    old_f = open_heap[child][0]
                    new_f = child.f(h_weight)
                    if old_f > new_f:
                        del open_heap[child]
                        open_heap[child] = (new_f, child.state[0])
                elif child in close_dct.keys():
                    assert(child not in open_heap.keys(), "Child is in both open and close! Ad Matay?!#!@#!@")
                    old_f = close_dct[child][0]
                    new_f = child.f(h_weight)
                    if old_f > new_f:
                        close_dct.pop(child)
                        open_heap[child] = (new_f, child.state[0])

        if actions_to_solutions:
            cost = self.step_by_solution(actions_to_solutions, epochs, created_nodes)
            return (actions_to_solutions, cost, epochs)
        return ([-1,-1,-1], -1, epochs)
    
    
class AStarEpsilonAgent(Agent):
    def __init__(self) -> None:
        Agent.__init__(self)

    def search(self, env: DragonBallEnv, epsilon) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        epochs = 0
        created_nodes = 1
        actions_to_solutions = []
        state = self.env.get_initial_state()
        open_heap = heapdict()
        close_dct = {}
        node = Node(state,h=get_houristic(state, env), g=0)
        open_heap[node] = node.f()
        while open_heap:
            # node = open_heap.popitem()[0]
            focal = [node for node, f_value in open_heap.items() if f_value <= open_heap.peekitem()[1] * (1 + epsilon)]
            min_g = focal[0].g
            node = focal[0]
            for n in focal:
            # node = random.choice(focal)
                if n.g < min_g:
                    min_g = n.g
                    node = n
     
            del open_heap[node]
            close_dct[node] = node.f()
            if self.env.is_final_state(node.state):
                actions_to_solutions = node.solution()
                break
            epochs += 1
            for succ, action in self.expand(node):
                child = Node(succ[0], h=get_houristic(succ[0], env), g=node.g + succ[1])
                created_nodes += 1 
                child.father = node
                child.pre_action = action
                if child not in close_dct.keys() and child not in open_heap.keys():
                    open_heap[child] = child.f()
                elif child in open_heap.keys():
                    old_f = open_heap[child]
                    new_f = child.f()
                    if old_f > new_f:
                        del open_heap[child]
                        open_heap[child] = new_f
                elif child in close_dct.keys():
                    assert(child not in open_heap.keys(), "Child is in both open and close! Ad Matay?!#!@#!@")
                    old_f = close_dct[child]
                    new_f = child.f()
                    if old_f > new_f:
                        close_dct.pop(child)
                        open_heap[child] = new_f

        if actions_to_solutions:
            cost = self.step_by_solution(actions_to_solutions, epochs, created_nodes)
            return (actions_to_solutions, cost, epochs)
        return ([-1,-1,-1], -1, epochs)
    