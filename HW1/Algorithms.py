import numpy as np
from IPython.display import clear_output
import time
from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
import heapdict

class Node:
    def __init__(self, state):
        self.pre_action = None
        self.state = state
        self.father = None
        
    def solution(self):
        action_list_from_init = []
        curr_node = self
        while curr_node.father is not None:
            action_list_from_init.insert(0, curr_node.pre_action)
            curr_node = curr_node.father
        return action_list_from_init
            
    def __eq__(self,other):
        return other.state == self.state

class Agent:
    def __init__(self):
        self.env = None

    def animation(self, epochs: int ,state: int, action: List[int], total_cost: int) -> None:
        clear_output(wait=True)
        print(self.env.render())
        print(f"Timestep: {epochs}")
        print(f"State: {state}")
        print(f"Action: {action}")
        print(f"Total Cost: {total_cost}")
        time.sleep(1)
    
    def search(self, env: DragonBallEnv):
        raise NotImplementedError
    
    def expand(self, node :Node):
        succs = self.env.succ(node.state) #there might be a problem with remembering what balls i already picked up
        expanded = []
        for action, successor in succs.items():
            new_state_index = successor[0][0]
            is_d1 = self.env.d1[0] == successor[0][0]
            is_d2 = self.env.d2[0] == successor[0][0]
            new_state = (new_state_index, is_d1 or node.state[1], is_d2 or node.state[2])
            new_successor = (new_state, successor[1], successor[2])
            if not successor[1] == np.inf:
                expanded.append((new_successor, action))
        return expanded
            
class BFSAgent(Agent):
    def __init__(self):
        Agent.__init__(self)
        
    def search(self, env: DragonBallEnv):
        self.env = env
        self.env.reset()
        epochs = 0
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
                cost = self.step_by_solution(actions_to_solutions, epochs)
                return (actions_to_solutions, cost, epochs)
        return ([-1,-1,-1], -1, epochs)

    def step_by_solution(self, action_list_from_init, epochs):
        state = self.env.get_initial_state()
        cost = 0
        terminated = False
        for action in action_list_from_init:
            # print(state)
            state, new_cost, terminated = self.env.step(action)
            cost = cost + new_cost
            self.animation(epochs,state,action,cost)
        return cost
    

class WeightedAStarAgent():
    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        raise NotImplementedError



class AStarEpsilonAgent():
    def __init__(self) -> None:
        raise NotImplementedError
        
    def ssearch(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        raise NotImplementedError
