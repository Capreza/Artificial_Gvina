import numpy as np

from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
import heapdict


class BFSAgent(Agent):

    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        epochs = 0

        state = self.env.get_initial_state()
        open_list = []
        close_list = []
        node = Node(state)
        open_list.append(node)
        while open_list:
            node = open_list.pop(0)
            epochs += 1
            close_list.append(node)
            for succ, action in self.expand(node.state):
                child = Node(succ[0])
                if child not in close_list and child not in open_list:
                    # TODO: track path
                    # see if problem is solved, if so return solution
                    # add to open list 
                    child.father = node
                    child.pre_action = action
                    if self.env.is_final_state(child.state):
                        actions_to_solutions = child.solution()
                    else:
                        open_list.append(child)
        # self.animation(epochs,state,action,total_cost)
        #return (actions, total_cost)


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
    
class Agent():
    def __init__(self):
        self.env = None

    def animation(self, epochs: int ,state: int, action: List[int], total_cost: int) -> None:
        clear_output(wait=True)
        print(self.env.render())
        print(f"Timestep: {epochs}")
        print(f"State: {state}")
        print(f"Action: {action}")
        print(f"Total Cost: {total_cost}")
    
    def search(self, DragonBallEnv: env) -> Tuple[List[int],int]:
        raise NotImplementedError
    
    def expand(self, Node: node):
        succs = env.successor(node.state) #there might be a problem with remembering what balls i already picked up
        expanded = []
        for action, successor in succs.items():
            successor[0][1] = node.state[1] or env.d1 == successor[0][0]
            successor[0][2] = node.state[2] or env.d2 == successor[0][0]
            if not successor[1] == np.inf:
                expanded.append((successor, action))
        return expanded
            
        

class Node():
    def __init__(self,state):
        self.pre_action = None
        self.state = state
        self.father = None
        
    def solution(self):
        action_list_from_init = []
        curr_node = self
        while curr_node.father is not None:
            action_list_from_init.insert(index=0, curr_node.pre_action)
            curr_node = curr_node.father
        return action_list_from_init
            
    def __eq__(self,other):
        return other.state == self.state
