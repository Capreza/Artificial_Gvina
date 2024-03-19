import math
import time
import timeit

from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random


# TODO: section a : 3
def get_dist_curr_goal(env: WarehouseEnv, robot_id: int):
    distances = []
    our_robot = env.get_robot(robot_id)
    if our_robot.package is not None:
        return manhattan_distance(our_robot.position,our_robot.package.destination)
    else:
        for package in env.packages:
            distances.append(manhattan_distance(our_robot.position,package.position))
        return min(distances)


def get_dist_charging_station(env: WarehouseEnv, robot_id: int) -> int:
    our_robot = env.get_robot(robot_id)
    return min(
        manhattan_distance(our_robot.position,env.charge_stations[0].position),
        manhattan_distance(our_robot.position,env.charge_stations[1].position)
    )


def smart_heuristic(env: WarehouseEnv, robot_id: int):
    our_robot = env.get_robot(robot_id)
    other_robot = env.get_robot(1-robot_id)
    dist_from_current_goal = get_dist_curr_goal(env,robot_id)
    heuristic_val = (our_robot.credit * 100 - other_robot.credit * 100 +
            (90 if our_robot.package is not None else 0) - dist_from_current_goal) + 0.1 * our_robot.position[1]
    return heuristic_val

class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start_time = time.time()

        curr_max = (0, None)
        max_depth = 1
        while True:  #TODO: change cond

            operators = env.get_legal_operators(agent_id)
            children = [env.clone() for _ in operators]
            curr_max = -math.inf, None
            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
                curr_val = self.minimax(child, agent_id, 1 - agent_id, time_limit, start_time, max_depth-1)
                if curr_val >= curr_max[0]:
                    curr_max = curr_val,op
            max_depth += 1

        return curr_max[1]

    def minimax(self,env,agent,turn_id,time_limit, start_time,depth_limit):
        if time.time() - start_time >= time_limit - 0.03 or depth_limit == 0:
            return self.heuristic(env,agent)
        if env.done():
            if env.get_robot(agent).credit > env.get_robot(1 - agent).credit:
                return math.inf
            elif env.get_robot(agent).credit < env.get_robot(1-agent).credit:
                return -math.inf
            else:
                return 0
        operators = env.get_legal_operators(turn_id)
        children = [env.clone() for _ in operators]
        if turn_id == agent:
            curr_max = -math.inf
            for child, op in zip(children,operators):
                child.apply_operator(turn_id,op)
                curr_val = self.minimax(child, agent, 1 - turn_id, time_limit, start_time)
                curr_max = max(curr_val,curr_max)
            return curr_max
        else:
            curr_min = math.inf
            for child, op in zip(children, operators):
                child.apply_operator(turn_id, op)
                curr_val = self.minimax(child, agent, 1 - turn_id,time_limit, start_time)
                curr_min = min(curr_val,curr_min)
            return curr_min


class AgentAlphaBeta(Agent):
    # TODO: section b : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start_time = time.time()
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        curr_max = -math.inf, None
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
            curr_val = self.minimax(child, agent_id, 1 - agent_id, time_limit, start_time, alpha=-math.inf,beta=math.inf,depth=10)
            if curr_val >= curr_max[0]:
                curr_max = curr_val,op
        return curr_max[1]

    def minimax(self,env,agent,turn_id,time_limit, start_time, alpha, beta,depth):
        if time.time() - start_time >= time_limit - 0.03 or depth==0:
            return self.heuristic(env,agent)
        if env.done():
            if env.get_robot(agent).credit > env.get_robot(1 - agent).credit:
                return math.inf
            elif env.get_robot(agent).credit < env.get_robot(1-agent).credit:
                return -math.inf
            else:
                return 0
        operators = env.get_legal_operators(turn_id)
        children = [env.clone() for _ in operators]
        if turn_id == agent:
            curr_max = -math.inf
            for child, op in zip(children,operators):
                child.apply_operator(turn_id,op)
                curr_val = self.minimax(child, agent, 1 - turn_id, time_limit, start_time, alpha, beta,depth-1)
                curr_max = max(curr_val,curr_max)
                alpha = max(curr_max,alpha)
                if curr_max >= beta:
                    return math.inf
            return curr_max
        else:
            curr_min = math.inf
            for child, op in zip(children, operators):
                child.apply_operator(turn_id, op)
                curr_val = self.minimax(child, agent, 1 - turn_id,time_limit, start_time, alpha, beta,depth-1)
                curr_min = min(curr_val,curr_min)
                beta = min(curr_min,beta)
                if curr_min <= alpha:
                    return -math.inf
            return curr_min


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)