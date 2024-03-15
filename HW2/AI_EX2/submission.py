from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random


# TODO: section a : 3
def get_dist_curr_goal(env: WarehouseEnv, robot_id: int):
    distances = []
    our_robot = env.get_robot(robot_id)
    if our_robot.position == (0,4):
        print("foo")
    if our_robot.package is not None:
        return manhattan_distance(our_robot.position,our_robot.package.destination)
    else:
        for package in env.packages:
            distances.append(manhattan_distance(our_robot.position,package.position))
        print("get dist curr goal")
        print("_"*100)
        print("curr location = ", our_robot.position)
        print(min(distances))
        print("_"*100)
        return min(distances)


def get_dist_charging_station(env: WarehouseEnv, robot_id: int) -> int:
    our_robot = env.get_robot(robot_id)
    return min(
        manhattan_distance(our_robot.position,env.charge_stations[0].position),
        manhattan_distance(our_robot.position,env.charge_stations[1].position)
    )


def smart_heuristic(env: WarehouseEnv, robot_id: int):
    our_robot = env.get_robot(robot_id)
    dist_from_current_goal = get_dist_curr_goal(env,robot_id)
    return -dist_from_current_goal + our_robot.credit*10 + (9 if our_robot.package is not None else 0)

class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


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