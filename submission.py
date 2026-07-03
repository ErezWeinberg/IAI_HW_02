from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import time
import math


# -----------------------------------------------------------------
# Heuristic Function
# -----------------------------------------------------------------

def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    rival = env.get_robot(1 - robot_id)

    # Feature 1: Credit difference
    credit_diff = robot.credit - rival.credit

    # Feature 2: Battery level
    battery = robot.battery

    # Feature 3: Distance to target (package or destination)
    dist_to_target = 0
    if robot.package is not None:
        # If holding a package, the target is the package's destination
        dist_to_target = manhattan_distance(robot.position, robot.package.destination)
    else:
        # If empty-handed, find the closest package currently on the board
        packages_on_board = [p for p in env.packages if p.on_board]
        if packages_on_board:
            distances = [manhattan_distance(robot.position, p.position) for p in packages_on_board]
            dist_to_target = min(distances)

    # Calculate exact heuristic value according to the dry report formula: w1=2, w2=2, w3=1
    h = (2 * credit_diff) + (2 * battery) - (1 * dist_to_target)

    return h


# -----------------------------------------------------------------
# Global Decision Functions (As required by the assignment format)
# -----------------------------------------------------------------

def minimax_decision(env: WarehouseEnv, robot_id: int, depth: int, heuristic_fn=None, time_limit=None):
    if heuristic_fn is None:
        heuristic_fn = smart_heuristic

    def rb_minimax(current_env: WarehouseEnv, current_depth: int, current_robot: int):
        if time_limit is not None and time.time() > time_limit:
            raise TimeoutError()

        if current_depth == 0 or current_env.done():
            return heuristic_fn(current_env, robot_id), None

        operators = current_env.get_legal_operators(current_robot)
        if not operators:
            return heuristic_fn(current_env, robot_id), None

        if current_robot == robot_id:
            curr_max = -math.inf
            best_op = None
            for op in operators:
                child = current_env.clone()
                child.apply_operator(current_robot, op)
                v, _ = rb_minimax(child, current_depth - 1, 1 - current_robot)
                if v > curr_max:
                    curr_max = v
                    best_op = op
            return curr_max, best_op
        else:
            curr_min = math.inf
            best_op = None
            for op in operators:
                child = current_env.clone()
                child.apply_operator(current_robot, op)
                v, _ = rb_minimax(child, current_depth - 1, 1 - current_robot)
                if v < curr_min:
                    curr_min = v
                    best_op = op
            return curr_min, best_op

    try:
        _, best_move = rb_minimax(env, depth, robot_id)
        return best_move
    except TimeoutError:
        return None


def alphabeta_decision(env: WarehouseEnv, robot_id: int, depth: int, heuristic_fn=None, time_limit=None):
    if heuristic_fn is None:
        heuristic_fn = smart_heuristic

    def rb_alpha_beta(current_env: WarehouseEnv, current_depth: int, current_robot: int, alpha: float, beta: float):
        if time_limit is not None and time.time() > time_limit:
            raise TimeoutError()

        if current_depth == 0 or current_env.done():
            return heuristic_fn(current_env, robot_id), None

        operators = current_env.get_legal_operators(current_robot)
        if not operators:
            return heuristic_fn(current_env, robot_id), None

        if current_robot == robot_id:
            curr_max = -math.inf
            best_op = None
            for op in operators:
                child = current_env.clone()
                child.apply_operator(current_robot, op)
                v, _ = rb_alpha_beta(child, current_depth - 1, 1 - current_robot, alpha, beta)
                if v > curr_max:
                    curr_max = v
                    best_op = op
                alpha = max(curr_max, alpha)
                if curr_max >= beta:
                    return curr_max, best_op
            return curr_max, best_op
        else:
            curr_min = math.inf
            best_op = None
            for op in operators:
                child = current_env.clone()
                child.apply_operator(current_robot, op)
                v, _ = rb_alpha_beta(child, current_depth - 1, 1 - current_robot, alpha, beta)
                if v < curr_min:
                    curr_min = v
                    best_op = op
                beta = min(currMin, beta)
                if curr_min <= alpha:
                    return curr_min, best_op
            return curr_min, best_op

    try:
        _, best_move = rb_alpha_beta(env, depth, robot_id, -math.inf, math.inf)
        return best_move
    except TimeoutError:
        return None


def expectimax_decision(env: WarehouseEnv, robot_id: int, depth: int, heuristic_fn=None, time_limit=None):
    if heuristic_fn is None:
        heuristic_fn = smart_heuristic

    def rb_expectimax(current_env: WarehouseEnv, current_depth: int, current_robot: int):
        if time_limit is not None and time.time() > time_limit:
            raise TimeoutError()

        if current_depth == 0 or current_env.done():
            return heuristic_fn(current_env, robot_id), None

        operators = current_env.get_legal_operators(current_robot)
        if not operators:
            return heuristic_fn(current_env, robot_id), None

        if current_robot == robot_id:
            curr_max = -math.inf
            best_op = None
            for op in operators:
                child = current_env.clone()
                child.apply_operator(current_robot, op)
                v, _ = rb_expectimax(child, current_depth - 1, 1 - current_robot)
                if v > curr_max:
                    curr_max = v
                    best_op = op
            return curr_max, best_op
        else:
            total_weight = 0
            weights = []
            for op in operators:
                w = 4 if op in ['move north', 'charge'] else 1
                weights.append(w)
                total_weight += w

            expected_value = 0
            for op, w in zip(operators, weights):
                child = current_env.clone()
                child.apply_operator(current_robot, op)
                v, _ = rb_expectimax(child, current_depth - 1, 1 - current_robot)
                probability = w / total_weight
                expected_value += (probability * v)

            return expected_value, operators[0]

    try:
        _, best_move = rb_expectimax(env, depth, robot_id)
        return best_move
    except TimeoutError:
        return None


# -----------------------------------------------------------------
# Agent Classes
# -----------------------------------------------------------------

class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        limit = time.time() + time_limit - 0.015
        moves = env.get_legal_operators(agent_id)
        if not moves:
            return 'park'
        moves_return = random.choice(moves)
        depth = 1
        while time.time() < limit and depth <= 4:
            if depth > 2 * env.get_robot(agent_id).battery:
                return moves_return
            op = minimax_decision(env, agent_id, depth, time_limit=limit)
            if op is not None:
                moves_return = op
                depth += 1
            else:
                break
        return moves_return


class AgentAlphaBeta(Agent):
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        limit = time.time() + time_limit - 0.015
        moves = env.get_legal_operators(agent_id)
        if not moves:
            return 'park'
        moves_return = random.choice(moves)
        depth = 1
        while time.time() < limit and depth <= 4:
            if depth > 2 * env.get_robot(agent_id).battery:
                return moves_return
            op = alphabeta_decision(env, agent_id, depth, time_limit=limit)
            if op is not None:
                moves_return = op
                depth += 1
            else:
                break
        return moves_return


class AgentExpectimax(Agent):
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        limit = time.time() + time_limit - 0.015
        moves = env.get_legal_operators(agent_id)
        if not moves:
            return 'park'
        moves_return = random.choice(moves)
        depth = 1
        while time.time() < limit and depth <= 4:
            if depth > 2 * env.get_robot(agent_id).battery:
                return moves_return
            op = expectimax_decision(env, agent_id, depth, time_limit=limit)
            if op is not None:
                moves_return = op
                depth += 1
            else:
                break
        return moves_return


class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        self.trajectory = ["move south", "move west", "move north", "move east", "move north", "move north", "pick_up",
                           "move east", "move east",
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