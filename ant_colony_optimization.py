import numpy as np
import random
from math import inf


class Ant:
    """
    Represents an Ant of the Ant Colony Optimization algorithm.
    """

    def __init__(self, lower_bound, upper_bound):
        """
        Creates an Ant of the Ant Colony Optimization algorithm.

        :param lower_bound: lower bound of the ant position.
        :type lower_bound: numpy array.
        :param upper_bound: upper bound of the ant position.
        :type upper_bound: numpy array.
        """
        # Todo
        self.dimension = np.size(lower_bound)
        self.position = np.zeros(self.dimension)

        self.position = [
            random.uniform(lower_bound[idx], upper_bound[idx])
            for idx in range(self.dimension)
        ]
        # for idx in range(self.dimension):
        #    self.position[idx] = random.uniform(lower_bound[idx], upper_bound[idx])

        self.path = [self.position]
        self.reward = -inf


class AntColonyOptimization:
    """
    Represents the Ant Colony Optimization algorithm.
    Hyperparameters:
        num_ants: number of ants.
        evaporation_rate: evaporation rate.
        pheromone_intensity: pheromone intensity.
        alpha: pheromone influence.
        beta: heuristic influence.
        num_best_solutions: number of solutions used to build Gaussian Mixture

    :param hyperparams: hyperparameters used by Ant Colony Optimization.
    :type hyperparams: Params.
    :param lower_bound: lower bound of ant position.
    :type lower_bound: numpy array.
    :param upper_bound: upper bound of ant position.
    :type upper_bound: numpy array.
    """

    def __init__(self, hyperparams, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.num_ants = hyperparams.num_ants
        self.num_best_solutions = hyperparams.num_best_solutions
        self.evaporation_rate = hyperparams.evaporation_rate
        self.pheromone_intensity = hyperparams.pheromone_intensity
        self.alpha = hyperparams.alpha
        self.beta = hyperparams.beta

        self.ants = [
            Ant(self.lower_bound, self.upper_bound) for _ in range(self.num_ants)
        ]

        self.current_ant = 0
        self.positions_current_generation = []
        self.rewards_current_generation = []

        self.best_positions = [None for _ in range(self.num_best_solutions)]
        self.best_rewards = [-inf for _ in range(self.num_best_solutions)]

    def get_best_position(self):
        """
        Obtains the best position so far found by the algorithm.

        :return: the best position.
        :rtype: numpy array.
        """
        # Todo

        return self.best_positions[np.argmax(self.best_rewards)]

    def get_best_value(self):
        """
        Obtains the value of the best position so far found by the algorithm.

        :return: value of the best position.
        :rtype: float.
        """
        # Todo
        return np.max(self.best_rewards)

    def get_position_to_evaluate(self):
        """
        Obtains a new position to evaluate.

        :return: position to evaluate.
        :rtype: numpy array.
        """
        # Todo
        return self.ants[self.current_ant].position

    def advance_generation(self):
        """
        Advances the generation of Ants. Auxiliary method to be used by notify_evaluation().
        """
        # Todo
        k = self.num_best_solutions
        indexes = np.argsort(self.rewards_current_generation)[::-1]
        sorted_generation_rewards = np.sort(self.rewards_current_generation)
        sorted_generation_positions = self.positions_current_generation[indexes]

        merged_best_rewards = self.best_rewards + sorted_generation_rewards[:k]
        merged_best_positions = self.best_positions + sorted_generation_positions[:k]

        indexes = np.argsort(merged_best_positions)[::-1]

        sorted_merged_best_rewards = np.sort(merged_best_rewards)
        sorted_merged_best_positions = self.merged_best_positions[indexes]

        self.best_rewards = sorted_merged_best_rewards[:k]
        self.best_positions = sorted_merged_best_positions[:k]

        self.positions_current_generation = []
        self.rewards_current_generation = []

    def notify_evaluation(self, value):
        """
        Notifies the algorithm that a ant position evaluation was completed.

        :param value: quality of the ant position.
        :type value: float.
        """
        # Todo

        ant = self.ants[self.current_ant]
        self.positions_current_generation.append(ant.position)
        self.rewards_current_generation.append(value)

        self.current_ant += 1
        if self.current_ant == self.num_ants:
            self.current_ant = 0
            self.advance_generation()
