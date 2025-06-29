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
        self.dimension = np.size(lower_bound)
        self.position = np.zeros(self.dimension)

        self.position = [
            random.uniform(lower_bound[idx], upper_bound[idx])
            for idx in range(self.dimension)
        ]

        self.path = [self.position]
        self.reward = -inf


class AntColonyOptimization:
    """
    Represents the Ant Colony Optimization algorithm.
    Hyperparameters:
        num_ants: number of ants.
        num_best_solutions: number of solutions used to build Gaussian Mixture
        evaporation_rate: evaporation rate (xi), controls the search shrinkage
        q: selection parameter that controls the selection pressure

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
        self.dimension = len(lower_bound)

        self.num_ants = hyperparams.num_ants
        self.num_best_solutions = hyperparams.num_best_solutions
        self.evaporation_rate = hyperparams.evaporation_rate
        self.q = hyperparams.q

        self.ants = [
            Ant(self.lower_bound, self.upper_bound) for _ in range(self.num_ants)
        ]

        self.current_ant = 0
        self.positions_current_generation = []
        self.rewards_current_generation = []

        self.best_positions = np.zeros((self.num_best_solutions, self.dimension))
        self.best_rewards = np.full(self.num_best_solutions, -inf)

    def get_best_position(self):
        """
        Obtains the best position so far found by the algorithm.

        :return: the best position.
        :rtype: numpy array.
        """
        if np.all(self.best_rewards == -inf):
            return None
        return self.best_positions[np.argmax(self.best_rewards)]

    def get_best_value(self):
        """
        Obtains the value of the best position so far found by the algorithm.

        :return: value of the best position.
        :rtype: float.
        """
        return np.max(self.best_rewards)

    def get_position_to_evaluate(self):
        """
        Obtains a new position to evaluate.

        :return: position to evaluate.
        :rtype: numpy array.
        """
        return self.ants[self.current_ant].position

    def _update_ant_positions(self):
        """
        Main logic of ACO_R: Updates the position of all ants for the next generation.
        """
        k = self.num_best_solutions

        valid_solutions_count = np.sum(self.best_rewards > -inf)
        if valid_solutions_count < 2:
            return

        weights = (1 / (self.q * k * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * (np.arange(k) ** 2) / (self.q**2 * k**2)
        )
        probabilities = weights / np.sum(weights)

        for ant in self.ants:
            selected_index = np.random.choice(k, p=probabilities)
            guide_solution = self.best_positions[selected_index]

            new_position = np.zeros(self.dimension)
            for dim in range(self.dimension):
                mu = guide_solution[dim]

                sigma = (
                    self.evaporation_rate
                    * np.sum(np.abs(self.best_positions[:, dim] - mu))
                    / (k - 1 + 1e-9)
                )

                new_position[dim] = random.gauss(mu, sigma)

            ant.position = np.clip(new_position, self.lower_bound, self.upper_bound)
            ant.path.append(ant.position)

    def advance_generation(self):
        """
        Advances the generation of Ants. Auxiliary method to be used by notify_evaluation().
        """
        current_pos_arr = np.array(self.positions_current_generation)
        current_rew_arr = np.array(self.rewards_current_generation)

        valid_best_indices = np.where(self.best_rewards > -inf)[0]

        if valid_best_indices.size > 0:
            combined_positions = np.vstack(
                (self.best_positions[valid_best_indices], current_pos_arr)
            )
            combined_rewards = np.hstack(
                (self.best_rewards[valid_best_indices], current_rew_arr)
            )
        else:
            combined_positions = current_pos_arr
            combined_rewards = current_rew_arr

        sorted_indices = np.argsort(combined_rewards)[::-1]

        num_to_keep = min(self.num_best_solutions, len(sorted_indices))
        top_indices = sorted_indices[:num_to_keep]

        self.best_rewards = np.full(self.num_best_solutions, -inf)
        self.best_positions = np.zeros((self.num_best_solutions, self.dimension))

        self.best_rewards[:num_to_keep] = combined_rewards[top_indices]
        self.best_positions[:num_to_keep] = combined_positions[top_indices]

        self.positions_current_generation = []
        self.rewards_current_generation = []

        self._update_ant_positions()

    def notify_evaluation(self, value):
        """
        Notifies the algorithm that a ant position evaluation was completed.

        :param value: quality of the ant position.
        :type value: float.
        """
        ant = self.ants[self.current_ant]
        self.positions_current_generation.append(ant.position)
        self.rewards_current_generation.append(value)
        ant.reward = value

        self.current_ant += 1
        if self.current_ant == self.num_ants:
            self.current_ant = 0
            self.advance_generation()
