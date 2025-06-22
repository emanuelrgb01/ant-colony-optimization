import numpy as np
import random
from math import inf


class Particle:
    """
    Represents a particle of the Particle Swarm Optimization algorithm.
    """

    def __init__(self, lower_bound, upper_bound):
        """
        Creates a particle of the Particle Swarm Optimization algorithm.

        :param lower_bound: lower bound of the particle position.
        :type lower_bound: numpy array.
        :param upper_bound: upper bound of the particle position.
        :type upper_bound: numpy array.
        """
        # Todo: implement
        self.dimension = np.size(lower_bound)  # = np.size(upper_bound)

        self.position = np.zeros(self.dimension)
        self.velocity = np.zeros(self.dimension)

        for idx in range(self.dimension):
            delta = upper_bound[idx] - lower_bound[idx]
            self.position[idx] = random.uniform(lower_bound[idx], upper_bound[idx])
            self.velocity[idx] = random.uniform(-delta, delta)

        self.best_position = None
        self.best_reward = -inf


class ParticleSwarmOptimization:
    """
    Represents the Particle Swarm Optimization algorithm.
    Hyperparameters:
        inertia_weight: inertia weight.
        cognitive_parameter: cognitive parameter.
        social_parameter: social parameter.

    :param hyperparams: hyperparameters used by Particle Swarm Optimization.
    :type hyperparams: Params.
    :param lower_bound: lower bound of particle position.
    :type lower_bound: numpy array.
    :param upper_bound: upper bound of particle position.
    :type upper_bound: numpy array.
    """

    def __init__(self, hyperparams, lower_bound, upper_bound):
        # Todo: implement

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.num_particles = hyperparams.num_particles
        self.inertia_weight = hyperparams.inertia_weight
        self.cognitive_parameter = hyperparams.cognitive_parameter
        self.social_parameter = hyperparams.social_parameter

        self.particles = [
            Particle(self.lower_bound, self.upper_bound)
            for _ in range(self.num_particles)
        ]

        self.current_particle = 0

        self.pso_best_position = None
        self.pso_best_reward = -inf

    def get_best_position(self):
        """
        Obtains the best position so far found by the algorithm.

        :return: the best position.
        :rtype: numpy array.
        """
        # Todo: implement

        return self.pso_best_position

    def get_best_value(self):
        """
        Obtains the value of the best position so far found by the algorithm.

        :return: value of the best position.
        :rtype: float.
        """
        # Todo: implement

        return self.pso_best_reward

    def get_position_to_evaluate(self):
        """
        Obtains a new position to evaluate.

        :return: position to evaluate.
        :rtype: numpy array.
        """
        # Todo: implement

        return self.particles[self.current_particle].position

    def advance_generation(self):
        """
        Advances the generation of particles. Auxiliary method to be used by notify_evaluation().
        """
        # Todo: implement

        w = self.inertia_weight
        phi_p = self.cognitive_parameter
        phi_g = self.social_parameter
        max_velocity = self.upper_bound - self.lower_bound
        min_velocity = -(self.upper_bound - self.lower_bound)

        for particle in self.particles:
            rp = random.uniform(0.0, 1.0)
            rg = random.uniform(0.0, 1.0)

            particle.velocity = (
                w * particle.velocity
                + phi_p * rp * (particle.best_position - particle.position)
                + phi_g * rg * (self.pso_best_position - particle.position)
            )

            # v_min <= v <= v_max ; utils.clamp()
            for i in range(particle.dimension):
                particle.velocity[i] = min(
                    max_velocity[i], max(min_velocity[i], particle.velocity[i])
                )

            particle.position = particle.position + particle.velocity
            # lower_bound <= position <= upper_bount ; utils.clamp()
            for i in range(particle.dimension):
                particle.position[i] = max(
                    self.lower_bound[i],
                    min(self.upper_bound[i], particle.position[i]),
                )

    def notify_evaluation(self, value):
        """
        Notifies the algorithm that a particle position evaluation was completed.

        :param value: quality of the particle position.
        :type value: float.
        """
        # Todo: implement

        particle = self.particles[self.current_particle]

        if value > particle.best_reward:
            particle.best_reward = value
            particle.best_position = particle.position

        if value > self.pso_best_reward:
            self.pso_best_reward = value
            self.pso_best_position = particle.position

        self.current_particle += 1
        if self.current_particle == self.num_particles:
            self.current_particle = 0
            self.advance_generation()
