import numpy as np
from math import inf


class Particle:

    def __init__(self):
        """
        Implementation of particle in PSO
        PSO solves a problem by having a population of candidate solutions, here dubbed particles.
        Each particle's movement is influenced by its local best known position, but is also guided toward
        the best known positions in the search-space, which are updated as better positions are found by other particles.

        Attributes:
            position : (list) current particle position
            best_position : (list) best particle position
            velocity : (list) particle velocity
            cost : (int) current particle cost
            best_cost (int) best particle cost
        """

        self.position = []
        self.best_position = []
        self.velocity = []
        self.cost = -1
        self.best_cost = -1


class PSO:

    def __init__(self, cost_function, num_var, num_particles=30, iter_max=100, var_min=-10, var_max=10,
                 wi=0.9, wf=0.4, cpi=2.5, cpf=0.5, csi=0.5, csf=2.5):
        """
        Implementation of  Particle swarm optimization algorithm (PSO)

        Attributes:
            cost_function : (function) Cost function that needs to be minimized
            num_var : (int) Number of variables
            num_particles : (int) Population size
            iter_max : (int) Maximum number of iterations
            var_min : (int) Lower bound of decision variables
            var_max : (int) Upper bound of decision variables
            wi : (int) Initial inertia coefficient
            wf : (int) Final inertia coefficient
            cpi : (int) Initial personal (cognitive) acceleration coefficient
            cpf : (int) Final personal (cognitive) acceleration coefficient
            csi : (int) Initial social acceleration coefficient
            csf : (int) Final social acceleration coefficient
        """

        self.cost_function = cost_function
        self.num_var = num_var

        self.num_particles = num_particles
        self.iter_max = iter_max
        self.var_min = var_min
        self.var_max = var_max
        self.wi = wi
        self.wf = wf
        self.cpi = cpi
        self.cpf = cpf
        self.csi = csi
        self.csf = csf

        self.global_best_cost = inf
        self.global_best_position = None

    def linrate(self, x_max, x_min, iteration):
        """
        Calculate PSO parameters for each iteration
        (inertia factor, cognitive acceleration coefficient and social acceleration coefficient)
        """

        return x_min + ((x_max - x_min) / (self.iter_max - 0)) * (self.iter_max - iteration)

    def optimize(self):
        """
        Minimize cost function (evaluation of artificial neural network performance)

        Return:
             global_best_position : (list)
        """

        # Initialize population
        # -----------------------------------------------------------------------------------------------
        particle_array = [Particle() for i in range(self.num_particles)]

        for i in range(self.num_particles):

            particle_array[i].position = np.random.uniform(self.var_min, self.var_max, self.num_var)
            particle_array[i].velocity = np.zeros(self.num_var)
            particle_array[i].cost = self.cost_function(particle_array[i].position)
            particle_array[i].best_position = particle_array[i].position
            particle_array[i].best_cost = particle_array[i].cost
            self.global_best_position = particle_array[i].best_position

            # Update global best
            if particle_array[i].best_cost < self.global_best_cost:
                self.global_best_cost = particle_array[i].best_cost
                self.global_best_position = particle_array[i].best_position
        # -----------------------------------------------------------------------------------------------

        # The main loop
        for iteration in range(self.iter_max):

            # Calculate inertia factor
            w = self.linrate(self.wf, self.wi, iteration)
            # Calculate personal acceleration coefficient
            cp = self.linrate(self.cpi, self.cpf, iteration)
            # Calculate social acceleration coefficient
            cs = self.linrate(self.csi, self.csf, iteration)

            for i in range(self.num_particles):

                # Update velocity
                particle_array[i].velocity = np.multiply(w, particle_array[i].velocity) + \
                                             np.multiply(cp * np.random.random(self.num_var),
                                                         np.subtract(particle_array[i].best_position,
                                                                     particle_array[i].position)) + \
                                             np.multiply(cs * np.random.random(self.num_var),
                                                         np.subtract(self.global_best_position,
                                                                     particle_array[i].position))

                # Update position
                particle_array[i].position = particle_array[i].position + particle_array[i].velocity

                # Evaluate cost function (current fitness)
                particle_array[i].cost = self.cost_function(particle_array[i].position)

                if particle_array[i].cost < particle_array[i].best_cost:

                    # Update personal best
                    particle_array[i].best_position = particle_array[i].position
                    particle_array[i].best_cost = particle_array[i].cost

                    # Update global best
                    if particle_array[i].best_cost < self.global_best_cost:
                        self.global_best_cost = particle_array[i].best_cost
                        self.global_best_position = particle_array[i].best_position

            # Display info about current iteration
            print(f'Iteration {iteration + 1}: Best cost = {self.global_best_cost}')

        return self.global_best_position
