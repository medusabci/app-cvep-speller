import copy
import numpy as np

class OptimizeLayoutGA:

    def __init__(self, seq_len, lag_array, matrix_size, n_chromosomes=50,
                 k=2, prob_mut=None, prob_cross=0.9, n_iter=1000,
                 initial_state=None):
        """ Genetic Algorithm coded to optimize the layout for c-VEP systems
        by maximizing the distance of continuous pair-wise lags accross a
        m-sequence.

        Parameters
        -------------
        seq_len : int
            Sequence length.
        lag_array : ndarray or list
            Flattened list of lags with dimensions (n,) where n is the number
            of commands.
        matrix_size : tuple or list
            2D tuple indicating the layout size (e.g., (4,4) for a 4x4 matrix).
        n_chromosomes: int
            Number of chromosomes per population (default: 50).
        k : int
            Number of selections for the tournament selection algorithm (
            default: 2).
        prob_mut : float
            Probability of mutation. If None, the algorithm will take
            prob_mut = 1/n, where n is the number of commands. The mutations
            is a single-couple swap mutation (default: None).
        prob_cross : float
            Probability of crossover. The algorithm implemented is an order
            crossover (OX) by Davis et al. (1999) (default: 0.9).
        n_iter : int
            Number of generations to compute (default: 1000).
        initial_state : ndarray
            Optimal layout to start iterating if any. Elitism will be applied
            to the initial population to include this solution. If None,
            initial population would be entirely random. Note that the
            dimensions of this initial_state must match the matrix_size
            parameter (default: None).

        Example of use
        ---------------
        >> lag_array = np.arange(0, 63, 4)
        >> matrix_dims = (4,4)
        >> seq_len = 63
        >> opt = OptimizeLayoutGA(seq_len, lag_array, matrix_dims)
        >> best_sol, best_fitness = opt.start()
        >> print(best_sol)

        """
        self.seq_len = seq_len
        self.lag_array = np.array(lag_array).astype(int)
        self.matrix_size = matrix_size
        assert len(self.matrix_size) == 2, ValueError("[OptimizeLayoutGA] "
                                                      "Matrix can only be 2D")
        self.N = len(lag_array)
        if np.prod(self.matrix_size) != len(self.lag_array):
            raise ValueError("[OptimizeLayoutGA] The product of matrix "
                             "dimensions must match the length of the "
                             "lag_array!")
        self.n_chromosomes = n_chromosomes
        self.prob_cross = prob_cross
        self.prob_mut = prob_mut
        self.k = k
        self.initial_state = initial_state
        if prob_mut is None:
            self.prob_mut = 1 / self.N
        self.n_iter = n_iter

        # Initialization
        self.curr_gen = 0
        self.curr_best_fit = -1
        self.curr_best_sol = None
        self.population = np.empty((self.n_chromosomes, self.matrix_size[0],
                                    self.matrix_size[1]), dtype=int)

    def start(self, verbose=True):
        """ Starts the iteration process.

        Returns
        ----------------
        tuple (best solution, best fitness)
            Tuple containing the best layout found so far, as well as the
            associated fitness.
        """
        # Initialization
        for i in range(self.n_chromosomes):
            np.random.shuffle(self.lag_array)
            self.population[i, :, :] = self.lag_array.reshape(
                self.matrix_size[0], self.matrix_size[1])

        # Maybe we want to include the last best layout to start with (elitism)
        self.population[0, :, :] = np.array(self.initial_state)

        # Iterations
        while self.curr_gen < self.n_iter:
            self.curr_gen += 1

            # Evaluate the fitness
            fit = self.fitness(self.population)

            # Maximize the fitness
            if np.max(fit) > self.curr_best_fit:
                self.curr_best_fit = np.max(fit)
                self.curr_best_sol = self.population[np.argmax(fit), :, :]
            if verbose:
                print("[OptimizeLayoutGA] >  Gen %i\t Fitness: %.2f (best) - "
                      "%.2f (local)" %  (self.curr_gen, self.curr_best_fit,
                                         np.max(fit)))

            # Selection via k-tournament selection
            n_childs = int(np.ceil(self.n_chromosomes/2))
            p1 = self.tournament_selection(
                self.k, n_childs, self.population, fit)
            p2 = self.tournament_selection(
                self.k, n_childs, self.population, fit)

            # Cross-over
            child_pop = self.ox_crossover(self.prob_cross, p1, p2)

            # Mutation
            child_pop = self.mutation(self.prob_mut, child_pop)

            # Elitism
            self.population = np.concatenate(
                (child_pop[:self.n_chromosomes-1,:,:],
                 np.expand_dims(self.curr_best_sol, axis=0)
                 ),
                axis=0)

        return (self.curr_best_sol, self.curr_best_fit)

    def fitness(self, population):
        """ Fitness function that computes the sum of all consecutive lag
        distances in horizontal, vertical and diagonal planes. """
        fit = np.empty((self.n_chromosomes,))
        for c in range(population.shape[0]):
            fit[c] = self._fitness_chromosome(population[c, :, :])
        return fit

    def _fitness_chromosome(self, ch):
        def point_exists(ch, i, j):
            if i < 0 or j < 0 or i >= ch.shape[0] or j >= ch.shape[1]:
                return False
            return True

        def l_dist(seq_len, lag1, lag2):
            """ Checks the distance as a circular array. E.g., For a seq_len of 63,
            the distance between lag 1 and 63 is 1.

            Parameters
            --------------
            seq_len : int
                Sequence length.
            lag1 :  int
                First lag point.
            lag2 : int
                Second lag point.

            Returns
            --------------
            int : distance
            """
            if np.abs(lag1 - lag2) < seq_len / 2:
                return np.abs(lag1 - lag2)
            else:
                return (seq_len - np.max([lag1, lag2])) + np.min([lag1, lag2])

        fit = np.zeros(ch.shape)
        for i in range(ch.shape[0]):
            for j in range(ch.shape[1]):
                if i == j:
                    continue
                # Sum of distance of consecutive rows
                if point_exists(ch, i + 1, j):
                    fit[i, j] += l_dist(self.seq_len, ch[i, j], ch[i + 1, j])
                if point_exists(ch, i - 1, j):
                    fit[i, j] += l_dist(self.seq_len, ch[i, j], ch[i - 1, j])

                # Sum of distance of consecutive columns
                if point_exists(ch, i, j + 1):
                    fit[i, j] += l_dist(self.seq_len, ch[i, j], ch[i, j + 1])
                if point_exists(ch, i, j - 1):
                    fit[i, j] += l_dist(self.seq_len, ch[i, j], ch[i, j - 1])

                # Sum of distance of consecutive diagonals
                if point_exists(ch, i + 1, j + 1):
                    fit[i, j] += l_dist(self.seq_len, ch[i, j],
                                        ch[i + 1, j + 1])
                if point_exists(ch, i - 1, j - 1):
                    fit[i, j] += l_dist(self.seq_len, ch[i, j],
                                        ch[i - 1, j - 1])
                if point_exists(ch, i + 1, j - 1):
                    fit[i, j] += l_dist(self.seq_len, ch[i, j],
                                        ch[i + 1, j - 1])
                if point_exists(ch, i - 1, j + 1):
                    fit[i, j] += l_dist(self.seq_len, ch[i, j],
                                        ch[i - 1, j + 1])
        return np.sum(fit.flatten())

    @staticmethod
    def ox_crossover(prob_cross, pop1, pop2):
        """ Order crossover according to Davis et al. (1999). It is a kind of
        variation of PMX with a different repairing procedure. For each
        child, select a substring from a parent at random and produce a
        proto-child by copying the substring into the corresponding position.
        Takes the second parent and selects only the lags which are not
        included in the proto-child. Then, places those lags into the unfixed
        positions of the proto-child in order. For the second child, repeat
        the procedure using the other parent. Finally, childs are combined
        into a new population. """
        dims = pop1.shape
        q1 = np.empty(dims)
        q2 = np.empty(dims)
        for c in range(dims[0]):
            if np.random.rand() < prob_cross:
                # order cross over
                p2_ = pop2[c, :, :].flatten()
                p1_ = pop1[c, :, :].flatten()
                cuts = np.sort(np.random.randint(0, len(p2_), 2))

                # first child
                q_ = copy.copy(p1_)
                to_place = [i for i in p2_ if i not in q_[cuts[0]:cuts[1]]]
                q_[0:cuts[0]] = to_place[:cuts[0]]
                q_[cuts[1]:] = to_place[cuts[0]:]
                q1[c, :, :] = q_.reshape((dims[1], dims[2])).astype(int)
                if len(np.unique(q_)) != dims[1] * dims[2]:
                    raise ValueError("[ox_crossover] Something happened, "
                                     "child has not all unique values")

                # second child
                q_ = copy.copy(p2_)
                to_place = [i for i in p1_ if i not in q_[cuts[0]:cuts[1]]]
                q_[0:cuts[0]] = to_place[:cuts[0]]
                q_[cuts[1]:] = to_place[cuts[0]:]
                q2[c, :, :] = q_.reshape((dims[1], dims[2])).astype(int)
                if len(np.unique(q_)) != dims[1] * dims[2]:
                    raise ValueError("[ox_crossover] Something happened, "
                                     "child has not all unique values")
            else:
                # do nothing
                q1[c, :, :] = pop1[c, :, :]
                q2[c, :, :] = pop2[c, :, :]
        return np.concatenate((q1, q2), axis=0)

    @staticmethod
    def mutation(prob_mut, pop):
        """ Single-couple swap mutation. The algorithm takes randomly a
        couple of lags and swaps their position.
        """
        for c in range(pop.shape[0]):
            for i in range(pop.shape[1]):
                for j in range(pop.shape[2]):
                    if np.random.rand() < prob_mut:
                        switch = pop[c, i, j]
                        ri = np.random.randint(0, pop.shape[1], 1)
                        rj = np.random.randint(0, pop.shape[2], 1)
                        pop[c, i, j] = pop[c, ri, rj]
                        pop[c, ri, rj] = switch
        return pop

    @staticmethod
    def tournament_selection(k, n_selections, population, fitness):
        """ K-tournament selection algorithm. For each child chromosome,
        takes k-parents randomly and compares their fitnesses. The parent
        with highest fitness wins. """
        child = np.empty((n_selections, population.shape[1],
                          population.shape[2]))
        for i in range(n_selections):
            competition = np.random.randint(0, population.shape[0], k)
            try:
                winner = np.argmax(fitness[competition])
            except Exception as e:
                print(e)
            child[i, :, :] = population[competition[winner], :, :]
        return child


if __name__ == "__main__":
    lag_array = np.arange(0, 63, 4)
    matrix_dims = (4,4)
    seq_len = 63
    opt = OptimizeLayoutGA(
        seq_len, lag_array, matrix_dims,
        initial_state=[[52., 28., 56., 32.],
                       [60., 24.,  0., 20.],
                       [16., 36.,  8., 40.],
                       [4., 44., 12., 48.]]
    )
    best_sol, _ = opt.start()
    print(best_sol)
