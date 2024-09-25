from Toeplitz import Toeplitz
#from scipy.sparse.linalg import gmres
from cgmres import gmres
import multiprocessing
import os
import numpy as np
import scipy
import numpy.linalg as LA
from time import time

import Toeplitz, kernels, grids, graphs
from grids import generate_grid
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator
import timeit

import functools
print = functools.partial(print, flush=True)

class Solver_exact:
    def __init__(self, N, NU_COEFF, equation_utils, grid='chebyshev',\
                system_solver='gmres', no_init=False, use_precomputed=True, stats=False, verbose=True):
        self.N = N
        self.NU_COEFF = NU_COEFF
        self.K = equation_utils.K
        if isinstance(grid, str):
            self.grid_type = grid
            self.grid_params = {}
        elif isinstance(grid, tuple):
            self.grid_type = grid[0]
            self.grid_params = grid[1]
        elif isinstance(grid, np.ndarray):
            self.grid_type = 'custom'
            self.X = grid
        self.no_init = no_init
        self.use_precomputed = use_precomputed
        self.equation_id = equation_utils.equation_id
        self.exact_int = equation_utils.exact_int
        self.calc_type = np.double
        self.stats_mode = stats
        self.verbose = verbose

        if 'SLURM_NTASKS' in os.environ:
            self.num_processes = int(os.environ['SLURM_NTASKS'])
            print('in slurm')
        else:
            self.num_processes = 2#os.cpu_count()
        #print(f'esolver numproc = {self.num_processes}')


        if stats:
            self.stats_mode = True
            self.stats = defaultdict(int)
            self.stats['N'] = self.N[0]
            self.stats['build_time_exact'] = time()

        self.parameters_str = os.path.join('precomputed', str(self.equation_id), 'exact',f'{N}_{self.grid_type}')
        if use_precomputed:
            if not os.path.isdir('precomputed'):
                os.mkdir('precomputed')
            if not os.path.isdir(os.path.join('precomputed', str(self.equation_id))):
                os.mkdir(os.path.join('precomputed', str(self.equation_id)))

            if not os.path.isdir(os.path.join('precomputed', str(self.equation_id), 'exact')):
                os.mkdir(os.path.join('precomputed', str(self.equation_id), 'exact'))
            if not os.path.isdir(self.parameters_str):
                os.mkdir(self.parameters_str)


        if self.use_precomputed and os.path.isdir(self.parameters_str) and os.path.isfile(os.path.join(self.parameters_str, \
                                                    f'A_{self.N[0]}_{self.grid_type}.npy')):
            self.A = np.load(os.path.join(self.parameters_str, f'A_{self.N[0]}_{self.grid_type}.npy'))
            self.Operator = lambda x: self.A @ x
            self.a = np.load(os.path.join(self.parameters_str, f'a_{self.N[0]}_{self.grid_type}.npy'))
            self.b = np.load(os.path.join(self.parameters_str, f'b_{self.N[0]}_{self.grid_type}.npy'))
            self.X = np.load(os.path.join(self.parameters_str, f'X_{self.N[0]}_{self.grid_type}.npy'))
            self.x_cap = np.load(os.path.join(self.parameters_str, f'x_cap_{self.N[0]}_{self.grid_type}.npy'))
            self.y = np.load(os.path.join(self.parameters_str, f'y_{self.N[0]}_{self.grid_type}.npy'))
            if self.stats_mode:
                self.stats['build_time_exact'] = float(np.load(os.path.join(self.parameters_str, f'build_time_exact_{self.N[0]}_{self.grid_type}.npy')))
                self.stats['cond'] = float(np.load(os.path.join(self.parameters_str, f'cond_{self.N[0]}_{self.grid_type}.npy')))
            if self.verbose:
                print('loaded precomputed exact')
        else:
            self.build_grids()
            if not self.no_init:
                self.build_A()
        #print('A constructed')

        if system_solver == 'gmres':
            self.system_solver = gmres
        else:
            self.system_solver = system_solver



    def build_grids(self):
        #Границы области (a - левый нижний, b - правый верхний)
        self.a = np.array([0, 0])
        self.b = np.array([1, 1])

        self.X = [np.empty(shape=(self.N[0]), dtype=self.calc_type), np.empty(shape=(self.N[1]), dtype=self.calc_type)]
        self.x_cap = [np.empty(shape=(self.N[0]), dtype=self.calc_type), np.empty(shape=(self.N[1]), dtype=self.calc_type)]
        self.y = [np.empty(shape=(self.N[0]), dtype=self.calc_type), np.empty(shape=(self.N[1]), dtype=self.calc_type)]


        for i in range(2):
            self.X[i] = generate_grid(self.N[i], self.a[i], self.b[i], grid_type=self.grid_type)
            self.x_cap[i] = self.X[i][:-1] + np.diff(self.X[i]) / 2
            self.y[i] = (self.X[i][:-1] + np.diff(self.X[i]) / 2)

        if self.use_precomputed:
            if not os.path.isdir(self.parameters_str):
                os.mkdir(self.parameters_str)
            np.save(os.path.join(self.parameters_str, f'a_{self.N[0]}_{self.grid_type}'), self.a)
            np.save(os.path.join(self.parameters_str, f'b_{self.N[0]}_{self.grid_type}'), self.b)
            np.save(os.path.join(self.parameters_str, f'X_{self.N[0]}_{self.grid_type}'), self.X)
            np.save(os.path.join(self.parameters_str, f'x_cap_{self.N[0]}_{self.grid_type}'), self.x_cap)
            np.save(os.path.join(self.parameters_str, f'y_{self.N[0]}_{self.grid_type}'), self.y)

    def build_K(self):
        pass

    def build_P_QX(self):
        pass



    def operate_row_compute_rhs(self, j):
        #print(multiprocessing.current_process())
        j_s = np.asarray([j] * self.N[0] * self.N[1])
        row_A = self.exact_int(self.X[0][self.i_s // self.N[1]], self.X[0][(self.i_s) // self.N[1] + 1], self.X[1][(self.i_s) % self.N[1]], \
        self.X[1][(self.i_s) % self.N[1] + 1], self.x_cap[0][(j_s) // self.N[1]], self.x_cap[1][(j_s) % self.N[1]])
        return row_A @ self._cur_vec

    def operate_batch_compute_A(self, start_ind):
        #print (multiprocessing.current_process())
        start = self._batch_starts[start_ind]
        end = self._batch_starts[start_ind + 1]
        return self.exact_int(self.X[0][self.i_s[start:end] // self.N[1]], self.X[0][(self.i_s[start:end]) // self.N[1] + 1],
                                self.X[1][(self.i_s[start:end]) % self.N[1]], self.X[1][(self.i_s[start:end]) % self.N[1] + 1],
                                self.x_cap[0][(self.j_s[start:end]) // self.N[1]], self.x_cap[1][(self.j_s[start:end]) % self.N[1]])

    def build_A(self):
        inds = np.arange((self.N[0] * self.N[1]) ** 2)
        self.j_s = inds // (self.N[0] * self.N[1])
        self.i_s = inds % (self.N[0] * self.N[1])

        BATCH_SIZE = min(1000000, (self.N[0] * self.N[1]) ** 2 // os.cpu_count())
        num_batches = len(self.i_s) // BATCH_SIZE
        if len(self.i_s) % BATCH_SIZE != 0:
          num_batches += 1
        self._batch_starts = np.hstack((np.arange(0, len(self.i_s), BATCH_SIZE), len(self.i_s)))

        self.A = np.empty(shape=(self.N[0] * self.N[1]) ** 2, dtype=np.double)

        with multiprocessing.Pool(self.num_processes) as pool:
            self.A = np.concatenate(pool.map(self.operate_batch_compute_A, np.arange(len(self._batch_starts))[:-1]), axis=0)

        self.A = self.A.reshape(-1, self.N[0] * self.N[1])
        matvec = lambda x: self.A @ x
        self.Operator = LinearOperator((self.A.shape[0], self.A.shape[1]), matvec=matvec, rmatvec=matvec)
        #self.operator = LinearOperator((self.A.shape[0], self.A.shape[1]), matvec=self.matrix.mv, rmatvec=self.matrix.mv)

        if self.stats:
            self.stats['build_time_exact'] = np.round(time() - self.stats['build_time_exact'], 2)
            self.stats['cond'] = int(LA.cond(self.A))

        if self.use_precomputed:
            np.save(os.path.join(self.parameters_str, f'A_{self.N[0]}_{self.grid_type}.npy'), self.A)
            np.save(os.path.join(self.parameters_str, f'build_time_exact_{self.N[0]}_{self.grid_type}.npy'), self.stats['build_time_exact'])
            np.save(os.path.join(self.parameters_str, f'cond_{self.N[0]}_{self.grid_type}.npy'), self.stats['cond'])





    def compute_rhs(self, vec):
        if len(vec) != self.N[0] * self.N[1]:
            raise 'Несовпадение размерностей'
            return
        numrows = len(vec)
        self._cur_vec = vec
        if self.use_precomputed and os.path.isdir(self.parameters_str) and os.path.isfile(os.path.join(self.parameters_str, \
                                f'rhs_{self.N[0]}_{self.grid_type}.npy')) and os.path.isfile(os.path.join(self.parameters_str, \
                                                        f'mult_time_rhs_{self.N[0]}_{self.grid_type}.npy')):
            if self.verbose:
                print('loaded precomputed rhs')
            if self.stats_mode:
                self.stats['mult_time'] = float(np.load(os.path.join(self.parameters_str, f'mult_time_rhs_{self.N[0]}_{self.grid_type}.npy')))
            return np.load(os.path.join(self.parameters_str, f'rhs_{self.N[0]}_{self.grid_type}.npy'))
        elif hasattr(self, 'A'):
            rhs = self.A @ vec
            A = self.A
            if self.stats_mode:
                def time_mult():
                    return A @ vec
                num_iterations, duration = timeit.Timer(stmt='time_mult()', globals=locals()).autorange()
                duration /= num_iterations


            if self.stats_mode:
                duration = np.round(duration * numrows, 6)
                self.stats['mult_time'] = duration
                np.save(os.path.join(self.parameters_str, f'mult_time_rhs_{self.N[0]}_{self.grid_type}.npy'), duration)
            if self.use_precomputed:
                np.save(os.path.join(self.parameters_str, f'rhs_{self.N[0]}_{self.grid_type}.npy'), rhs)

            return rhs
        else:
            rhs = np.empty(numrows, dtype=self.calc_type)
            self.i_s = np.arange(numrows)
            duration = 0

            with multiprocessing.Pool(self.num_processes) as pool:
                rhs = pool.map(self.operate_row_compute_rhs, range(len(rhs)))


            if self.stats_mode:
                ### START OF COMMENTING ###
                j_s = np.asarray([0] * numrows) #Умножение тестируется на нулевой строке
                i_s = self.i_s
                row_A = self.exact_int(self.X[0][i_s // self.N[1]], self.X[0][(i_s) // self.N[1] + 1], self.X[1][(i_s) % self.N[1]], \
                                     self.X[1][(i_s) % self.N[1] + 1], self.x_cap[0][(j_s) // self.N[1]], self.x_cap[1][(j_s) % self.N[1]])
                ### END OF COMMENTING ###

                def time_mult():
                    return row_A @ vec

                num_iterations, duration = timeit.Timer(stmt='time_mult()', globals=locals()).autorange()
                duration /= num_iterations

            if self.stats_mode:
                self.stats['mult_time'] = np.round(duration * numrows, 6)
                if self.use_precomputed:
                    np.save(os.path.join(self.parameters_str, f'mult_time_rhs_{self.N[0]}_{self.grid_type}.npy'), self.stats['mult_time'])
            if self.use_precomputed:
                np.save(os.path.join(self.parameters_str, f'rhs_{self.N[0]}_{self.grid_type}.npy'), rhs)
            return rhs
            
    def solve(self, rhs=None):
        tol = min(1e-3, 0.1 / self.N[0])
        maxiter = max(5000, 5 * self.N[0])
        if rhs is None:
            rhs = np.ones(self.N[0] * self.N[1])

        start_time = time()
        sol, num_iter = self.system_solver(self.Operator, rhs, tol=tol, maxiter=maxiter)
        end_time = time()
        if num_iter == maxiter:
            print('maxiter reached!')
        if self.stats_mode:
            self.stats['time'] = np.round(end_time - start_time, 4)
            """
            start_time = time()
            class gmres_counter(object):
                def __init__(self):
                    self.niter = 0
                def __call__(self, rk=None):
                    self.niter += 1
            iter_counter = gmres_counter()
            sol, info = self.system_solver(self.Operator, rhs, tol=tol, maxiter=maxiter, callback=iter_counter)
            """
            self.stats['num_iter'] = num_iter
            self.stats['iter_time'] = np.round(self.stats['time'] / num_iter, 5)

        return sol
