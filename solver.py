from Toeplitz import Toeplitz
#from scipy.sparse.linalg import gmres, lgmres
from cgmres import gmres
from numba import jit, njit
import numpy as np
from scipy.sparse import load_npz, save_npz
import numpy.linalg as LA
from time import time
import Toeplitz, kernels, grids, graphs
from grids import generate_grid
from collections import defaultdict
from itertools import product
from bisect import bisect_left
import os
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator
from datetime import datetime
from batched_product import batched_product
import sgd2
import multiprocessing
import timeit
import pickle
import functools

print = functools.partial(print, flush=True)

class Solver:
    #@profile
    def __init__(self, N, NU_COEF, exact_dist, equation_utils, grid='chebyshev', \
                system_solver='gmres', use_precomputed=True, debug=False, stats=False, verbose=True):
        self.N = N
        self.NU_COEFF = NU_COEF
        if isinstance(exact_dist, int):
            self.EXACT_DIST = exact_dist
        elif isinstance(exact_dist, tuple):
            if exact_dist[0] == 'log':
                self.EXACT_DIST = int(float(exact_dist[1]) * np.log(N[0]))
            elif exact_dist[0] == 'sqrt':
                self.EXACT_DIST = int(float(exact_dist[1]) * np.sqrt(N[0]))
        else:
            self.EXACT_DIST = int(3 * np.log(N))

        self.NU = [int(N[0] * NU_COEF[0]), int(N[1] * NU_COEF[1])]
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

        self.use_precomputed = use_precomputed
        self.equation_id = equation_utils.equation_id
        self.exact_int = equation_utils.exact_int
        self.calc_type = np.double
        self.debug_mode = debug
        self.verbose = verbose

        if 'SLURM_NTASKS' in os.environ:
            self.num_processes = int(os.environ['SLURM_NTASKS'])
        else:
            self.num_processes = os.cpu_count()

        #os.environ['OMP_NUM_THREADS'] = str(self.num_processes)
        #print(f'effsolver numproc = {self.num_processes}')
        if debug:
            self.debug_folder = os.path.join('debug_info', datetime.now().strftime("%m:%d:%H:%M:%S"))
            os.mkdir(f'{self.debug_folder}')

        def init_stats():
            self.stats_mode = True
            self.stats = defaultdict(int)
            self.stats['N'] = self.N[0]
            self.stats['NU_COEF'] = self.NU_COEFF[0]
            self.stats['EXACT_DIST'] = self.EXACT_DIST
            self.stats['grid_type'] = self.grid_type
            self.stats['build_time'] = time()
        if stats:
            init_stats()


        self.parameters_str = os.path.join('precomputed', str(self.equation_id), 'eff',f'{N}_{NU_COEF}_{self.EXACT_DIST}_{self.grid_type}')
        if use_precomputed:
            os.makedirs(self.parameters_str, exist_ok=True)

        if self.use_precomputed and os.path.isdir(self.parameters_str):
            try:
                self.K_u_big_toepl = np.load(os.path.join(self.parameters_str, f'K_{self.N[0]}_{self.NU_COEFF[0]}_{self.EXACT_DIST}_{self.grid_type}.npy'))
                self.P = load_npz(os.path.join(self.parameters_str, f'P_{self.N[0]}_{self.NU_COEFF[0]}_{self.EXACT_DIST}_{self.grid_type}.npz'))
                self.QX = load_npz(os.path.join(self.parameters_str, f'QX_{self.N[0]}_{self.NU_COEFF[0]}_{self.EXACT_DIST}_{self.grid_type}.npz'))
                self.R = load_npz(os.path.join(self.parameters_str, f'R_{self.N[0]}_{self.NU_COEFF[0]}_{self.EXACT_DIST}_{self.grid_type}.npz'))

                #self.P_star = load_npz(os.path.join(self.parameters_str, f'P_star_{self.N[0]}_{self.NU_COEFF[0]}_{self.EXACT_DIST}_{self.grid_type}.npz'))
                #self.QX_star = load_npz(os.path.join(self.parameters_str, f'QX_star_{self.N[0]}_{self.NU_COEFF[0]}_{self.EXACT_DIST}_{self.grid_type}.npz'))
                #self.R_star = load_npz(os.path.join(self.parameters_str, f'R_star_{self.N[0]}_{self.NU_COEFF[0]}_{self.EXACT_DIST}_{self.grid_type}.npz'))

                self.a = np.load(os.path.join(self.parameters_str, f'a_{self.N[0]}_{self.NU_COEFF[0]}_{self.EXACT_DIST}_{self.grid_type}.npy'))
                self.b = np.load(os.path.join(self.parameters_str, f'b_{self.N[0]}_{self.NU_COEFF[0]}_{self.EXACT_DIST}_{self.grid_type}.npy'))
                self.X = np.load(os.path.join(self.parameters_str, f'X_{self.N[0]}_{self.NU_COEFF[0]}_{self.EXACT_DIST}_{self.grid_type}.npy'))
                self.XU_big = np.load(os.path.join(self.parameters_str, f'XU_big_{self.N[0]}_{self.NU_COEFF[0]}_{self.EXACT_DIST}_{self.grid_type}.npy'))
                self.x_cap = np.load(os.path.join(self.parameters_str, f'x_cap_{self.N[0]}_{self.NU_COEFF[0]}_{self.EXACT_DIST}_{self.grid_type}.npy'))
                self.y = np.load(os.path.join(self.parameters_str, f'y_{self.N[0]}_{self.NU_COEFF[0]}_{self.EXACT_DIST}_{self.grid_type}.npy'))
                self.H = np.load(os.path.join(self.parameters_str, f'H_{self.N[0]}_{self.NU_COEFF[0]}_{self.EXACT_DIST}_{self.grid_type}.npy'))
                if self.stats_mode:
                    self.stats['build_time'] = float(np.load(os.path.join(self.parameters_str, f'build_time_{self.N[0]}_{self.NU_COEFF[0]}_{self.EXACT_DIST}_{self.grid_type}.npy')))
                    self.stats['fullness'] = '{:.2e}'.format(float(np.load(os.path.join(self.parameters_str, f'fullness_{self.N[0]}_{self.NU_COEFF[0]}_{self.EXACT_DIST}_{self.grid_type}.npy'))))
                if self.verbose:
                    print('loaded precomputed')
            except FileNotFoundError:
                pass

        if not hasattr(self, 'H'):
            if not self.stats_mode:
                init_stats()

            self.build_grids()
            self.build_K()
            self.build_P_QX()
            print('K, P, QX built')
            self.build_R()
            print('R constructed')

        self.matrix = Toeplitz.Toeplitz(self.K_u_big_toepl, self.P, self.QX, self.R, num_processes=self.num_processes, f=1)
        #print(self.R is self.matrix.R)
        #self.matrix_star = Toeplitz.Toeplitz(self.K_u_big_toepl, self.QX_star, self.P_star, self.R_star, f=1)
        print('created Toeplitz matrix')

        self.operator = LinearOperator((self.R.shape[0], self.R.shape[1]), matvec=self.matrix.mv, rmatvec=self.matrix.mv)

        if self.use_precomputed and self.stats_mode:
            try:
                approx_err, exact_norm = np.load(os.path.join(self.parameters_str, f'approx_err_exact_norm_{self.N[0]}_{self.NU_COEFF[0]}_{self.EXACT_DIST}_{self.grid_type}.npy'))
                print('using precomputed approx_err')

                self.stats['approx_err'] = '{:.2e}'.format(approx_err)
                self.stats['exact_norm'] = '{:.2e}'.format(exact_norm)
            except FileNotFoundError:
                self.calc_approximation()

        if system_solver == 'gmres':
            self.system_solver = gmres
        elif system_solver == 'lgmres':
            self.system_solver = lgmres
        elif system_solver == 'gd2':
            sgd2_solver = sgd2.sgd2(self.matrix, self.matrix_star)
            self.system_solver = sgd2_solver.solve
        else:
            self.system_solver = system_solver

        if self.stats_mode:
            self.stats['system_solver'] = str(system_solver)
        print('basic construction finished')

    def build_grids(self):
        #Границы области (a - левый нижний, b - правый верхний)
        self.a = np.array([0, 0])
        self.b = np.array([1, 1])

        if self.grid_type != 'custom':
            self.X = [np.empty(shape=(self.N[0]), dtype=self.calc_type), np.empty(shape=(self.N[1]), dtype=self.calc_type)]
        #self.XU = [np.empty(shape=(self.N[0]), dtype=self.calc_type), np.empty(shape=(self.N[1]), dtype=self.calc_type)]
        self.XU_big = [np.empty(shape=(self.NU[0]), dtype=self.calc_type), np.empty(shape=(self.NU[1]), dtype=self.calc_type)]
        self.x_cap = [np.empty(shape=(self.N[0]), dtype=self.calc_type), np.empty(shape=(self.N[1]), dtype=self.calc_type)]
        self.y = [np.empty(shape=(self.N[0]), dtype=self.calc_type), np.empty(shape=(self.N[1]), dtype=self.calc_type)]
        self.H = np.empty(2, dtype=self.calc_type) #Шаг сетки с большим числом узлов по всем осям

        for i in range(2):
            if self.grid_type != 'custom':
                self.X[i] = generate_grid(self.N[i], self.a[i], self.b[i], grid_type=self.grid_type)
            #self.XU[i] = generate_grid(self.N[i], self.a[i], self.b[i])
            self.XU_big[i] = generate_grid(self.NU[i], self.a[i], self.b[i])
            if self.grid_type == 'chebyshev':
                self.x_cap[i] = (1 - np.cos(np.pi * (np.arange(1, self.N[i] + 1) - 0.5) / self.N[i])) / 2
            else:
                self.x_cap[i] = self.X[i][:-1] + np.diff(self.X[i]) / 2
            self.y[i] = (self.X[i][:-1] + np.diff(self.X[i]) / 2)
            self.H[i] = self.XU_big[i][1] - self.XU_big[i][0]

        if self.use_precomputed:
            os.makedirs(self.parameters_str, exist_ok=True)
            np.save(os.path.join(self.parameters_str, f'a_{self.N[0]}_{self.NU_COEFF[0]}_{self.EXACT_DIST}_{self.grid_type}'), self.a)
            np.save(os.path.join(self.parameters_str, f'b_{self.N[0]}_{self.NU_COEFF[0]}_{self.EXACT_DIST}_{self.grid_type}'), self.b)
            np.save(os.path.join(self.parameters_str, f'X_{self.N[0]}_{self.NU_COEFF[0]}_{self.EXACT_DIST}_{self.grid_type}'), self.X)
            np.save(os.path.join(self.parameters_str, f'XU_big_{self.N[0]}_{self.NU_COEFF[0]}_{self.EXACT_DIST}_{self.grid_type}'), self.XU_big)
            np.save(os.path.join(self.parameters_str, f'x_cap_{self.N[0]}_{self.NU_COEFF[0]}_{self.EXACT_DIST}_{self.grid_type}'), self.x_cap)
            np.save(os.path.join(self.parameters_str, f'y_{self.N[0]}_{self.NU_COEFF[0]}_{self.EXACT_DIST}_{self.grid_type}'), self.y)
            np.save(os.path.join(self.parameters_str, f'H_{self.N[0]}_{self.NU_COEFF[0]}_{self.EXACT_DIST}_{self.grid_type}'), self.H)

    def build_K(self):
        self.K_u_big_toepl = np.zeros((2 * (self.NU[0] + 1) - 1, 2 * (self.NU[1] + 1) - 1), dtype=self.calc_type)
        for block in range((self.NU[0] + 1)):
            row = np.array([self.K(np.array([self.XU_big[0][0], self.XU_big[1][0]]),
                              np.array([self.XU_big[0][block], self.XU_big[1][i]])) for i in range(self.NU[1]+1)])
            self.K_u_big_toepl[self.NU[0] + block, :] = np.hstack((row[::-1][:-1], row))
            self.K_u_big_toepl[self.NU[0] - block, :] = np.hstack((row[::-1][:-1], row))
        if self.debug_mode:
            np.save(os.path.join(f'{self.debug_folder}', f'K_u_big_toepl_{self.N[0]}.npy'), self.K_u_big_toepl)
        if self.use_precomputed:
            np.save(os.path.join(self.parameters_str, f'K_{self.N[0]}_{self.NU_COEFF[0]}_{self.EXACT_DIST}_{self.grid_type}'), self.K_u_big_toepl)

    def build_P_QX(self):
        x_cap_p_inds = [np.zeros(shape=(self.N[0]), dtype=self.calc_type), np.zeros(shape=(self.N[1]), dtype=self.calc_type)]#проекция xcap на индексы большой равномерной сетки
        y_p_inds = [np.zeros(shape=(self.N[0]), dtype=self.calc_type), np.zeros(shape=(self.N[1]), dtype=self.calc_type)]#проекция xcap на индексы большой равномерной сетки
        alphas = [np.zeros(shape=(self.N[0]), dtype=self.calc_type), np.zeros(shape=(self.N[1]), dtype=self.calc_type)] #альфы для сеток
        betas = [np.zeros(shape=(self.N[0]), dtype=self.calc_type), np.zeros(shape=(self.N[1]), dtype=self.calc_type)] #беты для сеток
        for i in range(2):
          x_cap_p_inds[i] = np.floor((self.x_cap[i] - self.XU_big[i][0]) / self.H[i]).astype('int64')
          x_cap_p = np.array(self.XU_big[i][x_cap_p_inds[i]])

          y_p_inds[i] = np.floor((self.y[i] - self.XU_big[i][0]) / self.H[i]).astype('int64')
          y_p = np.array(self.XU_big[i][y_p_inds[i]])

          #alphas[i] = np.round((self.x_cap[i] - x_cap_p) / self.H[i], self.num_exact)
          alphas[i] = (self.x_cap[i] - x_cap_p) / self.H[i]
          #betas[i] = np.round((self.y[i] - y_p) / self.H[i], self.num_exact)
          betas[i] = (self.y[i] - y_p) / self.H[i]

        #times.append(time() - start) #1 - заполнение сеток

        P_row_inds = np.repeat(np.arange(self.N[0] * self.N[1]), 4)

        j0s = x_cap_p_inds[0][P_row_inds // self.N[1]]
        j0s[2:len(j0s):4] += 1
        j0s[3:len(j0s):4] += 1

        j1s = x_cap_p_inds[1][P_row_inds % self.N[1]]
        j1s[1:len(j0s):4] += 1
        j1s[3:len(j0s):4] += 1


        P_col_inds = j0s * (self.NU[1] + 1) + j1s
        QX_col_inds = P_row_inds
        QX_row_inds = P_col_inds

        j0s = alphas[0][P_row_inds // self.N[1]]
        j0s[0:len(j0s):4] = 1 - j0s[0:len(j0s):4]
        j0s[1:len(j0s):4] = 1 - j0s[1:len(j0s):4]

        j1s = alphas[1][P_row_inds % self.N[1]]
        j1s[0:len(j1s):4] = 1 - j1s[0:len(j1s):4]
        j1s[2:len(j1s):4] = 1 - j1s[2:len(j1s):4]

        P_data = j0s * j1s

        del j0s, j1s

        i0s = betas[0][QX_col_inds // self.N[1]]
        i0s[0:len(i0s):4] = 1 - i0s[0:len(i0s):4]
        i0s[1:len(i0s):4] = 1 - i0s[1:len(i0s):4]

        i1s = betas[1][QX_col_inds % self.N[1]]
        i1s[0:len(i1s):4] = 1 - i1s[0:len(i1s):4]
        i1s[2:len(i1s):4] = 1 - i1s[2:len(i1s):4]

        Xdiff0 = np.diff(self.X[0])
        Xdiff1 = np.diff(self.X[1])
        i0s_ind = QX_col_inds // self.N[1]
        i1s_ind = QX_col_inds % self.N[1]

        QX_data = -i0s * i1s * Xdiff0[i0s_ind] * Xdiff1[i1s_ind]

        del i0s, i1s, Xdiff0, Xdiff1
        del alphas, betas
        del i0s_ind, i1s_ind


        self.P = csr_matrix((P_data, (P_row_inds, P_col_inds)), shape=(self.N[0] * self.N[1], (self.NU[0] + 1) * (self.NU[1] + 1)), dtype=self.calc_type)
        self.QX = csr_matrix((QX_data, (QX_row_inds, QX_col_inds)), shape=((self.NU[0] + 1) * (self.NU[1] + 1), self.N[0] * self.N[1]), dtype=self.calc_type)
        #self.P_star = csr_matrix((np.conj(P_data), (P_col_inds, P_row_inds)))
        #self.QX_star = csr_matrix((np.conj(QX_data), (QX_col_inds, QX_row_inds)))

        self.x_cap_p_inds = x_cap_p_inds

        self.y_p_inds = y_p_inds
        self.P_data = P_data
        self.P_col_inds = P_col_inds
        self.P_row_inds = P_row_inds

        self.QX_data = QX_data
        self.QX_row_inds = QX_row_inds
        self.QX_col_inds = QX_col_inds

        if self.debug_mode:
            np.save(os.path.join(f'{self.debug_folder}', f'P_{self.N[0]}.npy'), self.P.todense())
            np.save(os.path.join(f'{self.debug_folder}', f'QX_{self.N[0]}.npy'), self.QX.todense())

        if self.use_precomputed:
            save_npz(os.path.join(self.parameters_str, f'P_{self.N[0]}_{self.NU_COEFF[0]}_{self.EXACT_DIST}_{self.grid_type}'), self.P)
            save_npz(os.path.join(self.parameters_str, f'QX_{self.N[0]}_{self.NU_COEFF[0]}_{self.EXACT_DIST}_{self.grid_type}'), self.QX)
            #save_npz(os.path.join(self.parameters_str, f'P_star_{self.N[0]}_{self.NU_COEFF[0]}_{self.EXACT_DIST}_{self.grid_type}'), self.P_star)
            #save_npz(os.path.join(self.parameters_str, f'QX_star_{self.N[0]}_{self.NU_COEFF[0]}_{self.EXACT_DIST}_{self.grid_type}'), self.QX_star)
    #@profile
    def operate_d0d1_build_R(self, d0d1):
        d0, d1 = d0d1
        NU0 = self.NU[0]
        NU1 = self.NU[1]
        dists_pairs = self._dists_pairs
        X = self.X
        x_cap = self.x_cap
        K_u_big_toepl = self.K_u_big_toepl
        N = self.N
        P_data = self.P_data
        P_col_inds = self.P_col_inds
        P_row_inds = self.P_row_inds

        QX_data = self.QX_data
        QX_row_inds = self.QX_row_inds
        QX_col_inds = self.QX_col_inds

        R_data_full =  []
        R_row_inds_full = []
        R_col_inds_full = []

        @njit
        def get_K_elem(x):
            j = x[0]
            i = x[1]
            return K_u_big_toepl[NU0 - j // (NU0 + 1) + i // (NU1 + 1), \
                                 NU1 - j % (NU0 + 1) + i % (NU1 + 1)]

        for fix_inds in batched_product(dists_pairs[0][d0], dists_pairs[1][d1]):
            fix_inds = np.asarray(fix_inds, dtype=np.int32)
            j0s = fix_inds[:,0,0]
            j1s = fix_inds[:,1,0]
            i0s = fix_inds[:,0,1]
            i1s = fix_inds[:,1,1]
            del fix_inds

            R_row_inds = j0s * N[1] + j1s
            R_col_inds = i0s * N[1] + i1s


            R_data = self.exact_int(X[0][i0s], X[0][i0s + 1], X[1][i1s], X[1][i1s + 1], \
                               x_cap[0][j0s], x_cap[1][j1s])

            del j0s, j1s, i0s, i1s

            P_locs_data = np.empty(4 * len(R_row_inds), dtype=np.double)
            QX_locs_data = np.empty(4 * len(R_col_inds), dtype=np.double)
            rows_all = np.empty(4 * len(R_row_inds), dtype=np.int32)
            columns_all = np.empty(4 * len(R_col_inds), dtype=np.int32)


            R_row_inds4 = np.repeat(4 * R_row_inds, 4)
            R_row_inds4[1::4] += 1
            R_row_inds4[2::4] += 2
            R_row_inds4[3::4] += 3
            P_locs_data = P_data[R_row_inds4].reshape(-1, 4)
            rows_all = P_col_inds[R_row_inds4].reshape(-1, 4)

            R_row_inds_full.extend(R_row_inds)
            del R_row_inds4, R_row_inds

            R_col_inds4 = np.repeat(4 * R_col_inds, 4)
            R_col_inds4[1::4] += 1
            R_col_inds4[2::4] += 2
            R_col_inds4[3::4] += 3
            QX_locs_data = QX_data[R_col_inds4].reshape(-1, 4).T
            columns_all = QX_row_inds[R_col_inds4].reshape(-1, 4)

            R_col_inds_full.extend(R_col_inds)
            del R_col_inds4, R_col_inds

            """
            for ind, _ in enumerate(R_row_inds):
                ind4 = 4 * ind
                R_row_inds_ind4 = 4 * R_row_inds[ind]
                R_col_inds_ind4 = 4 * R_col_inds[ind]

                P_locs_data[ind4:ind4+4] = P_data[R_row_inds_ind4:R_row_inds_ind4+4]
                QX_locs_data[ind4:ind4+4] = QX_data[R_col_inds_ind4:R_col_inds_ind4+4]
                rows_all[ind4:ind4+4] = P_col_inds[R_row_inds_ind4:R_row_inds_ind4+4]
                columns_all[ind4:ind4+4] = QX_row_inds[R_col_inds_ind4:R_col_inds_ind4+4]

            P_locs_data = P_locs_data.reshape(-1, 4)
            QX_locs_data = QX_locs_data.reshape(-1, 4).T
            rows_all = rows_all.reshape(-1, 4)
            columns_all = columns_all.reshape(-1, 4)
            """
            """
            for ind, _ in enumerate(R_data):
                K_loc = np.asarray([get_K_elem(x) for x in product(rows_all[ind], columns_all[ind])]).reshape(4, 4)
                R_data[ind] -= (P_locs_data[ind] @ K_loc) @ QX_locs_data[:, ind]
            """
            R_data -= np.asarray([(P_locs_data[ind] @ np.asarray([get_K_elem(x) for x in product(rows_all[ind], columns_all[ind])]).reshape(4, 4)) @ \
                    QX_locs_data[:, ind] for ind, _ in enumerate(R_data)])
            del P_locs_data, QX_locs_data, rows_all, columns_all


            R_data_full.extend(R_data)

        return (R_data_full, R_row_inds_full, R_col_inds_full)

    #@profile
    def build_R(self):
        NU0 = self.NU[0]
        NU1 = self.NU[1]
        N = self.N
        #N1 = self.N[1]
        X = self.X
        x_cap = self.x_cap
        x_cap_p_inds = self.x_cap_p_inds
        y_p_inds = self.y_p_inds

        EXACT_DIST = self.EXACT_DIST
        x_j = x_cap_p_inds
        y_i = y_p_inds

        dists_pairs = [defaultdict(list), defaultdict(list)]

        for axis in [0, 1]:
          #left - первый у, который попадает в зону текущего х. right - первый, который не годится после left
          left = right = 0
          for x_j_ind, x_j_cur in enumerate(x_j[axis]):
            while left < len(y_i[axis]) and y_i[axis][left] < x_j_cur - EXACT_DIST:
              left += 1
            while right < len(y_i[axis]) and y_i[axis][right] <= x_j_cur + EXACT_DIST:
              right += 1
            for y_i_ind in range(left, right):
              dists_pairs[axis][np.abs(y_i[axis][y_i_ind] - x_j[axis][x_j_ind])].append((x_j_ind, y_i_ind))
        print('calculated dists_pairs')
        self._dists_pairs = dists_pairs

        R_row_inds_full = []
        R_col_inds_full = []
        R_data_full = []

        if self.verbose:
            print(f'number of iterations <= {np.ceil(len(dists_pairs[0]) * len(dists_pairs[1]))}')
        d0_d1s = filter(lambda x: x[0] + x[1] <= EXACT_DIST, product(dists_pairs[0], dists_pairs[1]))
        print('running calculating R_data, etc.')
        with multiprocessing.Pool(self.num_processes) as pool:
            R_data_R_row_inds_R_col_inds = pool.map(self.operate_d0d1_build_R, d0_d1s)
        del d0_d1s, self._dists_pairs

        print('collecting R_data')
        for R_data, R_row_inds, R_col_inds in R_data_R_row_inds_R_col_inds:
            R_data_full.extend(R_data)
            R_row_inds_full.extend(R_row_inds)
            R_col_inds_full.extend(R_col_inds)
        del R_data_R_row_inds_R_col_inds

        self.R = csr_matrix((R_data_full, (R_row_inds_full, R_col_inds_full)), shape=(self.N[0] * self.N[1], self.N[0] * self.N[1]), dtype=self.calc_type)
        #self.R_star = csr_matrix((np.conj(R_data_full), (R_col_inds_full, R_row_inds_full)))

        if self.stats_mode:
            self.stats['fullness'] = '{:.2e}'.format(len(R_data_full) / N[0]**2 / N[1]**2)
            self.stats['build_time'] = np.round(time() - self.stats['build_time'], 2)

        del R_row_inds_full, R_col_inds_full, R_data_full
        #print(self.R is self.matrix.R)
        if self.debug_mode:
            np.save(os.path.join(f'{self.debug_folder}', f'R_{self.N[0]}.npy'), self.R.todense())
            #np.save(os.path.join(f'{self.debug_folder}', f'R_data_{self.N[0]}.npy'), R_data.reshape(self.N[0]**2, self.N[0]**2))
        if self.use_precomputed:
            save_npz(os.path.join(self.parameters_str, f'R_{self.N[0]}_{self.NU_COEFF[0]}_{self.EXACT_DIST}_{self.grid_type}'), self.R)
            #save_npz(os.path.join(self.parameters_str, f'R_star_{self.N[0]}_{self.NU_COEFF[0]}_{self.EXACT_DIST}_{self.grid_type}'), self.R_star)
            np.save(os.path.join(self.parameters_str, f'build_time_{self.N[0]}_{self.NU_COEFF[0]}_{self.EXACT_DIST}_{self.grid_type}'), self.stats['build_time'])
            np.save(os.path.join(self.parameters_str, f'fullness_{self.N[0]}_{self.NU_COEFF[0]}_{self.EXACT_DIST}_{self.grid_type}'), self.stats['fullness'])
        #times.append(time() - start) #7 - R сформировано

    def operate_col_approximation(self, col):
        X = self.X
        x_cap = self.x_cap
        BATCH_SIZE = 100000
        row_norm = 0.0
        row_err = 0.0
        col_true = col
        #print(f'{multiprocessing.current_process()} reached')
        for col in range(col_true * self.N[0], col_true * self.N[0] + self.N[0]):
            unit_vec = np.zeros(self.N[0] ** 2, dtype=np.bool_)
            unit_vec[col] = 1
            operator_unit_vec = self.operator(unit_vec).real
            del unit_vec
            i0s = col // self.N[0]
            i1s = col % self.N[0]
            batch = 0
            while batch < self.N[0] ** 2:
            #for batch in range(0, self.N[0] ** 2, BATCH_SIZE):
                j0s_batch = np.arange(batch, min(batch + BATCH_SIZE, self.N[0] ** 2), dtype=np.int32) // self.N[0]
                j1s_batch = np.arange(batch, min(batch + BATCH_SIZE, self.N[0] ** 2), dtype=np.int32) % self.N[0]
                exact_col_batch = self.exact_int(X[0][i0s], X[0][i0s + 1], X[1][i1s],\
                                    X[1][i1s + 1], x_cap[0][j0s_batch], x_cap[1][j1s_batch])
                row_norm += np.sum(exact_col_batch ** 2)
                row_err += np.sum((operator_unit_vec[batch:batch+BATCH_SIZE] - exact_col_batch) ** 2)
                batch += BATCH_SIZE
        #self._mutex.acquire()
        #self._row_norm += row_norm
        #self._row_err += row_err
        #self._mutex.release()
        return (row_err, row_norm)

    #Возвращает относительную норму ошибки и норму точной матрицы
    def calc_approximation(self):
        print('running calculation of approx')
        #[200~approx_err_exact_norm_400_1_20_chebyshev.npy
        if os.path.isfile(os.path.join(self.parameters_str, f'approx_err_exact_norm_{self.N[0]}_{self.NU_COEFF[0]}_{self.EXACT_DIST}_{self.grid_type}.npy')):
            print('loaded precomputed approx err, exact_norm')
            results = np.load(os.path.join(self.parameters_str, f'approx_err_exact_norm_{self.N[0]}_{self.NU_COEFF[0]}_{self.EXACT_DIST}_{self.grid_type}.npy'))
            return results[0], results[1]
        #self._manager = multiprocessing.Manager()
        #self._mutex = self._manager.Lock()
        #self._row_norm = 0.0
        #self._row_err = 0.0
        if self.N[0] <= 600:
            num_processes = self.num_processes
        elif self.N[0] <= 700:
            num_processes = min(30, self.num_processes)
        elif self.N[0] <= 800:
            num_processes = min(15, self.num_processes)
        elif self.N[0] <= 900:
             num_processes = min(10, self.num_processes)

        if self.N[0] <= 900:
            with multiprocessing.Pool(num_processes) as pool:
                results = pool.map(self.operate_col_approximation, np.arange(self.N[0], dtype=np.int16))
        else:
            results = list(map(self.operate_col_approximation, np.arange(self.N[0], dtype=np.int16)))
        print('approx multiprocessing finished')
        results = np.sqrt(np.sum(results, axis=0))
        results[0] /= results[1]
        err, exact_norm = results[0], results[1]
        #err, exact_norm = self._row_err, self._row_norm
        print('approximation calculated')
        if self.stats_mode:
            self.stats['approx_err'] = '{:.2e}'.format(err / exact_norm)
            self.stats['exact_norm'] = '{:.2e}'.format(exact_norm)
            if not os.path.isdir(self.parameters_str):
                os.makedirs(self.parameters_str)
            np.save(os.path.join(self.parameters_str, f'approx_err_exact_norm_{self.N[0]}_{self.NU_COEFF[0]}_{self.EXACT_DIST}_{self.grid_type}.npy'), results)
        print('finishing calc approx')
        return err, exact_norm

    #@profile
    def solve(self, rhs=None, tol=None, maxiter=None):
        if tol is None:
            tol = min(1e-3, 0.1 / self.N[0])
        if maxiter is None:
            maxiter = max(5000, 5 * self.N[0])
        if rhs is None:
            rhs = np.ones(self.N[0] * self.N[1])

        start_time = time()
        sol, num_iter = self.system_solver(self.operator, rhs, tol=tol, maxiter=maxiter)
        end_time = time()
        if num_iter == maxiter:
            print('maxiter reached')

        if self.stats_mode:
            #self.stats['time'] = np.round(end_time - start_time, 4)
            """
            class gmres_counter(object):
                def __init__(self):
                    self.niter = 0
                def __call__(self, rk=None):
                    self.niter += 1
            iter_counter = gmres_counter()

            start_time = time()
            sol, num_iter = self.system_solver(self.operator, rhs, tol=tol, maxiter=maxiter, callback=iter_counter)
            end_time = time()
            if num_iter == maxiter:
                print('maxiter reached!')
            """
            self.stats['time'] = (end_time - start_time)
            self.stats['tol'] = tol
            self.stats['num_iter'] = num_iter
            self.stats['R_elements'] = '{:.2e}'.format(self.R.nnz)

            mv = self.matrix.mv

            def time_mult():
                return mv(rhs)

            num_iterations, duration = timeit.Timer(stmt='time_mult()', globals=locals()).autorange()
            duration /= num_iterations

            duration = np.round(duration, 6)


            self.stats['mult_time'] = '{:.2e}'.format(duration)
            self.stats['iter_time'] = '{:.2e}'.format(self.stats['time'] / self.stats['num_iter'])
            self.stats['time'] = '{:.2e}'.format(self.stats['time'])
        if self.use_precomputed:
            np.save(os.path.join(f'{self.parameters_str}', f'sol_{self.N[0]}.npy'), sol)

        return sol

        #@njit
        #def make_loc_matrix(x, y):

        #    return np.vstack((np.repeat(x, 4), np.hstack((y, y, y, y)))).T

        #@njit
        #def calc_PKQX(ind):
            #print(rows_all[ind], columns_all[ind
        #    K_loc = np.asarray([get_K_elem(x) for x in product(rows_all[ind], columns_all[ind])]).reshape(4, 4)
        #    return (P_locs_data[ind] @ K_loc) @ QX_locs_data[:, ind]
