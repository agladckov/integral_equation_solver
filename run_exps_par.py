import solver, solver_exact, equation_utils_base
import numpy as np
import scipy
import numpy.linalg as LA
from itertools import product
import pandas as pd
import os
import time
from datetime import datetime
import functools
from time import time
import matplotlib.pyplot as plt
print = functools.partial(print, flush=True)
from scipy.interpolate import RegularGridInterpolator
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.pyplot import figure
from matplotlib.ticker import LinearLocator
import multiprocessing
from itertools import chain



equtils = equation_utils_base.equation_utils_base(0.002, 20)
'''
Ns = (50, 100, 200, 300, 40) #500, 600)
NU_coeffs = (1, 1.5, 2) #5, 10, 20, 30)
exact_dists = (10, 15, 20)#(50, 100, 150, 200, 250, 300)
grids = ('chebyshev', )
solvers = ('gmres', )
tols = (1e-2, 1e-3, 1e-4)
'''
Ns = (55, 65, 75) #500, 600)
NU_coeffs = (1,) #5, 10, 20, 30)
exact_dists = (10, )#(50, 100, 150, 200, 250, 300)
grids = ('chebyshev', )
solvers = ('gmres', )
tols = (1e-4,)

parameters = product(Ns, NU_coeffs, exact_dists, grids, solvers)


#Сетка базового точного решения
x_grid_base = np.load(os.path.join('precomputed', '1', 'X_grid.npy'))
y_grid_base = np.load(os.path.join('precomputed', '1', 'Y_grid.npy'))
xy_grid_base = np.meshgrid(x_grid_base, y_grid_base)
exactsol_base = np.load(os.path.join('precomputed', '1', 'base_sol.npy'))
interp = RegularGridInterpolator((x_grid_base, y_grid_base), exactsol_base.reshape(int(exactsol_base.shape[0] ** 0.5),-1),
                                  bounds_error=False, method='linear', fill_value=None)

#Подготовка точных решателей для вычиления правых частей
print('Подготовка rhs')
for N, grid in product(Ns, grids):
    start_time = time()
    print("===============================")
    print(f'Running N: {N}, grid:{grid}')
    esolver = solver_exact.Solver_exact([N] * 2, [1] * 2, equtils, grid=grid, system_solver='gmres',
                                       no_init=True, stats=True, verbose=True, use_precomputed=True)

    xy_grid = np.meshgrid(esolver.x_cap[0], esolver.x_cap[1])
    exactsol = interp((xy_grid[0], xy_grid[1])).flatten()
    print('computed exact sol')
    rhs = esolver.compute_rhs(exactsol)
    print(f'computed rhs, time = {(time() - start_time):.3f}')
    print("===============================")
    del esolver, xy_grid, exactsol, rhs
#Подготовка решателей
print('Подготовка решателей эффективным методом')
for parameter in parameters:
    start_time = time()
    N = parameter[0]
    NU_coeff = parameter[1]
    exact_dist = parameter[2]
    grid = parameter[3]
    system_solver = parameter[4]
    print(f'Running N: {N}, NU_coeff: {NU_coeff}, exact_dist: {exact_dist}, grid:{grid}')
    equtils = equation_utils_base.equation_utils_base(1 / N / NU_coeff, exact_dist)
    effsolver = solver.Solver([N] * 2, [NU_coeff] * 2, exact_dist, equtils, system_solver=system_solver,
                                        grid=grid, debug=False, stats=True, verbose=True, use_precomputed=True)
    print(f'created effsolver, time = {(time() - start_time):.3f}')
    print('approx err, matrix norm =', effsolver.calc_approximation())
    print("===============================")
    del equtils, effsolver


#Выполнение экспериментов
print('Начало экспериментов')
def operate_parameter(parameter):
    #print('cur proc=', multiprocessing.current_process())
    N, NU_coeff, exact_dist, grid, system_solver, tol = parameter
    print(f'Running N: {N}, NU_coeff: {NU_coeff}, exact_dist: {exact_dist}, grid:{grid}')
    equtils = equation_utils_base.equation_utils_base(1 / N / NU_coeff, exact_dist)
    results = []
    effsolver = solver.Solver([N] * 2, [NU_coeff] * 2, exact_dist, equtils, system_solver=system_solver,
                                        grid=grid, debug=False, stats=True, verbose=True, use_precomputed=True)
    print('created effsolver')
    xy_grid = np.meshgrid(effsolver.x_cap[0], effsolver.x_cap[1])
    exactsol = interp((xy_grid[0], xy_grid[1])).flatten()
    parameters_str = os.path.join('precomputed', '1', 'exact', f'[{N}, {N}]_{grid}')
    rhs = np.load(os.path.join(parameters_str, f'rhs_{N}_{grid}.npy'))
    print('loaded precomputed rhs')
    effsol = np.real(effsolver.solve(rhs, tol=tol))
    del rhs, xy_grid
    print('Effective', effsolver.stats)

    relerror = np.round(LA.norm(effsol - exactsol) / LA.norm(exactsol), 4)
    print('effsol err', relerror)

    mult_time_exact = float(np.load(os.path.join(parameters_str, f'mult_time_rhs_{N}_{grid}.npy')))
    result = {'N': N, 'NU_coeff': NU_coeff, 'exact_dist': effsolver.EXACT_DIST,
                    'relerror': relerror, 'approx_err': effsolver.stats['approx_err'], 'fullness': effsolver.stats['fullness'],
                    'mult_time_eff': effsolver.stats['mult_time'], 'mult_time_exact': mult_time_exact,
                    'time_eff': effsolver.stats['time'], 'iter_time_eff': effsolver.stats['iter_time'],
                    'num_iter_eff': effsolver.stats['num_iter'], 'exact_norm': effsolver.stats['exact_norm'],
                    'build_time': effsolver.stats['build_time'], 'grid_type': effsolver.stats['grid_type'],
                    'memory': (float(effsolver.stats['R_elements']) + 8 * N ** 2 + (2 * int(N * NU_coeff) + 1) ** 2) / N ** 4,
                    'tol': tol}
    results.append(result)
    print("===============================")

    parameters_str = os.path.join('precomputed', '1', 'eff', f'{[N, N]}_{[NU_coeff, NU_coeff]}_{effsolver.EXACT_DIST}_{effsolver.grid_type}')
    cur_time = datetime.now().strftime("%Y:%m:%d:%H:%M:%S")
    stats_grid_type = effsolver.stats['grid_type']
    results_file_name = f'res_{N}_{NU_coeff}_{effsolver.EXACT_DIST}_{stats_grid_type}_{cur_time}.csv'
    results_file_name = os.path.join(parameters_str, results_file_name)

    if os.path.isfile(results_file_name):
        df = pd.read_csv(results_file_name)
        df = pd.concat([df, pd.DataFrame([result])])
    else:
        df = pd.DataFrame([result])

    df.to_csv(results_file_name, index=False)

    parameters_str = os.path.join('pictures', '1',f'{N}_{NU_coeff}_{exact_dist}_{grid}')
    if not os.path.isdir(parameters_str):
        os.makedirs(parameters_str)

    grid_symbol = {'chebyshev': 'сетка чебышёвская', 'uniform': 'сетка равномерная'}
    NU_coeff_str = '$\\frac{N_u}{N}$'

    #Сечение точного и приближённого решений
    fig, ax = plt.subplots(figsize=(10,6))
    x_s = effsolver.x_cap[0]
    z_s_exact = exactsol[(N + 1) // 2 * N : ((N + 1) // 2 + 1) * N]
    z_s_eff = effsol[(N + 1) // 2 * N : ((N + 1) // 2 + 1) * N]
    plt.scatter(x_s, z_s_exact, s=2, c='red', label='точное решение')
    plt.scatter(x_s, z_s_eff, s=2, c='blue', label='приближённое решение')
    ax.set_title(f'Сечение y = {effsolver.x_cap[0][(N + 1) // 2]:.2f}, \n N={N}, {NU_coeff_str}={NU_coeff:.1f}, ' + \
                 f'$D_e$={exact_dist}, {grid_symbol[grid]}', fontsize=14)
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('U', fontsize=14)
    plt.savefig(os.path.join(parameters_str, f'exactsol_1D_{N}.png'))
    plt.close()

    """
    #Сечение точного решения
    fig, ax = plt.subplots(figsize=(10,6))
    x_s = effsolver.x_cap[0]
    z_s = exactsol[(N + 1) // 2 * N : ((N + 1) // 2 + 1) * N]
    plt.scatter(x_s, z_s, s=2, c='red')
    ax.set_title(f'Точное решение, сечение y = {effsolver.x_cap[0][(N + 1) // 2]:.2f}, \n N={N}, {NU_coeff_str}={NU_coeff:.1f}, ' + \
                 f'$D_e$={exact_dist}, {grid_symbol[grid]}', fontsize=14)
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('U', fontsize=14)
    plt.savefig(os.path.join(parameters_str, f'exactsol_1D_{N}.png'))
    plt.close()

    #Сечение приближённого решения
    fig, ax = plt.subplots(figsize=(10,6))
    x_s = effsolver.x_cap[0]
    z_s = effsol[(N + 1) // 2 * N : ((N + 1) // 2 + 1) * N]
    plt.scatter(x_s, z_s, s=2, c='red')
    ax.set_title(f'Решение эффективным методом, сечение y = {effsolver.x_cap[0][(N + 1) // 2]:.2f},\n N={N}, {NU_coeff_str}={NU_coeff:.1f}, ' + \
                 f'$D_e$={exact_dist}, {grid_symbol[grid]}', fontsize=14)
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('U', fontsize=14)
    plt.savefig(os.path.join(parameters_str, f'effsol_1D_{N}.png'))
    plt.close()
    """

    ax = plt.figure(figsize=(10,8)).add_subplot(projection='3d')
    ax.set_title(f'Референсное решение,\n y = {effsolver.x_cap[0][(N + 1) // 2]:.2f}, N={N}, {NU_coeff_str}={NU_coeff:.1f}, ' + \
                 f'$D_e$={exact_dist}, {grid_symbol[grid]}', fontsize=16)
    g = ax.contour3D(effsolver.x_cap[0], effsolver.x_cap[1], exactsol.reshape(int(exactsol.shape[0] ** 0.5), -1), 20, cmap=cm.jet)
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    ax.set_zlabel('Z', fontsize=14)
    #fig.colorbar(g, ax=ax)
    plt.savefig(os.path.join(parameters_str, f'exactsol_3D_{N}.png'))
    plt.close()

    ax = plt.figure(figsize=(10,8)).add_subplot(projection='3d')
    ax.set_title(f'Решение эффективным методом,\n y = {effsolver.x_cap[0][(N + 1) // 2]:.2f}, N={N}, {NU_coeff_str}={NU_coeff:.1f}, ' + \
                 f'$D_e$={exact_dist}, {grid_symbol[grid]}', fontsize=16)
    g = ax.contour3D(effsolver.x_cap[0], effsolver.x_cap[1], effsol.reshape(int(effsol.shape[0] ** 0.5), -1), 20, cmap=cm.jet)
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    ax.set_zlabel('Z', fontsize=14)
    #fig.colorbar(g, ax=ax)
    plt.savefig(os.path.join(parameters_str, f'effsol_3D_{N}.png'))
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f'Референсное решение,\n y = {effsolver.x_cap[0][(N + 1) // 2]:.2f}, N={N}, {NU_coeff_str}={NU_coeff:.1f}, ' + \
                 f'$D_e$={exact_dist}, {grid_symbol[grid]}', fontsize=15)
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    g = ax.imshow(effsol.reshape(int(exactsol.shape[0] ** 0.5), -1))
    fig.colorbar(g, ax=ax)
    plt.savefig(os.path.join(parameters_str, f'exactsol_2D_{N}.png'))
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f'Решение эффективным методом,\n y = {effsolver.x_cap[0][(N + 1) // 2]:.2f}, N={N}, {NU_coeff_str}={NU_coeff:.1f}, ' + \
                 f'$D_e$={exact_dist}, {grid_symbol[grid]}', fontsize=15)
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    g = ax.imshow(effsol.reshape(int(effsol.shape[0] ** 0.5), -1))
    fig.colorbar(g, ax=ax)
    plt.savefig(os.path.join(parameters_str, f'effsol_2D_{N}.png'))
    plt.close()
    del effsolver
    return results

parameters = list(product(Ns, NU_coeffs, exact_dists, grids, solvers, tols))

if 'SLURM_NTASKS' in os.environ:
    num_processes = int(os.environ['SLURM_NTASKS'])
else:
    num_processes = os.cpu_count()
with multiprocessing.Pool(num_processes) as pool:
    results = pool.map(operate_parameter, parameters)
results = list(chain.from_iterable(results))

results_file_name = 'R1709.csv'
if os.path.isfile(results_file_name):
    df = pd.read_csv(results_file_name)
    df = pd.concat([df, pd.DataFrame(results)])
else:
    df = pd.DataFrame(results)

df.to_csv(results_file_name, index=False)
