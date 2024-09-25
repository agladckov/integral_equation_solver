import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def generate_grid(N, a, b, grid_type='uniform', **kwargs):
    """
    N - количество промежутков, т.е. количество узлов N + 1
    """
    ans = np.empty(N+1, dtype=np.double)
    if grid_type == 'uniform':
        ans = np.linspace(a, b, num=N + 1)
    elif grid_type == 'chebyshev':
        ans = np.array([a + (b - a) * 0.5 * (1 - np.cos(np.pi * i / N)) for i in range(N + 1)])
    elif grid_type == 'exp':
        ans = np.log(np.linspace(np.exp(a) , np.exp(b), num = N + 1))
    elif grid_type == 'power':
        ans = np.sqrt(np.linspace(a ** 2 , b ** 2, num = N + 1))
    return ans

def visualize_grid(N, a, b, **kwargs):
    X = generate_grid(N, a, b, **kwargs)

    fig, ax = plt.subplots(figsize=(15, 0.25))
    ax.set_xlabel('X')
    ax.scatter(X, np.zeros_like(X), s=0.05)
    plt.show()
