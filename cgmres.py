import numpy as np
import scipy.linalg as slinalg
import scipy
import ctypes
import time

def apply_givens_rotation(h, cs, sn, k):
    for i in range(k - 1):
        slinalg.lapack.zrot(h[i:i + 1], h[i + 1:i + 2], cs[i], sn[i], n=1, overwrite_x=1, overwrite_y=1)
    cs_k, sn_k = slinalg.blas.zrotg(h[k - 1], h[k])
    cs_k = cs_k.real

    slinalg.lapack.zrot(h[k - 1:k], h[k:k + 1], cs_k, sn_k, n=1, overwrite_x=1, overwrite_y=1)
    return h, cs_k, sn_k


def gmres(apply_operator, b, tol=1e-5, maxiter=5000, actual_res_config=None, callback=None):
    n = b.shape[0]
    r = b.copy()
    b_norm = np.linalg.norm(b)
    err = np.linalg.norm(r) / b_norm
    sn = np.zeros(maxiter, dtype=np.complex128)
    cs = np.zeros(maxiter, dtype=np.float64)
    e1 = np.zeros(maxiter + 1, dtype=np.complex128)

    e1[0] = 1
    r_norm = np.linalg.norm(r)
    beta = r_norm * e1

    Q = np.empty((n, maxiter + 1), dtype=np.complex128, order='F')
    Q[:, 0] = r / r_norm
    H = np.zeros((maxiter + 1, maxiter), dtype=np.complex128)


    err_hist = []
    actual_err_hist = []
    QCq = np.empty(maxiter + 1, dtype=np.complex128)

    for k in range(maxiter):
        q = apply_operator(Q[:, k])
        for _ in range(2):
            QCq = scipy.linalg.blas.zgemv(1.0, Q[:, :k+1], q, trans=2)
            H[:k+1, k] += QCq[:k+1]
            q = q - Q[:, :k+1] @ QCq[:k+1]

        h = np.linalg.norm(q)
        q = q / h
        H[k + 1, k] = h
        Q[:, k + 1] = q

        h, c, s = apply_givens_rotation(H[:k + 2, k], cs, sn, k + 1)
        H[:k + 2, k] = h
        cs[k] = c
        sn[k] = s

        slinalg.lapack.zrot(beta[k:k + 1], beta[k + 1:k + 2], cs[k], sn[k], n=1, overwrite_x=1, overwrite_y=1)

        err = np.abs(beta[k + 1]) / b_norm
        err_hist.append(err)
        #print(f'{k+1}) {err}')

        if actual_res_config is not None:
            if k % actual_res_config[0] == 0:
                tmp_y = slinalg.solve_triangular(H[:k + 1, :k + 1], beta[:k + 1])
                tmp_y = Q[:, :k + 1] @ tmp_y
                tmp_x = actual_res_config[1](tmp_y)
                curr_res = np.linalg.norm(actual_res_config[2] @ tmp_x - b) / np.linalg.norm(b)
                actual_err_hist.append(curr_res)
                #print(f'Actual error {k+1}) {curr_res}')

        if err <= tol:
            break
    y = slinalg.solve_triangular(H[:k + 1, :k + 1], beta[:k + 1])
    if callback is not None:
        callback()
    return (Q[:, :k + 1] @ y).real, len(err_hist)
    #return Q[:, :k + 1] @ y, err_hist, actual_err_hist
