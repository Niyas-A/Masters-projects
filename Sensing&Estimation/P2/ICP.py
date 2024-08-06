import numpy as np
from scipy.spatial import KDTree

def icp_iter(M, Z):
    assert len(M) == len(Z)
    mean_M = np.mean(M, axis=0)
    mean_Z = np.mean(Z, axis=0)
    delM = M - mean_M
    delZ = Z - mean_Z
    Q = np.dot(delM.T, delZ)
    U, S, Vt = np.linalg.svd(Q)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
       Vt[2,:] *= -1
       R = np.dot(Vt.T, U.T)
    t = mean_Z.T - np.dot(R, mean_M.T)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def icp(Source, Target, T_init=None, max_iterations=20, tolerance=0.001):
    src = np.ones((4, Source.shape[0]))
    dst = np.ones((4, Target.shape[0]))
    src[:3, :] = np.copy(Source.T)
    dst[:3, :] = np.copy(Target.T)
    if T_init is not None:
        src = np.dot(T_init, src)
    prev_error = 0
    for i in range(max_iterations):
        dst_tree = KDTree(dst[:3, :].T)
        distances, indices = dst_tree.query(src[:3, :].T, k=1)
        T = icp_iter(src[:3, :].T, dst[:3, indices].squeeze().T)
        src = np.dot(T, src)
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    T = icp_iter(Source, src[:3, :].T)
    return T, distances, i + 1