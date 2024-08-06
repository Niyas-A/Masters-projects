# generate the probability matrix
import numpy as np
import utils
import scipy.stats as stats
from tqdm import tqdm
import os
pf_filename = "pf_matrix.npy"

horizon = 1

def create_grid(nx, ny, n_theta):
    x_range = np.linspace(-3, 3, nx)
    y_range = np.linspace(-3, 3, ny)
    theta_range = np.linspace(-np.pi, np.pi, n_theta)
    return x_range, y_range, theta_range

def find_grid_index(x, y, theta, x_range, y_range, theta_range):
    ix = np.searchsorted(x_range, x, side='right') - 1
    iy = np.searchsorted(y_range, y, side='right') - 1
    itheta = np.searchsorted(theta_range, theta, side='right') - 1
    return ix, iy, itheta


def initialize_matrix(nx, ny, n_theta, nv, n_omega):
    return np.zeros((nx, ny, n_theta, nv, n_omega, 8, 4))


def compute_transition_probabilities(x_range, y_range, theta_range, sigma):
    # Define standard deviations for each state variable
    sigma_x, sigma_y, sigma_theta = sigma

    # Calculate total grid points
    nx, ny, n_theta = len(x_range), len(y_range), len(theta_range)

    # Create an empty matrix
    transition_matrix = np.zeros((nx, ny, n_theta, 8, 4))

    # Neighbor offsets for the 8 surrounding cells + current cell
    neighbor_offsets = [(dx, dy, dtheta) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dtheta in (-1, 0, 1) if
                        not (dx == 0 and dy == 0 and dtheta == 0)]

    # Iterate through each discretized state
    for ix in range(nx):
        for iy in range(ny):
            for itheta in range(n_theta):
                current_state = (x_range[ix], y_range[iy], theta_range[itheta])
                probabilities = []
                states = []

                # Calculate probabilities for each neighboring state
                for (dx, dy, dtheta) in neighbor_offsets:
                    neighbor_index = (ix + dx, iy + dy, itheta + dtheta)
                    if 0 <= neighbor_index[0] < nx and 0 <= neighbor_index[1] < ny and 0 <= neighbor_index[2] < n_theta:
                        neighbor_state = (
                        x_range[neighbor_index[0]], y_range[neighbor_index[1]], theta_range[neighbor_index[2]])
                        # Calculate the Gaussian probability density
                        probability = stats.multivariate_normal.pdf(neighbor_state, mean=current_state, cov=np.diag(
                            [sigma_x ** 2, sigma_y ** 2, sigma_theta ** 2]))
                        probabilities.append(probability)
                        states.append(neighbor_state)

                # Normalize probabilities
                total_prob = sum(probabilities)
                normalized_probabilities = [p / total_prob for p in probabilities]

                # Populate transition matrix
                for j, (prob, state) in enumerate(zip(normalized_probabilities, states)):
                    transition_matrix[ix, iy, itheta, j, 0:3] = state
                    transition_matrix[ix, iy, itheta, j, 3] = prob

    return transition_matrix


def error_model(t, err_t, u_t):
    cur_ref_state = utils.lissajous(t)
    cur_state = err_t + cur_ref_state
    next_state = utils.car_next_state(t, cur_state, u_t, noise=False)
    next_ref_state = utils.lissajous(t + 1)
    err_t1 = next_state - next_ref_state
    return err_t1

def nearest(err_t, r_x, r_y, r_theta):
    err_t_near = []
    err_t_near.append(err_t)
    err_t_near.append(err_t + [r_x, 0, 0])
    err_t_near.append(err_t + [-r_x, 0, 0])
    err_t_near.append(err_t + [0, r_y, 0])
    err_t_near.append(err_t + [0, -r_y, 0])
    err_t_near.append(err_t + [0, 0, r_theta])
    err_t_near.append(err_t + [0, 0, -r_theta])
    return np.array(err_t_near)


def find_grid_positions(points, x_range, y_range, theta_range):
    # find the nearest grid index
    # enforce wrap around
    def nearest_index(value, grid):
        return np.argmin(np.abs(grid - value))

    grid_positions = []
    for point in points:
        x_idx = nearest_index(point[0], x_range)
        y_idx = nearest_index(point[1], y_range)
        theta_idx = nearest_index(point[2], theta_range)
        grid_positions.append((x_idx, y_idx, theta_idx))

    return grid_positions



    # x_range, y_range, theta_range = create_grid(nx, ny, n_theta)
    # print(x_range, y_range, theta_range)

def generate_pf(self, x=5, ny=5, n_theta=5, nv=5, nw=5):

    pf = np.zeros((nx, ny, n_theta, nv, nw, 7, 4))
    print(pf.shape)
    t = 0
    for i, vi in tqdm(enumerate(v_range)):
        for j, wi in enumerate(w_range):
            for k, xi in enumerate(x_range):
                for l, yi in enumerate(y_range):
                    for m, thetai in enumerate(theta_range):
                        err_t = np.array([xi, yi, thetai])
                        u_t = np.array([vi, wi])
                        # print(err_t, u_t)
                        err_t1 = error_model(t, err_t, u_t)
                        # print(err_t1)
                        # find 6 nearest neighbours
                        err_t_near = nearest(err_t1, r_x, r_y, r_theta)
                        # print(err_t_near)
                        grid_positions = find_grid_positions(err_t_near, x_range, y_range, theta_range)
                        # print(grid_positions)
                        probabilities = []
                        for g in grid_positions:
                            g_e = [x_range[g[0]], y_range[g[1]], theta_range[g[2]]]
                            probability = stats.multivariate_normal.pdf(g_e, mean=err_t1, cov=np.diag(
                                [sigma_x ** 2, sigma_y ** 2, sigma_theta ** 2]))
                            probabilities.append(probability)
                            # print(g_e,probability)
                        total_probability = sum(probabilities)
                        normalized_probabilities = [p / total_probability for p in probabilities]
                        # print(normalized_probabilities)
                        for q, g in enumerate(grid_positions):
                            pf[k, l, m, i, j, q, :] = [g[0], g[1], g[2], normalized_probabilities[q]]
                        # print(x_range[grid_positions[0][0]],y_range[grid_positions[0][1]],theta_range[grid_positions[0][2]])
                        #
    return pf


def generalizedPI():
    nx = 5
    ny = 5
    n_theta = 5
    nv = 5
    nw = 5
    x_min = -3
    x_max = 3
    y_min = -3
    y_max = 3
    theta_min = -np.pi
    theta_max = np.pi
    v_min = 0
    v_max = 1
    w_min = -1
    w_max = 1
    r_x = 6 / nx
    r_y = 6 / ny
    r_theta = 2 * np.pi / n_theta
    sigma_x = 0.04
    sigma_y = 0.04
    sigma_theta = 0.004

    x_range = np.linspace(x_min, x_max, nx)
    y_range = np.linspace(y_min, y_max, ny)
    theta_range = np.linspace(theta_min, theta_max, n_theta)
    v_range = np.linspace(v_min, v_max, nv)
    w_range = np.linspace(w_min, w_max, nw)



    if os.path.exists(pf_filename):
        pf = np.load(pf_filename)
        print("Loaded pf matrix from file.")
    else:
        pf = generate_pf(nx, ny, n_theta, nv, nw)
        np.save(pf_filename, pf)
        print("Saved pf matrix to file.")

    print('Policy Iteration:')

    def nearest_index(value, grid):
        return np.argmin(np.abs(grid - value))

    def H_function(err_t, u_t):
        Q = np.eye(2)
        R = np.eye(2)
        q = 1.0
        gamma = 0.9
        cost = err_t[0:2].T @ Q @ err_t[0:2] + q * (1 - np.cos(err_t[2])) ** 2 + u_t.T @ R @ u_t
        # convert err_t and u_t to index
        xj = nearest_index(err_t[0], x_range)
        yj = nearest_index(err_t[1], y_range)
        thetaj = nearest_index(err_t[2], theta_range)
        vj = nearest_index(u_t[0], v_range)
        wj = nearest_index(u_t[1], w_range)
        neighbors = pf[xj, yj, thetaj, vj, wj, :, 0:3]
        probabilities = pf[xj, yj, thetaj, vj, wj, :, 3]
        for s, n in enumerate(neighbors):
            xs = nearest_index(n[0], x_range)
            ys = nearest_index(n[1], y_range)
            thetas = nearest_index(n[2], theta_range)
            # print('DEBUG :', n)
            cost += gamma * probabilities[s] * V[xs, ys, thetas]

        return cost

    # Policy iteration
    x_min = -3
    x_max = 3
    y_min = -3
    y_max = 3
    theta_min = -np.pi
    theta_max = np.pi
    v_min = 0
    v_max = 1
    w_min = -1
    w_max = 1
    r_x = 6/nx
    r_y = 6/ny
    r_theta = 2*np.pi/n_theta
    sigma_x = 0.04
    sigma_y = 0.04
    sigma_theta = 0.004

    x_range = np.linspace(x_min, x_max, nx)
    y_range = np.linspace(y_min, y_max, ny)
    theta_range = np.linspace(theta_min, theta_max, n_theta)
    v_range = np.linspace(v_min, v_max, nv)
    w_range = np.linspace(w_min, w_max, nw)

    V = np.zeros((nx, ny, n_theta, 1))
    V_new = np.zeros((nx, ny, n_theta, 1))
    V_iter = np.zeros((nx, ny, n_theta, 1))
    Q = np.zeros((nx, ny, n_theta, nv, nw, 1))
    pi = np.zeros((nx, ny, n_theta, 2))
    gamma = 0.9
    max_iter = 5
    eval_iter = 2

    for k1 in range(max_iter):
        # policy improvement
        for k, xi in enumerate(x_range):
            for l, yi in enumerate(y_range):
                for m, thetai in enumerate(theta_range):
                    for i, vi in enumerate(v_range):
                        for j, wi in enumerate(w_range):
                            err_t = [xi, yi, thetai]
                            u_t = [vi, wi]
                            cost = H_function(np.array(err_t), np.array(u_t))
                            Q[k, l, m, i, j, 0] = cost
                    # print('DEBUG :', Q[k, l, m, :, :])
                    # print('DEBUG :', np.unravel_index(np.argmin(Q[k, l, m, :, :]), (nv, nw)))
                    print("DEBUG: Q :", Q[k, l, m, :, :, 0])
                    pi[k, l, m, :] = np.unravel_index(np.argmin(Q[k, l, m, :, :, 0]), (nv, nw))
                    print(f"DEBUG: pi at {k}, {l}, {m} = ", pi[k, l, m, :])
                    print("DEBUG: control", v_range[int(pi[k, l, m, 0])], w_range[int(pi[k, l, m, 1])])
                    # v_range[pi[k, l, m, 0]], v_range[pi[k, l, m, 1]]
                    #V[xi, yi, thetai, 1] = np.min(Q[xi, yi, thetai, :, :])
        # policy evaluation
        for k2 in range(eval_iter):
            for k, xi in enumerate(x_range):
                for l, yi in enumerate(y_range):
                    for m, thetai in enumerate(theta_range):
                        err_t = [xi, yi, thetai]
                        u_t = pi[k, l, m, :]
                        cost = H_function(np.array(err_t), np.array(u_t))
                        vi = nearest_index(u_t[0], v_range)
                        wi = nearest_index(u_t[1], w_range)
                        Q[k, l, m, vi, wi, 0] = cost
                        V_iter[k, l, m, 0] = cost
            V_new = np.copy(V_iter)
        if np.array_equal(V, V_new):
            print('V equal V_new at k = ',k)
            break
        else:
            V = np.copy(V_new)

    # print('V: ', V.T)
    # print('pi: ', pi.T)
    return None

def main():
    generalizedPI()
    pass


if __name__ == '__main__':
    main()
