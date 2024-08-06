# generate the probability matrix
import numpy as np
import utils
import scipy.stats as stats
from tqdm import tqdm
import os
from time import time

class generalizedPI():
    def __init__(self, nx, ny, n_theta, nv, nw, set_range):
        self.nx = nx
        self.ny = ny
        self.n_theta = n_theta
        self.nv = nv
        self.nw = nw
        self.x_min = -3
        self.x_max = 3
        self.y_min = -3
        self.y_max = 3
        self.theta_min = -np.pi
        self.theta_max = np.pi
        self.v_min = 0
        self.v_max = 1
        self.w_min = -1
        self.w_max = 1
        self.r_x = (self.x_max - self.x_min) / self.nx
        self.r_y = (self.y_max - self.y_min) / self.ny
        self.r_theta = (self.theta_max - self.theta_min) / self.n_theta
        self.sigma_x = 0.04
        self.sigma_y = 0.04
        self.sigma_theta = 0.004
        if set_range:
            self.x_range = np.array([-3, -1.5, -0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9, 1.5, 3])
            self.y_range = np.array([-3, -1.5, -0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9, 1.5, 3])
            self.theta_range = np.pi * np.array([-1, -1/2, -1/4, -1/6, -1/12, 0, 1/12, 1/6, 1/4, 1/2, 1])
            self.v_range = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
            self.w_range = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
            if self.x_range.shape[0] != self.nx or self.y_range.shape[0] != self.ny or self.theta_range.shape[0] != self.n_theta or self.v_range.shape[0] != self.nv or self.w_range.shape[0] != self.nw:
                print('DEBUG: dimensions mismatch')
                raise NotImplementedError
        else:
            self.x_range = np.linspace(self.x_min, self.x_max, self.nx)
            self.y_range = np.linspace(self.y_min, self.y_max, self.ny)
            self.theta_range = np.linspace(self.theta_min, self.theta_max, self.n_theta)
            self.v_range = np.linspace(self.v_min, self.v_max, self.nv)
            self.w_range = np.linspace(self.w_min, self.w_max, self.nw)
        self.pf = np.zeros((self.nx, self.ny, self.n_theta, self.nv, self.nw, 7, 4))
        self.gamma = 0.9
        self.max_iter = 100
        self.eval_iter = 10


    def find_index(self, value, grid):
        return np.argmin(np.abs(grid - value))

    def load_pf(self, t):
        pf_filename = f"saved/trial2/pf_matrix{t}.npy"
        if os.path.exists(pf_filename):
            pf = np.load(pf_filename)
            print("Loaded pf matrix from file.")
        else:
            pf = self.generate_pf(t)
            np.save(pf_filename, pf)
            print("Saved pf matrix to file.")
        return pf

    def nearest_index(self, value, grid):
        return np.argmin(np.abs(grid - value))

    def test_control(self, nt):
        # Obstacles in the environment
        obstacles = np.array([[-2, -2, 0.5], [1, 2, 0.5]])
        # Params
        traj = utils.lissajous
        ref_traj = []
        error_trans = 0.0
        error_rot = 0.0
        car_states = []
        times = []
        # Start main loop
        main_loop = time()  # return time in sec
        # Initialize state
        cur_state = np.array([utils.x_init, utils.y_init, utils.theta_init])
        cur_iter = 0
        pf_filename = f"saved/trial2/Pi_matrix.npy"
        if os.path.exists(pf_filename):
            Pi = np.load(pf_filename)
            print("Loaded pf matrix from file.")
        else:
            print("file not found.")
            raise NotImplementedError
        # Main loop
        while cur_iter * utils.time_step < min(utils.sim_time, nt * utils.time_step):
            t1 = time()
            # Get reference state
            cur_time = cur_iter * utils.time_step
            cur_ref = traj(cur_iter)
            # Save current state and reference state for visualization
            ref_traj.append(cur_ref)
            car_states.append(cur_state)

            ################################################################
            # Generate control input
            # TODO: Replace this simple controller with your own controller
            cur_err = cur_state - cur_ref
            xi = self.find_index(cur_err[0], self.x_range)
            yi = self.find_index(cur_err[1], self.y_range)
            thetai = self.find_index(cur_err[2], self.theta_range)
            control = Pi[cur_iter % 100, xi, yi, thetai, :]
            # control = fcec(cur_iter, cur_state, cur_ref)
            # control = utils.simple_controller(cur_state, cur_ref)
            print("[v,w]", control)
            ################################################################

            # Apply control input
            next_state = utils.car_next_state(utils.time_step, cur_state, control, noise=True)
            # Update current state
            cur_state = next_state
            # Loop time
            t2 = utils.time()
            print(cur_iter)
            print(t2 - t1)
            times.append(t2 - t1)
            cur_err = cur_state - cur_ref
            cur_err[2] = np.arctan2(np.sin(cur_err[2]), np.cos(cur_err[2]))
            error_trans = error_trans + np.linalg.norm(cur_err[:2])
            error_rot = error_rot + np.abs(cur_err[2])
            print(cur_iter, cur_err, error_trans, error_rot)
            print("======================")
            cur_iter = cur_iter + 1

        main_loop_time = time()
        print("\n\n")
        print("Total time: ", main_loop_time - main_loop)
        print("Average iteration time: ", np.array(times).mean() * 1000, "ms")
        print("Final error_trains: ", error_trans)
        print("Final error_rot: ", error_rot)

        # Visualization
        ref_traj = np.array(ref_traj)
        car_states = np.array(car_states)
        times = np.array(times)
        utils.visualize(car_states, ref_traj, obstacles, times, utils.time_step, save=False)
        pass

    def lissajous(self, k, T, time_step):
        xref_start = 0
        yref_start = 0
        A = 2
        B = 2
        a = 2 * np.pi / (T * time_step)
        b = 3 * a
        delta = np.pi / 2

        k = k % T

        xref = xref_start + A * np.sin(a * k * time_step + delta)
        yref = yref_start + B * np.sin(b * k * time_step)
        vx = A * a * np.cos(a * k * time_step + delta)
        vy = B * b * np.cos(b * k * time_step)
        thetaref = np.arctan2(vy, vx)

        return np.vstack((xref, yref, thetaref)).T

    def G_function(self, time_step, X, U, X_ref, t):
        theta = X[:, 2]
        x_ref = X_ref[t, :]
        # print(x_ref)
        # print(x_ref[2])
        cos_theta = np.cos(theta+x_ref[2])
        sin_theta = np.sin(theta+x_ref[2])

        G = np.zeros((X.shape[0], 3, 2))
        G[:, 0, 0] = cos_theta
        G[:, 1, 0] = sin_theta
        G[:, 2, 1] = 1

        # Transform control inputs
        f = np.einsum('ijk,lk->ijl', G, U)
        # print(f.shape)

        X_expanded = np.repeat(X[:, :, np.newaxis], U.shape[0], axis=2)
        # print(X_expanded.shape)

        ref_diff = X_ref[t % 100, :] - X_ref[(t+1) % 100, :]
        ref_diff = ref_diff[np.newaxis, :, np.newaxis]
        # print(ref_diff.shape)

        # Compute new state
        X_new = X_expanded + ref_diff + f * time_step

        return X_new

    def generate_pf2(self, t):
        # initialize pf
        pf = np.zeros((self.nx, self.ny, self.n_theta, self.nv, self.nw, 7, 4))
        # create X
        x_mesh, y_mesh, theta_mesh = np.meshgrid(self.x_range, self.y_range, self.theta_range)
        X = np.vstack((x_mesh.ravel(), y_mesh.ravel(), theta_mesh.ravel())).T
        # create U
        v_mesh, w_mesh = np.meshgrid(self.v_range, self.w_range)
        U = np.vstack((v_mesh.ravel(), w_mesh.ravel())).T
        # create ref state
        t_range = np.linspace(0, 99, num=100)
        X_ref = self.lissajous(t_range, utils.T, utils.time_step)
        X_new = self.G_function(utils.time_step, X, U, X_ref, t)
        X_new = np.transpose(X_new, (0, 2, 1))

        # gernerate new 6 points
        original_points = X_new.reshape(self.nx*self.ny*self.n_theta, self.nv*self.nw, 1, 3)
        perturbations = np.array([
            [0, 0, 0],
            [self.r_x, 0, 0],
            [-self.r_x, 0, 0],
            [0, self.r_y, 0],
            [0, -self.r_y, 0],
            [0, 0, self.r_theta],
            [0, 0, -self.r_theta]
        ])
        all_points = original_points + perturbations
        all_points[:, :, :, 2] = np.mod(all_points[:, :, :, 2] + np.pi, 2 * np.pi) - np.pi
        x_indices = np.digitize(all_points[..., 0], self.x_range) - 1
        y_indices = np.digitize(all_points[..., 1], self.y_range) - 1
        theta_indices = np.digitize(all_points[..., 2], self.theta_range) - 1
        x_indices = np.clip(x_indices, 0, self.nx - 1)
        y_indices = np.clip(y_indices, 0, self.ny - 1)
        theta_indices = np.clip(theta_indices, 0, self.n_theta - 1)
        all_points[..., 0] = self.x_range[x_indices]
        all_points[..., 1] = self.y_range[y_indices]
        all_points[..., 2] = self.theta_range[theta_indices]
        sigma = [0.04/np.square(self.r_x), 0.04/np.square(self.r_y), 0.004/np.square(self.r_theta)]
        # probabilities = stats.norm.pdf(all_points, loc=original_points, scale=sigma).prod(axis=3)
        # pf = np.concatenate([all_points, probabilities[..., np.newaxis]], axis=3)
        prob_x = stats.norm.pdf(all_points[..., 0], loc=original_points[..., 0], scale=sigma[0])
        prob_y = stats.norm.pdf(all_points[..., 1], loc=original_points[..., 1], scale=sigma[1])
        prob_theta = stats.norm.pdf(all_points[..., 2], loc=original_points[..., 2], scale=sigma[2])
        probabilities = prob_x * prob_y * prob_theta
        probabilities = probabilities.reshape(1210, 66, 7)
        prob_sums = probabilities.sum(axis=2, keepdims=True)
        zero_sum_mask = (prob_sums == 0)
        print(np.sum(zero_sum_mask))
        prob_sums[zero_sum_mask] = 1
        normalized_probabilities = probabilities / prob_sums
        pf = np.concatenate([all_points, normalized_probabilities[..., np.newaxis]], axis=3)
        self.pf = pf
        return pf

    def nearest_index_vectorized(self, values, grid):
        """ Vectorized nearest index finding in a grid. """
        indices = np.searchsorted(grid, values) - 1
        indices = np.clip(indices, 0, len(grid) - 1)
        return indices

    def H_function2(self, t, V):
        # V = np.zeros((self.nx, self.ny, self.n_theta, 1))
        V = V.reshape((self.nx, self.ny, self.n_theta, 1))
        # print(self.pf.shape)
        # create X
        x_mesh, y_mesh, theta_mesh = np.meshgrid(self.x_range, self.y_range, self.theta_range)
        X = np.vstack((x_mesh.ravel(), y_mesh.ravel(), theta_mesh.ravel())).T
        # create U
        v_mesh, w_mesh = np.meshgrid(self.v_range, self.w_range)
        U = np.vstack((v_mesh.ravel(), w_mesh.ravel())).T

        Q = np.eye(2)
        R = np.eye(2)
        q = 1.0
        gamma = 0.9
        cos_err_t2 = np.cos(X[:, 2])
        err_t = np.repeat(X[:, np.newaxis, :], U.shape[0], axis=1)
        # print(err_t.shape)
        u_t = np.repeat(U[np.newaxis, :, :], X.shape[0], axis=0)
        quad_err_t = np.einsum('ijk,kl,ijl->ij', err_t[:, :, :2], Q, err_t[:, :, :2])
        cos_err_t2 = cos_err_t2[:, np.newaxis]
        linear_err_t = q * (1 - cos_err_t2) ** 2
        quad_u_t = np.einsum('ijk,kl,ijl->ij', u_t, R, u_t)
        cost = quad_err_t + linear_err_t + quad_u_t
        # print("DEBUG: cost", cost.shape)
        stage_cost = cost.reshape(self.nx, self.ny, self.n_theta, self.nv, self.nw)
        x_indices = self.nearest_index_vectorized(err_t[..., 0], self.x_range)
        # print(x_indices.shape)
        y_indices = self.nearest_index_vectorized(err_t[..., 1], self.y_range)
        theta_indices = self.nearest_index_vectorized(err_t[..., 2], self.theta_range)
        v_indices = self.nearest_index_vectorized(u_t[..., 0], self.v_range)
        w_indices = self.nearest_index_vectorized(u_t[..., 1], self.w_range)
        neighbors = self.pf[x_indices, y_indices, theta_indices, v_indices, w_indices, :, :3]
        probabilities = self.pf[x_indices, y_indices, theta_indices, v_indices, w_indices, :, 3]
        # print("DEBUG: probabilities", probabilities.shape)
        xs, ys, thetas = self.nearest_index_vectorized(neighbors[..., 0], self.x_range), \
            self.nearest_index_vectorized(neighbors[..., 1], self.y_range), \
            self.nearest_index_vectorized(neighbors[..., 2], self.theta_range)
        # print("DEBUG: xs", xs.shape)
        future_values = V[xs, ys, thetas]
        # print("DEBUG: future_values", future_values.shape)
        expected_future_cost = gamma * np.einsum('ijk,ijkl->ij', probabilities, future_values)
        # print("DEBUG: expected_future_cost", expected_future_cost.shape)
        cost += expected_future_cost
        return cost

    def generate_stage_cost(self):
        # create X
        x_mesh, y_mesh, theta_mesh = np.meshgrid(self.x_range, self.y_range, self.theta_range)
        X = np.vstack((x_mesh.ravel(), y_mesh.ravel(), theta_mesh.ravel())).T
        # create U
        v_mesh, w_mesh = np.meshgrid(self.v_range, self.w_range)
        U = np.vstack((v_mesh.ravel(), w_mesh.ravel())).T

        Q = np.eye(2)
        R = np.eye(2)
        q = 1.0
        gamma = 0.9
        cos_err_t2 = np.cos(X[:, 2])
        err_t = np.repeat(X[:, np.newaxis, :], U.shape[0], axis=1)
        u_t = np.repeat(U[np.newaxis, :, :], X.shape[0], axis=0)
        quad_err_t = np.einsum('ijk,kl,ijl->ij', err_t[:, :, :2], Q, err_t[:, :, :2])
        cos_err_t2 = cos_err_t2[:, np.newaxis]
        linear_err_t = q * (1 - cos_err_t2) ** 2
        quad_u_t = np.einsum('ijk,kl,ijl->ij', u_t, R, u_t)
        cost = quad_err_t + linear_err_t + quad_u_t
        cost = cost.reshape(self.nx, self.ny, self.n_theta, self.nv, self.nw)
        return cost

    def PI2(self, t):
        # Policy iteration
        print('Policy Iteration:')
        self.pf = self.load_pf(t)

        V = np.zeros((self.nx, self.ny, self.n_theta, 1))
        V_new = np.zeros((self.nx, self.ny, self.n_theta, 1))
        V_iter = np.zeros((self.nx, self.ny, self.n_theta, 1))
        Q = np.zeros((self.nx, self.ny, self.n_theta, self.nv, self.nw, 1))
        pi = np.zeros((self.nx, self.ny, self.n_theta, 2))
        # control_space =
        v_mesh, w_mesh = np.meshgrid(self.v_range, self.w_range)
        U = np.vstack((v_mesh.ravel(), w_mesh.ravel())).T

        V = np.zeros((self.nx * self.ny * self.n_theta, 1))
        V_new = np.zeros((self.nx * self.ny * self.n_theta, 1))
        V_iter = np.zeros((self.nx * self.ny * self.n_theta, 1))
        for k1 in range(self.max_iter):
            cost = self.H_function2(t, V) # shape (1210, 66)
            # print("DEBUG: cost ", cost.shape)
            # check if there is any nan
            control_indices = np.argmin(cost, axis=-1)
            # print("DEBUG: control_indices ", control_indices.shape)
            # print(control_indices)
            # print("DEBUG: U ", U.shape)
            pi = U[control_indices, :]
            # print("DEBUG: pi", pi.shape)
            # policy evaluation
            for k2 in range(self.eval_iter):
                cost = self.H_function2(t, V_new)
                row_indices = np.arange(cost.shape[0])
                V_iter = cost[row_indices, control_indices]
                # print("DEBUG: V_iter", V_iter.shape)
                V_new = np.copy(V_iter)
            # if np.array_equal(V, V_new):
            # print("DEBUG: V_new", np.abs(V_new))
            delta_V = np.sum(np.abs(V_new - V))
            print(f'Iteration {k1}: Change in V: {delta_V}')
            if delta_V < 1e-4:
                print('Convergence achieved at iteration: ', k1)
                break
            else:
                V = np.copy(V_new)
        # print("DEBUG: pi", pi)
        return pi

    def PI3(self, t):
        print('Policy Iteration:')
        self.pf = self.load_pf(t)

        # Load previous policy if exists, otherwise initialize randomly
        prev_pi_filename = f"saved/trial2/pi_matrix{t - 1}.npy"
        if t > 0 and os.path.exists(prev_pi_filename):
            pi = np.load(prev_pi_filename)
            print("Loaded previous policy from file.")
        else:
            pi = np.random.choice(self.nv * self.nw, size=(self.nx, self.ny, self.n_theta, 1))
            print("Initialized random policy.")

        # Reshape pi to match the U index directly if needed
        v_mesh, w_mesh = np.meshgrid(self.v_range, self.w_range)
        U = np.vstack((v_mesh.ravel(), w_mesh.ravel())).T

        # Initialize V based on the existing or random policy
        V = np.zeros((self.nx * self.ny * self.n_theta, 1))
        for k2 in range(self.eval_iter):
            cost = self.H_function2(t, V, pi)  # Adjusted to pass pi
            row_indices = np.arange(cost.shape[0])
            control_indices = pi.flatten()  # Ensure pi is being used properly
            V = cost[row_indices, control_indices].reshape(-1, 1)

        V_new = np.zeros_like(V)

        for k1 in range(self.max_iter):
            cost = self.H_function2(t, V)  # shape (1210, 66)
            control_indices = np.argmin(cost, axis=-1)
            pi = U[control_indices, :]

            # Policy evaluation
            for k2 in range(self.eval_iter):
                cost = self.H_function2(t, V_new, pi)  # Adjusted to pass pi
                row_indices = np.arange(cost.shape[0])
                V_new = cost[row_indices, control_indices].reshape(-1, 1)

            # Convergence check based on the sum of absolute changes in V
            delta_V = np.sum(np.abs(V_new - V))
            print(f'Iteration {k1}: Change in V: {delta_V}')

            if delta_V < 1e-3:  # Convergence threshold, adjust as necessary
                print(f'Convergence achieved at iteration {k1}')
                break
            V = np.copy(V_new)

        # Optionally save the policy for the next step
        np.save(f"saved/trial2/pi_matrix{t}.npy", pi)
        print("DEBUG: pi", pi)
        return pi


def main():
    nx = 11
    ny = 11
    n_theta = 10
    nv = 6
    nw = 11
    nt = 100
    gpi = generalizedPI(nx, ny, n_theta, nv, nw, set_range=False)
    create_pf = False
    gpi_train = False
    gpi_test = True
    if create_pf:
        for t in tqdm(range(nt)):
            pf = gpi.generate_pf2(t)
            pf = pf.reshape(nx, ny, n_theta, nv, nw, 7, 4)
            pf_filename = f"saved/trial2/pf_matrix{t}.npy"
            np.save(pf_filename, pf)
    if gpi_train:
        Pi = np.zeros((nt, nx, ny, n_theta, 2))
        for t in tqdm(range(nt)):
            pi = gpi.PI2(t)
            pi = pi.reshape(nx, ny, n_theta, 2)
            Pi[t, :, :, :, :] = pi
        pf_filename = f"saved/trial2/Pi_matrix.npy"
        np.save(pf_filename, Pi)
    if gpi_test:
        gpi.test_control(nt)
    pass


if __name__ == '__main__':
    main()
