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
        self.eval_iter = 20

    def error_model(self, t, err_t, u_t):
        cur_ref_state = utils.lissajous(t)
        cur_state = err_t + cur_ref_state
        # theta bounding
        cur_state[2] = (cur_state[2] + np.pi) % (2 * np.pi) - np.pi
        next_state = utils.car_next_state(utils.time_step, cur_state, u_t, noise=False)
        # theta bounding
        next_state[2] = (next_state[2] + np.pi) % (2 * np.pi) - np.pi
        next_ref_state = utils.lissajous(t + 1)
        err_t1 = next_state - next_ref_state
        # theta bounding
        err_t1[2] = (err_t1[2] + np.pi) % (2 * np.pi) - np.pi
        return err_t1

    def nearest(self, err_t):
        err_t_near = []
        err_t_near.append(err_t)
        err_t_near.append(err_t + [self.r_x, 0, 0])
        err_t_near.append(err_t + [-self.r_x, 0, 0])
        err_t_near.append(err_t + [0, self.r_y, 0])
        err_t_near.append(err_t + [0, -self.r_y, 0])
        err_t_near.append(err_t + [0, 0, self.r_theta])
        err_t_near.append(err_t + [0, 0, -self.r_theta])
        return np.array(err_t_near)

    def find_index(self, value, grid):
        return np.argmin(np.abs(grid - value))
    def find_grid_positions(self, points):
        # find the nearest grid index
        # enforce wrap around
        def nearest_index(value, grid):
            return np.argmin(np.abs(grid - value))

        grid_positions = []
        for point in points:
            x_idx = nearest_index(point[0], self.x_range)
            y_idx = nearest_index(point[1], self.y_range)
            theta_idx = nearest_index(point[2], self.theta_range)
            grid_positions.append((x_idx, y_idx, theta_idx))

        return grid_positions

    def generate_pf(self, t):

        pf = np.zeros((self.nx, self.ny, self.n_theta, self.nv, self.nw, 7, 4))
        print(pf.shape)
        for i, vi in tqdm(enumerate(self.v_range)):
            for j, wi in enumerate(self.w_range):
                for k, xi in enumerate(self.x_range):
                    for l, yi in enumerate(self.y_range):
                        for m, thetai in enumerate(self.theta_range):
                            err_t = np.array([xi, yi, thetai])
                            u_t = np.array([vi, wi])
                            # print(err_t, u_t)
                            err_t1 = self.error_model(t, err_t, u_t)
                            # print(err_t1)
                            # find 6 nearest neighbours
                            err_t_near = self.nearest(err_t1)
                            grid_positions = self.find_grid_positions(err_t_near)
                            # print(grid_positions)
                            probabilities = []
                            for g in grid_positions:
                                g_e = [self.x_range[g[0]], self.y_range[g[1]], self.theta_range[g[2]]]
                                probability = stats.multivariate_normal.pdf(g_e, mean=err_t1, cov=np.diag(
                                    [self.sigma_x, self.sigma_y, self.sigma_theta]))
                                probabilities.append(probability)
                                # print(g_e,probability)
                            total_probability = sum(probabilities)
                            if total_probability == 0:
                                print('DEBUG total probability zero:')
                                print(err_t, u_t)
                                print(err_t_near)
                                print(probabilities)
                                raise NotImplementedError
                            normalized_probabilities = [p / total_probability for p in probabilities]
                            # print(normalized_probabilities)
                            for q, g in enumerate(grid_positions):
                                pf[k, l, m, i, j, q, :] = [g[0], g[1], g[2], normalized_probabilities[q]]

        return pf

    def load_pf(self, t):
        pf_filename = f"saved/pf_matrix{t}.npy"
        if os.path.exists(pf_filename):
            pf = np.load(pf_filename)
            print("Loaded pf matrix from file.")
        else:
            pf = self.generate_pf(t)
            np.save(pf_filename, pf)
            print("Saved pf matrix to file.")

        print('Policy Iteration:')
        return pf

    def nearest_index(self, value, grid):
        return np.argmin(np.abs(grid - value))

    def H_function(self, err_t, u_t, V):
        Q = np.eye(2)
        R = np.eye(2)
        q = 1.0
        gamma = 0.9
        cost = err_t[0:2].T @ Q @ err_t[0:2] + q * (1 - np.cos(err_t[2])) ** 2 + u_t.T @ R @ u_t
        # convert err_t and u_t to index
        xj = self.nearest_index(err_t[0], self.x_range)
        yj = self.nearest_index(err_t[1], self.y_range)
        thetaj = self.nearest_index(err_t[2], self.theta_range)
        vj = self.nearest_index(u_t[0], self.v_range)
        wj = self.nearest_index(u_t[1], self.w_range)
        neighbors = self.pf[xj, yj, thetaj, vj, wj, :, 0:3]
        probabilities = self.pf[xj, yj, thetaj, vj, wj, :, 3]
        # print(probabilities)
        for s, n in enumerate(neighbors):
            xs = self.nearest_index(n[0], self.x_range)
            ys = self.nearest_index(n[1], self.y_range)
            thetas = self.nearest_index(n[2], self.theta_range)
            # print('DEBUG :', n)
            cost += gamma * probabilities[s] * V[xs, ys, thetas]

        return cost

    def PI(self, t):

        # Policy iteration
        self.pf = self.load_pf(t)

        V = np.zeros((self.nx, self.ny, self.n_theta, 1))
        V_new = np.zeros((self.nx, self.ny, self.n_theta, 1))
        V_iter = np.zeros((self.nx, self.ny, self.n_theta, 1))
        Q = np.zeros((self.nx, self.ny, self.n_theta, self.nv, self.nw, 1))
        pi = np.zeros((self.nx, self.ny, self.n_theta, 2))


        for k1 in range(self.max_iter):
            # policy improvement
            for k, xi in enumerate(self.x_range):
                for l, yi in enumerate(self.y_range):
                    for m, thetai in enumerate(self.theta_range):
                        for i, vi in enumerate(self.v_range):
                            for j, wi in enumerate(self.w_range):
                                err_t = [xi, yi, thetai]
                                u_t = [vi, wi]
                                cost = self.H_function(np.array(err_t), np.array(u_t), V)
                                if cost == "nan":
                                    print(f"BREAK at t = {t}, err_t = {err_t}, u_t = {u_t}")
                                    # raise BreakOutOfNestedLoops
                                    raise NotImplementedError
                                Q[k, l, m, i, j, 0] = cost
                        # print('DEBUG :', Q[k, l, m, :, :])
                        # print('DEBUG :', np.unravel_index(np.argmin(Q[k, l, m, :, :]), (nv, nw)))
                        print("DEBUG: Q :", Q[k, l, m, :, :, 0])
                        # pi[k, l, m, :] = np.unravel_index(np.argmin(Q[k, l, m, :, :, 0]), (self.nv, self.nw))
                        control_index = np.unravel_index(np.argmin(Q[k, l, m, :, :, 0]), (self.nv, self.nw))
                        print("DEBUG: control_index", control_index)
                        # print("DEBUG: control_index[0]", control_index[0])
                        # print("DEBUG: control_index[1]", control_index[1])
                        # print("DEBUG: v_range", self.v_range)
                        # print("DEBUG: w_range", self.w_range)
                        pi[k, l, m, :] = np.array([self.v_range[control_index[0]], self.w_range[control_index[1]]])
                        print(f"DEBUG: pi at {k}, {l}, {m} = ", pi[k, l, m, :])
                        # print("DEBUG: control", self.v_range[int(pi[k, l, m, 0])], self.w_range[int(pi[k, l, m, 1])])
                        # print("DEBUG: control", self.v_range[control_index[0]], self.w_range[control_index[1]])
                        # raise NotImplementedError
                        # v_range[pi[k, l, m, 0]], v_range[pi[k, l, m, 1]]
                        #V[xi, yi, thetai, 1] = np.min(Q[xi, yi, thetai, :, :])
            # policy evaluation
            for k2 in range(self.eval_iter):
                for k, xi in enumerate(self.x_range):
                    for l, yi in enumerate(self.y_range):
                        for m, thetai in enumerate(self.theta_range):
                            err_t = [xi, yi, thetai]
                            u_t = pi[k, l, m, :]
                            cost = self.H_function(np.array(err_t), np.array(u_t), V_new)
                            vi = self.nearest_index(u_t[0], self.v_range)
                            wi = self.nearest_index(u_t[1], self.w_range)
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
        return pi

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
        pf_filename = f"Pi_matrix.npy"
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

def main():
    nx = 5
    ny = 5
    n_theta = 5
    nv = 5
    nw = 5
    nt = 100
    gpi = generalizedPI(nx, ny, n_theta, nv, nw, set_range=False)
    train = True
    test = True
    if train:
        Pi = np.zeros((nt, nx, ny, n_theta, 2))
        for t in range(nt):
        # for t in [8]:
            print("#####################################################")
            print(f"           Time t = {t} ")
            print("#####################################################")
            pi = gpi.PI(t)
            Pi[t, :, :, :, :] = pi
            print(pi.shape)
        print(pi[0, 0, :, :])
        print(Pi[nt-1, 0, 0, :, :])
        pf_filename = f"saved/Pi_matrix.npy"
        np.save(pf_filename, Pi)
    if test:
        gpi.test_control(nt)
    pass


if __name__ == '__main__':
    main()

# tasks
# increase max_iter
# check policy evaluation by linear equation
# save stage cost
# vectorize Pf calculation
