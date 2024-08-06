import casadi as ca
import numpy as np
import utils
from time import time


class CEC:
    def __init__(self) -> None:
        self.Q = np.eye(3)  # State weighting in the cost function
        self.R = np.eye(2)  # Control weighting in the cost function
        self.q = 1.0        # Additional scalar weight

    def __call__(self, t: int, cur_state: np.ndarray, cur_ref_state: np.ndarray) -> np.ndarray:
        # Define the decision variables for the control inputs
        u = ca.MX.sym('u', 2)

        # Calculate the current penalties from state and reference
        pt_t = ca.MX(cur_state[:2] - cur_ref_state[:2])
        print(pt_t)
        theta_t = cur_state[2] - cur_ref_state[2]
        next_state = utils.car_next_state(utils.time_step, cur_state, [u[0],u[1]], noise=False)
        print(next_state)
        next_ref_state = utils.lissajous(t+1)
        pt_t1 = pt_t.__copy__()
        pt_t1[0] = next_state[0] - next_ref_state[0]
        pt_t1[1] = next_state[1] - next_ref_state[1]
        theta_t1 = next_state[2] - next_ref_state[2]

        # print(ca.MX.size(pt_t))
        cost = ca.mtimes([pt_t.T, self.Q[:2, :2], pt_t]) + self.q * (1 - ca.cos(theta_t))**2
        cost += ca.mtimes([u.T, self.R, u])
        cost += ca.mtimes([pt_t1.T, self.Q[:2, :2], pt_t1]) + self.q * (1 - ca.cos(theta_t1))**2

        # Set the optimization problem
        nlp = {'x': u, 'f': cost}
        solver = ca.nlpsol('solver', 'ipopt', nlp)

        # Solve the problem
        sol = solver(lbx=[utils.v_min, utils.w_min], ubx=[utils.v_max, utils.w_max], x0=[0, 0])

        # Extract the control input from the solution
        u_opt = sol['x'].full().flatten()
        return u_opt


def main():
    cec = CEC()  # Instantiate your controller
    main_loop = time()
    # Remaining setup...
    obstacles = np.array([[-2, -2, 0.5], [1, 2, 0.5]])
    ref_traj = []
    error_trans = 0.0
    error_rot = 0.0
    car_states = []
    times = []

    cur_state = np.array([utils.x_init, utils.y_init, utils.theta_init])
    cur_iter = 0

    while cur_iter * utils.time_step < utils.sim_time:
        t1 = time()
        cur_time = cur_iter * utils.time_step
        cur_ref = utils.lissajous(cur_iter)

        ref_traj.append(cur_ref)
        car_states.append(cur_state)

        # Generate control input using CEC
        control = cec(cur_iter, cur_state, cur_ref)
        print("[v,w]", control)

        # Apply control input and update state
        next_state = utils.car_next_state(utils.time_step, cur_state, control, noise=True)
        cur_state = next_state

        t2 = time()
        times.append(t2 - t1)

        cur_err = cur_state - cur_ref
        cur_err[2] = np.arctan2(np.sin(cur_err[2]), np.cos(cur_err[2]))
        error_trans += np.linalg.norm(cur_err[:2])
        error_rot += abs(cur_err[2])

        print(cur_err, error_trans, error_rot)
        print("======================")
        cur_iter += 1

    main_loop_time = time()
    print("Total time: ", main_loop_time - main_loop)
    print("Average iteration time: ", np.mean(times) * 1000, "ms")
    print("Final error_trans: ", error_trans)
    print("Final error_rot: ", error_rot)

    utils.visualize(np.array(car_states), np.array(ref_traj), obstacles, times, utils.time_step, save=True)

if __name__ == "__main__":
    main()
