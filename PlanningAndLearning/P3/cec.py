import casadi as ca
import numpy as np
import utils
import main

class CEC:
    def __init__(self, horizon) -> None:
        # raise NotImplementedError
        self.Q = 10*np.eye(2)
        self.R = 1*np.eye(2)
        self.q = 20
        self.horizon = horizon
        self.obstacle1_center = ca.DM([-2, -2])
        self.obstacle2_center = ca.DM([1, 2])
        self.radius_squared = 0.5 ** 2
        pass

    def __call__(self, t: int, cur_state: np.ndarray, cur_ref_state: np.ndarray) -> np.ndarray:
        """
        Given the time step, current state, and reference state, return the control input.
        Args:
            t (int): time step
            cur_state (np.ndarray): current state
            cur_ref_state (np.ndarray): reference state
        Returns:
            np.ndarray: control input
        """
        # TODO: define optimization variables
        u = ca.MX.sym('u', 2, self.horizon)
        et, constraints, lbg, ubg = self.error_model(t, cur_state, u_t=u)
        print(et.size())

        # TODO: define optimization constraints and optimization objective
        cost = self.cost_function(et, ut=u)
        # print(cost)

        u_vector = ca.reshape(u, -1, 1)
        # TODO: define optimization solver
        nlp = {'x': u_vector, 'f': cost, 'g': ca.vertcat(*constraints)}
        solver = ca.nlpsol('solver', 'ipopt', nlp)
        sol = solver(
            x0=ca.MX.zeros(2 * self.horizon), # utils.simple_controller(cur_state, cur_ref_state), # TODO: initial guess
            lbx=[utils.v_min, utils.w_min] * self.horizon, # TODO: lower bound on optimization variables
            ubx=[utils.v_max, utils.w_max] * self.horizon, # TODO: upper bound on optimization variables
            lbg=ca.vertcat(*lbg), # TODO: lower bound on optimization constraints
            ubg=ca.vertcat(*ubg), # TODO: upper bound on optimization constraints
        )
        x = sol["x"]  # get the solution

        print(type(x))
        print(x)

        # TODO: extract the control input from the solution
        # if isinstance(x, ca.MX):
        #     x = x[0]
        # u = x.full().flatten()
        # print(x)
        # u = x.full().flatten()
        f_x = ca.Function('f_x', [u_vector], [x])  # u_vector is the variable vector used in the solver
        numeric_x = f_x(0)
        print(numeric_x)
        print(type(numeric_x.full()))
        return numeric_x[0:2].full().flatten()

    def cost_function(self, et, ut):
        # print(et.size(), ut.size())
        cost = ca.sum2(ca.sum1(ca.mtimes([et[0:2, 0:-1].T, self.Q, et[0:2, 0:-1]]))) + ca.sum2(self.q * (1 - ca.cos(et[2, 0:-1])) ** 2)
        cost += ca.sum2(ca.sum1(ca.mtimes([ut.T, self.R, ut])))
        cost += ca.sum2(ca.sum1(ca.mtimes([et[:2, -1].T, self.Q, et[:2, -1]]))) + self.q * (1 - ca.cos(et[2, -1])) ** 2
        # print(cost.size())
        return cost

    def error_model(self, t, cur_state, u_t):
        constraints = []
        lbg = []
        ubg = []
        err_t1 = ca.MX.zeros(3, self.horizon+1)
        cur_ref_state = utils.lissajous(t + 0)
        err_t = cur_state - cur_ref_state
        err_t1[:, 0] = err_t
        cur_state_mx = ca.vertcat(*cur_state)
        dist1 = ca.sumsqr(cur_state_mx[:2] - self.obstacle1_center)
        constraints.append(dist1 - self.radius_squared)
        dist2 = ca.sumsqr(cur_state_mx[:2] - self.obstacle2_center)
        constraints.append(dist2 - self.radius_squared)
        lbg.extend([0, 0])
        ubg.extend([ca.inf, ca.inf])


        for i in range(1, self.horizon+1):
            next_state = utils.car_next_state(utils.time_step, cur_state, [u_t[0, i-1], u_t[1, i-1]], noise=False)
            next_ref_state = utils.lissajous(t + i)
            err_t1[:, i] = ca.vertcat(*(next_state - next_ref_state))
            cur_state = next_state
            cur_state_mx = ca.vertcat(*cur_state)
            dist1 = ca.sumsqr(cur_state_mx[:2] - self.obstacle1_center)
            constraints.append(dist1 - self.radius_squared)
            dist2 = ca.sumsqr(cur_state_mx[:2] - self.obstacle2_center)
            constraints.append(dist2 - self.radius_squared)
            lbg.extend([0, 0])
            ubg.extend([ca.inf, ca.inf])

        return err_t1, constraints, lbg, ubg

if __name__ == "__main__":
    # cec = CEC()
    main.main()

