from casadi import *
import casadi as cd
from time import time
import numpy as np
import utils
import cec

def casadi_test2():
    # Import libraries
    import casadi as cd
    import numpy as np
    import matplotlib.pyplot as plt

    # Decision variables
    x = cd.SX.sym("x", 3)

    # Parameters
    p = [5.00, 1.00]

    # Objective function
    f = x[0] * x[0] + x[1] * x[1] + x[2] * x[2]

    # Concatenate nonlinear constraints
    g = cd.vertcat((6 * x[0] + 3 * x[1] + 2 * x[2] - p[0], p[1] * x[0] + x[1] - x[2] - 1))

    # Nonlinear bounds
    lbg = [0.00, 0.00]
    ubg = [0.00, 0.00]

    # Input bounds for the optimization variables
    lbx = [0.00, 0.00, 0.00]
    ubx = [cd.inf, cd.inf, cd.inf]

    # Initial guess for the decision variables
    x0 = [0.15, 0.15, 0.00]

    # Create NLP solver
    nlp = cd.SXFunction(cd.nlpIn(x=x), cd.nlpOut(f=f, g=g))
    solver = cd.NlpSolver("ipopt", nlp)
    # Initialize solver
    solver.init()

    # Pass the bounds and the initial values
    solver.setInput(x0, "x0")
    solver.setInput(lbx, "lbx")
    solver.setInput(ubx, "ubx")
    solver.setInput(lbg, "lbg")
    solver.setInput(ubg, "ubg")

    # Solve NLP
    solver.evaluate()

    # Print the solution
    print("----")
    print("Minimal cost ", solver.getOutput("f"))
    print("----")
    print("Optimal solution")
    print("x = ", solver.output("x").data())
    print("----")

def casadi_test():

    # Symbols/expressions
    x = MX.sym('x')
    y = MX.sym('y')
    z = MX.sym('z')
    f = x**2+100*z**2
    g = z+(1-x)**2-y

    nlp = {}                 # NLP declaration
    nlp['x']= vertcat(x,y,z) # decision vars
    nlp['f'] = f             # objective
    nlp['g'] = g             # constraints

    # Create solver instance
    F = nlpsol('F','ipopt',nlp);

    # Solve the problem using a guess
    res = F(x0=[2.5,3.0,0.75],ubg=0,lbg=0)

    print(res)


def main():
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
    fcec = cec.CEC()
    # Main loop
    while cur_iter * utils.time_step < utils.sim_time:
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
        control = fcec(cur_time, cur_state, cur_ref)
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
        print(cur_err, error_trans, error_rot)
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
    utils.visualize(car_states, ref_traj, obstacles, times, utils.time_step, save=True)


if __name__ == "__main__":
    casadi_test2()
    main()


