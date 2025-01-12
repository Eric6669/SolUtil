import numpy as np
from Solverz.solvers.solution import daesol
from Solverz import Rodas, TimeSeriesParam, Opt, load, save_result, fdae_solver, implicit_trapezoid, save
from Solverz.variable.variables import Vars
from Solverz.utilities.address import Address
from copy import deepcopy
import tqdm

from time import perf_counter


def create_yc(y1, y2, sys1_param, sys2_param):
    # coupling variable
    a = Address()  # address
    for varname in sys1_param:
        a.add(varname, y2[varname].shape[0])
    for varname in sys2_param:
        a.add(varname, y1[varname].shape[0])

    array = np.zeros((a.total_size,))
    for varname in sys1_param:
        array[a[varname]] = y2[varname]
    for varname in sys2_param:
        array[a[varname]] = y1[varname]

    return Vars(a, array)


def update_yc(yc, y1, y2, sys1_param, sys2_param):
    yc1 = deepcopy(yc)
    for varname in sys1_param:
        yc1[varname] = y2[varname]
    for varname in sys2_param:
        yc1[varname] = y1[varname]
    return yc1


def alternate_sol(sys1,
                  sys2,
                  y0_1,
                  y0_2,
                  solver1,
                  solver2,
                  Dt,
                  tspan,
                  sys1_param,
                  sys2_param,
                  ite_tol=1e-5):
    """
    sys1 small step size, sys2 big step size
    """
    bar = tqdm.tqdm(total=tspan[-1] - tspan[0])

    start = perf_counter()

    flag = False

    t0 = tspan[0]
    tend = tspan[1]
    step = 0
    t = t0
    T = tend
    sol1 = daesol()
    sol2 = daesol()

    yc0 = create_yc(y0_1, y0_2, sys1_param, sys2_param)

    while t < T:
        err = 1
        ite = 0

        y1_1 = deepcopy(y0_1)
        y1_2 = deepcopy(y0_2)

        while err > 1e-5:
            ite = ite + 1

            sol2 = solver2(sys2, [t, t + Dt], y0_2, Opt(step_size=Dt))
            y1_2 = sol2.Y[-1]

            for varname in sys1_param:
                sys1.p[varname] = TimeSeriesParam(varname,
                                                  sol2.Y[varname].reshape((-1, )),
                                                  sol2.T)

            sol1 = solver1(sys1,
                           [t, t + Dt],
                           y0_1,
                           Opt(step_size=Dt))
            y1_1 = sol1.Y[-1]

            yc1 = update_yc(yc0, y1_1, y1_2, sys1_param, sys2_param)

            err = np.max(np.abs(yc0 - yc1))

            yc0.array[:] = yc1.array[:]

            for varname in sys2_param:
                sys2.p[varname] = TimeSeriesParam(varname,
                                                  sol1.Y[varname].reshape((-1, )),
                                                  sol1.T)

            if ite > 100:
                print("Cannot converge within 100 iterations!")
                break

        if step == 0:
            sol1.append(sol1)
            sol2.append(sol2)
        else:
            sol1.append(sol1[-1:])
            sol2.append(sol2[-1:])
        step = step + 1
        y0_1 = deepcopy(y1_1)
        y0_2 = deepcopy(y1_2)

        t = t + Dt
        bar.update(Dt)

        if flag:
            bar.close()
            break

    end = perf_counter()
    print(f'Time elapsed: {end - start}s!')

    return sol1, sol2
