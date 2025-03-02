"""
Solve gas flow equations using optimization
"""
import os

from pyomo.environ import (Reals, Var, AbstractModel, Set, Param, Constraint, minimize,
                           SolverFactory, Objective, PositiveReals)
from Solverz import Var as SolVar, Param as SolParam, Eqn, Model, made_numerical, nr_method, module_printer, Sign
import numpy as np
import pandas as pd
from ..sysparser import load_ngs
from .hydraulic_mdl import HydraFlow

__all__ = ['GasFlow', 'mdl_ngs']


class GasFlow:
    """
    Perform natural gas flow with models and algorithms from [1]_.

    References
    ==========
    .. [1] W. Jia, T. Ding, Y. Yuan, and H. Zhang, “Fast Probabilistic Energy Flow Calculation for Natural Gas Systems: A Convex Multiparametric Programming Approach,” IEEE Trans. Automat. Sci. Eng., pp. 1–11, 2024, doi: 10.1109/TASE.2024.3454750.

    """

    def __init__(self,
                 file: str):
        self.G = None
        self.Pi_square = None
        self.delta = None
        self.C = None
        self.non_slack_node = None
        self.slack = None
        self.n_node = None
        self.n_pipe = None
        self.pipe_to = None
        self.pipe_from = None
        self.results = None
        self.f = None
        self.Pi = None
        self.Piset = None
        self.fs = None
        self.fl = None
        self.pinloop = []
        self.gc = load_ngs(file)
        self.__dict__.update(self.gc)
        self.gas_mdl = HydraFlow(self.slack,
                                 self.non_slack_node,
                                 self.C,
                                 self.fs,
                                 self.fl,
                                 self.Piset ** 2,
                                 self.delta,
                                 self.pinloop,
                                 self.G,
                                 2)

    def run(self, tee=False):
        """
        Run gas flow
        """
        self.gas_mdl.update_fs(self.fs)
        self.gas_mdl.update_fl(self.fl)
        self.gas_mdl.update_Hset(self.Piset**2)

        self.gas_mdl.run(tee)

        self.f = self.gas_mdl.f
        self.fs = self.gas_mdl.fs
        if np.any(self.gas_mdl.H < 0):
            raise ValueError('Negative Square Pressure! Physically infeasible.')
        else:
            self.Pi = np.sqrt(self.gas_mdl.H)

    def output_results(self, file):
        fnd = []
        tnd = []
        for i, j in self.cgmdl.Arcs:
            self.f.append(self.cgmdl.f[i, j].value)
            fnd.append(i)
            tnd.append(j)

        pipe = {'idx': np.arange(len(self.f)),
                'fnd': fnd,
                'tnd': tnd,
                'f': self.f}
        node = {'idx': np.arange(len(self.Pi)),
                'Pi': self.Pi}
        pipe_df = pd.DataFrame(pipe)
        node_df = pd.DataFrame(node)

        with pd.ExcelWriter(file, engine='openpyxl') as writer:
            # Write each DataFrame to a different sheet
            pipe_df.to_excel(writer, sheet_name='pipe', index=False)
            node_df.to_excel(writer, sheet_name='node', index=False)


def mdl_ngs(gc, module_name, jit=True):
    """
    Full NGS model using Solverz
    """
    m = Model()
    m.Pi = SolVar("Pi", gc['Pi'])
    m.f = SolVar('f', np.zeros(gc['n_pipe']))
    minset = np.zeros(gc['n_node'])
    minset[gc['non_slack_node']] = gc['non_slack_fin_set']
    m.minset = SolParam('minset', minset)
    m.c = SolParam('c', gc['C'])
    m.Piset = SolParam('Piset', gc['Pi'])

    # mass flow continuity
    for node in gc['non_slack_node']:
        rhs = - m.minset[node]
        for edge in gc['G'].in_edges(node, data=True):
            pipe = edge[2]['idx']
            rhs = rhs + m.f[pipe]

        for edge in gc['G'].out_edges(node, data=True):
            pipe = edge[2]['idx']
            rhs = rhs - m.f[pipe]
        m.__dict__[f"Mass_flow_continuity_{node}"] = Eqn(f"Mass_flow_continuity_{node}",
                                                         rhs)

    # mass-flow & pressure
    for edge in gc['G'].edges(data=True):
        fnode = edge[0]
        tnode = edge[1]
        idx = edge[2]['idx']
        pi = m.Pi[fnode]
        pj = m.Pi[tnode]
        fij = m.f[idx]
        rhs = m.c[idx] * fij ** 2 * Sign(fij) - (pi ** 2 - pj ** 2)
        m.__dict__[f'p_q_{fnode}_{tnode}'] = Eqn(f"p_q_{fnode}_{tnode}", rhs)

    # node pressure
    for node in gc['slack']:
        m.__dict__[f'pressure_{node}'] = Eqn(f'pressure_{node}', m.Pi[node] - m.Piset[node])

    # %% create instance
    gas, y0 = m.create_instance()
    pyprinter = module_printer(gas, y0, module_name, jit=jit)
    pyprinter.render()
