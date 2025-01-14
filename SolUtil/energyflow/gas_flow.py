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
from .hydraulic_mdl import hydraulic_opt_mdl

__all__ = ['GasFlow', 'mdl_ngs']


def ae_pi(f, gc):
    """
    The linear algebraic equations about node pressure, with pipe flow given as parameters.
    """
    m = Model()
    m.f = SolParam('f', f)
    m.p_square = SolVar('p_square', np.zeros(gc['n_node']))
    m.c = SolParam('c', gc['C'])
    m.Pi_slack = SolParam('Pi_slack', gc['Pi'][gc['slack']])

    for edge in gc['G'].edges(data=True):
        fnode = edge[0]
        tnode = edge[1]
        idx = edge[2]['idx']
        if len(gc['pinloop']) > 0:
            # if there exists loop, omit the last pipe in the loop
            if idx != gc['pinloop'][-1]:
                pi = m.p_square[fnode]
                pj = m.p_square[tnode]
                fij = m.f[idx]
                rhs = m.c[idx] * fij ** 2 * Sign(fij) - (pi - pj)
                m.__dict__[f'p_q_{fnode}_{tnode}'] = Eqn(f"p_f_{fnode}_{tnode}", rhs)
        else:
            pi = m.p_square[fnode]
            pj = m.p_square[tnode]
            fij = m.f[idx]
            rhs = m.c[idx] * fij ** 2 * Sign(fij) - (pi - pj)
        m.__dict__[f'p_q_{fnode}_{tnode}'] = Eqn(f"p_f_{fnode}_{tnode}", rhs)

    # node pressure equations
    # for multi-slack cases, preserve only the first one, so that the equations would not be overdetermined
    for node in gc['slack'][0:1]:
        idx_slack = gc['slack'].tolist().index(node)
        m.__dict__[f'pressure_{node}'] = Eqn(f'pressure_{node}',
                                             m.p_square[node] - m.Pi_slack[idx_slack] ** 2)

    sae, y0 = m.create_instance()
    ae = made_numerical(sae, y0, sparse=True)

    return ae, y0


class GasFlow:

    def __init__(self,
                 file: str):
        self.Pi_square = None
        self.delta = None
        self.C = None
        self.non_slack_node = None
        self.slack = None
        self.n_node = None
        self.n_pipe = None
        self.pipe_to = None
        self.pipe_from = None
        self.fin = None
        self.gc = load_ngs(file)
        self.gas_mdl = hydraulic_opt_mdl(2)
        self.results = None
        self.f = np.zeros(self.gc['n_pipe'])
        self.Pi = np.zeros(self.gc['n_node'])
        self.Pi_slack = self.gc['Pi'][self.gc['slack']]
        self.fs = self.gc['fs']
        self.fl = self.gc['fl']
        self.__dict__.update(self.gc)

        # print("Creating pf model of node pressure!")
        self.ae, self.y0 = ae_pi(self.f, self.gc)

    def run(self, tee=True):

        arcs = [(i, j) for i, j in zip(self.pipe_from, self.pipe_to)]
        nodes = np.arange(self.n_node)
        slack_nodes = self.slack
        non_slack_nodes = self.non_slack_node
        c = dict(zip(arcs, self.C))
        fs = dict(zip(nodes, self.fs))
        fl = dict(zip(nodes, self.fl))
        Hset = dict(zip(slack_nodes, self.Pi_slack**2))
        delta = dict(zip(arcs, self.delta))
        data_dict = {
            None: {
                'Nodes': {None: nodes},
                'slack_nodes': {None: slack_nodes},
                'non_slack_nodes': {None: non_slack_nodes},
                'Arcs': {None: arcs},
                'fs': fs,
                'fl': fl,
                'c': c,
                'Hset': Hset,
                'delta': delta
            }
        }

        # print("Creating optimization model instance!")
        self.cgmdl = self.gas_mdl.create_instance(data_dict)
        opt = SolverFactory('ipopt')
        self.results = opt.solve(self.cgmdl, tee=tee)

        if self.results.solver.status == 'ok' and tee:
            print('Solution found')

        self.f = []
        for i, j in self.cgmdl.Arcs:
            self.f.append(self.cgmdl.f[i, j].value)

        self.f = np.array(self.f)
        self.fin = self.A @ self.f
        self.fs = - self.fin + self.fl

        self.ae.p['f'] = self.f
        self.ae.p['c'] = self.C
        self.ae.p['Pi_slack'] = self.Pi[self.slack]
        sol = nr_method(self.ae, self.y0)
        self.Pi_square = sol.y['p_square']
        self.Pi = sol.y['p_square'] ** (1 / 2)

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
