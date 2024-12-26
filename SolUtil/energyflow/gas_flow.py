"""
Solve gas flow equations using optimization
"""
import os

from pyomo.environ import (Reals, Var, AbstractModel, Set, Param, Constraint, minimize,
                           SolverFactory, Objective)
from Solverz import Var as SolVar, Param as SolParam, Eqn, Model, made_numerical, nr_method, module_printer, Sign
import numpy as np
import pandas as pd
import networkx as nx
from ..sysparser import load_ngs

__all__ = ['GasFlow', 'mdl_ngs']


def gas_flow_mdl():
    m = AbstractModel()
    m.Nodes = Set()
    m.Arcs = Set(dimen=2)
    m.non_slack_nodes = Set()

    def NodesIn_init(m, node):
        for i, j in m.Arcs:
            if j == node:
                yield i

    m.NodesIn = Set(m.Nodes, initialize=NodesIn_init)

    def NodesOut_init(m, node):
        for i, j in m.Arcs:
            if i == node:
                yield j

    m.NodesOut = Set(m.Nodes, initialize=NodesOut_init)

    m.f = Var(m.Arcs, domain=Reals)
    m.c = Param(m.Arcs)
    m.minset = Param(m.Nodes, mutable=True)

    def m_continuity_rule(m, node):
        return m.minset[node] == sum(m.f[i, node] for i in m.NodesIn[node]) - sum(
            m.f[node, j] for j in m.NodesOut[node])

    m.m_balance = Constraint(m.non_slack_nodes, rule=m_continuity_rule)

    def obj(m):
        return sum(1 / m.c[i, j] * abs(m.f[i, j]) ** 3 / 3 for i, j in m.Arcs)

    m.Obj = Objective(rule=obj, sense=minimize)

    return m


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
        pi = m.p_square[fnode]
        pj = m.p_square[tnode]
        fij = m.f[idx]
        rhs = m.c[idx] * fij ** 2*Sign(fij) - (pi - pj)
        m.__dict__[f'p_q_{fnode}_{tnode}'] = Eqn(f"p_f_{fnode}_{tnode}", rhs)

    # node pressure
    for node in gc['slack']:
        m.__dict__[f'pressure_{node}'] = Eqn(f'pressure_{node}',
                                             m.p_square[node] - m.Pi_slack ** 2)

    sae, y0 = m.create_instance()
    ae = made_numerical(sae, y0, sparse=True)

    return ae, y0


class GasFlow:

    def __init__(self,
                 file: str):
        self.gc = load_ngs(file)
        self.gas_mdl = gas_flow_mdl()
        self.results = None
        self.f = np.zeros(self.gc['n_pipe'])
        self.Pi = np.zeros(self.gc['n_node'])
        self.Pi_slack = self.gc['Pi'][self.gc['slack']]

        # print("Creating pf model of node pressure!")
        self.ae, self.y0 = ae_pi(self.f, self.gc)

    def run(self, tee=True):

        arcs = [(i, j) for i, j in zip(self.gc['pipe_from'], self.gc['pipe_to'])]
        c = dict(zip(arcs, self.gc['C']))
        minset = np.zeros(self.gc['n_node'])
        minset[self.gc['non_slack_node']] = self.gc['non_slack_fin_set']
        minset = dict(zip(np.arange(self.gc['n_node']), minset))
        data_dict = {
            None: {
                'Nodes': {None: np.arange(self.gc['n_node'])},
                'non_slack_nodes': {None: self.gc['non_slack_node']},
                'Arcs': {None: arcs},
                'minset': minset,
                'c': c
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
        self.fin = self.gc['A'] @ self.f

        self.ae.p['f'] = self.f
        self.ae.p['c'] = self.gc['C']
        self.ae.p['Pi_slack'] = self.gc['Pi'][self.gc['slack']]
        sol = nr_method(self.ae, self.y0)
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
    m.Pi_slack = SolParam('Pi_slack', gc['Pi'][gc['slack']])

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
        m.__dict__[f'pressure_{node}'] = Eqn(f'pressure_{node}', m.Pi[node] - m.Pi_slack)

    # %% create instance
    gas, y0 = m.create_instance()
    pyprinter = module_printer(gas, y0, module_name, jit=jit)
    pyprinter.render()
