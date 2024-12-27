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

__all__ = ['GasFlow', 'mdl_ngs']


def gas_flow_mdl():
    m = AbstractModel()
    m.Nodes = Set()
    m.Arcs = Set(dimen=2)
    m.slack_nodes = Set()
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
    m.c = Param(m.Arcs, mutable=True)
    m.fs = Param(m.Nodes, mutable=True)
    m.fs_slack = Var(m.slack_nodes, domain=Reals)
    m.fl = Param(m.Nodes, mutable=True)
    m.Piset = Param(m.slack_nodes, mutable=True, domain=Reals)
    m.delta = Param(m.Arcs, domain=Reals)

    def m_continuity_rule1(m, node):
        return m.fl[node] - m.fs_slack[node] == sum(m.f[i, node] for i in m.NodesIn[node]) - sum(
            m.f[node, j] for j in m.NodesOut[node])

    m.m_balance1 = Constraint(m.slack_nodes, rule=m_continuity_rule1)

    def m_continuity_rule2(m, node):
        return m.fl[node] - m.fs[node] == sum(m.f[i, node] for i in m.NodesIn[node]) - sum(
            m.f[node, j] for j in m.NodesOut[node])

    m.m_balance2 = Constraint(m.non_slack_nodes, rule=m_continuity_rule2)

    def obj(m):
        obj_ = sum(m.c[i, j] * abs(m.f[i, j]) ** 3 / 3 - m.delta[i, j] * m.f[i, j] for i, j in m.Arcs)
        obj_ -= sum(m.Piset[i] ** 2 * m.fs_slack[i] for i in m.slack_nodes)
        return obj_

    m.Obj = Objective(rule=obj, sense=minimize)

    return m


def pressure_mdl():
    m = AbstractModel()
    m.Nodes = Set()
    m.Arcs = Set(dimen=2)
    m.slack_nodes = Set()

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

    m.f = Param(m.Arcs, domain=Reals)
    m.c = Param(m.Arcs, mutable=True)
    m.Piset = Param(m.slack_nodes, mutable=True, domain=Reals)
    m.Pi2 = Var(m.Nodes, domain=PositiveReals)
    m.delta = Param(m.Arcs, domain=Reals)

    def f_pi_rule(m, i, j):
        return m.c[i, j] * m.f[i, j] ** 2 * abs(m.f[i, j]) == m.delta[i, j] + m.Pi2[i] - m.Pi2[j]

    m.f_pi = Constraint(m.Arcs, rule=f_pi_rule)

    def pi_rule(m, node):
        return m.Pi2[node] - m.Piset[node] ** 2 == 0

    m.m_balance1 = Constraint(m.slack_nodes, rule=pi_rule)

    def obj(m):
        obj_ = m.Pi2[0]
        return obj_

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
        self.gc = load_ngs(file)
        self.gas_mdl = gas_flow_mdl()
        self.pi_mdl = pressure_mdl()
        self.results = None
        self.f = np.zeros(self.gc['n_pipe'])
        self.Pi = np.zeros(self.gc['n_node'])
        self.Pi_slack = self.gc['Pi'][self.gc['slack']]

        # print("Creating pf model of node pressure!")
        self.ae, self.y0 = ae_pi(self.f, self.gc)

    def run(self, tee=True):

        arcs = [(i, j) for i, j in zip(self.gc['pipe_from'], self.gc['pipe_to'])]
        nodes = np.arange(self.gc['n_node'])
        slack_nodes = self.gc['slack']
        non_slack_nodes = self.gc['non_slack_node']
        c = dict(zip(arcs, self.gc['C']))
        fs = dict(zip(nodes, self.gc['fs']))
        fl = dict(zip(nodes, self.gc['fl']))
        Piset = dict(zip(slack_nodes, self.Pi_slack))
        delta = dict(zip(arcs, self.gc['delta']))
        data_dict = {
            None: {
                'Nodes': {None: nodes},
                'slack_nodes': {None: slack_nodes},
                'non_slack_nodes': {None: non_slack_nodes},
                'Arcs': {None: arcs},
                'fs': fs,
                'fl': fl,
                'c': c,
                'Piset': Piset,
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
        self.fin = self.gc['A'] @ self.f

        self.ae.p['f'] = self.f
        self.ae.p['c'] = self.gc['C']
        self.ae.p['Pi_slack'] = self.gc['Pi'][self.gc['slack']]
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
