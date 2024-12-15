"""
Solve gas flow equations using optimization
"""
import os

from pyomo.environ import (Reals, Var, AbstractModel, Set, Param, Constraint, minimize,
                           SolverFactory, Objective)
from Solverz import Var as SolVar, Param as SolParam, Eqn, Model, made_numerical, nr_method
import numpy as np
import pandas as pd
import networkx as nx
from ..sysparser import load_ngs

__all__ = ['GasFlow']


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

def mdl_Pi():
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

class GasFlow:

    def __init__(self,
                 file: str):
        self.gc = load_ngs(file)
        self.gas_mdl = gas_flow_mdl()
        self.results = None
        self.f = None
        self.Pi = None

        arcs = [(i, j) for i, j in zip(self.gc['pipe_from'], self.gc['pipe_to'])]
        # fset_ = dict(zip(self.gc['ns_node'], self.gc['finset']))
        # Piset_ = dict(zip(self.gc['s_node'], self.gc['Piset']))
        c_ = dict(zip(arcs, self.gc['C']))
        # compress_fac = dict(zip(arcs, self.gc['compress_fac']))
        minset = np.zeros(self.gc['n_node'])
        minset[self.gc['non_slack_node']] = self.gc['non_slack_fin_set']
        minset = dict(zip(np.arange(self.gc['n_node']), minset))
        data_dict = {
            None: {
                'Nodes': {None: np.arange(self.gc['n_node'])},
                'non_slack_nodes': {None: self.gc['non_slack_node']},
                'Arcs': {None: arcs},
                'minset': minset,
                'c': c_
            }
        }

        self.cgmdl = self.gas_mdl.create_instance(data_dict)


    def run(self):
        opt = SolverFactory('ipopt')
        self.results = opt.solve(self.cgmdl, tee=True)

        self.f = []
        # self.Pi = []

        for i, j in self.cgmdl.Arcs:
            self.f.append(self.cgmdl.f[i, j].value)
        # for node in self.cgmdl.Nodes:
        #     self.Pi.append(self.cgmdl.Pi[node].value)

        self.f = np.array(self.f)
        self.fin = self.gc['A']@self.f
        # self.Pi = np.array(self.Pi)

        m = Model()
        m.f = SolParam('f', self.f)
        m.p_square = SolVar('p_square', np.zeros(self.gc['n_node']))
        m.C = SolParam('C',self.gc['C'])

        for edge in self.gc['G'].edges(data=True):
            fnode = edge[0]
            tnode = edge[1]
            idx = edge[2]['idx']
            pi = m.p_square[fnode]
            pj = m.p_square[tnode]
            fij = m.f[idx]
            rhs = m.C[idx] * fij ** 2 - (pi - pj)
            # rhs = (m.C[idx]) ** (1 / 2) * fij - (pi ** 2 - pj ** 2) ** (1 / 2)
            m.__dict__[f'p_q_{fnode}_{tnode}'] = Eqn(f"p_f_{fnode}_{tnode}", rhs)
        # node pressure
        for node in self.gc['slack']:
            m.__dict__[f'pressure_{node}'] = Eqn(f'pressure_{node}',
                                                 m.p_square[node] - self.gc['Pi'][node] ** 2)

        sae, y0 = m.create_instance()
        ae = made_numerical(sae, y0, sparse=True)
        sol = nr_method(ae, y0)
        self.Pi = sol.y['p_square']**(1/2)

    def output_results(self, file):

        fnd = []
        tnd = []
        for i, j in self.cgmdl.Arcs:
            self.f.append(self.cgmdl.f[i, j].value)
            fnd.append(i)
            tnd.append(j)
        # for node in self.cgmdl.Nodes:
        #     self.Pi.append(self.cgmdl.Pi[node].value)
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
