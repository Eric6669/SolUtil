import warnings

from pyomo.environ import (Reals, Var, AbstractModel, Set, Param, Constraint, minimize,
                           SolverFactory, Objective, PositiveReals)
import numpy as np
from Solverz import Var as SolVar, Param as SolParam, Eqn, Model, made_numerical, nr_method, module_printer, Sign


def hydraulic_opt_mdl(alpha):
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
    m.Hset = Param(m.slack_nodes, mutable=True, domain=Reals)
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
        obj_ = sum(
            m.c[i, j] * abs(m.f[i, j]) ** (alpha + 1) / (alpha + 1) - m.delta[i, j] * m.f[i, j] for i, j in m.Arcs)
        obj_ -= sum(m.Hset[i] * (m.fs_slack[i] - m.fl[i]) for i in m.slack_nodes)
        return obj_

    m.Obj = Objective(rule=obj, sense=minimize)

    return m


class HydraFlow:

    def __init__(self,
                 pipe_from,
                 pipe_to,
                 slack_nodes,
                 non_slack_nodes,
                 c,
                 fs,
                 fl,
                 Hset,
                 delta,
                 pinloop,
                 alpha=2):
        self.mdl = hydraulic_opt_mdl(alpha)
        self.cmdl = None
        self.pipe_from = pipe_from
        self.pipe_to = pipe_to
        if len(self.pipe_from) != len(self.pipe_to):
            raise ValueError(f'Length of pipe_from and pipe_to not equal')
        self.n_pipe = len(self.pipe_from)
        self.slack_nodes = slack_nodes
        self.non_slack_nodes = non_slack_nodes
        self.n_node = len(self.slack_nodes) + len(self.non_slack_nodes)
        self.c = c
        self.fs = fs
        self.fl = fl
        self.Hset = Hset
        self.delta = delta
        self.pinloop = pinloop
        self.f = np.zeros(self.n_pipe)
        self.H = np.zeros(self.n_node)
        self.res = None

        arcs = [(i, j) for i, j in zip(self.pipe_from, self.pipe_to)]

        nodes = np.arange(self.n_node)
        slack_nodes = self.slack_nodes
        non_slack_nodes = self.non_slack_nodes
        c = dict(zip(arcs, self.c))
        fs = dict(zip(nodes, self.fs))
        fl = dict(zip(nodes, self.fl))
        Hset = dict(zip(slack_nodes, self.Hset))
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

        self.cmdl = self.mdl.create_instance(data_dict)
        self.Hmdl, self.y0 = self.derive_H_mdl()

    def update_fs(self, fs1):
        for i in range(len(fs1)):
            self.cmdl.fs[i].value = fs1[i]

    def update_fl(self, fl1):
        for i in range(len(fl1)):
            self.cmdl.fl[i].value = fl1[i]

    def run(self, tee=False):
        opt = SolverFactory('ipopt')
        self.res = opt.solve(self.cmdl, tee=tee)

        self.f = []
        for i, j in self.cmdl.Arcs:
            self.f.append(self.cmdl.f[i, j].value)
        self.f = np.array(self.f)

        for i in self.slack_nodes:
            self.fs[i] = self.cmdl.fs_slack[i].value

        if len(self.slack_nodes) > 0:
            self.Hmdl.p['f'] = self.f
            sol = nr_method(self.Hmdl, self.y0)
            self.H = sol.y['H']
        else:
            warnings.warn("No slack nodes! Cannot calculate H distribution!")

    def derive_H_mdl(self):
        """
        The linear algebraic equations about node pressure, with pipe flow given as parameters.
        """
        m = Model()
        m.f = SolParam('f', self.f)
        m.H = SolVar('H', np.zeros(self.n_node))
        m.c = SolParam('c', self.c)
        m.Hset = SolParam('Hset', self.Hset)

        for edge in range(self.n_pipe):
            fnode = self.pipe_from[edge]
            tnode = self.pipe_to[edge]
            idx = edge
            include_pipe = True

            if len(self.pinloop) > 0:
                # if there exists loop, omit the last pipe in the loop
                if idx == self.pinloop[-1]:
                    include_pipe = False

            if include_pipe:
                Hi = m.H[fnode]
                Hj = m.H[tnode]
                fij = m.f[idx]
                rhs = m.c[idx] * fij ** 2 * Sign(fij) - (Hi - Hj)
                m.__dict__[f'H_f_{fnode}_{tnode}'] = Eqn(f"H_f_{fnode}_{tnode}", rhs)

        # node pressure equations
        # for multi-slack cases, preserve only the first one, so that the equations would not be overdetermined
        for node in self.slack_nodes[0:1]:
            idx_slack = self.slack_nodes.tolist().index(node)
            m.__dict__[f'H_{node}'] = Eqn(f'H_{node}',
                                          m.H[node] - m.Hset[idx_slack])

        sae, y0 = m.create_instance()
        ae = made_numerical(sae, y0, sparse=True)

        return ae, y0
