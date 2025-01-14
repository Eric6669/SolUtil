from pyomo.environ import (Reals, Var, AbstractModel, Set, Param, Constraint, minimize,
                           SolverFactory, Objective, PositiveReals)


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
        obj_ = sum(m.c[i, j] * abs(m.f[i, j]) ** (alpha+1) / (alpha+1) - m.delta[i, j] * m.f[i, j] for i, j in m.Arcs)
        obj_ -= sum(m.Hset[i] * m.fs_slack[i] for i in m.slack_nodes)
        return obj_

    m.Obj = Objective(rule=obj, sense=minimize)

    return m
