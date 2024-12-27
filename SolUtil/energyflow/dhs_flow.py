import os

from pyomo.environ import (Reals, Var, AbstractModel, Set, Param, Constraint, minimize,
                           SolverFactory, Objective, PositiveReals)
from Solverz import Var as SolVar, Param as SolParam, Eqn, Model, made_numerical, nr_method, module_printer, Sign, \
    heaviside, exp, Abs
from SolUtil.sysparser import load_hs
import numpy as np
import pandas as pd
from ..sysparser import load_ngs

__all__ = ["DhsFlow", "mdl_dhs"]

class DhsFlow:

    def __init__(self,
                 file: str):
        self.hc = load_hs(file)


def mdl_dhs(hc, module_name, jit=True):
    m = Model()
    Tamb = hc['Ta']
    m.m = SolVar('m', hc['m'])
    m.Ts = SolVar('Ts', hc['Ts'] - Tamb)
    m.Tr = SolVar('Tr', hc['Tr'] - Tamb)
    m.Touts = SolVar('Touts', hc['Ts'] - Tamb)
    m.Toutr = SolVar('Toutr', hc['Tr'] - Tamb)
    # m.min = SolVar('min', np.zeros(hc['n_node']))
    m.min = SolVar('min', 0.1 * np.ones(hc['n_node']))
    m.phi_slack = SolVar('phi_slack', np.zeros(1))
    m.Tsource = SolParam('Tsource', hc['Ts'] - Tamb)
    m.Tload = SolParam('Tload', hc['Tr'] - Tamb)
    m.lam = SolParam('lam', hc['lam'])
    m.L = SolParam('L', hc['L'])
    m.Cp = SolParam('Cp', 4182)
    m.phi = SolParam('phi', hc['phi'])

    # mass flow continuity
    for node in range(hc['n_node']):
        rhs = - m.min[node]
        for edge in hc['G'].in_edges(node, data=True):
            pipe = edge[2]['idx']
            rhs = rhs + m.m[pipe]

        for edge in hc['G'].out_edges(node, data=True):
            pipe = edge[2]['idx']
            rhs = rhs - m.m[pipe]
        m.__dict__[f"Mass_flow_continuity_{node}"] = Eqn(f"Mass_flow_continuity_{node}", rhs)

    # loop pressure
    m.K = SolParam('K', hc['K'])
    rhs = 0
    if len(hc['pinloop']) > 0:
        for pipe in hc['pinloop']:
            i = np.abs(pipe)
            rhs += m.K[i] * m.m[i] ** 2 * Sign(m.m[i])
        m.loop_pressure = Eqn("loop_pressure", rhs)

    # Supply temperature
    for node in range(hc['n_node']):
        lhs = 0
        rhs = 0

        if node in hc['s_node']:
            lhs += Abs(m.min[node])
            rhs += m.Tsource[node] * Abs(m.min[node])

        for edge in hc['G'].in_edges(node, data=True):
            pipe = edge[2]['idx']
            lhs += heaviside(m.m[pipe]) * Abs(m.m[pipe])
            rhs += heaviside(m.m[pipe]) * (m.Touts[pipe] * Abs(m.m[pipe]))

        for edge in hc['G'].out_edges(node, data=True):
            pipe = edge[2]['idx']
            lhs += (1 - heaviside(m.m[pipe])) * Abs(m.m[pipe])
            rhs += (1 - heaviside(m.m[pipe])) * (m.Touts[pipe] * Abs(m.m[pipe]))

        lhs *= m.Ts[node]

        m.__dict__[f"Ts_{node}"] = Eqn(f"Ts_{node}", lhs - rhs)

    # Return temperature
    for node in range(hc['n_node']):
        lhs = 0
        rhs = 0

        if node in hc['l_node']:
            lhs += Abs(m.min[node])
            rhs += m.Tload[node] * Abs(m.min[node])

        for edge in hc['G'].out_edges(node, data=True):
            pipe = edge[2]['idx']
            lhs += heaviside(m.m[pipe]) * Abs(m.m[pipe])
            rhs += heaviside(m.m[pipe]) * (m.Toutr[pipe] * Abs(m.m[pipe]))

        for edge in hc['G'].in_edges(node, data=True):
            pipe = edge[2]['idx']
            lhs += (1 - heaviside(m.m[pipe])) * Abs(m.m[pipe])
            rhs += (1 - heaviside(m.m[pipe])) * (m.Toutr[pipe] * Abs(m.m[pipe]))

        lhs *= m.Tr[node]

        m.__dict__[f"Tr_{node}"] = Eqn(f"Tr_{node}", lhs - rhs)

    # Temperature drop
    for edge in hc['G'].edges(data=True):
        fnode = edge[0]
        tnode = edge[1]
        pipe = edge[2]['idx']
        attenuation = exp(- m.lam[pipe] * m.L[pipe] / (m.Cp * Abs(m.m[pipe])))
        Tstart = m.Ts[fnode] * heaviside(m.m[pipe]) + m.Ts[tnode] * (1 - heaviside(m.m[pipe]))
        rhs = m.Touts[pipe] - Tstart * attenuation
        m.__dict__[f"Touts_{pipe}"] = Eqn(f"Touts_{pipe}", rhs)
        Tstart = m.Tr[tnode] * heaviside(m.m[pipe]) + m.Tr[fnode] * (1 - heaviside(m.m[pipe]))
        rhs = m.Toutr[pipe] - Tstart * attenuation
        m.__dict__[f"Toutr_{pipe}"] = Eqn(f"Toutr_{pipe}", rhs)

    # heat power
    for node in range(hc['n_node']):
        if node in hc['slack_node']:
            phi = m.phi_slack
        else:
            phi = m.phi[node]

        if node in hc['s_node']:
            rhs = phi - m.Cp / 1e6 * Abs(m.min[node]) * (m.Tsource[node] - m.Tr[node])
        elif node in hc['l_node']:
            rhs = phi - m.Cp / 1e6 * Abs(m.min[node]) * (m.Ts[node] - m.Tload[node])
        elif node in hc['I_node']:
            rhs = m.min[node]

        m.__dict__[f'phi_{node}'] = Eqn(f"phi_{node}", rhs)

    heat, y0 = m.create_instance()
    pyprinter = module_printer(heat,
                               y0,
                               module_name,
                               jit=jit)
    pyprinter.render()
