import os

from pyomo.environ import (Reals, Var, AbstractModel, Set, Param, Constraint, minimize,
                           SolverFactory, Objective, PositiveReals)
from Solverz import Var as SolVar, Param as SolParam, Eqn, Model, made_numerical, nr_method, module_printer, Sign, \
    heaviside, exp, Abs
from SolUtil.sysparser import load_hs
import numpy as np
import pandas as pd
from .hydraulic_mdl import hydraulic_opt_mdl

__all__ = ["DhsFlow", "generate_dhs_module", "generate_temp_module"]


class DhsFlow:

    def __init__(self,
                 file: str,
                 symmetric_hydraulic: bool = True,
                 heatmdl=None,
                 tempmdl=None):
        """
        Parameter
        ---------

        symmetric_hydraulic :

            If True, assume that the supply and return hydraulics, apart from the directions, are the same.

        """

        self.non_slack_nodes = None
        self.Toutr = None
        self.Touts = None
        self.Tr = None
        self.phi = None
        self.slack_node = None
        self.A = None
        self.Ci = None
        self.n_node = None
        self.n_pipe = None
        self.Ta = None
        self.Tload = None
        self.Ts = None
        self.Cl = None
        self.Cs = None
        self.Tsource = None
        self.L = None
        self.lam = None
        self.pipe_from = None
        self.pipe_to = None
        self.G = None
        self.s_node = None
        self.l_node = None
        self.I_node = None
        self.pinloop = []
        self.K = None
        self.S = None
        self.delta_node = None
        self.delta_pipe = None
        self.hc = load_hs(file)
        self.__dict__.update(self.hc)
        if symmetric_hydraulic:
            self.hydraulic_mdl = hydraulic_opt_mdl(2)
            self.delta = np.zeros(self.n_node)
            self.Hset = np.zeros_like(self.slack_node)
        else:
            self.hydraulic_mdl = hydraulic_opt_mdl(2)
        self.chydrmdl = None
        self.hyd_res = None
        self.m = np.zeros(self.n_pipe)
        self.minset = np.zeros(self.n_node)
        self.run_succeed = False

        # print("Creating pf model of node pressure!")
        if heatmdl is None:
            self.heatmdl, self.hy0 = inline_dhs_mdl(self.hc)
        else:
            self.heatmdl = heatmdl['mdl']
            self.hy0 = heatmdl['y0']
        if tempmdl is None:
            self.tempmdl, self.ty0 = inline_temp_mdl(self.hc)
        else:
            self.tempmdl = tempmdl['mdl']
            self.ty0 = tempmdl['y0']

    def run(self, tee=True):
        """
        Run heat flow based on IPOPT
        """

        # opt model initialization
        arcs = [(i, j) for i, j in zip(self.pipe_from, self.pipe_to)]
        nodes = np.arange(self.n_node)
        slack_nodes = self.slack_node
        non_slack_nodes = self.non_slack_nodes
        c = dict(zip(arcs, self.K))
        fs = dict(zip(nodes, np.zeros((self.n_node,))))
        fl = dict(zip(nodes, np.zeros((self.n_node,))))
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

        self.chydrmdl = self.hydraulic_mdl.create_instance(data_dict)

        self.run_succeed = False
        done = False
        Tsource = (self.Tsource - self.Ta) * np.ones(self.n_node)
        Tload = (self.Tload - self.Ta) * np.ones(self.n_node)
        Ts = np.ones(self.n_node) * Tsource
        Tr = np.ones(self.n_node) * Tload
        nt = 0
        while not done:
            nt += 1
            phi = self.phi
            dT = np.sum(self.Cs, axis=0) * Tsource + (self.Cl + self.Ci) @ Ts \
                 - (self.Ci + self.Cs) @ Tr - np.sum(self.Cl, axis=0) * Tload
            minset = phi * 1e6 / (4182 * dT)

            for i in self.s_node.tolist() + self.slack_node.tolist():
                self.chydrmdl.fs[i].value = minset[i]

            for i in self.I_node.tolist() + self.l_node.tolist():
                self.chydrmdl.fl[i].value = minset[i]

            opt = SolverFactory('ipopt')
            self.hyd_res = opt.solve(self.chydrmdl, tee=tee)

            if self.hyd_res.solver.status == 'ok' and tee:
                print('Solution found')

            m = []
            for i, j in self.chydrmdl.Arcs:
                m.append(self.chydrmdl.f[i, j].value)
            m = np.array(m)

            self.tempmdl.p['m'] = m
            minset = self.A @ m
            self.tempmdl.p['min'] = minset
            tsol = nr_method(self.tempmdl, self.ty0)
            if not tsol.stats.succeed:
                print("Temperature not found")
                break

            Ts = tsol.y['Ts']
            Tr = tsol.y['Tr']
            Touts = tsol.y['Touts']
            Toutr = tsol.y['Toutr']

            self.hy0['m'] = m
            self.hy0['min'] = minset
            self.hy0['Ts'] = Ts
            self.hy0['Tr'] = Tr
            self.hy0['Touts'] = Touts
            self.hy0['Toutr'] = Toutr
            self.hy0['phi_slack'] = (4182 * abs(self.hy0['min'][self.slack_node]) *
                                     (Tsource[self.slack_node] - Tr[self.slack_node]) / 1e6)
            self.heatmdl.p['phi'] = phi
            F = self.heatmdl.F(self.hy0, self.heatmdl.p)
            dF = np.max(np.abs(F))
            if dF < 1e-5:
                done = True
                self.run_succeed = True
            if nt > 100:
                done = True
                self.run_succeed = False

        if self.run_succeed:
            self.Ts = Ts + self.Ta
            self.Tr = Tr + self.Ta
            self.m = m
            self.minset = minset
            self.phi[self.slack_node] = self.hy0['phi_slack']
            self.Touts = Touts + self.Ta
            self.Toutr = Toutr + self.Ta
        else:
            print("Solution not found")


def mdl_temp(hc):
    """
    Temperature model based on Solverz, with mass flow as parameters
    """
    m = Model()
    Tamb = hc['Ta']
    m.m = SolParam('m', hc['m'])
    m.Ts = SolVar('Ts', hc['Ts'] - Tamb)
    m.Tr = SolVar('Tr', hc['Tr'] - Tamb)
    m.Touts = SolVar('Touts', hc['Ts'][0] * np.ones(hc['n_pipe']) - Tamb)
    m.Toutr = SolVar('Toutr', hc['Tr'][0] * np.ones(hc['n_pipe']) - Tamb)
    m.min = SolParam('min', 0.1 * np.ones(hc['n_node']))
    m.Tsource = SolParam('Tsource', hc['Ts'] - Tamb)
    m.Tload = SolParam('Tload', hc['Tr'] - Tamb)
    m.lam = SolParam('lam', hc['lam'])
    m.L = SolParam('L', hc['L'])
    m.Cp = SolParam('Cp', 4182)

    # Supply temperature
    for node in range(hc['n_node']):
        lhs = 0
        rhs = 0

        if node in hc['s_node'].tolist() + hc['slack_node'].tolist():
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

    temp, y0 = m.create_instance()
    return temp, y0


def mdl_dhs(hc):
    """
    Full DHS model using Solverz
    """
    m = Model()
    Tamb = hc['Ta']
    m.m = SolVar('m', hc['m'])
    m.Ts = SolVar('Ts', hc['Ts'] - Tamb)
    m.Tr = SolVar('Tr', hc['Tr'] - Tamb)
    m.Touts = SolVar('Touts', hc['Ts'][0] * np.ones(hc['n_pipe']) - Tamb)
    m.Toutr = SolVar('Toutr', hc['Tr'][0] * np.ones(hc['n_pipe']) - Tamb)
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
        for i in range(hc['n_pipe']):
            rhs += m.K[i] * m.m[i] ** 2 * Sign(m.m[i]) * hc['pinloop'][i]
        m.loop_pressure = Eqn("loop_pressure", rhs)

    # Supply temperature
    for node in range(hc['n_node']):
        lhs = 0
        rhs = 0

        if node in hc['s_node'].tolist() + hc['slack_node'].tolist():
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

        if node in hc['s_node'].tolist() + hc['slack_node'].tolist():
            rhs = phi - m.Cp / 1e6 * Abs(m.min[node]) * (m.Tsource[node] - m.Tr[node])
        elif node in hc['l_node']:
            rhs = phi - m.Cp / 1e6 * Abs(m.min[node]) * (m.Ts[node] - m.Tload[node])
        elif node in hc['I_node']:
            rhs = m.min[node]

        m.__dict__[f'phi_{node}'] = Eqn(f"phi_{node}", rhs)

    heat, y0 = m.create_instance()
    return heat, y0


def inline_dhs_mdl(hc):
    heat, y0 = mdl_dhs(hc)
    nheat = made_numerical(heat, y0, sparse=True)
    return nheat, y0


def inline_temp_mdl(hc):
    temp, y0 = mdl_temp(hc)
    ntemp = made_numerical(temp, y0, sparse=True)
    return ntemp, y0


def generate_dhs_module(hc, module_name, jit=True):
    heat, y0 = mdl_dhs(hc)
    pyprinter = module_printer(heat,
                               y0,
                               module_name,
                               jit=jit)
    pyprinter.render()


def generate_temp_module(hc, module_name, jit=True):
    temp, y0 = mdl_temp(hc)
    pyprinter = module_printer(temp,
                               y0,
                               module_name,
                               jit=jit)
    pyprinter.render()
