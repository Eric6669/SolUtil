#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：SolUtil 
@File    ：power_N_1_flow.py
@Author  ：He Xing
@Date    ：2025/9/23 16:41 
"""
import os

from Solverz import Var as SolVar, Param as SolParam, Eqn, Model, made_numerical, nr_method, module_printer, sin, cos
from Solverz.solvers.solution import aesol
from SolUtil.sysparser import load_mpc, load_N_1_mpc
import numpy as np
import pandas as pd

__all__ = ["Power_N_1_Flow"]


class Power_N_1_Flow:
    """
    The electric power flow with Newton-Raphson method.
    """
    def __init__(self,
                 file: str,
                 lines_to_remove,
                 mdl=None):

        self.Vm = None
        self.Va = None
        self.Pg = None
        self.Pd = None
        self.Qg = None
        self.Qd = None
        self.nb = None
        self.Ybus = None
        self.idx_slack = None
        self.idx_pv = None
        self.idx_pq = None
        self.sol = None
        self.baseMVA = None
        self.U = None
        self.S = None
        self.ux = None
        self.uy = None
        self.ix = None
        self.iy = None

        self.mpc = load_N_1_mpc(file, lines_to_remove)
        self.__dict__.update(self.mpc)
        self.Gbus = self.Ybus.real
        self.Bbus = self.Ybus.imag

        self.run_succeed = False

        if mdl is None:
            self.pfmdl, self.y0 = inline_pf_mdl(self)
        else:
            self.pfmdl = mdl['mdl']
            self.y0 = mdl['y0']

    def mdlpf(self):
        Vm = self.Vm
        Va = self.Va
        nb = self.nb
        Ybus = self.Ybus
        G = Ybus.real
        B = Ybus.imag
        ref = self.idx_slack.tolist()
        pv = self.idx_pv.tolist()
        pq = self.idx_pq.tolist()
        Pg = self.Pg
        Qg = self.Qg
        Pd = self.Pd
        Qd = self.Qd

        m = Model()
        m.Va = SolVar('Va', Va[pv + pq])
        m.Vm = SolVar('Vm', Vm[pq])
        m.Pg = SolParam('Pg', Pg)
        m.Qg = SolParam('Qg', Qg)
        m.Pd = SolParam('Pd', Pd)
        m.Qd = SolParam('Qd', Qd)

        def get_Vm(idx):
            if idx in ref + pv:
                return Vm[idx]
            elif idx in pq:
                return m.Vm[pq.index(idx)]

        def get_Va(idx):
            if idx in ref:
                return Va[idx]
            elif idx in pv + pq:
                return m.Va[(pv + pq).index(idx)]

        for i in pv + pq:
            expr = 0
            Vmi = get_Vm(i)
            Vai = get_Va(i)
            for j in range(nb):
                Vmj = get_Vm(j)
                Vaj = get_Va(j)
                expr += Vmi * Vmj * (G[i, j] * cos(Vai - Vaj) + B[i, j] * sin(Vai - Vaj))
            m.__dict__[f'P_eqn_{i}'] = Eqn(f'P_eqn_{i}', expr + m.Pd[i] - m.Pg[i])

        for i in pq:
            expr = 0
            Vmi = get_Vm(i)
            Vai = get_Va(i)
            for j in range(nb):
                Vmj = get_Vm(j)
                Vaj = get_Va(j)
                expr += Vmi * Vmj * (G[i, j] * sin(Vai - Vaj) - B[i, j] * cos(Vai - Vaj))
            m.__dict__[f'Q_eqn_{i}'] = Eqn(f'Q_eqn_{i}', expr + m.Qd[i] - m.Qg[i])

        spf, y0 = m.create_instance()

        return spf, y0

    def run(self):
        self.pfmdl.p['Pg'] = self.Pg
        self.pfmdl.p['Qg'] = self.Qg
        self.pfmdl.p['Pd'] = self.Pd
        self.pfmdl.p['Qd'] = self.Qd
        self.sol = nr_method(self.pfmdl, self.y0)
        if self.sol.stats.succeed:
            self.run_succeed = True
        self.parse_data_post_pf(self.sol)

    def parse_data_post_pf(self, sol: aesol):
        nb = self.nb
        Vm = self.Vm
        Va = self.Va
        Ybus = self.Ybus
        G = Ybus.real
        B = Ybus.imag
        ref = self.idx_slack.tolist()
        pv = self.idx_pv.tolist()
        pq = self.idx_pq.tolist()
        Pg = self.Pg
        Qg = self.Qg
        Pd = self.Pd
        Qd = self.Qd
        Vm[pq] = sol.y['Vm']
        Va[pv + pq] = sol.y['Va']

        # update slack pg qg

        for i in ref:
            Pinj = 0
            Vmi = Vm[i]
            Vai = Va[i]
            for j in range(nb):
                Vmj = Vm[j]
                Vaj = Va[j]
                Pinj += Vmi * Vmj * (G[i, j] * np.cos(Vai - Vaj) + B[i, j] * np.sin(Vai - Vaj))
            Pg[i] = Pinj + Pd[i]

        for i in ref + pv:
            Qinj = 0
            Vmi = Vm[i]
            Vai = Va[i]
            for j in range(nb):
                Vmj = Vm[j]
                Vaj = Va[j]
                Qinj += Vmi * Vmj * (G[i, j] * np.sin(Vai - Vaj) - B[i, j] * np.cos(Vai - Vaj))
            Qg[i] = Qinj + Qd[i]

        self.Vm = Vm
        self.Va = Va
        self.Pg = Pg
        self.Qg = Qg

        self.U: np.ndarray = self.Vm * np.exp(1j * self.Va)
        self.S: np.ndarray = (self.Pg - self.Pd) + 1j * (self.Qg - self.Qd)
        I = (self.S / self.U).conjugate()
        self.ux = self.U.real
        self.uy = self.U.imag
        self.ix = I.real
        self.iy = I.imag


def generate_pf_module(pf: Power_N_1_Flow, module_name, jit=True):
    spf, y0 = pf.mdlpf()
    pyprinter = module_printer(spf,
                               y0,
                               module_name,
                               jit=jit)
    pyprinter.render()


def inline_pf_mdl(pf: Power_N_1_Flow):
    spf, y0 = pf.mdlpf()
    npf = made_numerical(spf, y0, sparse=True)
    return npf, y0
