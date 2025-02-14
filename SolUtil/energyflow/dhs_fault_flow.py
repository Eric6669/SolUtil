from .dhs_flow import DhsFlow
from .hydraulic_mdl import HydraFlow
from Solverz.num_api.Array import Array
from Solverz import Var as SolVar, Param as SolParam, Model, heaviside, Abs, Eqn, exp, made_numerical, nr_method
import networkx as nx
import numpy as np
import copy


# %%
class DhsFaultFlow:

    def __init__(self,
                 df: DhsFlow,
                 fault_pipe,
                 fault_location,
                 fault_sys='s'):
        """

        :param df:
        :param fault_pipe: 故障管道在正常网络中的编号
        :param fault_location: 始端节点到故障位置的距离占管道总长度的百分比
        :param fault_sys: 故障在供水网络还是回水网络
        """
        if not df.run_succeed:
            df.run()

        self.HydraSup = df.HydraFlow
        self.HydraRet = HydraFlow(df.slack_node,
                                  df.non_slack_nodes,
                                  self.HydraSup.c,
                                  self.HydraSup.fl,
                                  self.HydraSup.fs,
                                  self.HydraSup.Hset,
                                  -self.HydraSup.delta,
                                  [1],
                                  self.HydraSup.G.reverse(),
                                  2)
        self.HydraRet.run()

        self.dH = self.HydraSup.H[df.slack_node[0]] - self.HydraRet.H[df.slack_node[0]]

        self.df = df
        self.Hset = df.Hset
        self.fault_sys = fault_sys
        self.fault_location = fault_location
        self.fault_pipe = fault_pipe
        self.G = df.G

        # run power flow considering leakage
        self.HydraFault = self.HydraFault_mdl()
        self.fs_injection = None
        self.stemp = None
        self.temp_mdl = None
        self.y0 = None
        self.HydraSup.run()
        self.HydraRet.run()
        self.HydraFault.run()
        self.mdl_temp()

    def run_hydraulic(self):
        if self.fault_sys == 's':
            self.HydraFault.run()
            self.HydraRet.run()
        else:
            self.HydraSup.run()
            self.HydraFault.run()
        self.fs_injection = self.HydraFault.f[-1]

    def run(self):

        if self.fault_sys == 'r':
            # Hset_fault = self.HydraFault.Hset.copy()
            # Hset_fault[0] = self.HydraSup.H[self.df.non_slack_nodes[0:1]]
            # self.HydraFault.update_Hset(Hset_fault)
            self.run_hydraulic()
            self.temp_mdl.p['m_fault_downstream'] = self.HydraFault.f[-2]
            ms = self.HydraSup.f[0:self.df.n_pipe]
            mr = self.HydraFault.f[0:self.df.n_pipe]
            self.temp_mdl.p['ms'] = ms
            self.temp_mdl.p['mr'] = mr
            minset = self.df.A @ ms
            self.temp_mdl.p['min'] = minset
            sol = nr_method(self.temp_mdl, self.y0)
            self.Ts = sol.y['Ts']
            self.Tr = sol.y['Tr']
            self.ms = ms
            self.mr = mr
            self.min = self.df.minset
            self.min[self.df.slack_node] -= self.fs_injection
            self.phi_slack = (4182 * abs(self.min[self.df.slack_node]) *
                              (self.df.Tsource - self.Tr[self.df.slack_node]) / 1e6)
            # calculate dH
            self.dH_post_fault = self.HydraFault.H[self.df.slack_node[0]] - self.HydraRet.H[self.df.slack_node[0]]

        else:
            done = False
            Tsource = (self.df.Tsource - self.df.Ta) * np.ones(self.df.n_node)
            Tload = (self.df.Tload - self.df.Ta) * np.ones(self.df.n_node)
            Ts = np.ones(self.df.n_node) * Tsource
            Tr = np.ones(self.df.n_node) * Tload
            min_slack_0 = self.df.minset[self.df.slack_node]
            nt = 0
            while not done:
                nt += 1
                phi = self.df.phi
                dT = np.sum(self.df.Cs, axis=0) * Tsource + (self.df.Cl + self.df.Ci) @ Ts \
                     - (self.df.Ci + self.df.Cs) @ Tr - np.sum(self.df.Cl, axis=0) * Tload
                minset = phi * 1e6 / (4182 * dT)

                fs = np.zeros(self.df.n_node, dtype=np.float64)
                for i in self.df.s_node.tolist() + self.df.slack_node.tolist():
                    fs[i] = minset[i]
                fl = np.zeros(self.df.n_node, dtype=np.float64)
                for i in self.df.I_node.tolist() + self.df.l_node.tolist():
                    fl[i] = minset[i]

                self.HydraFault.update_fs(fs)
                self.HydraFault.update_fl(fl)
                self.HydraRet.update_fs(fl)
                self.HydraRet.update_fl(fs)

                self.run_hydraulic()
                self.temp_mdl.p['m_fault_downstream'] = self.HydraFault.f[-2]

                ms = self.HydraFault.f[0:self.df.n_pipe]
                mr = self.HydraRet.f[0:self.df.n_pipe]
                self.temp_mdl.p['ms'] = ms
                self.temp_mdl.p['mr'] = mr
                minset = self.df.A @ mr

                minset[self.df.slack_node] -= self.fs_injection

                self.temp_mdl.p['min'] = minset

                sol = nr_method(self.temp_mdl, self.y0)

                if not sol.stats.succeed:
                    print("Temperature not found")
                    break

                Ts = sol.y['Ts']
                Tr = sol.y['Tr']
                Touts = sol.y['Touts']
                Toutr = sol.y['Toutr']

                phi_slack = (4182 * abs(minset[self.df.slack_node]) *
                             (self.df.Tsource - Tr[self.df.slack_node]) / 1e6)

                dF = np.abs(minset[self.df.slack_node] - min_slack_0)

                if dF < 1e-5:
                    done = True
                    self.run_succeed = True
                if nt > 100:
                    done = True
                    self.run_succeed = False

                min_slack_0 = minset[self.df.slack_node]

            if self.run_succeed:
                self.Ts = Ts + self.df.Ta
                self.Tr = Tr + self.df.Ta
                self.ms = ms
                self.mr = mr
                self.minset = minset
                self.phi = self.df.phi
                self.phi[self.df.slack_node] = phi_slack
                self.phi_slack = phi_slack
                self.Touts = Touts + self.df.Ta
                self.Toutr = Toutr + self.df.Ta

                # # calculate dH
                # Hset_ret =self.HydraFault.H[self.df.non_slack_nodes[0:1]]
                # self.HydraRet.update_Hset(Hset_ret)
                # self.HydraRet.run()
                # self.dH_post_fault = self.HydraFault.H[self.df.slack_node[0]] - self.HydraRet.H[self.df.slack_node[0]]
            else:
                print("Solution not found")

    def HydraFault_mdl(self):

        if self.fault_sys == 's':
            G = copy.deepcopy(self.HydraSup.G)
        elif self.fault_sys == 'r':
            G = copy.deepcopy(self.HydraRet.G)

        nfault = G.number_of_nodes()
        for u, v, data in G.edges(data=True):
            if data.get('idx') == self.fault_pipe:
                edge_to_remove = [u, v, data]

        u, v, data = edge_to_remove
        c = data['c']
        G.remove_edge(u, v)

        f = u
        t = nfault
        G.add_edge(f,
                   t,
                   idx=self.fault_pipe,
                   c=c * self.fault_location)

        f = nfault
        t = v
        G.add_edge(f,
                   t,
                   idx=G.number_of_edges(),
                   c=c * (1 - self.fault_location))

        nenviron = G.number_of_nodes()
        g = 10
        G.add_edge(nfault,
                   nenviron,
                   idx=G.number_of_edges() + 1,
                   c=1 / (2 * g * (np.pi * (self.df.D[self.fault_pipe] / 2)) ** 2))

        c = [np.array(k['c']).reshape(-1) for i, j, k in
             sorted(G.edges(data=True), key=lambda edge: edge[2].get('idx', 1))]
        c = np.asarray(c).reshape(-1)

        delta = np.zeros(G.number_of_edges())

        slack_node = [*self.df.slack_node, nenviron]
        non_slack_node = np.setdiff1d(np.arange(G.number_of_nodes()), slack_node)

        if self.fault_sys == 's':
            fs = self.HydraSup.fs
            fl = self.HydraSup.fl
        elif self.fault_sys == 'r':
            fs = self.HydraRet.fs
            fl = self.HydraRet.fl
        else:
            raise ValueError(f'Unkown fault sys type{self.fault_sys}')

        # assign fs and fl of node nfault and nenviron
        fs = np.append(fs, [0, 0])
        fl = np.append(fl, [0, 0])

        if self.fault_sys == 's':
            Hslack = np.append(self.HydraSup.Hset, [0])
        else:
            Hslack = np.append(self.HydraRet.Hset, [0])
        return HydraFlow(slack_node,
                         non_slack_node,
                         c,
                         fs,
                         fl,
                         Hslack,
                         delta,
                         [1],
                         G,
                         2)

    def mdl_temp(self):
        """
        Temperature model based on Solverz, with mass flow as parameters
        """
        m = Model()
        Tamb = self.df.Ta
        if self.fault_sys == 's':
            HydraSup = self.HydraFault
            HydraRet = self.HydraRet
        else:
            HydraSup = self.HydraSup
            HydraRet = self.HydraFault

        m.ms = SolParam('ms', HydraSup.f[0: self.df.n_pipe])
        m.mr = SolParam('mr', HydraRet.f[0: self.df.n_pipe])
        m.m_fault_downstream = SolParam('m_fault_downstream', self.HydraFault.f[self.df.n_pipe])
        m.Ts = SolVar('Ts', self.df.Ts - Tamb)
        m.Tr = SolVar('Tr', self.df.Tr - Tamb)
        m.Touts = SolVar('Touts', self.df.Touts - Tamb)
        m.Toutr = SolVar('Toutr', self.df.Toutr - Tamb)
        m.min = SolParam('min', self.df.minset)
        m.Tsource = SolParam('Tsource', self.df.Tsource * np.ones(self.df.n_node) - Tamb)
        m.Tload = SolParam('Tload', self.df.Tload * np.ones(self.df.n_node) - Tamb)
        m.lam = SolParam('lam', self.df.lam)
        m.L = SolParam('L', self.df.L)
        m.Cp = SolParam('Cp', 4182)

        # Supply temperature
        for node in range(self.df.n_node):
            lhs = 0
            rhs = 0

            if node in self.df.s_node.tolist() + self.df.slack_node.tolist():
                lhs += Abs(m.min[node])
                rhs += m.Tsource[node] * Abs(m.min[node])

            for edge in self.df.G.in_edges(node, data=True):
                pipe = edge[2]['idx']
                lhs += heaviside(m.ms[pipe]) * Abs(m.ms[pipe])
                rhs += heaviside(m.ms[pipe]) * (m.Touts[pipe] * Abs(m.ms[pipe]))

            for edge in self.df.G.out_edges(node, data=True):
                pipe = edge[2]['idx']
                lhs += (1 - heaviside(m.ms[pipe])) * Abs(m.ms[pipe])
                rhs += (1 - heaviside(m.ms[pipe])) * (m.Touts[pipe] * Abs(m.ms[pipe]))

            lhs *= m.Ts[node]

            m.__dict__[f"Ts_{node}"] = Eqn(f"Ts_{node}", lhs - rhs)

        # Return temperature
        for node in range(self.df.n_node):
            lhs = 0
            rhs = 0

            if node in self.df.l_node:
                lhs += Abs(m.min[node])
                rhs += m.Tload[node] * Abs(m.min[node])

            for edge in self.df.G.out_edges(node, data=True):
                pipe = edge[2]['idx']
                lhs += heaviside(m.mr[pipe]) * Abs(m.mr[pipe])
                rhs += heaviside(m.mr[pipe]) * (m.Toutr[pipe] * Abs(m.mr[pipe]))

            for edge in self.df.G.in_edges(node, data=True):
                pipe = edge[2]['idx']
                lhs += (1 - heaviside(m.mr[pipe])) * Abs(m.mr[pipe])
                rhs += (1 - heaviside(m.mr[pipe])) * (m.Toutr[pipe] * Abs(m.mr[pipe]))

            lhs *= m.Tr[node]

            m.__dict__[f"Tr_{node}"] = Eqn(f"Tr_{node}", lhs - rhs)

        # Temperature drop
        for edge in self.df.G.edges(data=True):
            fnode = edge[0]
            tnode = edge[1]
            pipe = edge[2]['idx']

            if pipe == self.fault_pipe and self.fault_sys == 's':
                attenuation = exp(- m.lam[pipe] * m.L[pipe] * self.fault_location / (m.Cp * Abs(m.ms[pipe])))
                attenuation *= exp(
                    - m.lam[pipe] * m.L[pipe] * (1 - self.fault_location) / (m.Cp * Abs(m.m_fault_downstream)))
            else:
                attenuation = exp(- m.lam[pipe] * m.L[pipe] / (m.Cp * Abs(m.ms[pipe])))

            Tstart = m.Ts[fnode] * heaviside(m.ms[pipe]) + m.Ts[tnode] * (1 - heaviside(m.ms[pipe]))
            rhs = m.Touts[pipe] - Tstart * attenuation
            m.__dict__[f"Touts_{pipe}"] = Eqn(f"Touts_{pipe}", rhs)

            if pipe == self.fault_pipe and self.fault_sys == 'r':
                attenuation = exp(- m.lam[pipe] * m.L[pipe] * self.fault_location / (m.Cp * Abs(m.mr[pipe])))
                attenuation *= exp(
                    - m.lam[pipe] * m.L[pipe] * (1 - self.fault_location) / (m.Cp * Abs(m.m_fault_downstream)))
            else:
                attenuation = exp(- m.lam[pipe] * m.L[pipe] / (m.Cp * Abs(m.mr[pipe])))

            Tstart = m.Tr[tnode] * heaviside(m.mr[pipe]) + m.Tr[fnode] * (1 - heaviside(m.mr[pipe]))
            rhs = m.Toutr[pipe] - Tstart * attenuation
            m.__dict__[f"Toutr_{pipe}"] = Eqn(f"Toutr_{pipe}", rhs)

        stemp, y0 = m.create_instance()
        self.stemp = stemp
        temp = made_numerical(stemp, y0, sparse=True)
        self.temp_mdl = temp
        self.y0 = y0


def remove_edge_by_idx(G, target_idx):
    """
    从图 G 中移除具有给定 idx 属性值的边。

    :param G: networkx.Graph 或其子类的实例
    :param target_idx: 要移除的边的 idx 属性值
    """
    edge_to_remove = None
    for u, v, data in G.edges(data=True):
        if data.get('idx') == target_idx:
            edge_to_remove = (u, v)
            break

    if edge_to_remove is not None:
        G.remove_edge(*edge_to_remove)
        print(f"Edge with idx={target_idx} removed.")
    else:
        print(f"No edge found with idx={target_idx}.")


def cal_H(slack_nodes,
          Hset,
          G: nx.DiGraph):
    """
    calculate H using depth-first search, for graph without self-loop and multiple edges
    """
    H = dict()
    slack_nodes = slack_nodes.tolist()
    H[slack_nodes[0]] = Hset[slack_nodes.index(0)]

    def dfs(i):
        for neighbor in list(G.successors(i)) + list(G.predecessors(i)):
            if neighbor not in H:
                if neighbor in G.successors(i):
                    f = i
                    t = neighbor
                    fij = G[f][t]['f']
                    cij = G[f][t]['c']
                    H[t] = H[f] - cij * fij ** 2 * np.sign(fij)
                elif neighbor in G.predecessors(i):
                    f = neighbor
                    t = i
                    fij = G[f][t]['f']
                    cij = G[f][t]['c']
                    H[f] = H[t] + cij * fij ** 2 * np.sign(fij)
                else:
                    raise ValueError(f'{neighbor} neither successor nor predecessor!')
                dfs(neighbor)

    dfs(slack_nodes[0])

    sorted_H = sorted(H.items(), key=lambda item: item[0])
    H = [np.array(item[1]).reshape(-1) for item in sorted_H]

    return np.asarray(H).reshape(-1)
