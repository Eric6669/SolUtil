from .dhs_flow import DhsFlow
from .hydraulic_mdl import HydraFlow
from Solverz.num_api.Array import Array
import networkx as nx
import numpy as np


# %%
class DhsFaultFlow:

    def __init__(self,
                 df: DhsFlow,
                 Hslack,
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
        self.HydraRet = HydraFlow(df.non_slack_nodes[0:1],
                                  np.setdiff1d(np.arange(df.n_node), df.non_slack_nodes[0]),
                                  self.HydraSup.c,
                                  self.HydraSup.fl,
                                  self.HydraSup.fs,
                                  self.HydraSup.H[df.non_slack_nodes[0:1]],
                                  -self.HydraSup.delta,
                                  [1],
                                  self.HydraSup.G,
                                  2)
        self.HydraRet.run()

        self.dH = self.HydraSup.H[df.slack_node[0]] - self.HydraRet.H[df.slack_node[0]]

        self.df = df
        self.Hslack = Array(Hslack, dim=1)
        self.fault_sys = fault_sys
        self.fault_location = fault_location
        self.fault_pipe = fault_pipe
        self.G = df.G

        # run power flow considering leakage
        self.HydraFault = self.HydraFault_mdl()
        self.HydraFault.run()

    def run(self):
        pass

    def HydraFault_mdl(self):

        # if self.fault_sys == 's':
        #     Hydra

        nfault = self.G.number_of_nodes()
        for u, v, data in self.G.edges(data=True):
            if data.get('idx') == self.fault_pipe:
                edge_to_remove = [u, v, data]

        u, v, data = edge_to_remove
        c = data['c']
        self.G.remove_edge(u, v)

        f = u
        t = nfault
        self.G.add_edge(f,
                        t,
                        idx=self.fault_pipe,
                        c=c * self.fault_location)

        f = nfault
        t = v
        self.G.add_edge(f,
                        t,
                        idx=self.fault_pipe,
                        c=c * (1 - self.fault_location))

        nenviron = self.G.number_of_nodes()
        g = 10
        self.G.add_edge(nfault,
                        nenviron,
                        idx=self.G.number_of_edges() + 1,
                        c=1 / (2 * g * (np.pi * (self.df.D[self.fault_pipe] / 2)) ** 2))

        c = [np.array(k['c']).reshape(-1) for i, j, k in
             sorted(self.G.edges(data=True), key=lambda edge: edge[2].get('idx', 1))]
        c = np.asarray(c).reshape(-1)

        delta = np.zeros(self.G.number_of_edges())

        slack_node = [*self.df.slack_node, nenviron]
        non_slack_node = np.setdiff1d(np.arange(self.G.number_of_nodes()), slack_node)

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

        return HydraFlow(slack_node,
                         non_slack_node,
                         c,
                         fs,
                         fl,
                         [*self.Hslack, 0],
                         delta,
                         [1],
                         self.G,
                         2)


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
    calculate H using depth-first search, for graph without loop and multiple edges
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
