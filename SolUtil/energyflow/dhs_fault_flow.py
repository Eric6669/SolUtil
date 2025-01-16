from .dhs_flow import DhsFlow
from .hydraulic_mdl import HydraFlow
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

        # create asymmetric hydraulic networks
        self.G = nx.DiGraph()

        # add symmetric edge
        for i in range(len(df.pipe_from)):
            if i != fault_pipe:
                f = df.pipe_from[i]
                t = df.pipe_to[i]
                self.G.add_node(f)
                self.G.add_node(t)
                self.G.add_edge(f, t, idx=i, f=df.m[i], c=df.K[i])

                f = df.pipe_to[i] + df.n_node
                t = df.pipe_from[i] + df.n_node
                self.G.add_node(f)
                self.G.add_node(t)
                self.G.add_edge(f, t, idx=i + df.n_pipe, f=df.m[i], c=df.K[i])

        # add fault edge
        nfault = 2 * np.int64(df.n_node)
        self.G.add_node(nfault)

        if fault_sys == 's':
            f = df.pipe_from[fault_pipe]
            t = df.pipe_to[fault_pipe]
            self.G.add_edge(f,
                            nfault,
                            idx=fault_pipe,
                            f=df.m[fault_pipe],
                            c=df.K[fault_pipe] * fault_location)
            self.G.add_edge(nfault,
                            t,
                            idx=2 * df.n_pipe,
                            f=df.m[fault_pipe],
                            c=df.K[fault_pipe] * (1 - fault_location))
            self.G.add_edge(t + df.n_node,
                            f + df.n_node,
                            idx=fault_pipe + df.n_pipe,
                            f=df.m[fault_pipe],
                            c=df.K[fault_pipe])
        elif fault_sys == 'r':
            f = df.pipe_to[fault_pipe]
            t = df.pipe_from[fault_pipe]
            self.G.add_edge(f + df.n_node,
                            nfault,
                            idx=fault_pipe + df.n_pipe,
                            f=df.m[fault_pipe],
                            c=df.K[fault_pipe] * fault_location)
            self.G.add_edge(nfault,
                            t + df.n_node,
                            idx=2 * df.n_pipe,
                            f=df.m[fault_pipe],
                            c=df.K[fault_pipe] * (1 - fault_location))
            self.G.add_edge(t,
                            f,
                            idx=fault_pipe,
                            f=df.m[fault_pipe],
                            c=df.K[fault_pipe])
        else:
            raise ValueError(f'Unknown fault sys: {fault_sys}.')

        # create inner edge from node, except for the slack_node
        for i in range(df.n_node):
            f = np.int64(i)
            t = np.int64(i + df.n_node)
            if i not in df.slack_node.tolist()[0:1]:
                self.G.add_edge(f,
                                t,
                                idx=i + 2 * df.n_pipe + 1,
                                f=df.minset[i],
                                c=np.array([0]))
            else:
                self.G.add_node(t)

        self.n_node = len(self.G.nodes)
        self.n_pipe = len(self.G.edges)
        self.Hest = df.Hset

        slack_node = df.slack_node
        non_slack_nodes = np.setdiff1d(np.arange(self.n_node), slack_node)

        # 确定ΔH
        # remove edge between slack and its return dounterpart
        self.H = cal_H(slack_node, [0], self.G)
        self.dH = self.H[slack_node[0]] - self.H[slack_node[0] + df.n_node]

        # 进行潮流分布计算
        self.delta = np.zeros(self.n_node)

        # on-fault hydraulic model
        f = slack_node[0]
        t = slack_node[0] + df.n_node
        self.G.add_edge(f,
                        t,
                        idx=f + 2 * df.n_pipe + 1,
                        f=df.minset[slack_node[0]],
                        c=np.array([0]))

        nenviron = self.G.number_of_nodes()
        g = 10
        self.G.add_edge(nfault,
                        nenviron,
                        idx=2 * df.n_pipe + df.n_node + 1,
                        f=0,
                        c=1 / (2 * g * (np.pi * (df.D[fault_pipe] / 2)) ** 2))

        self.K = [np.array(k['c']).reshape(-1) for i, j, k in sorted(self.G.edges(data=True), key=lambda edge: edge[2].get('idx', 1))]
        self.K = np.asarray(self.K).reshape(-1)

        delta = np.zeros(self.G.number_of_edges())
        delta[slack_node[0] + 2*df.n_pipe + 1] = self.dH

        self.HydraFlow = HydraFlow(np.array(slack_node.tolist() + [nenviron]),
                                   non_slack_nodes,
                                   self.K,
                                   np.zeros(self.G.number_of_nodes()),
                                   np.zeros(self.G.number_of_nodes()),
                                   [Hslack, 0],
                                   delta,
                                   [1],
                                   self.G,
                                   2)
        self.HydraFlow.run()

        # run power flow considering leakage

    def run(self):
        pass


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
