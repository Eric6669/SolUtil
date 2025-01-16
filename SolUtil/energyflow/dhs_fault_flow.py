from .dhs_flow import DhsFlow
from .hydraulic_mdl import HydraFlow
import networkx as nx
import numpy as np


# %%
class DhsFaultFlow:

    def __init__(self,
                 df: DhsFlow,
                 fault_pipe,
                 fault_location,
                 fault_sys='s'):
        # create asymmetric hydraulic networks
        self.G = nx.DiGraph()

        # add symmetric edge
        for i in range(len(df.pipe_from)):
            if i != fault_pipe:
                f = df.pipe_from[i]
                t = df.pipe_to[i]
                self.G.add_node(f)
                self.G.add_node(t)
                self.G.add_edge(f, t, idx=i)

                f = df.pipe_to[i] + df.n_node
                t = df.pipe_from[i] + df.n_node
                self.G.add_node(f)
                self.G.add_node(t)
                self.G.add_edge(f, t, idx=i + df.n_pipe)

        # add fault edge
        nfault = 2 * np.int64(df.n_node)
        self.G.add_node(nfault)
        f = df.pipe_from[fault_pipe]
        t = df.pipe_to[fault_pipe]

        if fault_sys == 's':
            self.G.add_edge(f, nfault, idx=fault_pipe)
            self.G.add_edge(nfault, t, idx=2 * df.n_pipe)
            self.G.add_edge(t, f, idx=fault_pipe + df.n_pipe)
        elif fault_sys == 'r':
            self.G.add_edge(f, nfault, idx=fault_pipe + df.n_pipe)
            self.G.add_edge(nfault, t, idx=2 * df.n_pipe)
            self.G.add_edge(t, f, idx=fault_pipe)
        else:
            raise ValueError(f'Unknown fault sys: {fault_sys}.')

        # create inner edge from node
        for i in range(df.n_node):
            f = np.int64(i)
            t = np.int64(i + df.n_node)
            self.G.add_edge(f, t)

        A = nx.incidence_matrix(self.G,
                                nodelist=np.arange(2 * df.n_node + 1),
                                edgelist=sorted(self.G.edges(data=True), key=lambda edge: edge[2].get('idx', 1)),
                                oriented=True)

        self.n_node = len(self.G.nodes)
        self.Hest = df.Hset

        edgelist = sorted(self.G.edges(data=True), key=lambda edge: edge[2].get('idx', 1))

        slack_node = df.slack_node
        non_slack_nodes = 0
        self.HydraFlow = HydraFlow(self.pipe_from,
                                   self.pipe_to,
                                   self.slack_node,
                                   self.non_slack_nodes,
                                   self.K,
                                   np.zeros(self.n_node),
                                   np.zeros(self.n_node),
                                   self.Hset,
                                   self.delta,
                                   self.pinloop,
                                   2)

        # 确定ΔH

        # run power flow considering leakage

    def relabel(self):
        pass
