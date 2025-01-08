import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def visualize_ngs(G: nx.DiGraph):
    virtual_pipe = [(u, v) for (u, v, d) in G.edges(data=True) if d["type"] == 0]
    real_pipe = [(u, v) for (u, v, d) in G.edges(data=True) if d["type"] == 1]
    s_node = [n for (n, data) in G.nodes(data=True) if data['type'] == 2]
    ns_node = [n for (n, data) in G.nodes(data=True) if data['type'] == 1]

    # Visualize the graph and the minimum spanning tree
    pos = nx.planar_layout(G, scale=100)
    # pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, nodelist=s_node, node_color='#ff7f0e')
    nx.draw_networkx_nodes(G, pos, nodelist=ns_node, node_color='#17becf')
    nx.draw_networkx_labels(G, pos, font_family="sans-serif")
    nx.draw_networkx_edges(G, pos, edgelist=virtual_pipe, edge_color='grey', alpha=0.5)
    nx.draw_networkx_edges(G, pos, edgelist=real_pipe, edge_color='green')
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels={(u, v): d["idx"] for u, v, d in G.edges(data=True)},
        verticalalignment='top'
    )
    ax = plt.gca()
    ax.margins(10)
    plt.axis("off")
    plt.show()


def load_ngs(filename):
    df = pd.read_excel(filename,
                       sheet_name=None,
                       engine='openpyxl',
                       index_col=None
                       )
    gc = dict()

    type_node = np.asarray(df['node']['type'])
    type_pipe = np.asarray(df['pipe']['type'])
    idx_node = np.asarray(df['node']['idx'])
    idx_pipe = np.asarray(df['pipe']['idx'])
    idx_from = np.asarray(df['pipe']['fnode'])
    compress_fac = np.asarray(df['pipe']['Compress factor'])
    gc['compress_fac'] = compress_fac
    idx_to = np.asarray(df['pipe']['tnode'])
    ns_node_cond = df['node']['type'] == 1
    ns_node = np.asarray(df['node'][ns_node_cond]['idx'])
    gc['ns_node'] = ns_node
    s_node_cond = df['node']['type'] != 1
    s_node = np.asarray(df['node'][s_node_cond]['idx'])
    gc['s_node'] = s_node
    slack_cond = df['node']['type'] == 3
    slack_node = np.asarray(df['node'][slack_cond]['idx'])
    non_slack_source_node = np.setdiff1d(s_node, slack_node)
    non_slack_cond = df['node']['type'] != 3
    non_slack_node = np.setdiff1d(np.arange(len(df['node'])), slack_node)
    gc['slack'] = slack_node
    gc['n_slack'] = len(slack_node)
    gc['non_slack_source'] = non_slack_source_node
    gc['non_slack_node'] = non_slack_node
    n_node = len(df['node'])
    gc['n_node'] = n_node
    gc['n_pipe'] = len(df['pipe'])
    gc['fs'] = np.asarray(df['node']['fs'])
    gc['fl'] = np.asarray(df['node']['fl'])
    gc['delta'] = np.asarray(df['pipe']['delta'])

    # loop detection and conversion
    if 'loop' in df:
        pipe_in_loop = df['loop']['loop1'] == 1
        pinloop = np.asarray(df['loop'][pipe_in_loop]['idx'])
        gc['pinloop'] = pinloop
    else:
        gc['pinloop'] = []

    G = nx.DiGraph()

    for i in range(len(idx_pipe)):
        G.add_node(idx_from[i], type=type_node[idx_from[i]])
        G.add_node(idx_to[i], type=type_node[idx_to[i]])
        G.add_edge(idx_from[i], idx_to[i], idx=idx_pipe[i], type=type_pipe[i])

    A = nx.incidence_matrix(G,
                            nodelist=idx_node,
                            edgelist=sorted(G.edges(data=True), key=lambda edge: edge[2].get('idx', 1)),
                            oriented=True)
    gc['A'] = A
    rpipe_from = []
    rpipe_to = []
    idx_rpipe = []
    pipe_from = []
    pipe_to = []
    idx_pipe = []
    for x, y, z in sorted(G.edges(data=True), key=lambda edge: edge[2].get('idx', 1)):
        if z['type'] == 1:
            rpipe_from.append(x)
            rpipe_to.append(y)
            idx_rpipe.append(z['idx'])
        pipe_from.append(x)
        pipe_to.append(y)
        idx_pipe.append(z['idx'])

    gc['rpipe_from'] = rpipe_from
    gc['rpipe_to'] = rpipe_to
    gc['idx_rpipe'] = idx_rpipe
    gc['pipe_from'] = pipe_from
    gc['pipe_to'] = pipe_to
    gc['idx_pipe'] = idx_pipe
    gc['G'] = G

    lam = np.asarray(df['pipe']['Friction'])
    D = np.asarray(df['pipe']['Diameter'])
    Pi = np.array(df['node']['p'])
    finset = np.array(df['node'][ns_node_cond]['q'])
    non_slack_fin_set = np.array(df['node'][non_slack_cond]['q'])
    Piset = np.array(df['node'][s_node_cond]['p'])
    L = np.asarray(df['pipe']['Length'])
    va = 340
    S = np.pi * (D / 2) ** 2
    C = lam * va ** 2 * L / D / S ** 2 / (1e6 ** 2)
    gc['lam'] = lam
    gc['D'] = D
    gc['Pi'] = Pi
    gc['finset'] = finset
    gc['non_slack_fin_set'] = non_slack_fin_set
    gc['Piset'] = Piset
    gc['L'] = L
    gc['va'] = va
    gc['S'] = S
    gc['C'] = C
    return gc


def load_GT(filename):
    df = pd.read_excel(filename,
                       sheet_name=None,
                       engine='openpyxl',
                       index_col=None
                       )
    gtc = dict()

    for column_name in df['GT'].columns:
        gtc[column_name] = np.asarray(df['GT'][column_name])

    return gtc


def load_P2G(filename):
    df = pd.read_excel(filename,
                       sheet_name=None,
                       engine='openpyxl',
                       index_col=None
                       )
    p2gc = dict()

    for column_name in df['P2G'].columns:
        p2gc[column_name] = np.asarray(df['P2G'][column_name])

    return p2gc

