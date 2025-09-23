import warnings

import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import csc_array


def load_hs(filename):
    df = pd.read_excel(filename,
                       sheet_name=None,
                       engine='openpyxl',
                       index_col=None
                       )
    hc = dict()

    type_node = np.asarray(df['node']['type'])
    idx_node = np.asarray(df['node']['idx'])
    idx_pipe = np.asarray(df['pipe']['idx'])
    idx_from = np.asarray(df['pipe']['fnode'])
    idx_to = np.asarray(df['pipe']['tnode'])
    I_node_cond = df['node']['type'] == 1
    I_node = np.asarray(df['node'][I_node_cond]['idx'])
    hc['I_node'] = I_node
    s_node_cond = df['node']['type'] == 0
    s_node = np.asarray(df['node'][s_node_cond]['idx'])
    hc['s_node'] = s_node
    l_node_cond = df['node']['type'] == 2
    l_node = np.asarray(df['node'][l_node_cond]['idx'])
    hc['l_node'] = l_node
    if len(l_node) == 0:
        warnings.warn('No DHS load node!')
    slack_node_cond = df['node']['type'] == 3
    slack_node = np.asarray(df['node'][slack_node_cond]['idx'])
    hc['slack_node'] = slack_node
    n_node = len(df['node'])
    hc['n_node'] = n_node
    n_pipe = len(df['pipe'])
    hc['n_pipe'] = n_pipe
    non_slack_nodes = np.setdiff1d(np.arange(n_node), slack_node)
    hc['non_slack_nodes'] = non_slack_nodes

    if 'delta' in df['node'].columns:
        delta_node = np.asarray(df['node']['delta'])
    else:
        delta_node = np.zeros(n_node)
    hc['delta_node'] = delta_node

    if 'delta' in df['pipe'].columns:
        delta_pipe = np.asarray(df['pipe']['delta'])
    else:
        delta_pipe = np.zeros(n_pipe)
    hc['delta_pipe'] = delta_pipe

    # loop detection and conversion
    if 'loop' in df:
        hc['pinloop'] = np.asarray(df['loop']['loop1'])
    else:
        hc['pinloop'] = np.zeros(n_pipe)

    lam = np.asarray(df['pipe']['lambda (W/mK)'])
    D = np.asarray(df['pipe']['D (mm)']) / 1000
    Ts = np.array(df['node']['Ts'])
    Tr = np.array(df['node']['Tr'])
    L = np.asarray(df['pipe']['L (m)'])
    S = np.pi * (D / 2) ** 2
    m = np.asarray(df['pipe']['Massflow (kg/s)'])
    hc['m'] = m

    density = 958.4
    g = 10
    mu = 0.294e-6
    epsilon = 1.25e-3

    # calculate K
    # Calculate absolute velocity
    v = np.abs(m) / (density * np.pi * (D ** 2) / 4)

    # Calculate Reynolds number
    Re = v * D / mu

    # Initialize friction factor array
    f = np.zeros(hc['n_pipe'])

    # Indices where Re < 2300
    low_Re_indices = np.where(Re < 2300)[0]
    f[low_Re_indices] = 64. / Re[low_Re_indices]

    # Indices where 2300 <= Re <= 4000
    mid_Re_indices = np.where((Re >= 2300) & (Re <= 4000))[0]
    f[mid_Re_indices] = (((colebrook(4000, epsilon / D[mid_Re_indices]) - 64 / 2300) / (4000 - 2300)) *
                         (Re[mid_Re_indices] - 2300) + 64 / 2300)

    # Indices where Re > 4000
    high_Re_indices = np.where(Re > 4000)[0]
    f[high_Re_indices] = colebrook(Re[high_Re_indices], epsilon / D[high_Re_indices])

    # Calculate K
    K = 8 * L * f / (D ** 5 * density ** 2 * np.pi ** 2 * g)

    hc['K'] = K

    G = nx.DiGraph()

    for i in range(len(idx_pipe)):
        G.add_node(idx_from[i], type=type_node[idx_from[i]])
        G.add_node(idx_to[i], type=type_node[idx_to[i]])
        G.add_edge(idx_from[i],
                   idx_to[i],
                   idx=idx_pipe[i],
                   c=K[i])

    A = nx.incidence_matrix(G,
                            nodelist=idx_node,
                            edgelist=sorted(G.edges(data=True), key=lambda edge: edge[2].get('idx', 1)),
                            oriented=True)
    hc['A'] = A
    pipe_from = []
    pipe_to = []
    idx_pipe = []
    for x, y, z in sorted(G.edges(data=True), key=lambda edge: edge[2].get('idx', 1)):
        pipe_from.append(x)
        pipe_to.append(y)
        idx_pipe.append(z['idx'])

    hc['pipe_from'] = pipe_from
    hc['pipe_to'] = pipe_to
    hc['idx_pipe'] = idx_pipe
    hc['G'] = G

    hc['lam'] = lam
    hc['D'] = D
    hc['Ts'] = Ts
    hc['Tr'] = Tr
    hc['L'] = L
    hc['S'] = S
    hc['Ta'] = np.asarray(df['setting']['Ta'])
    hc['Tsource'] = np.asarray(df['setting']['Tsource'])
    hc['Tload'] = np.asarray(df['setting']['Tload'])
    hc['phi'] = np.asarray(df['node']['phi (MW)'], dtype=float)
    if 'Hset' in df['node']:
        hc['Hset'] = np.asarray(df['node']['Hset'][slack_node])
    else:
        hc['Hset'] = np.zeros(slack_node.shape[0])

    s_slack_node = s_node.tolist() + slack_node.tolist()
    Cs = csc_array((np.ones(len(s_slack_node)), (s_slack_node, s_slack_node)),
                   shape=(hc['n_node'], hc['n_node']))
    Cl = csc_array((np.ones(len(hc['l_node'])), (hc['l_node'], hc['l_node'])),
                   shape=(hc['n_node'], hc['n_node']))
    Ci = csc_array((np.ones(len(hc['I_node'])), (hc['I_node'], hc['I_node'])),
                   shape=(hc['n_node'], hc['n_node']))
    hc['Cs'] = Cs
    hc['Cl'] = Cl
    hc['Ci'] = Ci

    return hc


def load_N_1_hs(filename, pipes_to_remove):
    df = pd.read_excel(filename,
                       sheet_name=None,
                       engine='openpyxl',
                       index_col=None
                       )

    new_node_df, new_pipe_df, new_loop_df, node_map, pipe_map = ex2in(
        original_nodedata=df['node'],
        original_pipedata=df['pipe'],
        original_loop=df['loop'],
        pipes_to_remove=pipes_to_remove)

    df['node'] = new_node_df
    df['pipe'] = new_pipe_df
    df['loop'] = new_loop_df

    hc = dict()

    type_node = np.asarray(df['node']['type'])
    idx_node = np.asarray(df['node']['idx'])
    idx_pipe = np.asarray(df['pipe']['idx'])
    idx_from = np.asarray(df['pipe']['fnode'])
    idx_to = np.asarray(df['pipe']['tnode'])
    I_node_cond = df['node']['type'] == 1
    I_node = np.asarray(df['node'][I_node_cond]['idx'])
    hc['I_node'] = I_node
    s_node_cond = df['node']['type'] == 0
    s_node = np.asarray(df['node'][s_node_cond]['idx'])
    hc['s_node'] = s_node
    l_node_cond = df['node']['type'] == 2
    l_node = np.asarray(df['node'][l_node_cond]['idx'])
    hc['l_node'] = l_node
    if len(l_node) == 0:
        warnings.warn('No DHS load node!')
    slack_node_cond = df['node']['type'] == 3
    slack_node = np.asarray(df['node'][slack_node_cond]['idx'])
    hc['slack_node'] = slack_node
    n_node = len(df['node'])
    hc['n_node'] = n_node
    n_pipe = len(df['pipe'])
    hc['n_pipe'] = n_pipe
    non_slack_nodes = np.setdiff1d(np.arange(n_node), slack_node)
    hc['non_slack_nodes'] = non_slack_nodes

    if 'delta' in df['node'].columns:
        delta_node = np.asarray(df['node']['delta'])
    else:
        delta_node = np.zeros(n_node)
    hc['delta_node'] = delta_node

    if 'delta' in df['pipe'].columns:
        delta_pipe = np.asarray(df['pipe']['delta'])
    else:
        delta_pipe = np.zeros(n_pipe)
    hc['delta_pipe'] = delta_pipe

    # loop detection and conversion
    if 'loop' in df:
        hc['pinloop'] = np.asarray(df['loop']['loop1'])
    else:
        hc['pinloop'] = np.zeros(n_pipe)

    lam = np.asarray(df['pipe']['lambda (W/mK)'])
    D = np.asarray(df['pipe']['D (mm)']) / 1000
    Ts = np.array(df['node']['Ts'])
    Tr = np.array(df['node']['Tr'])
    L = np.asarray(df['pipe']['L (m)'])
    S = np.pi * (D / 2) ** 2
    m = np.asarray(df['pipe']['Massflow (kg/s)'])
    hc['m'] = m

    density = 958.4
    g = 10
    mu = 0.294e-6
    epsilon = 1.25e-3

    # calculate K
    # Calculate absolute velocity
    v = np.abs(m) / (density * np.pi * (D ** 2) / 4)

    # Calculate Reynolds number
    Re = v * D / mu

    # Initialize friction factor array
    f = np.zeros(hc['n_pipe'])

    # Indices where Re < 2300
    low_Re_indices = np.where(Re < 2300)[0]
    f[low_Re_indices] = 64. / Re[low_Re_indices]

    # Indices where 2300 <= Re <= 4000
    mid_Re_indices = np.where((Re >= 2300) & (Re <= 4000))[0]
    f[mid_Re_indices] = (((colebrook(4000, epsilon / D[mid_Re_indices]) - 64 / 2300) / (4000 - 2300)) *
                         (Re[mid_Re_indices] - 2300) + 64 / 2300)

    # Indices where Re > 4000
    high_Re_indices = np.where(Re > 4000)[0]
    f[high_Re_indices] = colebrook(Re[high_Re_indices], epsilon / D[high_Re_indices])

    # Calculate K
    K = 8 * L * f / (D ** 5 * density ** 2 * np.pi ** 2 * g)

    hc['K'] = K

    G = nx.DiGraph()

    for i in range(len(idx_pipe)):
        G.add_node(idx_from[i], type=type_node[idx_from[i]])
        G.add_node(idx_to[i], type=type_node[idx_to[i]])
        G.add_edge(idx_from[i],
                   idx_to[i],
                   idx=idx_pipe[i],
                   c=K[i])

    A = nx.incidence_matrix(G,
                            nodelist=idx_node,
                            edgelist=sorted(G.edges(data=True), key=lambda edge: edge[2].get('idx', 1)),
                            oriented=True)
    hc['A'] = A
    pipe_from = []
    pipe_to = []
    idx_pipe = []
    for x, y, z in sorted(G.edges(data=True), key=lambda edge: edge[2].get('idx', 1)):
        pipe_from.append(x)
        pipe_to.append(y)
        idx_pipe.append(z['idx'])

    hc['pipe_from'] = pipe_from
    hc['pipe_to'] = pipe_to
    hc['idx_pipe'] = idx_pipe
    hc['G'] = G

    hc['lam'] = lam
    hc['D'] = D
    hc['Ts'] = Ts
    hc['Tr'] = Tr
    hc['L'] = L
    hc['S'] = S
    hc['Ta'] = np.asarray(df['setting']['Ta'])
    hc['Tsource'] = np.asarray(df['setting']['Tsource'])
    hc['Tload'] = np.asarray(df['setting']['Tload'])
    hc['phi'] = np.asarray(df['node']['phi (MW)'], dtype=float)
    if 'Hset' in df['node']:
        hc['Hset'] = np.asarray(df['node']['Hset'][slack_node])
    else:
        hc['Hset'] = np.zeros(slack_node.shape[0])

    s_slack_node = s_node.tolist() + slack_node.tolist()
    Cs = csc_array((np.ones(len(s_slack_node)), (s_slack_node, s_slack_node)),
                   shape=(hc['n_node'], hc['n_node']))
    Cl = csc_array((np.ones(len(hc['l_node'])), (hc['l_node'], hc['l_node'])),
                   shape=(hc['n_node'], hc['n_node']))
    Ci = csc_array((np.ones(len(hc['I_node'])), (hc['I_node'], hc['I_node'])),
                   shape=(hc['n_node'], hc['n_node']))
    hc['Cs'] = Cs
    hc['Cl'] = Cl
    hc['Ci'] = Ci

    hc['node_map'] = node_map
    hc['pipe_map'] = pipe_map

    return hc


def colebrook(R, K=None):
    """
    Compute the Darcy-Weisbach friction factor according to the Colebrook formula.

    Parameters:
    R : array_like
        Reynolds' number (should be > 2300).
    K : array_like or None
        Equivalent sand roughness height divided by the hydraulic diameter.
        If None, default value is set to 0.

    Returns:
    F : array_like
        Friction factor.

    Raises:
    ValueError
        If any value in R is non-positive or any value in K is negative.
    """

    # Check for input validity
    if np.any(R <= 0):
        raise ValueError("The Reynolds number must be positive (R>2300).")
    if K is None:
        K = np.zeros_like(R)
    elif np.any(K < 0):
        raise ValueError("The relative sand roughness must be non-negative.")

    # Constants used in initialization
    X1 = K * R * 0.123968186335417556  # X1 ≈ K * R * log(10) / 18.574
    X2 = np.log(R) - 0.779397488455682028  # X2 ≈ log(R*log(10) / 5.02)

    # Initial guess for F
    F = X2 - 0.2  # F ≈ X2 - 1/5

    # First iteration
    E = (np.log(X1 + F) + F - X2) / (1 + X1 + F)
    F = F - (1 + X1 + F + 0.5 * E) * E * (X1 + F) / (1 + X1 + F + E * (1 + E / 3))

    # Second iteration for higher accuracy
    E = (np.log(X1 + F) + F - X2) / (1 + X1 + F)
    F = F - (1 + X1 + F + 0.5 * E) * E * (X1 + F) / (1 + X1 + F + E * (1 + E / 3))

    # Finalizing the solution
    F = 1.151292546497022842 / F  # F ≈ 0.5 * log(10) / F
    F = F ** 2  # Square F to get the friction factor

    return F

def ex2in(original_nodedata: pd.DataFrame,
          original_pipedata: pd.DataFrame,
          original_loop: pd.DataFrame,
          pipes_to_remove: list,
          col_map: dict = None):
    """
    Removes specified pipes, cleans up isolated nodes, re-indexes the network,
    and correctly updates and returns the new loop data.

    Returns:
        tuple: A tuple containing:
            - new_node_data (pd.DataFrame): Re-indexed node data.
            - new_pipe_data (pd.DataFrame): Re-indexed pipe data (without loop columns).
            - new_loop_data (pd.DataFrame): Re-indexed and updated loop data.
            - node_map (pd.DataFrame): Mapping from new to original node IDs.
            - pipe_map (pd.DataFrame): Mapping from new to original pipe IDs.
    """
    # --- Define column names ---
    if col_map is None:
        col_map = {
            'node_id': 'idx', 'pipe_id': 'idx',
            'fnode': 'fnode', 'tnode': 'tnode',
            'loop_cols': ['loop1']  # List of loop column names
        }

    node_df = original_nodedata.copy()

    # --- Merge pipe and loop data for unified processing ---
    pipe_df = pd.merge(original_pipedata, original_loop, on=col_map['pipe_id'])

    # --- Check if the loop is broken BEFORE removing pipes ---
    is_loop_broken = False
    loop_cols = col_map.get('loop_cols', [])
    if loop_cols and pipes_to_remove:
        # Check across all specified loop columns
        is_in_loop_mask = (pipe_df[loop_cols] != 0).any(axis=1)
        original_loop_pipe_ids = set(pipe_df[is_in_loop_mask][col_map['pipe_id']])

        if original_loop_pipe_ids.intersection(set(pipes_to_remove)):
            is_loop_broken = True
            print("Info: A pipe within a loop was removed. The loop is now considered broken.")

    # --- Remove specified pipes ---
    if pipes_to_remove is None or not pipes_to_remove:
        remaining_pipes = pipe_df
    else:
        pipes_to_keep_mask = ~pipe_df[col_map['pipe_id']].isin(pipes_to_remove)
        remaining_pipes = pipe_df[pipes_to_keep_mask]

    if remaining_pipes.empty:
        # Return empty dataframes if all pipes are gone
        print("Warning: All pipes were removed, network is empty.")
        empty_node_df = original_nodedata.iloc[0:0]
        empty_pipe_df = original_pipedata.iloc[0:0]
        empty_loop_df = original_loop.iloc[0:0]
        empty_node_map = pd.DataFrame(columns=['NewNodeID', 'OriginalNodeID'])
        empty_pipe_map = pd.DataFrame(columns=['NewPipeID', 'OriginalPipeID'])
        return empty_node_df, empty_pipe_df, empty_loop_df, empty_node_map, empty_pipe_map

    # --- Remove isolated nodes and Re-index (same as before) ---
    fnodes = remaining_pipes[col_map['fnode']]
    tnodes = remaining_pipes[col_map['tnode']]
    active_original_node_ids = pd.unique(np.concatenate([fnodes, tnodes]))
    nodes_to_keep_mask = node_df[col_map['node_id']].isin(active_original_node_ids)
    remaining_nodes = node_df[nodes_to_keep_mask].reset_index(drop=True)

    original_node_ids = remaining_nodes[col_map['node_id']].copy()
    new_node_ids = np.arange(len(remaining_nodes))
    node_map = pd.DataFrame({'NewNodeID': new_node_ids, 'OriginalNodeID': original_node_ids})
    new_node_data = remaining_nodes.copy()
    new_node_data[col_map['node_id']] = new_node_ids

    original_pipe_ids = remaining_pipes[col_map['pipe_id']].copy()
    new_pipe_ids = np.arange(len(remaining_pipes))
    pipe_map = pd.DataFrame({'NewPipeID': new_pipe_ids, 'OriginalPipeID': original_pipe_ids})

    # Use the full remaining_pipes df for now, we will split it later
    new_full_pipe_data = remaining_pipes.copy().reset_index(drop=True)
    new_full_pipe_data[col_map['pipe_id']] = new_pipe_ids

    node_lookup = node_map.set_index('OriginalNodeID')['NewNodeID']
    new_full_pipe_data[col_map['fnode']] = new_full_pipe_data[col_map['fnode']].map(node_lookup)
    new_full_pipe_data[col_map['tnode']] = new_full_pipe_data[col_map['tnode']].map(node_lookup)
    new_full_pipe_data[[col_map['fnode'], col_map['tnode']]] = new_full_pipe_data[
        [col_map['fnode'], col_map['tnode']]].astype(int)

    # --- Apply the loop breaking logic ---
    if is_loop_broken:
        # If the loop was broken, set all loop values in the new data to 0
        for col in loop_cols:
            if col in new_full_pipe_data.columns:
                new_full_pipe_data[col] = 0

    # --- Create the final new_pipe_data and new_loop_data ---
    pipe_cols = original_pipedata.columns
    new_pipe_data = new_full_pipe_data[pipe_cols]

    loop_cols_to_keep = [col_map['pipe_id']] + loop_cols
    new_loop_data = new_full_pipe_data[loop_cols_to_keep]

    return new_node_data, new_pipe_data, new_loop_data, node_map, pipe_map