import pandas as pd
import numpy as np
import os


def load_dataset(directory):
    edges = pd.read_csv(os.path.join(directory, 'elliptic_txs_edgelist.csv'))
    features = pd.read_csv(os.path.join(directory, 'elliptic_txs_features.csv'), header=None)
    classes = pd.read_csv(os.path.join(directory, 'elliptic_txs_classes.csv'))

    return edges, features, classes


def split_by_time_step(edges, features, classes):
    min_time_step = features[1].min()
    max_time_step = features[1].max()

    features_by_timestep = {}
    edges_by_timestep = {}
    classes_by_timestep = {}
    for time_step in range(min_time_step, max_time_step):
        features_time_step = features[features.iloc[:, 1] == time_step]
        features_by_timestep[time_step] = features_time_step

        # According to description on Kaggle, there are no edges connecting transactions from different time steps.
        edges_time_step = edges[(edges.iloc[:, 0].isin(features_time_step[0])) | (edges.iloc[:, 0].isin(features_time_step[0]))]
        edges_by_timestep[time_step] = edges_time_step

        classes_time_step = classes[classes.iloc[:, 0].isin(features_time_step[0])]
        classes_by_timestep[time_step] = classes_time_step

    return edges_by_timestep, features_by_timestep, classes_by_timestep


def split_by_class(features, classes):
    classes_illicit = classes[classes['class'] == '1']
    classes_licit = classes[classes['class'] == '2']
    classes_unknown = classes[classes['class'] == 'unknown']

    features_illicit = features[features.iloc[:, 0].isin(classes_illicit['txId'])]
    features_licit = features[features.iloc[:, 0].isin(classes_licit['txId'])]
    features_unknown = features[features.iloc[:, 0].isin(classes_unknown['txId'])]

    return {
        'classes_illicit': classes_illicit,
        'classes_licit': classes_licit,
        'classes_unknown': classes_unknown,
        'features_illicit': features_illicit,
        'features_licit': features_licit,
        'features_unknown': features_unknown
    }


def get_adjacency_matrix(features, edges):
    adjacency_matrix = pd.DataFrame(np.zeros((features.shape[0], features.shape[0])), index=features[0], columns=features[0])
    for i in range(edges.shape[0]):
        # A[j, i] != A[i, j] if A[i, j] == 1 because it's a Directed Acyclic Graph (DAG)
        adjacency_matrix.loc[edges.iloc[i]['txId1'], edges.iloc[i]['txId2']] = 1

    return adjacency_matrix


def get_transactions_in_with_adjacency_matrix(adjacency_matrix, root_transaction, max_depth=None):
    transactions_in = [i for i in adjacency_matrix.index if adjacency_matrix.loc[i, root_transaction] == 1.0]
    transaction_trees_in = []
    if max_depth is None or max_depth > 0:
        for i in transactions_in:
            transaction_trees_in.append(get_transactions_in_with_adjacency_matrix(adjacency_matrix, i, max_depth - 1 if max_depth is not None else None))
    else:
        transaction_trees_in = 'max_depth reached'

    transactions = {
        'txId': root_transaction,
        'transaction_trees_in': transaction_trees_in
    }
    return transactions


def get_transactions_out_with_adjacency_matrix(adjacency_matrix, root_transaction, max_depth=None):
    transactions_out = [i for i in adjacency_matrix.columns if adjacency_matrix.loc[root_transaction, i] == 1.0]
    transaction_trees_out = []
    if max_depth is None or max_depth > 0:
        for i in transactions_out:
            transaction_trees_out.append(get_transactions_out_with_adjacency_matrix(adjacency_matrix, i, max_depth - 1 if max_depth is not None else None))
    else:
        transaction_trees_out = 'max_depth reached'

    transactions = {
        'txId': root_transaction,
        'transaction_trees_out': transaction_trees_out
    }
    return transactions


def get_adjacency_map(edges):
    adjacency_map = {}
    for _, row in edges.iterrows():
        if not row['txId1'] in adjacency_map:
            adjacency_map[row['txId1']] = []
        if not row['txId2'] in adjacency_map:
            adjacency_map[row['txId2']] = []
        adjacency_map[row['txId1']].append(row['txId2'])

    return adjacency_map


def get_transactions_out_with_adjacency_map(adjacency_map, root_transaction, max_depth=None, max_size=None):
    transactions_main_tree = {
        'txId': root_transaction,
        'transaction_trees_out': []
    }
    search_queue = [root_transaction]
    depth_queue = [0]
    transaction_trees_queue = [transactions_main_tree]
    size = 1

    while True:
        if len(search_queue) == 0:
            break

        transaction = search_queue.pop(0)
        depth = depth_queue.pop(0) + 1
        transaction_tree = transaction_trees_queue.pop(0)
        transactions_out = adjacency_map[transaction]

        if max_depth is not None and depth >= max_depth:
            transaction_tree['transaction_trees_out'] = 'max_depth reached'
        elif max_size is not None and size + len(transactions_out) >= max_size:
            transaction_tree['transaction_trees_out'] = 'max_size reached'
        else:
            for i in transactions_out:
                transaction_tree['transaction_trees_out'].append({
                    'txId': i,
                    'transaction_trees_out': []
                })
                search_queue.append(i)
                depth_queue.append(depth)
                transaction_trees_queue.append(transaction_tree['transaction_trees_out'][-1])
                size += 1
    
    return transactions_main_tree


def get_inversed_adjacency_map(edges):
    inversed_adjacency_map = {}
    for index, row in edges.iterrows():
        if not row['txId1'] in inversed_adjacency_map:
            inversed_adjacency_map[row['txId1']] = []
        if not row['txId2'] in inversed_adjacency_map:
            inversed_adjacency_map[row['txId2']] = []
        inversed_adjacency_map[row['txId2']].append(row['txId1'])

    return inversed_adjacency_map


def get_transactions_in_with_inversed_adjacency_map(inversed_adjacency_map, root_transaction, max_depth=None, max_size=None):
    transactions_main_tree = {
        'txId': root_transaction,
        'transaction_trees_in': []
    }
    search_queue = [root_transaction]
    depth_queue = [0]
    transaction_trees_queue = [transactions_main_tree]
    size = 1

    while True:
        if len(search_queue) == 0:
            break

        transaction = search_queue.pop(0)
        depth = depth_queue.pop(0) + 1
        transaction_tree = transaction_trees_queue.pop(0)
        transactions_in = inversed_adjacency_map[transaction]

        if max_depth is not None and depth >= max_depth:
            transaction_tree['transaction_trees_in'] = 'max_depth reached'
        elif max_size is not None and size + len(transactions_in) >= max_size:
            transaction_tree['transaction_trees_in'] = 'max_size reached'
        else:
            for i in transactions_in:
                transaction_tree['transaction_trees_in'].append({
                    'txId': i,
                    'transaction_trees_in': []
                })
                search_queue.append(i)
                depth_queue.append(depth)
                transaction_trees_queue.append(transaction_tree['transaction_trees_in'][-1])
                size += 1
    
    return transactions_main_tree