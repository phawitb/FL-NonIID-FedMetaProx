import torch
import numpy as np
import pandas as pd
from torch_geometric.datasets import Planetoid
import os
import numpy as np

def count_val(df):
    C = {}
    for c in df['client_id'].unique():
        d = dict(df[df['client_id']==c]['label'].value_counts())
        d = dict(sorted(d.items()))
        
        C[c] = d
    df_count = pd.DataFrame.from_dict(C, orient='index')
    df_count.columns = [f'class_{i}' for i in df_count.columns]
    df_count = df_count.reset_index().rename(columns={'index': 'client_id'}) 
    df_count = df_count[['client_id'] + [col for col in df_count.columns if col != 'client_id']]
    return df_count

def crete_data(data_set,n_client):
    dataset = Planetoid(root="data", name=data_set)
    data = dataset[0]

    features = data.x.numpy()
    labels = data.y.numpy()

    num_nodes = features.shape[0]
    num_clients = n_client 
    client_ids = np.array([i % num_clients for i in range(num_nodes)])

    feature_columns = [f"feature_{i}" for i in range(features.shape[1])]

    df = pd.DataFrame(features, columns=feature_columns)
    df["label"] = labels
    df["client_id"] = client_ids

    return df

def distribute_with_ratio(total, ratio):
    scaled_values = np.array(ratio, dtype=np.float64)
    scaled_values = scaled_values / scaled_values.sum() * total
    
    floored_values = np.floor(scaled_values).astype(int)
    remainder = total - floored_values.sum()
    
    for i in np.random.choice(len(ratio), remainder, replace=False):
        floored_values[i] += 1
    
    return floored_values.tolist()

def shift_left(lst, n=1):
    return lst[n:] + lst[:n]

def randomRow(df,n,target_client,target_cls):
    random_rows = df[df['label']==target_cls].sample(n=n, random_state=42) 
    random_rows['client_id'] = target_client

    df = df.drop(random_rows.index).reset_index(drop=True)
    
    return df,random_rows

DATA_SET = ['Citeseer','Cora','Pubmed']  # --------------------------------------------------------------


folder_name = "datas"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)


for data_set in DATA_SET:
    # initial_distribution = [4, 3, 2, 2, 2, 0, 0,0,0,0,0] 
    initial_distribution = [4,4,4,3,2] + [1]*25  # 30 clients
    n_client = len(initial_distribution)

    ## create data set
    df = crete_data(data_set,n_client)
    df.to_csv(f"datas/{data_set}_data_{n_client}client.csv", index=False)

    print('='*60)
    print(f"saved as {data_set}_data_{n_client}client.csv")
    print(count_val(df))

    ## create Non-IID data set
    S = {}
    d = dict(df['label'].value_counts())
    
    for i,k in enumerate(d):
        total_value = d[k]
        random_result = distribute_with_ratio(total_value, initial_distribution)
        random_result = shift_left(random_result, n=2*i)
        S[k] = random_result
    
    R = []
    orginal_df = df.copy()
    for c in S:
        target_cls = c
        for target_client,n in enumerate(S[c]):
            orginal_df,random_df = randomRow(orginal_df,n,target_client,target_cls)
            R.append(random_df)
    df_noniid = pd.concat(R, ignore_index=True) 
    df_noniid.to_csv(f"datas/{data_set}_data_nonIID_{n_client}client.csv", index=False)

    print('-'*20)
    print(f"saved as {data_set}_data_nonIID_{n_client}client.csv")
    print(count_val(df_noniid))