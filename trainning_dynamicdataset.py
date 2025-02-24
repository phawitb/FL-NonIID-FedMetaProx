import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import time

class FL_GNN(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(FL_GNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 16)
        self.fc2 = torch.nn.Linear(16, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train_federated(model, df_data, algorithm, num_rounds=10, epochs=10, lr=0.005, mu=0.01, meta_lr=0.001, clusters=3):

    df_trains = None
    n = int(df_data.shape[0]/num_rounds)
    df_sample,df_data = random_n_data(df_data,n)
    if not df_trains:
        df_trains = df_sample
    else:
        df_trains = pd.concat([df_trains, df_sample], ignore_index=True)

    while len(df_trains['client_id'].unique()) < len(df_data['client_id'].unique()):
        print("len(df_trains['client_id'].unique())",len(df_trains['client_id'].unique()),len(df_data['client_id'].unique()))
        df_sample,df_data = random_n_data(df_data,n)
        df_trains = pd.concat([df_trains, df_sample], ignore_index=True)

    X_train,y_train = df_to_XY(df_trains)
    X_train_tensor,y_train_tensor = XY_to_tensor(X_train,y_train)
    client_data = df_to_client_data(df_trains,X_train_tensor,y_train_tensor)

    global_model = copy.deepcopy(model)
    global_weights = global_model.state_dict()
    client_models = {client: copy.deepcopy(global_model) for client in client_data.keys()}

    H = {}
    H['algorithm'] = algorithm
    H['total_rounds'] = num_rounds
    H['epochs_per_round'] = epochs
    Acc = []
    n_data = []

    print(f'\nTraining with {algorithm} --------')

    for round_num in range(num_rounds):
        local_updates = []

        if round_num != 0:
            df_sample,df_data = random_n_data(df_data,n)
            df_trains = pd.concat([df_trains, df_sample], ignore_index=True)
            n_data.append(int(df_trains.shape[0]))
            
            X_train,y_train = df_to_XY(df_trains)
            X_train_tensor,y_train_tensor = XY_to_tensor(X_train,y_train)
            client_data = df_to_client_data(df_trains,X_train_tensor,y_train_tensor)

        for client_id, dataset in client_data.items():
            train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
            local_model = copy.deepcopy(client_models[client_id])
            optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)
            
            local_model.train()
            for _ in range(epochs):
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    output = local_model(X_batch)
                    loss = F.nll_loss(output, y_batch)
                    
                    if algorithm in ["FedProx", "FedMetaProx"]:
                        prox_term = sum(torch.norm(param - global_weights[k]) ** 2 for k, param in local_model.named_parameters())
                        loss += (mu / 2) * prox_term
                    
                    loss.backward()
                    optimizer.step()
            
            local_updates.append(copy.deepcopy(local_model.state_dict()))
        
        if algorithm in ["FedAvg", "FedProx"]:
            new_weights = copy.deepcopy(global_weights)
            for key in new_weights.keys():
                new_weights[key] = torch.mean(torch.stack([client[key] for client in local_updates]), dim=0)
            global_model.load_state_dict(new_weights)

        elif algorithm == "FedNova":
            total_steps = sum(len(dataset) for dataset in client_data.values())
            new_weights = copy.deepcopy(global_weights)
            for key in new_weights.keys():
                weighted_updates = sum(len(client_data[i]) * local_updates[i][key] for i in range(len(local_updates)))
                new_weights[key] += (weighted_updates / total_steps)
            global_model.load_state_dict(new_weights)

        elif algorithm == "SCAFFOLD":
            scaffold_updates = copy.deepcopy(global_weights)
            for key in scaffold_updates.keys():
                scaffold_updates[key] = torch.mean(torch.stack([client[key] for client in local_updates]), dim=0) 
            global_model.load_state_dict(scaffold_updates)

        elif algorithm == "FedMetaProx":
            clusters = 3
            kmeans = KMeans(n_clusters=clusters, random_state=0).fit(X)
            cluster_assignments = kmeans.labels_
            cluster_updates = {i: [] for i in range(clusters)}
            
            for client_id, update in enumerate(local_updates):
                cluster_id = cluster_assignments[client_id]
                cluster_updates[cluster_id].append(update)
            
            meta_weights = copy.deepcopy(global_weights)
            for key in meta_weights.keys():
                for cluster_id, updates in cluster_updates.items():
                    if updates:
                        meta_weights[key] = torch.mean(torch.stack([update[key] for update in updates]), dim=0)
            
            global_model.load_state_dict(meta_weights)

        test_acc = evaluate(global_model, TensorDataset(X_test_tensor,y_test_tensor))
        Acc.append(f"{test_acc:.4f}")
        print(f"- [Round {round_num + 1}] n_data: {int(df_trains.shape[0])} Test Accuracy: {test_acc:.4f}")
      
    H['trainng_accuracy'] = Acc
    H['n_data'] = n_data

    return global_model,H

def evaluate(model, test_data):
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            output = model(X_batch)
            pred = output.argmax(dim=1)
            correct += (pred == y_batch).sum().item()
            total += y_batch.size(0)
    return correct / total

def random_data_percen(df_data,percen):
    df_random = df_data.sample(frac=percen, random_state=42)
    df_data = df_data.drop(df_random.index)

    return df_random,df_data

def random_n_data(df_data,n):
    if n > df_data.shape[0]:
        n = df_data.shape[0]
    df_random = df_data.sample(n=n, random_state=42)
    df_data = df_data.drop(df_random.index)

    return df_random,df_data

def df_to_XY(df):
    scaler = StandardScaler()
    X = scaler.fit_transform(df.iloc[:, :-2].values)  
    y = df["label"].values 

    return X,y

def XY_to_tensor(X,y):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    return X_tensor,y_tensor

def df_to_client_data(df,X_tensor,y_tensor):
    clients = df["client_id"].values  
    num_clients = len(np.unique(clients))
    client_data = {}
    for client_id in range(num_clients):
        idx = np.where(clients == client_id)[0]
        client_data[client_id] = TensorDataset(X_tensor[idx], y_tensor[idx])

    return client_data

# ------------------------------------------------------------------------------------
DATA_SET = ['Citeseer','Cora','Pubmed']  
N_CLIENTS = 30
for data_set in DATA_SET:
    for ds in ['noniid','normal']:
        if ds == 'normal':
            df = pd.read_csv(f"datas/{data_set}_data_{N_CLIENTS}client.csv")
        elif ds == 'noniid':
            df = pd.read_csv(f"datas/{data_set}_data_nonIID_{N_CLIENTS}client.csv")

        print(df.shape)

        df_test,df_data = random_data_percen(df,0.2)

        X,y = df_to_XY(df_data)
        X_tensor,y_tensor = XY_to_tensor(X,y)
        client_data = df_to_client_data(df_data,X_tensor,y_tensor)

        X_test,y_test = df_to_XY(df_test)
        X_test_tensor,y_test_tensor = XY_to_tensor(X_test,y_test)

        # ---------------------------------
        algorithms = ["FedAvg","FedProx","FedNova","SCAFFOLD","FedMetaProx"]

        results = []
        idx = time.time()

        for algo in algorithms:

            start_time = time.time()

            model = FL_GNN(input_size=X.shape[1], num_classes=len(np.unique(y)))
            trained_model,H = train_federated(model, df_data, algo,num_rounds=20, epochs=20)
            
            test_acc = evaluate(trained_model, TensorDataset(X_test_tensor,y_test_tensor))

            H['final_accuracy'] = test_acc
            H['trainning_time'] = time.time() - start_time
            H['idx'] = idx
            H['dataset'] = data_set
            H['dataset_type'] = ds

            print('H',H)

            results.append(H)

            df = pd.DataFrame([H])
            file_path = 'datas/trainning_dynamic_historys.csv'
            if os.path.exists(file_path):
                df.to_csv(file_path, mode='a', header=False, index=False)
            else:
                df.to_csv(file_path, mode='w', header=True, index=False)

        print('results',results)
