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

def train_federated(model, client_data, algorithm, num_rounds=10, epochs=10, lr=0.005, mu=0.01, meta_lr=0.001, clusters=3):
    global_model = copy.deepcopy(model)
    global_weights = global_model.state_dict()
    client_models = {client: copy.deepcopy(global_model) for client in client_data.keys()}

    H = {}
    H['algorithm'] = algorithm
    H['total_rounds'] = num_rounds
    H['epochs_per_round'] = epochs
    Acc = []

    print(f'\nTraining with {algorithm} --------')
    for round_num in range(num_rounds):
        local_updates = []
        
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
        
        test_acc = evaluate(global_model, TensorDataset(X_tensor, y_tensor))

        Acc.append(f"{test_acc:.4f}")
        print(f"- [Round {round_num + 1}] Test Accuracy: {test_acc:.4f}")
    
    H['trainng_accuracy'] = Acc

    
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

# ------------------------------------------------------------------------------------
DATA_SET = ['Citeseer','Cora','Pubmed']

for data_set in DATA_SET:
    for ds in ['normal','noniid']:
        if ds == 'normal':
            df = pd.read_csv(f"datas/{data_set}_data.csv")
        elif ds == 'noniid':
            df = pd.read_csv(f"datas/{data_set}_data_nonIID.csv")

        # Normalize Features
        scaler = StandardScaler()
        X = scaler.fit_transform(df.iloc[:, :-2].values)
        y = df["label"].values
        clients = df["client_id"].values  

        # Convert to Torch Tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        # Define Dataset Splits per Client
        num_clients = len(np.unique(clients))
        client_data = {}
        for client_id in range(num_clients):
            idx = np.where(clients == client_id)[0]
            client_data[client_id] = TensorDataset(X_tensor[idx], y_tensor[idx])

        # Run Federated Training for Different Algorithms 
        algorithms = ["FedAvg", "FedProx","FedNova","SCAFFOLD","FedMetaProx"]
        results = []
        idx = time.time()

        for algo in algorithms:

            start_time = time.time()

            model = FL_GNN(input_size=X.shape[1], num_classes=len(np.unique(y)))
            trained_model,H = train_federated(model, client_data, algo, num_rounds=100, epochs=100)
            
            test_acc = evaluate(trained_model, TensorDataset(X_tensor, y_tensor))

            H['final_accuracy'] = test_acc
            H['trainning_time'] = time.time() - start_time
            H['idx'] = idx
            H['dataset'] = data_set
            H['dataset_type'] = ds

            print('H',H)

            results.append(H)

            # save history to csv
            df = pd.DataFrame([H])
            file_path = 'datas/trainning_historys.csv'
            if os.path.exists(file_path):
                df.to_csv(file_path, mode='a', header=False, index=False)
            else:
                df.to_csv(file_path, mode='w', header=True, index=False)

        print('results',results)

            