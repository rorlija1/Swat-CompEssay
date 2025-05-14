# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
from customDataset import CustDataset
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter_add
import shutil,os
from random import shuffle
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, ReduceLROnPlateau


# import argparse
# parser = argparse.ArgumentParser(description="Enter the images directory")
# parser.add_argument("-root_dir",help="images directory",type=str)
# args = parser.parse_args()
# args.root_dir = "graph_inputdata"
root_dir = "graph_inputdata"

def visualize_graph(G, color):
    plt.figure(figsize=(24,24))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                      node_color=color, cmap="Set2")
    plt.show()


class EdgeModel(torch.nn.Module):
    def __init__(self):
        super(EdgeModel, self).__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(2*(2*emb_dim+emb_dim), emb_dim), # 2*num_node_features + num_edge_features
                            # nn.BatchNorm1d(emb_dim),
                            nn.ReLU(),
                            nn.Linear(emb_dim, emb_dim),
                            nn.ReLU())

    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([src, dest, edge_attr], 1) # out = torch.cat([src, dest, edge_attr, u[batch]], 1)
        out = self.edge_mlp(out)
        # out = out  + edge_attr
        return out


class NodeModel(torch.nn.Module):
    def __init__(self):
        super(NodeModel, self).__init__()
        self.node_mlp_1 = nn.Sequential(nn.Linear(2*emb_dim+emb_dim, emb_dim),
                              # nn.BatchNorm1d(emb_dim),
                              nn.ReLU(),
                              nn.Linear(emb_dim, emb_dim),
                              nn.ReLU())
        self.node_mlp_2 = nn.Sequential(nn.Linear(2*emb_dim+emb_dim, emb_dim),
                              # nn.BatchNorm1d(emb_dim),
                              nn.ReLU(),
                              nn.Linear(emb_dim, emb_dim),
                              nn.ReLU())

    def forward(self, x, edge_index, edge_attr, u, batch):
        senders, receivers = edge_index
        out = torch.cat([x[senders], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_add(out, receivers, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1) # out = torch.cat([x, out, u[batch]], dim=1)
        out = self.node_mlp_2(out)
        # out = out + x
        return out


class GlobalModel(torch.nn.Module):
    def __init__(self):
        super(GlobalModel, self).__init__()
        """self.global_mlp = nn.Sequential(nn.Linear(emb_dim+1, emb_dim), #num_node_features + num_edge_features + 1
                              nn.BatchNorm1d(emb_dim),
                              nn.ReLU(),
                              nn.Linear(emb_dim, emb_dim))"""
        self.global_mlp = nn.Identity()

    def forward(self, x, edge_index, edge_attr, u, batch):
        # out = torch.cat([u, scatter_add(x, batch, dim=0)], dim=1)
        # return self.global_mlp(out)
        return self.global_mlp(u)


class InteractionNetwork(torch.nn.Module):
    def __init__(self):
        super(InteractionNetwork, self).__init__()

        MLP_dim = 256 #num. of neurons in fully-connected layers at the end

        self.fc1 = nn.Linear(node_dim, emb_dim)
        self.fc2 = nn.Linear(emb_dim, emb_dim)

        self.fc3 = nn.Linear(emb_dim, MLP_dim)
        self.fc4 = nn.Linear(MLP_dim, MLP_dim)
        self.fc5 = nn.Linear(MLP_dim, output_dim)

        self.edgefc1 = nn.Linear(edge_dim, emb_dim)
        self.edgefc2 = nn.Linear(emb_dim, emb_dim)

        self.act = nn.ReLU()

        self.interactionnetwork = MetaLayer(edge_model=EdgeModel(), node_model=NodeModel(), global_model=None)
        # self.bn = BatchNorm1d(inputs)

    def forward(self, data):

        MP_layers = 7 # num. of message-passing layers
        x, edge_index, edge_attr, globals, batch = data.x, data.edge_index, data.edge_attr, data.globals, data.batch #x = self.bn(x)

        """
        "... graph network architecture, consisting
        of several neural networks. We first embedded the node and edge labels in a
        high-dimensional vector-space using two encoder networks (we used standard multi-layer
        perceptrons). Next, we iteratively updated the embedded node and edge labels
        using two update networks visualized in Fig. 2b. At first, each edge updated based
        on its previous embedding and the embeddings of the two nodes it connected to. After
        all edges were updated in parallel using the same network, the nodes were also updated
        based on the sum of their neighboring edge embeddings and their previous embeddings,
        using a second network. We repeated this procedure several times (typically 7),
        allowing local information to propagate throughout the graph"
        Taken from: https://deepmind.google/discover/blog/towards-understanding-glasses-with-graph-neural-networks/
        """
        #encoder phase
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)

        edge_attr = self.edgefc1(edge_attr)
        edge_attr = self.act(edge_attr)
        edge_attr = self.edgefc2(edge_attr)
        edge_attr = self.act(edge_attr)

        save_x = x
        save_edge_attr = edge_attr

        #Message-passing layers
        for _ in range(MP_layers):
          #concatenate current values with 'save'd values.. concept of residual network
          x = torch.cat([x, save_x], dim=1)
          edge_attr = torch.cat([edge_attr, save_edge_attr], dim=1)

          x, edge_attr, globals = self.interactionnetwork(x=x, edge_index=edge_index, edge_attr=edge_attr, u=None, batch=batch)

        x = self.fc3(x)
        x = self.act(x)
        x = self.fc4(x)
        x = self.act(x)
        x = self.fc5(x)

        return x

class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}



# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Parameters and Hyperparameters
node_dim = 3 # node feature dimension
edge_dim = 2 # 2D relative position vector between nodes
output_dim = 1 # single-dim target (e.g. displacement)
edge_threshold = 5 # two nodes form an edge if separation shorter than threshold
learning_rate = 0.00001
num_epochs = 500
emb_dim = 64 # embedding dimension of the nodes and edges

dataset = CustDataset(h=edge_threshold, root_dir=root_dir)
data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)#, num_workers = batch_size)

cache_directory = 'cached_data'

if os.path.exists(cache_directory): shutil.rmtree(cache_directory)
os.makedirs(cache_directory, exist_ok=True)

# initialize array to store values for calculating mean, std
# NOTE: use only training data to avoid bias
train_targets = []

for i, (data, targets) in enumerate(data_loader):
    torch.save((data,targets), os.path.join(cache_directory, f'data_{i}.pt'))
    mask = np.array(data.train_mask).reshape(-1)
    train_targets.append(targets[0][mask])  # assuming batch_size=1 and targets.shape = (1, num_nodes)

# calculate mean, std of targets
train_targets_tensor = torch.cat(train_targets)
target_mean = train_targets_tensor.mean()
target_std = train_targets_tensor.std()

print(f"(training) target mean: {target_mean.item()}, std: {target_std.item()}")

if os.path.exists("savedmodel.pkl"):
    # resume training from previous model state if available
    print("Loading existing model...")
    model = torch.load("savedmodel.pkl", weights_only=False)
else: # otherwise start new model
    print("Initializing new model...")
    model = InteractionNetwork()

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    #model = nn.DataParallel(model)

model.to(device)

# Loss, optimizer, scheduler
# multistep decays LR by gamma at each milestone
milestones = [75, 150, 250]
gamma_multistep = 0.75
gamma_expo = 0.999

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

scheduler1 = MultiStepLR(optimizer, milestones=milestones, gamma=gamma_multistep)
scheduler2 = ExponentialLR(optimizer, gamma=gamma_expo)
scheduler_plateau = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, min_lr=1e-6)

# Initialize EMA
ema = EMA(model, decay=0.999)
ema.register()

def check_accuracy(cache_files, model, phase, target_mean, target_std, best_RMSE = 100, improvement_counter = 0, best_epoch = 1, epoch = 1):
  model.eval()
  quantify = 0.0
  count = 0
  writefile = f"Actual_and_predicted_values_{phase}"  # Differentiate train/val files

  # Open file in write mode for the first epoch so that older results are cleared and in append mode for subsequent epochs
  mode = "w" if epoch == 0 else "a"

  with open(writefile, mode) as file:
    with torch.no_grad():
      for cache_file in cache_files:
        data, targets = torch.load(os.path.join(cache_directory, cache_file),weights_only=False)
        data = data.to(device)
        #normalize targets
        targets = (targets - target_mean.to(targets.device)) / target_std.to(targets.device)
        targets = targets.to(device)

        scores = model(data)
        scores = scores.reshape(-1)
        targets = targets.reshape(-1)
        mask = np.array(data.train_mask if phase == "train" else data.val_mask).reshape(-1)
                
        # compute normalized RMSE
        quantify += criterion(scores[mask], targets[mask])
        
        count += 1

        for t in np.arange(mask.sum()):
          # write non normalized predictions and targets to file
          file.write("%s %s\n" % ((targets[mask])[t].item(), (scores[mask])[t].item()))
      ema.restore()

    rmse = quantify.item() / count
    file.write("RMSE is %s\n" % (rmse))
    file.close()

    # update best rmse if we are checking validation accuracy
    # do this based on the normalized rmse
    if phase == 'val':
        if rmse < best_RMSE: # save checkpoint and update rmse tracker
            torch.save(model, "bestmodel.pkl")
            print("new best val RMSE: %f" % rmse)
            improvement_counter = 0
            best_epoch = epoch
            return best_epoch, improvement_counter, rmse
        else:
            improvement_counter += 1
            print("no improvement in val RMSE for %d epochs" % improvement_counter)
            return best_epoch, improvement_counter, best_RMSE
            

def record_lr(cache_files, model, lr):
  writefile = "Learning_rates"

  # Open file in write mode for the first epoch so that older results are cleared and in append mode for subsequent epochs
  mode = "w" if epoch == 0 else "a"

  with open(writefile, mode) as file:
    file.write("%f\n" % lr)
    file.close()


# initalize best RMSE tracker for early stopping
best_RMSE = np.inf
# and counter to track how many epochs pass
# until we see improvement in RMSE
stopping_count = 0
# tolerance for how long we wait before early stopping
tol = 50
# track when (which epoch) we got best RMSE for early stopping
best_epoch = 1
# flag for early stop
early_stop = False

# Train Network
for epoch in range(num_epochs):
  if early_stop:
      break
  for phase in ['train', 'val']:
    if phase == 'train':
      model.train()  # Set model to training mode
    else:
      model.eval()   # Set model to evaluate mode
      ema.apply_shadow() # apply EMA to weights

    running_loss = 0.
    count = 0

    cache_files = [f for f in os.listdir(cache_directory) if f.endswith('.pt')]
    if phase == 'train': shuffle(cache_files)
    for cache_file in cache_files:
      #print(cache_file)
      data,targets = torch.load(os.path.join(cache_directory, cache_file),weights_only=False)
      data = data.to(device=device)
      
      # normalize targets
      targets = (targets - target_mean) / target_std
      targets = targets.to(device=device)

      # G = to_networkx(data, to_undirected=True)
      # visualize_graph(G, color=data.color); exit()

      if phase == 'train': mask = data.train_mask
      else: mask = data.val_mask

      # Determine the batch size for this graph
      if phase == 'train': batch_size = 32
      else: batch_size = np.sum(mask)
      true_indices = np.where(mask)[1]
      num_batches = (len(true_indices) + batch_size - 1) // batch_size
      # Iterate through nodes in batches
      for i in range(num_batches):
        optimizer.zero_grad()
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(true_indices))

        batch_indices = true_indices[start_idx:end_idx]
        node_batch = np.zeros(len(mask[0]), dtype=bool)
        node_batch[batch_indices] = True

        with torch.set_grad_enabled(phase == 'train'):
          optimizer.zero_grad()
          scores = model(data)
          scores = scores.reshape(-1); targets = targets.reshape(-1); node_batch = np.array(node_batch).reshape(-1)
          loss = criterion(scores[mask], targets[mask])
          # print(scores[mask].shape, targets[mask].shape)
          running_loss +=loss.item()
          count += 1

          # backward + optimize (update weights) only if in training phase
          if phase == 'train':
            loss.backward()
            #nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update() # update EMA after optimizer step
    
    rmse = running_loss/count
    print_rmse = np.format_float_scientific(rmse, unique=True, precision=8)
    print('{} Loss - epoch {}: {}'.format(phase, epoch, print_rmse))

    if phase == 'train':
        torch.save(model, "savedmodel.pkl")

        check_accuracy(cache_files, model, phase, target_mean, target_std)

    # update learning rate, restore EMA
    if phase == 'val':
        best_epoch, stopping_count, best_RMSE = check_accuracy(cache_files, model, phase, target_mean, target_std, best_RMSE, stopping_count, best_epoch, epoch)
        scheduler_plateau.step(loss.item()) # reduceLRonPlateau
        ema.restore()  # Restore original weights after validation
        if stopping_count > tol:
            print("Stopped training at epoch %d," % epoch)
            print("after no improvement in RMSE over %d epochs." % tol)
            print("Best validation RMSE occured at epoch %d." % best_epoch)
            early_stop = True
            
  # step LR with scheduler and write to file

  record_lr(cache_files, model, optimizer.param_groups[0]['lr'])
  scheduler1.step()
  scheduler2.step()

