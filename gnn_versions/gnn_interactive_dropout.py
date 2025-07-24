# This is interactive
# - modified from gnn_noninteractive_refined.py
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
from datetime import datetime


# import argparse
# parser = argparse.ArgumentParser(description="Enter the images directory")
# parser.add_argument("-root_dir",help="images directory",type=str)
# args = parser.parse_args()
# args.root_dir = "graph_inputdata"
root_dir = "graph_inputdata"

# inputs are read from gnn_input.txt file not from user input


task_input = input("Enter task description: ")
snapshot_input = input("Record (for your own benefit!) where you can find the current Snapshot file: ")
subfolder_name = input("Enter desired name of subdirectory for results: (will be created as 'output_[your name]') ")
resume_input = input("Would you like to resume training from a .pkl file? (yes/no) ")
num_epochs = int(input("How many epochs would you like to execute right now? "))
savedmodel_string = input("I would like to write a savedmodel.pkl checkpoint at each epoch: (True/False) ")
savedmodel_input = True if savedmodel_string.lower() == "true" else False

output_dir = f"output_{subfolder_name}"
print("Output will be saved to: ", output_dir)
print("\n")

MLP_dim = 256 #num. of neurons in fully-connected layers at the end
MP_layers = 3 # num. of message-passing layers
p = 0.25 # portion of neurons to discard for dropout

os.makedirs(output_dir, exist_ok=True)

"""
Helper functions:
"""

    
def save_normalization_params(target_mean, target_std, filename=os.path.join(output_dir, "normalization_params.txt")):
    with open(filename, "w") as f:
        f.write(f"target_mean {target_mean.item()}\n")
        f.write(f"target_std {target_std.item()}\n")
    print(f"Normalization parameters saved to {filename}")
    
def restart_prompt(response):

    if response not in ['yes', 'no']:
        raise ValueError("Uh oh! Please enter 'yes' or 'no'.")
    if response == 'no':
        return False, None

    model_choice = input("Would you like to use bestmodel.pkl (1) or savedmodel.pkl (2)? ")
    if model_choice not in ['1', '2']:
        raise ValueError("Uh oh! Please enter best (1) or saved (2).")
    filename = 'bestmodel.pkl' if model_choice == '1' else 'savedmodel.pkl'
    path = os.path.join(output_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Oops, {filename} not found in {output_dir}.")
    return True, path

def write_parameter_summary(output_dir, task, snapshot_path, resume_choice, model_choice, num_epochs, savedmodel_save,
                            MP_layers, MLP_dim, p, edge_threshold, learning_rate, milestones, gamma_multistep, gamma_expo, plateau_factor,
                            plateau_patience, plateau_min, ema_decay, tol):
    summary_file = os.path.join(output_dir, "gnn_parameter_choices.txt")
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d")
    print("\n")
    print(f"Writing parameter choices to {summary_file}")
    print("\n")
    with open(summary_file, "w") as f:
        f.write("----- Parameter Choices -----\n")
        f.write(f"TASK: {task}\n")
        f.write(f"GNN version: gnn_interactive_refined.py\n")
        f.write(f"Trained {date_string}\n\n")
        f.write(f"subdirectory name: {output_dir}\n")
        f.write(f"restart from .pkl file (yes/no): {resume_choice}\n")
        f.write(f"bestmodel.pkl (1) or savedmodel.pkl (2): {model_choice}\n")
        f.write(f"num_epochs: {num_epochs}\n\n")
        f.write(f"Snapshot path: {snapshot_path}\n\n")
        f.write(f"message passing layers: {MP_layers}\n")
        f.write(f"multi-layer perceptron dim: {MLP_dim}\n")
        f.write(f"dropout parameter: {p}")
        f.write(f"edge threshold: {edge_threshold}\n")
        f.write(f"embedded layers dim: {emb_dim}\n")
        f.write(f"learning rate: {learning_rate}\n\n")
        f.write("Scheduler Params\n\n")
        f.write("MultiStep:\n")
        f.write(f"milestones: {milestones}\n")
        f.write(f"decay (multistep): {gamma_multistep}\n\n")
        f.write("Exponential:\n")
        f.write(f"decay (exponential): {gamma_expo}\n\n")
        f.write("ReduceLROnPlateau:\n")
        f.write(f"decay (plateau): {plateau_factor}\n")
        f.write(f"patience: {plateau_patience}\n")
        f.write(f"minimum LR: {plateau_min}\n\n")
        f.write("EMA:\n")
        f.write(f"decay (EMA): {ema_decay}\n\n")
        f.write("Early stopping:\n")
        f.write(f"tolerance (epochs of no improvement): {tol}\n")
    return now

def record_lr(cache_files, model, lr):
  writefile = os.path.join(output_dir, "Learning_rates")

  # Open file in write mode for the first epoch so that older results are cleared and in append mode for subsequent epochs
  mode = "w" if epoch == 0 else "a"

  with open(writefile, mode) as file:
    file.write("%f\n" % lr)
    file.close()
    
"""
RMSE calculator, file writing helper:
"""

def check_accuracy(cache_files, model, phase, target_mean, target_std, best_RMSE = 100, improvement_counter = 0, best_epoch = 1, epoch = 1):
    model.eval()
    quantify = 0.0
    count = 0
    
    # File names for standardized data (existing functionality)
    writefile_std = os.path.join(output_dir, f"Actual_and_predicted_values_{phase}_standardized")
    # File names for original units data (new functionality)
    writefile_orig = os.path.join(output_dir, f"Actual_and_predicted_values_{phase}_original")

    # Open files in write mode for the first epoch, append mode for subsequent epochs
    mode = "w" if epoch == 1 else "a"

    # Store all predictions and actuals for batch processing
    all_predictions_std = []
    all_actuals_std = []
    all_predictions_orig = []
    all_actuals_orig = []

    with open(writefile_std, mode) as file_std, open(writefile_orig, mode) as file_orig:
        with torch.no_grad():
            for cache_file in cache_files:
                data, targets = torch.load(os.path.join(cache_directory, cache_file), weights_only=False)
                data = data.to(device)
                
                # Store original targets before normalization
                targets_orig = targets.clone()
                
                # Normalize targets for model prediction
                targets_std = (targets - target_mean.to(targets.device)) / target_std.to(targets.device)
                targets_std = targets_std.to(device)

                scores_std = model(data)
                scores_std = scores_std.reshape(-1)
                targets_std = targets_std.reshape(-1)
                
                # Convert standardized predictions back to original units
                scores_orig = scores_std * target_std.to(scores_std.device) + target_mean.to(scores_std.device)
                targets_orig = targets_orig.reshape(-1).to(device)
                
                mask = np.array(data.train_mask if phase == "train" else data.val_mask).reshape(-1)
                        
                # Compute normalized RMSE (for model performance tracking)
                quantify += criterion(scores_std[mask], targets_std[mask])
                count += 1

                # Collect masked predictions and actuals
                masked_targets_std = targets_std[mask]
                masked_scores_std = scores_std[mask]
                masked_targets_orig = targets_orig[mask]
                masked_scores_orig = scores_orig[mask]
                
                # Store for batch writing
                all_predictions_std.extend(masked_scores_std.cpu().numpy())
                all_actuals_std.extend(masked_targets_std.cpu().numpy())
                all_predictions_orig.extend(masked_scores_orig.cpu().numpy())
                all_actuals_orig.extend(masked_targets_orig.cpu().numpy())

        # Write all standardized data
        for actual, predicted in zip(all_actuals_std, all_predictions_std):
            file_std.write(f"{actual} {predicted}\n")
            
        # Write all original units data
        for actual, predicted in zip(all_actuals_orig, all_predictions_orig):
            file_orig.write(f"{actual} {predicted}\n")

        ema.restore()

    # Calculate and write RMSE
    rmse = quantify.item() / count
    
    # Append RMSE to both files
    with open(writefile_std, "a") as file_std:
        file_std.write(f"RMSE is {rmse}\n")
    
    with open(writefile_orig, "a") as file_orig:
        # Calculate RMSE in original units for reference
        rmse_orig = np.sqrt(np.mean((np.array(all_predictions_orig) - np.array(all_actuals_orig))**2))
        file_orig.write(f"RMSE is {rmse_orig}\n")

    # Update best RMSE logic remains the same (based on standardized RMSE)
    if phase == 'val':
        if rmse < best_RMSE:
            improvement_counter = 0
            best_epoch = epoch
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr': optimizer.param_groups[0]['lr'],
                'epoch': epoch,
                'rmse': rmse,
                'ema_state': ema.shadow,
                'scheduler1_state': scheduler1.state_dict(),
                'scheduler2_state': scheduler2.state_dict(),
                'scheduler_plateau_state': scheduler_plateau.state_dict(),
                'best_epoch': best_epoch,
                'improvement_counter': stopping_count}, 
                os.path.join(output_dir, 'bestmodel.pkl'))

            print(f"new best val RMSE: {rmse} (standardized), {rmse_orig} (original units)")
            return best_epoch, improvement_counter, rmse
        else:
            improvement_counter += 1
            print(f"no improvement in val RMSE for {improvement_counter} epochs...")
            return best_epoch, improvement_counter, best_RMSE

    return best_epoch, improvement_counter, best_RMSE

"""
Helpers to keep track of time and epochs spent
training, incorporating potential restarts:
"""

def training_meta(meta_path):
    if not os.path.exists(meta_path):
        return 0, 0  # epochs_trained, training_seconds
    with open(meta_path, "r") as f:
        lines = f.readlines()
    epochs_trained = int(lines[0].split(":")[1].strip())
    training_seconds = float(lines[1].split(":")[1].strip())
    return epochs_trained, training_seconds

def save_training_meta(meta_path, start_time, epochs_trained, training_seconds):
    with open(meta_path, "w") as f:
        f.write(f"epochs_trained: {epochs_trained}\n")
        f.write(f"training_seconds: {training_seconds:.3f}\n")
        f.write("NOTE: If you restart from a bestmodel.pkl,\n")
        f.write("     and no improvement occurs, then stop training\n")
        f.write("     and re-restart, then the epochs trained will reflect\n")
        f.write("     this (continue to augment), whereas the epoch counter\n")
        f.write("     printed to STDOUT will restart from the epoch\n")
        f.write("     at which the bestmodel.pkl file was obtained.")
        
def format_time(seconds):
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}h {int(m)}m {int(s)}s"


"""
Model architecture setup:
"""

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

        self.fc1 = nn.Linear(node_dim, emb_dim)
        self.fc2 = nn.Linear(emb_dim, emb_dim)

        self.fc3 = nn.Linear(emb_dim, MLP_dim)
        self.fc4 = nn.Linear(MLP_dim, MLP_dim)
        self.fc5 = nn.Linear(MLP_dim, output_dim)

        self.edgefc1 = nn.Linear(edge_dim, emb_dim)
        self.edgefc2 = nn.Linear(emb_dim, emb_dim)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p)

        self.interactionnetwork = MetaLayer(edge_model=EdgeModel(), node_model=NodeModel(), global_model=None)
        # self.bn = BatchNorm1d(inputs)

    def forward(self, data):

        x, edge_index, edge_attr, globals, batch = data.x, data.edge_index, data.edge_attr, data.globals, data.batch #x = self.bn(x)

        #encoder phase
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.dropout(x)

        edge_attr = self.edgefc1(edge_attr)
        edge_attr = self.act(edge_attr)
        edge_attr = self.dropout(edge_attr)
        edge_attr = self.edgefc2(edge_attr)
        edge_attr = self.act(edge_attr)
        edge_attr = self.dropout(edge_attr)

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
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.act(x)
        x = self.dropout(x)
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
    
    def to(self, device):
        for name in self.shadow:
            self.shadow[name] = self.shadow[name].to(device)
        for name in self.backup:
            self.backup[name] = self.backup[name].to(device)


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available(): 
    device = torch.device("mps") 
    print("MPS (GPU) is available! Nice new Mac?")
    print("Let's use MPS:")
elif torch.cuda.is_available():
    device = torch.device("cuda") 
    print("CUDA is online!") 
else:
    device = torch.device("cpu")
    print("GPU and CUDA unavailable...")
    print("using CPU:")

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    #model = nn.DataParallel(model)

print(f"device: {device}")
print("\n")

# Parameters and Hyperparameters
# NOTE: node_dim = 3 for no pins
#       node_dim = 4 for systems with pins
# (we are telling the model whether particles are pins)
node_dim = 3 # node feature dimension
edge_dim = 2 # 2D relative position vector between nodes
output_dim = 1 # single-dim target (e.g. displacement)
edge_threshold = 5 # two nodes form an edge if separation shorter than threshold
learning_rate = 0.0001
emb_dim = 64 # embedding dimension of the nodes and edges

dataset = CustDataset(h=edge_threshold, root_dir=root_dir)
data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)#, num_workers = batch_size)

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
save_normalization_params(target_mean, target_std) # save to file for ease of access

"""
Set parameters
"""

# Loss, optimizer, scheduler
# multistep decays LR by gamma at each milestone
milestones = [50, 100, 300, 400]
gamma_multistep = 0.9
gamma_expo = 0.999

plateau_factor = 0.9
plateau_patience = 6
plateau_min = 1e-6

ema_decay = 0.999


# tolerance for how long we wait before early stopping
tol = 50

criterion = nn.MSELoss()

resume, resume_path = restart_prompt(resume_input)

"""
Restart or new model logic
"""

if resume:
    print(f"Loading model from {resume_path}...")
    model = InteractionNetwork()
    pickup = torch.load(resume_path)
    model.load_state_dict(pickup['model_state_dict'])

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.load_state_dict(pickup['optimizer_state_dict'])
    
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    learning_rate = pickup['lr']
    old_epoch = pickup['epoch']
    best_RMSE = pickup['rmse']

    # Initialize schedulers before loading
    scheduler1 = MultiStepLR(optimizer, milestones=milestones, gamma=gamma_multistep)
    scheduler2 = ExponentialLR(optimizer, gamma=gamma_expo)
    scheduler_plateau = ReduceLROnPlateau(optimizer, 
                                          mode='min', factor=plateau_factor, 
                                          patience=plateau_patience, min_lr=plateau_min)

    try:
        scheduler1.load_state_dict(pickup['scheduler1_state'])
    except KeyError:
        print("⚠️ Warning: 'scheduler1_state' not found. Initialized from scratch.")

    try:
        scheduler2.load_state_dict(pickup['scheduler2_state'])
    except KeyError:
        print("⚠️ Warning: 'scheduler2_state' not found. Initialized from scratch.")

    try:
        scheduler_plateau.load_state_dict(pickup['scheduler_plateau_state'])
    except KeyError:
        print("⚠️ Warning: 'scheduler_plateau_state' not found. Initialized from scratch.")

    ema = EMA(model, decay=ema_decay)
    ema.register()
    try:
        ema.shadow = pickup['ema_state']
    except KeyError:
        print("⚠️ Warning: 'ema_state' not found. Using freshly initialized EMA.")
    ema.to(device)

    
    # best_epoch and improvement_counter
    best_epoch = pickup['best_epoch']

    if 'best_epoch' not in pickup:
        print("🟡 'best_epoch' not found in checkpoint — defaulting to old_epoch")
        best_epoch = old_epoch
    if 'improvement_counter' in pickup:
        stopping_count = pickup['improvement_counter']
    else:
        stopping_count = 0

    print(f"No improvement in val RMSE for {stopping_count} epochs")
    
    print(f"Running model for {num_epochs} new epochs.")

    num_epochs += old_epoch
    epoch_range = range(old_epoch + 1, num_epochs + 1)

    start_time = write_parameter_summary(output_dir, task_input, snapshot_input, resume_input,
                            resume_path, num_epochs, savedmodel_input,
                            MP_layers, MLP_dim, p, edge_threshold, learning_rate, milestones,
                            gamma_multistep, gamma_expo, plateau_factor,
                            plateau_patience, plateau_min, ema_decay, tol)
    model.to(device)

else:
    print("Initializing new model...")
    
    # If not resuming, wipe meta info (start fresh)
    meta_path = os.path.join(output_dir, "training_meta.txt")
    if os.path.exists(meta_path):
        print("Found and deleted old training_meta.txt file (new run).")
        os.remove(meta_path)

    model = InteractionNetwork()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler1 = MultiStepLR(optimizer, milestones=milestones, gamma=gamma_multistep)
    scheduler2 = ExponentialLR(optimizer, gamma=gamma_expo)
    scheduler_plateau = ReduceLROnPlateau(optimizer, 
                                          mode='min', factor=plateau_factor, 
                                          patience=plateau_patience, min_lr=plateau_min)
    ema = EMA(model, decay=ema_decay)
    best_RMSE = np.inf
    best_epoch = 1
    stopping_count = 0
    epoch_range = range(1, num_epochs + 1)

    start_time = write_parameter_summary(output_dir, task_input, snapshot_input, resume_input,
                            resume_path, num_epochs, savedmodel_input,
                            MP_layers, MLP_dim, p, edge_threshold, learning_rate, milestones,
                            gamma_multistep, gamma_expo, plateau_factor,
                            plateau_patience, plateau_min, ema_decay, tol)

    model.to(device)
    ema.register()

"""
Keeping track of epochs, time trained:
"""
meta_path = os.path.join(output_dir, "training_meta.txt")
prev_epochs, prev_time = training_meta(meta_path)

if prev_time == 0:
    total_epochs = len(epoch_range)
    total_seconds = 0.0
else:
    total_epochs = prev_epochs
    total_seconds = prev_time

"""
Training loop: where the magic happens
"""

# flag for early stop
early_stop = False

start_time = datetime.now()

# Train Network
for epoch in epoch_range:
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
      data = data.to(device)
      
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
        check_accuracy(cache_files, model, phase, target_mean, target_std, epoch=epoch)

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
    
    # save state after updating stopping count, epoch, etc
    if savedmodel_input:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr': optimizer.param_groups[0]['lr'],
            'epoch': epoch,
            'rmse': rmse,
            'ema_state': ema.shadow,
            'scheduler1_state': scheduler1.state_dict(),
            'scheduler2_state': scheduler2.state_dict(),
            'scheduler_plateau_state': scheduler_plateau.state_dict(),
            'best_epoch': best_epoch,
            'improvement_counter': stopping_count}, 
        os.path.join(output_dir, 'savedmodel.pkl'))
            
  # step LR with scheduler and write to file

  record_lr(cache_files, model, optimizer.param_groups[0]['lr'])
  scheduler1.step()
  scheduler2.step()

"""
Record time spent / epochs:
"""

end_time = datetime.now()
this_run_time = end_time - start_time
this_seconds = this_run_time.total_seconds()
total_epochs += len(epoch_range)
total_seconds += this_seconds

# save
save_training_meta(meta_path, start_time, total_epochs, total_seconds)

formatted_this = format_time(this_seconds)
formatted_total = format_time(total_seconds)

# write to file
summary_file = os.path.join(output_dir, "gnn_parameter_choices.txt")
with open(summary_file, "a") as f:
    f.write(f"\nthis run: {len(epoch_range)} epochs in {formatted_this}\n")
    f.write(f"total training: {total_epochs} epochs in {formatted_total}\n")

print(f"\ntime elapsed for this run: {formatted_this} — // total: {formatted_total}")

