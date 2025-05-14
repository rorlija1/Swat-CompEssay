import numpy as np
import os
import torch
from torch.utils.data import Dataset
# from torch_geometric.data import Dataset
from torch_geometric.data import Data

class CustDataset(Dataset):
  
    def __init__(self, h, root_dir):
        self.root_dir = root_dir
        self.h = h
        self.data_files = [filename for filename in os.listdir(root_dir) if filename.endswith('.graphdata')]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        # print(index,self.data_files[index])
        
        # atomtype,x,y,wall_flag,target,training_flag
        img_path = os.path.join(self.root_dir, self.data_files[index])
        x = np.loadtxt(img_path)
        h = self.h

        temp = np.array(x)
        box = temp[0,:2] # simulation box length in x and y dimensions
        positions = temp[1:,1:3]
        wall_flag = temp[1:,3]
        target = temp[1:,4]
        
        # identifying nodes that are part of training and validation data
        train_mask = (temp[1:,5]==1)
        val_mask = (temp[1:,5]==0)
        print(train_mask.sum(),val_mask.sum())

        # Atom type encoding (one-hot)
        atom_type = torch.tensor(temp[1:,0] - 1) # atom type (assuming 0-based indexing)
        atom_type = torch.nn.functional.one_hot(atom_type.to(torch.int64), num_classes=-1) #0->[1,0]; 1->[0,1]
        atom_type = atom_type.clone().detach().to(torch.float)
        
        # Wall flag encoding (binary feature)
        wall_flag = torch.tensor(wall_flag).unsqueeze(1).float()  # (n_nodes, 1)
    
        # Concatenate atom type and wall flag to create final node features
        node_features = torch.cat([atom_type, wall_flag], dim=1)  # Concatenate along the feature dimension
        
        
        y = torch.tensor(target, dtype=torch.float)

        # Calculate pairwise relative distances between particles: shape [n, n, 2].
        cross_positions = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
        # Enforce periodic boundary conditions in the x-direction only
        box_x = box[0]
        cross_positions[:, :, 0] += (cross_positions[:, :, 0] < -box_x / 2.) * box_x
        cross_positions[:, :, 0] -= (cross_positions[:, :, 0] > box_x / 2.) * box_x


        distances = np.linalg.norm(cross_positions, axis=-1)
        indices = np.where(distances < h)
        # senders = list(indices[0]); receivers = list(indices[1])
        mask = indices[0] != indices[1]
        senders = list(indices[0][mask]); receivers = list(indices[1][mask])
        edge_index = torch.tensor([senders,receivers],dtype=torch.long)
        
        edges = cross_positions[senders[0],receivers[0]]
        # edges1 = np.linalg.norm(cross_positions[senders[0],receivers[0]])
        for i,j in zip(senders[1:],receivers[1:]):
          edges = np.vstack([edges, cross_positions[i,j]])
 
        edge_attr = torch.tensor(edges, dtype=torch.float)

        globals = torch.zeros((1,1),dtype=torch.float)
            
        positions = torch.tensor(positions, dtype=torch.float)
          
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, globals=globals, train_mask=train_mask, val_mask=val_mask)
        
        # To visualize the graph structure
        """
        mask = temp[1:,0]
        mask = torch.tensor(mask, dtype=torch.int)
        mask = mask.bool()
        color = mask
        data = Data(color=color,x=node_features, edge_index=edge_index, edge_attr=edge_attr, globals=globals, train_mask=train_mask, val_mask=val_mask)
        """
        
        
        return (data,y)
