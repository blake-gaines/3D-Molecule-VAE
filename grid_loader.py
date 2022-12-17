import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from openbabel import pybel
import torch

pybel.ob.obErrorLog.StopLogging()

class MoleculeGridDataset(Dataset):
    stats_columns = ["index", 'rc_A','rc_B','rc_C','mu','alpha','homo','lumo','gap','r2','zpve','U0','U','H','G','Cv']
    atom_types = [1, 6, 7, 8, 9]

    def __init__(self, directory, filenames, resolution, atom_max_coord=10.6687871363, random_rotation=False):
        self.filenames = filenames
        self.directory = directory
        self.random_rotation = random_rotation
        self.ATOM_MAX_COORD = atom_max_coord
        self.resolution = resolution
        
        self.N_ATOM_TYPES = len(self.atom_types)
        self.type_to_index = pd.Series(range(self.N_ATOM_TYPES), index=self.atom_types)

    def __len__(self):
        return len(self.filenames)

    def get_mol_from_file(self, filename):
        with open(self.directory+filename) as file:
            text = file.readlines()
        mol = pybel.readstring("xyz",''.join(text))
        mol.data["filename"] = filename
        for property, value in zip(self.stats_columns, text[1].split()[1:]):
            mol.data[property] = value
        return mol

    def molecule_to_grid(self, mol):
        grid = np.zeros((self.N_ATOM_TYPES, self.resolution, self.resolution, self.resolution), dtype=np.float64)
        resolution = self.resolution - 1
        for atom in mol.atoms:
            coords = np.array(atom.coords)
            coords = (np.round(coords * (resolution // 2) / self.ATOM_MAX_COORD) + resolution // 2).astype(int)
            grid[self.type_to_index[atom.atomicnum], coords[0], coords[1], coords[2]] = 1.0
        return grid

    def grid_to_molecule(self, grid):
        string = ""
        nonzero = np.round(grid).nonzero()
        num_nonzero = len(nonzero[0])
        if num_nonzero == 0: return "No atoms predicted"
        if num_nonzero > 20: return f"Too many atoms predicted: {num_nonzero}"
        string = f"{num_nonzero}\n\n"
        resolution = self.resolution - 1
        for atom_index in zip(*nonzero):
            coords = ((np.array(atom_index[1:]) - resolution // 2) * self.ATOM_MAX_COORD) / (resolution // 2)
            type = self.atom_types[atom_index[0]]
            string += f"{type} {coords[0]} {coords[1]} {coords[2]}\n"
        return pybel.readstring("xyz", string)

    def random_rotation(self, grid):
        return grid

    def __getitem__(self, idx):
        assert self.random_rotation == False
        mol = self.get_mol_from_file(self.filenames[idx])
        if not mol.atoms: return None, None
        grid = self.molecule_to_grid(mol)
        return grid, float(mol.data["lumo"])

def collate_fn(data):
    # print(len(data[0]))
    grids, values = zip(*data)
    return torch.stack(tuple(torch.tensor(grid) for grid in grids if grid is not None)), torch.stack(tuple(torch.tensor(value) for value in values if value is not None))
    

def setup_data_loaders(directory, filenames, batch_size=1, shuffle=True, use_cuda=False, resolution=32):
    train_filenames, test_filenames = train_test_split(filenames, test_size=0.2, random_state=42)
    return DataLoader(MoleculeGridDataset(directory, train_filenames, resolution=resolution), batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn), DataLoader(MoleculeGridDataset(directory, test_filenames, resolution=resolution), batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
