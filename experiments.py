# %%
import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader

# Global seed config
seed = 127
np.random.seed(seed)
torch.manual_seed(seed)

# Global device config
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device("mps" if torch.mps.is_available() else "cpu")

# %%
all_train_data = pd.read_csv("data/train_set.csv")
test_data = pd.read_csv("data/test_set.csv")

all_train_data.head()

# %%
# Hay desbalanceo en las clases
# (1 para preguntas deshonestas, 0 para preguntas honestas)
class_distribution = all_train_data["target"].value_counts()
class_distribution
# %%
all_train_data["question_text"].str.split().apply(len).hist()
# %%
all_train_data["question_text"].str.split().apply(len).describe()
# %%
all_train_data["question_text"].loc[12]
# hay caracteres invisibles. Hay que limpiarlos.

# %%
from app.dataset import get_dataloader

dataloader = get_dataloader(all_train_data)

# %%
import torch

device = torch.device("mps")
x = torch.ones(5, device=device)  # input tensor
y = torch.zeros(3, device=device)  # expected output
w = torch.randn(5, 3, device=device, requires_grad=True)
b = torch.randn(3, device=device, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

# %%
loss.backward()
print(w.grad)
print(b.grad)
# %%
