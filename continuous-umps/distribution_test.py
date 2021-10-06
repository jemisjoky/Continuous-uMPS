import torch
import matplotlib.pyplot as plt
import numpy as np
from torchmps import ProbMPS
from torchmps.embeddings import DataDomain


def discreteEmbedding(input):
    input = input.cpu()
    d1 = torch.tensor(np.ones(input.shape) * (input.numpy() > 0.2))
    d2 = torch.tensor(np.ones(input.shape) * (input.numpy() < 0.2))
    ret = torch.stack([d1, d2], dim=-1)
    return ret.float()


discreteDomain = DataDomain(False, 2)


def sincosEmbedding(input):
    input = input.cpu()
    d1 = np.cos(input * np.pi / 2)
    d2 = np.sin(input * np.pi / 2)
    ret = torch.stack([d1, d2], dim=-1)
    return ret.float()


sincosDomain = DataDomain(True, 1, 0)


def compute(input, embedding, domain):

    bond_dim = 2
    input_dim = 2
    # batch_size = 100
    sequence_len = 2
    complex_params = False

    my_mps = ProbMPS(
        sequence_len,
        input_dim,
        bond_dim,
        complex_params,
        embed_fun=embedding,
        domain=domain,
    )
    probs = my_mps.forward(input)
    return torch.exp(probs).detach().numpy()


# def genProbs():


nPoints = 20
valX1 = np.linspace(0, 1, nPoints)
valX2 = np.linspace(0, 1, nPoints)

X1, X2 = np.meshgrid(valX1, valX2)

print(X1)
print(X2)

batchInputs = (
    torch.stack([torch.tensor(X1.flatten()), torch.tensor(X2.flatten())], dim=-1)
    .float()
    .transpose(0, 1)
)


probs = compute(batchInputs, sincosEmbedding, sincosDomain)
probs = probs.reshape(nPoints, nPoints)
print(probs)

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

ax.plot_surface(X1, X2, probs)
ax.set_zlim(bottom=0, top=np.max(probs))
plt.show()
