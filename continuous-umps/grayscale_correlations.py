from functools import partial

import torch
import matplotlib.pyplot as plt

from torchmps.embeddings import (
    unit_interval,
    trig_embed,
    legendre_embed,
    FixedEmbedding,
)


@torch.no_grad()
def make_covariance_mat():
    """
    Convert embedding function into pixel-pixel covariance matrix
    """
    global EMBEDDING_SPEC, EMBEDDING_DIM, FRAMEIFY, NUM_POINTS

    # Set up full embedding function, with option for frameified ingredients
    EMBEDDING_SPEC = EMBEDDING_SPEC.lower()
    assert EMBEDDING_SPEC in ["leg", "trig"]
    embed_map = {"leg": legendre_embed, "trig": trig_embed}
    raw_embed = partial(embed_map[EMBEDDING_SPEC], emb_dim=EMBEDDING_DIM)
    embed_fun = FixedEmbedding(raw_embed, unit_interval, frameify=FRAMEIFY)

    # Compute all inner products between embedding vectors
    points = torch.linspace(0.0, 1.0, steps=NUM_POINTS)
    emb_vecs = embed_fun(points)
    assert emb_vecs.shape == (NUM_POINTS, EMBEDDING_DIM)
    raw_inner_prods = emb_vecs @ emb_vecs.T.conj()

    # Convert the covariance matrix entries into conditional probabilities
    covariance_mat = raw_inner_prods.abs().square()
    assert covariance_mat.shape == (NUM_POINTS, NUM_POINTS)
    return covariance_mat


if __name__ == "__main__":
    # Global configuration options:
    # EMBEDDING_SPEC: String specifying function taking batches of greyscale
    #     values and mapping them to batches of embedded vectors.
    #     Must be either "leg" or "trig".
    # EMBEDDING_DIM: Dimension of the embedded vectors.
    # FRAMEIFY: Whether or not to convert embedding functions into frames.
    # NUM_POINTS: The number of grayscale points to evaluate the covariance
    #     matrix at.
    # BISTOCHASTIC: Whether to convert the covariance matrix into a conditional
    #   probability distribution.
    global EMBEDDING_SPEC, EMBEDDING_DIM, FRAMEIFY, NUM_POINTS
    EMBEDDING_SPEC = "trig"
    EMBEDDING_DIM = 20
    FRAMEIFY = True
    NUM_POINTS = 256
    BISTOCHASTIC = False

    # Prepare covariance matrix
    covariance_mat = make_covariance_mat()
    if BISTOCHASTIC:
        from sinkhorn_knopp import sinkhorn_knopp

        sk = sinkhorn_knopp.SinkhornKnopp()
        covariance_mat = sk.fit(covariance_mat.numpy())
        covariance_mat = torch.tensor(covariance_mat)

    # Plot covariance matrix
    plot_title = (
        f"{EMBEDDING_SPEC.title()} Grayscale-Grayscale Covariance, d={EMBEDDING_DIM}"
    )
    plt.imshow(covariance_mat, extent=[0, 1, 0, 1])
    plt.title(plot_title)
    plt.colorbar()
    plt.show()

    # Plot diagonals of covariance matrix
    emb_norms = covariance_mat.diag().sqrt()
    assert torch.all(emb_norms > 0)
    max_y = 1.05 * emb_norms.max()
    points = torch.linspace(0.0, 1.0, steps=NUM_POINTS)
    plot_title = f"{EMBEDDING_SPEC.title()} Norm of Embedding, d={EMBEDDING_DIM}"
    plt.plot(points, emb_norms)
    plt.ylim(0, max_y)
    plt.title(plot_title)
    plt.show()
