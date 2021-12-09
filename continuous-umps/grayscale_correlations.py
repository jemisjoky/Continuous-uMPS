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
    assert EMBEDDING_SPEC in ["legendre", "trigonometric"]
    embed_map = {"legendre": legendre_embed, "trigonometric": trig_embed}
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
    #     Must be one of "legendre", "trigonometric", or "binned".
    # EMBEDDING_DIM: Dimension of the embedded vectors.
    # FRAMEIFY: Whether or not to convert embedding functions into frames.
    # NUM_POINTS: The number of grayscale points to evaluate the covariance
    #     matrix at.
    # BISTOCHASTIC: Whether to convert the covariance matrix into a conditional
    #   probability distribution.
    global EMBEDDING_SPEC, EMBEDDING_DIM, FRAMEIFY, NUM_POINTS
    EMBEDDING_SPEC = "legendre"
    EMBEDDING_DIM = 10
    FRAMEIFY = False
    NUM_POINTS = 256
    BISTOCHASTIC = True

    # Prepare covariance matrix
    covariance_mat = make_covariance_mat()
    if BISTOCHASTIC:
        from sinkhorn_knopp import sinkhorn_knopp
        sk = sinkhorn_knopp.SinkhornKnopp()
        covariance_mat = sk.fit(covariance_mat.numpy())

    plot_title = f"{EMBEDDING_SPEC.title()} Grayscale-Grayscale Covariance, d={EMBEDDING_DIM}"
    plt.imshow(covariance_mat, extent=[0, 1, 0, 1])
    plt.title(plot_title)
    plt.show()