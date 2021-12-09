import torch
import matplotlib.pyplot as plt

from torchmps.embeddings import (
    unit_interval,
    trig_embed,
    legendre_embed,
    FixedEmbedding,
)


@torch.no_grad()
def make_covariance_mat(embedding_spec, embedding_dim, frameify=True, num_points=256):
    """
    Convert embedding function into pixel-pixel covariance matrix

    Args:
        embedding_spec: String specifying function taking batches of greyscale
            values and mapping them to batches of embedded vectors.
            Must be one of "legendre", "trig", or "binned".
        embedding_dim: Dimension of the embedded vectors.
        frameify: Whether or not to convert embedding functions into frames.
        num_points: The number of grayscale points to evaluate the covariance
            matrix at.
    """
    # Set up full embedding function, with option for frameified ingredients
    embedding_spec = embedding_spec.lower()
    assert embedding_spec in ["legendre", "trig"]
    embed_map = {"legendre": legendre_embed, "trig": trig_embed}
    raw_embed = partial(embed_map[embedding_spec], emb_dim=embedding_dim)
    embed_fun = FixedEmbedding(embed_fun, unit_interval, frameify=frameify)

    # Compute all inner products between embedding vectors
    points = torch.linspace(0.0, 1.0, steps=num_points)
    emb_vecs = embed_fun(points)
    assert emb_vecs.shape == (num_points, embedding_dim)
    raw_inner_prods = emb_vecs @ emb_vecs.T.conj()

    covariance_mat = raw_inner_prods.abs().square()
    assert covariance_mat.shape == (num_points, num_points)
    return covariance_mat
