import argparse
from pathlib import Path

import pandas as pd
import torch


def load_embeddings(input_path: str):
    """
    Load the saved multimodal embeddings file.
    Expected structure:
        {
            "embeddings": torch.Tensor of shape [N, D],
            "metadata": pandas.DataFrame
        }
    """
    data = torch.load(input_path, map_location="cpu", weights_only=False)

    if not isinstance(data, dict):
        raise ValueError("Expected a dict in the .pt file.")

    if "embeddings" not in data:
        raise KeyError("Missing key 'embeddings' in input file.")

    if "metadata" not in data:
        raise KeyError("Missing key 'metadata' in input file.")

    embeddings = data["embeddings"]
    metadata = data["metadata"]

    if not torch.is_tensor(embeddings):
        raise TypeError("'embeddings' must be a torch.Tensor.")

    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("'metadata' must be a pandas DataFrame.")

    return embeddings, metadata


def compute_cosine_similarity(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute full pairwise cosine similarity matrix.
    embeddings: [N, D]
    returns: [N, N]
    """
    embeddings = embeddings.float()

    norms = embeddings.norm(dim=1, keepdim=True).clamp(min=1e-12)
    embeddings_normalized = embeddings / norms

    cosine_sim = embeddings_normalized @ embeddings_normalized.T
    return cosine_sim


def compute_topk_neighbors(cosine_sim: torch.Tensor, k: int):
    """
    Compute top-k nearest neighbors per product, excluding self.
    """
    if k >= cosine_sim.shape[0]:
        raise ValueError(f"k must be smaller than number of products ({cosine_sim.shape[0]}).")

    sim_no_self = cosine_sim.clone()
    sim_no_self.fill_diagonal_(-float("inf"))

    topk_scores, topk_indices = torch.topk(sim_no_self, k=k, dim=1)
    return topk_scores, topk_indices


def build_neighbors_dataframe(metadata: pd.DataFrame,
                              topk_scores: torch.Tensor,
                              topk_indices: torch.Tensor) -> pd.DataFrame:
    """
    Build a readable neighbor table.
    Assumes metadata contains 'external_code' and 'release_date'.
    """
    rows = []
    k = topk_scores.shape[1]

    for i in range(len(metadata)):
        for rank in range(k):
            j = topk_indices[i, rank].item()
            score = topk_scores[i, rank].item()

            row = {
                "query_index": i,
                "neighbor_index": j,
                "rank": rank + 1,
                "cosine_similarity": score,
            }

            if "external_code" in metadata.columns:
                row["query_external_code"] = metadata.loc[i, "external_code"]
                row["neighbor_external_code"] = metadata.loc[j, "external_code"]

            if "release_date" in metadata.columns:
                row["query_release_date"] = metadata.loc[i, "release_date"]
                row["neighbor_release_date"] = metadata.loc[j, "release_date"]

            if "category" in metadata.columns:
                row["query_category"] = metadata.loc[i, "category"]
                row["neighbor_category"] = metadata.loc[j, "category"]

            rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Compute cosine similarity matrix from multimodal embeddings.")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to multimodal_embeddings.pt"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save cosine similarity output .pt file"
    )
    parser.add_argument(
        "--neighbors_csv",
        type=str,
        default=None,
        help="Optional path to save top-k neighbors as CSV"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=20,
        help="Number of nearest neighbors to keep per product"
    )

    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    print(f"Loading embeddings from: {input_path}")
    embeddings, metadata = load_embeddings(str(input_path))

    print(f"Embeddings shape: {tuple(embeddings.shape)}")
    print(f"Metadata shape: {metadata.shape}")

    cosine_sim = compute_cosine_similarity(embeddings)

    print(f"Cosine similarity matrix shape: {tuple(cosine_sim.shape)}")
    print(f"Min similarity: {cosine_sim.min().item():.6f}")
    print(f"Max similarity: {cosine_sim.max().item():.6f}")
    print("First 5 diagonal values:", cosine_sim.diag()[:5].tolist())

    topk_scores, topk_indices = compute_topk_neighbors(cosine_sim, k=args.k)
    print(f"Top-{args.k} scores shape: {tuple(topk_scores.shape)}")
    print(f"Top-{args.k} indices shape: {tuple(topk_indices.shape)}")

    output = {
        "cosine_similarity_matrix": cosine_sim,
        "topk_scores": topk_scores,
        "topk_indices": topk_indices,
        "metadata": metadata,
        "k": args.k,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    print(f"Saved similarity output to: {output_path}")

    if args.neighbors_csv is not None:
        neighbors_df = build_neighbors_dataframe(metadata, topk_scores, topk_indices)
        neighbors_csv_path = Path(args.neighbors_csv)
        neighbors_csv_path.parent.mkdir(parents=True, exist_ok=True)
        neighbors_df.to_csv(neighbors_csv_path, index=False)
        print(f"Saved neighbors CSV to: {neighbors_csv_path}")


if __name__ == "__main__":
    main()