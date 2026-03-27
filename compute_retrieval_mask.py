import argparse
from pathlib import Path

import pandas as pd
import torch


def load_embeddings(input_path: str):
    """
    Load multimodal embeddings file.
    Expected structure:
        {
            "embeddings": torch.Tensor of shape [N, D],
            "metadata": pandas.DataFrame
        }
    """
    data = torch.load(input_path, map_location="cpu", weights_only=False)

    if not isinstance(data, dict):
        raise ValueError("Expected a dict in the embeddings file.")

    if "embeddings" not in data:
        raise KeyError("Missing key 'embeddings' in embeddings file.")

    if "metadata" not in data:
        raise KeyError("Missing key 'metadata' in embeddings file.")

    embeddings = data["embeddings"]
    metadata = data["metadata"]

    if not torch.is_tensor(embeddings):
        raise TypeError("'embeddings' must be a torch.Tensor.")

    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("'metadata' must be a pandas DataFrame.")

    return embeddings, metadata


def load_similarity(similarity_path: str):
    """
    Load cosine similarity output file from compute_cosine_similarity.py.
    Expected structure:
        {
            "cosine_similarity_matrix": torch.Tensor [N, N],
            "topk_scores": ...,
            "topk_indices": ...,
            "metadata": pandas.DataFrame,
            "k": int
        }
    """
    data = torch.load(similarity_path, map_location="cpu", weights_only=False)

    if not isinstance(data, dict):
        raise ValueError("Expected a dict in the similarity file.")

    if "cosine_similarity_matrix" not in data:
        raise KeyError("Missing key 'cosine_similarity_matrix' in similarity file.")

    cosine_sim = data["cosine_similarity_matrix"]
    metadata = data.get("metadata", None)

    if not torch.is_tensor(cosine_sim):
        raise TypeError("'cosine_similarity_matrix' must be a torch.Tensor.")

    return cosine_sim, metadata


def build_retrieval_mask(metadata: pd.DataFrame, horizon_weeks: int) -> torch.Tensor:
    """
    Build binary retrieval mask m_ij^retr = 1[d_j + H <= d_i]

    Rows i = target/query products
    Columns j = candidate historical products
    """
    if "release_date" not in metadata.columns:
        raise KeyError("Metadata must contain 'release_date'.")

    release_dates = pd.to_datetime(metadata["release_date"])

    if release_dates.isna().any():
        raise ValueError("Some release_date values are missing or invalid.")

    # d_i as column vector, d_j as row vector
    d_i = release_dates.values[:, None]
    d_j = release_dates.values[None, :]

    horizon = pd.to_timedelta(horizon_weeks * 7, unit="D")

    admissible_np = (d_j + horizon) <= d_i
    retrieval_mask = torch.from_numpy(admissible_np)

    return retrieval_mask


def apply_retrieval_mask(cosine_sim: torch.Tensor, retrieval_mask: torch.Tensor) -> torch.Tensor:
    """
    Apply retrieval mask to similarity matrix.

    Inadmissible pairs are set to -inf so they cannot appear in top-k retrieval.
    """
    if cosine_sim.shape != retrieval_mask.shape:
        raise ValueError(
            f"Shape mismatch: cosine_sim has shape {tuple(cosine_sim.shape)}, "
            f"retrieval_mask has shape {tuple(retrieval_mask.shape)}."
        )

    masked_sim = cosine_sim.clone()
    masked_sim[~retrieval_mask] = -float("inf")

    return masked_sim


def compute_topk_admissible_neighbors(masked_sim: torch.Tensor, k: int):
    """
    Compute top-k admissible historical neighbors per product.
    """
    if k >= masked_sim.shape[0]:
        raise ValueError(f"k must be smaller than number of products ({masked_sim.shape[0]}).")

    topk_scores, topk_indices = torch.topk(masked_sim, k=k, dim=1)
    valid_mask = torch.isfinite(topk_scores)

    return topk_scores, topk_indices, valid_mask


def build_admissible_neighbors_dataframe(metadata: pd.DataFrame,
                                         topk_scores: torch.Tensor,
                                         topk_indices: torch.Tensor,
                                         valid_mask: torch.Tensor,
                                         horizon_weeks: int) -> pd.DataFrame:
    """
    Build readable CSV of admissible retrieval neighbors.
    """
    rows = []
    k = topk_scores.shape[1]

    for i in range(len(metadata)):
        query_release_date = pd.to_datetime(metadata.loc[i, "release_date"])

        for rank in range(k):
            is_valid = bool(valid_mask[i, rank].item())
            neighbor_idx = int(topk_indices[i, rank].item())

            row = {
                "query_index": i,
                "rank": rank + 1,
                "is_admissible_neighbor": is_valid,
                "forecast_horizon_weeks": horizon_weeks,
            }

            if "external_code" in metadata.columns:
                row["query_external_code"] = metadata.loc[i, "external_code"]

            row["query_release_date"] = query_release_date

            if is_valid:
                neighbor_release_date = pd.to_datetime(metadata.loc[neighbor_idx, "release_date"])
                score = float(topk_scores[i, rank].item())

                row["neighbor_index"] = neighbor_idx
                row["masked_cosine_similarity"] = score
                row["neighbor_release_date"] = neighbor_release_date
                row["neighbor_plus_horizon_date"] = neighbor_release_date + pd.to_timedelta(
                    horizon_weeks * 7, unit="D"
                )
                row["days_between_launches"] = (query_release_date - neighbor_release_date).days
                row["weeks_between_launches"] = (query_release_date - neighbor_release_date).days / 7.0

                if "external_code" in metadata.columns:
                    row["neighbor_external_code"] = metadata.loc[neighbor_idx, "external_code"]

                if "category" in metadata.columns:
                    row["query_category"] = metadata.loc[i, "category"]
                    row["neighbor_category"] = metadata.loc[neighbor_idx, "category"]
            else:
                row["neighbor_index"] = None
                row["masked_cosine_similarity"] = None
                row["neighbor_release_date"] = None
                row["neighbor_plus_horizon_date"] = None
                row["days_between_launches"] = None
                row["weeks_between_launches"] = None

                if "external_code" in metadata.columns:
                    row["neighbor_external_code"] = None

                if "category" in metadata.columns:
                    row["query_category"] = metadata.loc[i, "category"]
                    row["neighbor_category"] = None

            rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Compute retrieval admissibility mask and top-k admissible neighbors.")
    parser.add_argument(
        "--embeddings_path",
        type=str,
        required=True,
        help="Path to multimodal_embeddings.pt"
    )
    parser.add_argument(
        "--similarity_path",
        type=str,
        required=True,
        help="Path to cosine_similarities.pt"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save masked retrieval output .pt file"
    )
    parser.add_argument(
        "--neighbors_csv",
        type=str,
        required=True,
        help="Path to save admissible neighbors CSV"
    )
    parser.add_argument(
        "--horizon_weeks",
        type=int,
        default=12,
        help="Forecast horizon H in weeks"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=20,
        help="Number of admissible neighbors to retrieve per product"
    )

    args = parser.parse_args()

    embeddings_path = Path(args.embeddings_path)
    similarity_path = Path(args.similarity_path)
    output_path = Path(args.output_path)
    neighbors_csv_path = Path(args.neighbors_csv)

    print(f"Loading embeddings from: {embeddings_path}")
    _, metadata = load_embeddings(str(embeddings_path))

    print(f"Loading cosine similarities from: {similarity_path}")
    cosine_sim, similarity_metadata = load_similarity(str(similarity_path))

    if similarity_metadata is not None and len(similarity_metadata) != len(metadata):
        raise ValueError("Metadata length mismatch between embeddings file and similarity file.")

    print(f"Metadata shape: {metadata.shape}")
    print(f"Cosine similarity shape: {tuple(cosine_sim.shape)}")

    retrieval_mask = build_retrieval_mask(metadata, horizon_weeks=args.horizon_weeks)
    print(f"Retrieval mask shape: {tuple(retrieval_mask.shape)}")

    masked_sim = apply_retrieval_mask(cosine_sim, retrieval_mask)

    topk_scores, topk_indices, valid_mask = compute_topk_admissible_neighbors(
        masked_sim, k=args.k
    )

    print(f"Top-{args.k} admissible scores shape: {tuple(topk_scores.shape)}")
    print(f"Top-{args.k} admissible indices shape: {tuple(topk_indices.shape)}")

    output = {
        "retrieval_mask": retrieval_mask,
        "masked_retrieval_similarity": masked_sim,
        "topk_scores": topk_scores,
        "topk_indices": topk_indices,
        "topk_valid_mask": valid_mask,
        "metadata": metadata,
        "horizon_weeks": args.horizon_weeks,
        "k": args.k,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    print(f"Saved masked retrieval output to: {output_path}")

    neighbors_df = build_admissible_neighbors_dataframe(
        metadata=metadata,
        topk_scores=topk_scores,
        topk_indices=topk_indices,
        valid_mask=valid_mask,
        horizon_weeks=args.horizon_weeks
    )

    neighbors_csv_path.parent.mkdir(parents=True, exist_ok=True)
    neighbors_df.to_csv(neighbors_csv_path, index=False)
    print(f"Saved admissible neighbors CSV to: {neighbors_csv_path}")

    total_pairs = retrieval_mask.numel()
    admissible_pairs = int(retrieval_mask.sum().item())
    print(f"Admissible pairs: {admissible_pairs} / {total_pairs} ({admissible_pairs / total_pairs:.4%})")


if __name__ == "__main__":
    main()