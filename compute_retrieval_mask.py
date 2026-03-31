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


def build_retrieval_mask(
    metadata: pd.DataFrame,
    horizon_weeks: int,
    split_col: str = "split",
    candidate_split_value: str = "subtrain",
) -> torch.Tensor:
    """
    Build binary retrieval mask:

        m_ij^retr = 1[d_j + H <= d_i] * 1[split_j == candidate_split_value] * 1[i != j]

    Rows i = query/target products
    Columns j = candidate historical products
    """
    if "release_date" not in metadata.columns:
        raise KeyError("Metadata must contain 'release_date'.")

    if split_col not in metadata.columns:
        raise KeyError(f"Metadata must contain '{split_col}'.")

    release_dates = pd.to_datetime(metadata["release_date"])
    if release_dates.isna().any():
        raise ValueError("Some release_date values are missing or invalid.")

    d_i = release_dates.values[:, None]
    d_j = release_dates.values[None, :]
    horizon = pd.to_timedelta(horizon_weeks * 7, unit="D")

    temporal_mask_np = (d_j + horizon) <= d_i
    candidate_mask_np = (metadata[split_col].values == candidate_split_value)[None, :]

    admissible_np = temporal_mask_np & candidate_mask_np
    retrieval_mask = torch.from_numpy(admissible_np)

    # Never retrieve self.
    retrieval_mask.fill_diagonal_(False)
    return retrieval_mask


def apply_retrieval_mask(cosine_sim: torch.Tensor, retrieval_mask: torch.Tensor) -> torch.Tensor:
    """
    Apply admissibility mask to similarity matrix.
    Inadmissible pairs are set to -inf.
    """
    if cosine_sim.shape != retrieval_mask.shape:
        raise ValueError(
            f"Shape mismatch: cosine_sim has shape {tuple(cosine_sim.shape)}, "
            f"retrieval_mask has shape {tuple(retrieval_mask.shape)}."
        )

    masked_sim = cosine_sim.clone()
    masked_sim[~retrieval_mask] = -float("inf")
    return masked_sim


def apply_similarity_threshold(masked_sim: torch.Tensor, similarity_threshold: float) -> torch.Tensor:
    """
    Keep only admissible neighbors whose cosine similarity is above threshold.
    Non-qualifying pairs are set to -inf.
    """
    thresholded_sim = masked_sim.clone()
    below_threshold = torch.isfinite(thresholded_sim) & (thresholded_sim < similarity_threshold)
    thresholded_sim[below_threshold] = -float("inf")
    return thresholded_sim


def compute_thresholded_neighbors(similarity_tensor: torch.Tensor, max_k: int):
    """
    Retrieve up to max_k neighbors per row after admissibility + thresholding.
    Invalid slots remain -inf and are tracked via valid_mask.
    """
    if max_k <= 0:
        raise ValueError("max_k must be positive.")

    safe_k = min(max_k, similarity_tensor.shape[1])
    topk_scores, topk_indices = torch.topk(similarity_tensor, k=safe_k, dim=1)
    valid_mask = torch.isfinite(topk_scores)
    return topk_scores, topk_indices, valid_mask


def build_admissible_neighbors_dataframe(
    metadata: pd.DataFrame,
    topk_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    valid_mask: torch.Tensor,
    horizon_weeks: int,
    similarity_threshold: float,
) -> pd.DataFrame:
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
                "is_selected_neighbor": is_valid,
                "forecast_horizon_weeks": horizon_weeks,
                "similarity_threshold": similarity_threshold,
            }

            if "external_code" in metadata.columns:
                row["query_external_code"] = metadata.loc[i, "external_code"]

            row["query_release_date"] = query_release_date

            if is_valid:
                neighbor_release_date = pd.to_datetime(metadata.loc[neighbor_idx, "release_date"])
                score = float(topk_scores[i, rank].item())

                row["neighbor_index"] = neighbor_idx
                row["cosine_similarity"] = score
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
                row["cosine_similarity"] = None
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
    parser = argparse.ArgumentParser(
        description="Apply admissibility mask and cosine-similarity threshold for retrieval."
    )
    parser.add_argument("--embeddings_path", type=str, required=True, help="Path to multimodal_embeddings.pt")
    parser.add_argument("--similarity_path", type=str, required=True, help="Path to cosine_similarities.pt")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save thresholded retrieval output .pt file")
    parser.add_argument("--neighbors_csv", type=str, required=True, help="Path to save selected neighbors CSV")
    parser.add_argument("--horizon_weeks", type=int, default=12, help="Forecast horizon H in weeks")
    parser.add_argument("--similarity_threshold", type=float, required=True, help="Minimum cosine similarity required for retrieval")
    parser.add_argument("--max_k", type=int, default=20, help="Maximum number of retrieved neighbors to keep per product")

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
    if "split" in metadata.columns:
        print("Split counts:")
        print(metadata["split"].value_counts(dropna=False))

    print(f"Cosine similarity shape: {tuple(cosine_sim.shape)}")

    retrieval_mask = build_retrieval_mask(
        metadata=metadata,
        horizon_weeks=args.horizon_weeks,
        split_col="split",
        candidate_split_value="subtrain",
    )
    print(f"Retrieval mask shape: {tuple(retrieval_mask.shape)}")

    masked_sim = apply_retrieval_mask(cosine_sim, retrieval_mask)
    thresholded_sim = apply_similarity_threshold(masked_sim, args.similarity_threshold)

    topk_scores, topk_indices, valid_mask = compute_thresholded_neighbors(
        thresholded_sim, max_k=args.max_k
    )

    print(f"Selected-neighbor scores shape: {tuple(topk_scores.shape)}")
    print(f"Selected-neighbor indices shape: {tuple(topk_indices.shape)}")
    print(f"Queries with zero valid neighbors: {int((~valid_mask).all(dim=1).sum().item())}")

    output = {
        "retrieval_mask": retrieval_mask,
        "masked_retrieval_similarity": masked_sim,
        "thresholded_retrieval_similarity": thresholded_sim,
        "topk_scores": topk_scores,
        "topk_indices": topk_indices,
        "topk_valid_mask": valid_mask,
        "metadata": metadata,
        "horizon_weeks": args.horizon_weeks,
        "similarity_threshold": args.similarity_threshold,
        "k": topk_scores.shape[1],
        "max_k": args.max_k,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    print(f"Saved thresholded retrieval output to: {output_path}")

    neighbors_df = build_admissible_neighbors_dataframe(
        metadata=metadata,
        topk_scores=topk_scores,
        topk_indices=topk_indices,
        valid_mask=valid_mask,
        horizon_weeks=args.horizon_weeks,
        similarity_threshold=args.similarity_threshold,
    )

    neighbors_csv_path.parent.mkdir(parents=True, exist_ok=True)
    neighbors_df.to_csv(neighbors_csv_path, index=False)
    print(f"Saved selected neighbors CSV to: {neighbors_csv_path}")

    total_pairs = retrieval_mask.numel()
    admissible_pairs = int(retrieval_mask.sum().item())
    print(f"Admissible pairs: {admissible_pairs} / {total_pairs} ({admissible_pairs / total_pairs:.4%})")


if __name__ == "__main__":
    main()
