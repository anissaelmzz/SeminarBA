import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_retrieval_output(input_path: str):
    """
    Load retrieval output produced by compute_retrieval_mask.py.
    """
    data = torch.load(input_path, map_location="cpu", weights_only=False)

    required_keys = [
        "topk_scores",
        "topk_indices",
        "topk_valid_mask",
        "metadata",
        "k",
        "horizon_weeks",
    ]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing key '{key}' in retrieval output file.")

    topk_scores = data["topk_scores"]
    topk_indices = data["topk_indices"]
    topk_valid_mask = data["topk_valid_mask"]
    metadata = data["metadata"]

    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("'metadata' must be a pandas DataFrame.")

    return data


def load_sales_table(csv_paths, horizon: int, explicit_sales_cols=None):
    """
    Load one or more CSV files and extract the sales trajectory columns.

    Recommended case:
    - sales columns are '0', '1', ..., '11'

    If explicit_sales_cols is None, the script first tries ['0', ..., str(horizon-1)].
    If those are not present, it falls back to the first `horizon` columns.
    """
    dfs = []
    for path in csv_paths:
        df = pd.read_csv(path, parse_dates=["release_date"])
        dfs.append(df)

    sales_df = pd.concat(dfs, axis=0, ignore_index=True)

    if "external_code" not in sales_df.columns:
        raise KeyError("Sales table must contain 'external_code'.")

    if explicit_sales_cols is not None:
        sales_cols = explicit_sales_cols
    else:
        candidate_cols = [str(i) for i in range(horizon)]
        if all(col in sales_df.columns for col in candidate_cols):
            sales_cols = candidate_cols
        else:
            sales_cols = sales_df.columns[:horizon].tolist()

    missing = [col for col in sales_cols if col not in sales_df.columns]
    if missing:
        raise KeyError(f"Missing sales columns in sales table: {missing}")

    keep_cols = ["external_code", "release_date"] + sales_cols
    keep_cols = [col for col in keep_cols if col in sales_df.columns]

    sales_df = sales_df[keep_cols].copy()
    sales_df = sales_df.drop_duplicates(subset=["external_code"]).reset_index(drop=True)

    return sales_df, sales_cols


def align_sales_to_metadata(metadata: pd.DataFrame, sales_df: pd.DataFrame, sales_cols):
    """
    Align sales trajectories to the exact product order of retrieval metadata.
    """
    if "external_code" not in metadata.columns:
        raise KeyError("Metadata must contain 'external_code'.")

    merged = metadata[["external_code"]].merge(
        sales_df[["external_code"] + sales_cols],
        on="external_code",
        how="left",
        validate="one_to_one",
    )

    if merged[sales_cols].isna().any().any():
        missing_codes = merged.loc[merged[sales_cols].isna().any(axis=1), "external_code"].tolist()
        raise ValueError(
            f"Missing sales trajectories for {len(missing_codes)} products. "
            f"Examples: {missing_codes[:10]}"
        )

    sales_tensor = torch.tensor(merged[sales_cols].values, dtype=torch.float32)
    return sales_tensor


def build_neighbor_sales_tensor(
    sales_tensor: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_valid_mask: torch.Tensor,
):
    """
    Construct neighbor sales tensor of shape [N, K, H].

    Invalid top-k slots are filled with zeros.
    """
    n, k = topk_indices.shape
    h = sales_tensor.shape[1]

    neighbor_sales = torch.zeros((n, k, h), dtype=torch.float32)

    for i in range(n):
        for r in range(k):
            if bool(topk_valid_mask[i, r].item()):
                j = int(topk_indices[i, r].item())
                neighbor_sales[i, r] = sales_tensor[j]

    return neighbor_sales


def compute_similarity_weights(topk_scores: torch.Tensor, topk_valid_mask: torch.Tensor):
    """
    Compute normalized similarity weights over valid retrieved neighbors.

    Invalid slots receive weight 0.
    """
    masked_scores = topk_scores.clone()
    masked_scores[~topk_valid_mask] = -float("inf")

    weights = torch.softmax(masked_scores, dim=1)
    weights[~topk_valid_mask] = 0.0

    row_sums = weights.sum(dim=1, keepdim=True).clamp(min=1e-12)
    weights = weights / row_sums

    # If a query has zero valid neighbors, force all weights to 0
    no_valid = (~topk_valid_mask).all(dim=1)
    if no_valid.any():
        weights[no_valid] = 0.0

    return weights


def aggregate_neighbor_sales(neighbor_sales: torch.Tensor, similarity_weights: torch.Tensor):
    """
    Compute weighted average retrieved sales trajectory.

    neighbor_sales: [N, K, H]
    similarity_weights: [N, K]
    returns: [N, H]
    """
    weights_expanded = similarity_weights.unsqueeze(-1)  # [N, K, 1]
    retrieval_summary = (neighbor_sales * weights_expanded).sum(dim=1)
    return retrieval_summary


def build_neighbor_code_table(metadata: pd.DataFrame, topk_indices: torch.Tensor, topk_valid_mask: torch.Tensor):
    """
    Build [N, K] array of neighbor external codes for interpretability.
    """
    if "external_code" not in metadata.columns:
        return None

    codes = metadata["external_code"].tolist()
    n, k = topk_indices.shape
    neighbor_codes = np.empty((n, k), dtype=object)

    for i in range(n):
        for r in range(k):
            if bool(topk_valid_mask[i, r].item()):
                j = int(topk_indices[i, r].item())
                neighbor_codes[i, r] = codes[j]
            else:
                neighbor_codes[i, r] = None

    return neighbor_codes


def main():
    parser = argparse.ArgumentParser(description="Build explicit retrieval memory from admissible neighbors.")
    parser.add_argument(
        "--retrieval_path",
        type=str,
        required=True,
        help="Path to retrieval_masked_similarity_k?.pt"
    )
    parser.add_argument(
        "--sales_csv_paths",
        type=str,
        nargs="+",
        required=True,
        help="One or more CSV files containing sales trajectories, e.g. train.csv test.csv"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save retrieval memory .pt file"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=12,
        help="Sales horizon to retrieve"
    )
    parser.add_argument(
        "--sales_cols",
        type=str,
        nargs="+",
        default=None,
        help="Optional explicit list of sales columns, e.g. 0 1 2 3 4 5 6 7 8 9 10 11"
    )

    args = parser.parse_args()

    print(f"Loading retrieval output from: {args.retrieval_path}")
    retrieval_data = load_retrieval_output(args.retrieval_path)

    topk_scores = retrieval_data["topk_scores"]
    topk_indices = retrieval_data["topk_indices"]
    topk_valid_mask = retrieval_data["topk_valid_mask"]
    metadata = retrieval_data["metadata"]
    k = retrieval_data["k"]

    print(f"Loading sales tables from: {args.sales_csv_paths}")
    sales_df, sales_cols = load_sales_table(
        csv_paths=args.sales_csv_paths,
        horizon=args.horizon,
        explicit_sales_cols=args.sales_cols,
    )

    print(f"Using sales columns: {sales_cols}")

    sales_tensor = align_sales_to_metadata(
        metadata=metadata,
        sales_df=sales_df,
        sales_cols=sales_cols,
    )

    print(f"Aligned sales tensor shape: {tuple(sales_tensor.shape)}")

    neighbor_sales = build_neighbor_sales_tensor(
        sales_tensor=sales_tensor,
        topk_indices=topk_indices,
        topk_valid_mask=topk_valid_mask,
    )

    similarity_weights = compute_similarity_weights(
        topk_scores=topk_scores,
        topk_valid_mask=topk_valid_mask,
    )

    retrieval_summary = aggregate_neighbor_sales(
        neighbor_sales=neighbor_sales,
        similarity_weights=similarity_weights,
    )

    neighbor_codes = build_neighbor_code_table(
        metadata=metadata,
        topk_indices=topk_indices,
        topk_valid_mask=topk_valid_mask,
    )

    output = {
        "query_sales": sales_tensor,                    # [N, H]
        "neighbor_sales": neighbor_sales,               # [N, K, H]
        "topk_scores": topk_scores,                     # [N, K]
        "topk_indices": topk_indices,                   # [N, K]
        "topk_valid_mask": topk_valid_mask,             # [N, K]
        "similarity_weights": similarity_weights,       # [N, K]
        "retrieval_summary": retrieval_summary,         # [N, H]
        "neighbor_external_codes": neighbor_codes,      # [N, K] object array or None
        "metadata": metadata,
        "sales_columns": sales_cols,
        "horizon": args.horizon,
        "k": k,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)

    print(f"Saved retrieval memory to: {output_path}")
    print(f"neighbor_sales shape: {tuple(neighbor_sales.shape)}")
    print(f"retrieval_summary shape: {tuple(retrieval_summary.shape)}")
    print(f"Queries with zero valid neighbors: {int((~topk_valid_mask).all(dim=1).sum().item())}")


if __name__ == "__main__":
    main()