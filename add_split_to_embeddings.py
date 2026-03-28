import argparse
from pathlib import Path

import pandas as pd
import torch


def load_embeddings(input_path: str):
    data = torch.load(input_path, map_location="cpu", weights_only=False)

    if not isinstance(data, dict):
        raise ValueError("Expected embeddings file to contain a dict.")
    if "embeddings" not in data or "metadata" not in data:
        raise KeyError("Embeddings file must contain 'embeddings' and 'metadata'.")

    embeddings = data["embeddings"]
    metadata = data["metadata"]

    if not torch.is_tensor(embeddings):
        raise TypeError("'embeddings' must be a torch.Tensor.")
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("'metadata' must be a pandas.DataFrame.")

    return data, embeddings, metadata


def build_split_table(train_csv: str, test_csv: str):
    train_df = pd.read_csv(train_csv, parse_dates=["release_date"])
    test_df = pd.read_csv(test_csv, parse_dates=["release_date"])

    if "external_code" not in train_df.columns or "external_code" not in test_df.columns:
        raise KeyError("Both train.csv and test.csv must contain 'external_code'.")

    # Reproduce train.py logic exactly
    train_df = train_df.sort_values("release_date").reset_index(drop=True)
    val_size = max(1, int(0.15 * len(train_df)))

    subtrain_df = train_df.iloc[:-val_size].copy()
    val_df = train_df.iloc[-val_size:].copy()

    subtrain_df["split"] = "subtrain"
    val_df["split"] = "val"
    test_df["split"] = "test"

    split_df = pd.concat(
        [
            subtrain_df[["external_code", "split"]],
            val_df[["external_code", "split"]],
            test_df[["external_code", "split"]],
        ],
        axis=0,
        ignore_index=True,
    )

    if split_df["external_code"].duplicated().any():
        dupes = split_df.loc[split_df["external_code"].duplicated(), "external_code"].tolist()[:10]
        raise ValueError(f"Duplicate external_code values found across split table. Examples: {dupes}")

    return split_df


def attach_split_to_metadata(metadata: pd.DataFrame, split_df: pd.DataFrame):
    if "external_code" not in metadata.columns:
        raise KeyError("Embeddings metadata must contain 'external_code'.")

    merged = metadata.merge(split_df, on="external_code", how="left", validate="one_to_one")

    if merged["split"].isna().any():
        missing_codes = merged.loc[merged["split"].isna(), "external_code"].tolist()[:10]
        raise ValueError(
            f"Could not assign split labels to all embedding rows. Missing examples: {missing_codes}"
        )

    return merged


def main():
    parser = argparse.ArgumentParser(description="Add train/val/test split labels to multimodal embeddings metadata.")
    parser.add_argument("--embeddings_path", type=str, required=True, help="Path to multimodal_embeddings.pt")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to train.csv")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to test.csv")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save multimodal_embeddings_with_split.pt")
    args = parser.parse_args()

    print(f"Loading embeddings from: {args.embeddings_path}")
    data, embeddings, metadata = load_embeddings(args.embeddings_path)

    print(f"Embeddings shape: {tuple(embeddings.shape)}")
    print(f"Metadata shape before merge: {metadata.shape}")

    split_df = build_split_table(args.train_csv, args.test_csv)
    metadata_with_split = attach_split_to_metadata(metadata, split_df)

    print(f"Metadata shape after merge: {metadata_with_split.shape}")
    print("Split counts:")
    print(metadata_with_split["split"].value_counts(dropna=False))

    output = dict(data)
    output["metadata"] = metadata_with_split

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)

    print(f"Saved output to: {output_path}")


if __name__ == "__main__":
    main()