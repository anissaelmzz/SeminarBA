import os
import argparse
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset, DataLoader

from utils.data_multitrends import ZeroShotDataset
from models.GTM_retrieval import RetrievalAugmentedGTM


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class RetrievalAugmentedDataset(Dataset):
    """
    Wraps the existing ZeroShotDataset output and adds retrieval_summary
    aligned by external_code.
    """

    def __init__(self, base_dataset, retrieval_tensor):
        self.base_dataset = base_dataset
        self.retrieval_tensor = retrieval_tensor

        if len(self.base_dataset) != len(self.retrieval_tensor):
            raise ValueError(
                f"Length mismatch: base dataset has {len(self.base_dataset)} rows, "
                f"retrieval tensor has {len(self.retrieval_tensor)} rows."
            )

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        retrieval_summary = self.retrieval_tensor[idx]
        return (*item, retrieval_summary)


def build_retrieval_tensor_for_dataframe(df, retrieval_memory):
    """
    Align retrieval summaries to a dataframe using external_code.
    """
    metadata = retrieval_memory["metadata"]
    retrieval_summary = retrieval_memory["retrieval_summary"]

    if "external_code" not in df.columns:
        raise KeyError("Input dataframe must contain 'external_code'.")
    if "external_code" not in metadata.columns:
        raise KeyError("Retrieval memory metadata must contain 'external_code'.")

    retrieval_df = metadata[["external_code"]].copy()
    retrieval_df["retrieval_row"] = np.arange(len(retrieval_df))

    merged = df[["external_code"]].merge(
        retrieval_df,
        on="external_code",
        how="left",
        validate="one_to_one",
    )

    if merged["retrieval_row"].isna().any():
        missing_codes = merged.loc[merged["retrieval_row"].isna(), "external_code"].tolist()
        raise ValueError(
            f"Missing retrieval summary for {len(missing_codes)} products. "
            f"Examples: {missing_codes[:10]}"
        )

    row_idx = torch.tensor(merged["retrieval_row"].astype(int).values, dtype=torch.long)
    return retrieval_summary[row_idx]


def build_loader_with_retrieval(df, img_root, gtrends, cat_dict, col_dict, fab_dict,
                                trend_len, retrieval_memory, batch_size, train):
    """
    Build the original ZeroShotDataset, then augment it with retrieval summaries.
    """
    df_for_dataset = df.copy()

    zero_shot_dataset = ZeroShotDataset(
        df_for_dataset,
        img_root,
        gtrends,
        cat_dict,
        col_dict,
        fab_dict,
        trend_len
    ).preprocess_data()

    retrieval_tensor = build_retrieval_tensor_for_dataframe(df, retrieval_memory)

    augmented_dataset = RetrievalAugmentedDataset(
        base_dataset=zero_shot_dataset,
        retrieval_tensor=retrieval_tensor
    )

    return DataLoader(
        augmented_dataset,
        batch_size=batch_size if train else 1,
        shuffle=train,
        num_workers=2
    )


def run(args):
    print(args)
    pl.seed_everything(args.seed)

    # Load training data
    train_df = pd.read_csv(Path(args.data_folder) / "train.csv", parse_dates=["release_date"])

    # Sort by release date and create subtrain / val split
    train_df = train_df.sort_values("release_date").reset_index(drop=True)

    val_size = max(1, int(0.15 * len(train_df)))
    subtrain_df = train_df.iloc[:-val_size].copy()
    val_df = train_df.iloc[-val_size:].copy()

    # Load label encodings
    cat_dict = torch.load(Path(args.data_folder) / "category_labels.pt", weights_only=False)
    col_dict = torch.load(Path(args.data_folder) / "color_labels.pt", weights_only=False)
    fab_dict = torch.load(Path(args.data_folder) / "fabric_labels.pt", weights_only=False)

    # Load Google Trends
    gtrends = pd.read_csv(Path(args.data_folder) / "gtrends.csv", index_col=[0], parse_dates=True)

    # Load retrieval memory
    retrieval_memory = torch.load(args.retrieval_memory_path, map_location="cpu", weights_only=False)

    img_root = Path(args.data_folder) / "images"

    train_loader = build_loader_with_retrieval(
        df=subtrain_df,
        img_root=img_root,
        gtrends=gtrends,
        cat_dict=cat_dict,
        col_dict=col_dict,
        fab_dict=fab_dict,
        trend_len=args.trend_len,
        retrieval_memory=retrieval_memory,
        batch_size=args.batch_size,
        train=True,
    )

    val_loader = build_loader_with_retrieval(
        df=val_df,
        img_root=img_root,
        gtrends=gtrends,
        cat_dict=cat_dict,
        col_dict=col_dict,
        fab_dict=fab_dict,
        trend_len=args.trend_len,
        retrieval_memory=retrieval_memory,
        batch_size=1,
        train=False,
    )

    model = RetrievalAugmentedGTM(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_heads=args.num_attn_heads,
        num_layers=args.num_hidden_layers,
        cat_dict=cat_dict,
        col_dict=col_dict,
        fab_dict=fab_dict,
        use_text=args.use_text,
        use_img=args.use_img,
        trend_len=args.trend_len,
        num_trends=args.num_trends,
        gpu_num=args.gpu_num,
        retrieval_dim=args.output_dim,
        use_encoder_mask=args.use_encoder_mask,
        autoregressive=args.autoregressive,
    )

    model_savename = "GTM_retrieval_" + args.run_name
    dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=Path(args.log_dir) / "GTM_retrieval",
        filename=model_savename + "---{epoch}---" + dt_string,
        monitor="val_mae",
        mode="min",
        save_top_k=1,
    )

    tb_logger = pl.loggers.TensorBoardLogger(args.log_dir, name=model_savename)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=args.epochs,
        check_val_every_n_epoch=5,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("Best checkpoint:", checkpoint_callback.best_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train retrieval-augmented GTM")

    parser.add_argument("--data_folder", type=str, default="dataset/")
    parser.add_argument("--retrieval_memory_path", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default="log")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--gpu_num", type=int, default=0)

    parser.add_argument("--use_text", type=int, default=1)
    parser.add_argument("--use_img", type=int, default=1)
    parser.add_argument("--trend_len", type=int, default=52)
    parser.add_argument("--num_trends", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--output_dim", type=int, default=12)
    parser.add_argument("--use_encoder_mask", type=int, default=1)
    parser.add_argument("--autoregressive", type=int, default=0)
    parser.add_argument("--num_attn_heads", type=int, default=4)
    parser.add_argument("--num_hidden_layers", type=int, default=1)
    parser.add_argument("--run_name", type=str, default="Run1")

    args = parser.parse_args()
    run(args)