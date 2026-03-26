import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from models.GTM import GTM
from utils.data_multitrends import ZeroShotDataset


def load_model(checkpoint_path, args, cat_dict, col_dict, fab_dict):
    model = GTM.load_from_checkpoint(
        checkpoint_path,
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
        use_encoder_mask=args.use_encoder_mask,
        autoregressive=args.autoregressive,
        gpu_num=args.gpu_num,
        map_location=f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu",
    )
    model.eval()
    return model


def main(args):
    device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(Path(args.data_csv), parse_dates=["release_date"])
    df_original = df.copy()

    cat_dict = torch.load(Path(args.data_folder) / "category_labels.pt", weights_only=False)
    col_dict = torch.load(Path(args.data_folder) / "color_labels.pt", weights_only=False)
    fab_dict = torch.load(Path(args.data_folder) / "fabric_labels.pt", weights_only=False)
    gtrends = pd.read_csv(Path(args.data_folder) / "gtrends.csv", index_col=[0], parse_dates=True)

    dataset = ZeroShotDataset(
        df.copy(),
        Path(args.data_folder) / "images",
        gtrends,
        cat_dict,
        col_dict,
        fab_dict,
        args.trend_len,
    )

    loader = dataset.get_loader(batch_size=1, train=False)
    model = load_model(args.checkpoint_path, args, cat_dict, col_dict, fab_dict).to(device)

    embeddings = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting embeddings"):
            _, category, color, fabric, _, _, images = batch

            category = category.to(device)
            color = color.to(device)
            fabric = fabric.to(device)
            images = images.to(device)

            x_i = model.encode_multimodal_embedding(category, color, fabric, images)
            embeddings.append(x_i.squeeze(0).cpu())

    embeddings = torch.stack(embeddings)

    torch.save(
        {
            "embeddings": embeddings,
            "metadata": df_original,
        },
        args.output_path,
    )

    print(f"Saved embeddings with shape {embeddings.shape} to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--data_csv", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="multimodal_embeddings.pt")
    parser.add_argument("--gpu_num", type=int, default=0)

    parser.add_argument("--use_img", type=int, default=1)
    parser.add_argument("--use_text", type=int, default=1)
    parser.add_argument("--trend_len", type=int, default=52)
    parser.add_argument("--num_trends", type=int, default=3)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--output_dim", type=int, default=12)
    parser.add_argument("--use_encoder_mask", type=int, default=1)
    parser.add_argument("--autoregressive", type=int, default=0)
    parser.add_argument("--num_attn_heads", type=int, default=4)
    parser.add_argument("--num_hidden_layers", type=int, default=1)

    args = parser.parse_args()
    main(args)
