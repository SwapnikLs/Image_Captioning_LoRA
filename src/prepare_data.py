import argparse
import os
import re
from pathlib import Path

import pandas as pd


def strip_old_tokens(text: str) -> str:
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("<start>", "").replace("<end>", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_dataframe(captions_csv: str) -> pd.DataFrame:
    df = pd.read_csv(captions_csv)
    required = {"image_name", "caption"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["caption"] = df["caption"].astype(str).map(strip_old_tokens)
    df = df[df["caption"].str.len() > 2]
    return df[["image_name", "caption"]]


def split_by_image(df: pd.DataFrame, val_ratio: float, seed: int):
    image_ids = sorted(df["image_name"].unique().tolist())
    rng = pd.Series(image_ids).sample(frac=1.0, random_state=seed).tolist()
    n_val = max(1, int(len(rng) * val_ratio))

    val_images = set(rng[:n_val])
    train_df = df[~df["image_name"].isin(val_images)].reset_index(drop=True)
    val_df = df[df["image_name"].isin(val_images)].reset_index(drop=True)
    return train_df, val_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--captions_csv",
        type=str,
        default="/home/swapnik/MyStuff/ImageCaptioning/CapData/cleaned_captions.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/swapnik/MyStuff/ImageCaptioning/CapData/Captioning_LoRA_4GB/data/processed",
    )
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataframe(args.captions_csv)
    train_df, val_df = split_by_image(df, args.val_ratio, args.seed)

    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"Total rows: {len(df)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Val rows: {len(val_df)}")
    print(f"Saved: {train_path}")
    print(f"Saved: {val_path}")


if __name__ == "__main__":
    main()
