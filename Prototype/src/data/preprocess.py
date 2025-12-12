"""
Data loading and basic preprocessing.

Here we:
- Read True.csv and Fake.csv
- Combine title + text into one field
- Clean the text (lowercase, remove URLs, remove extra symbols)
- Create train/validation/test splits
"""

import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, remove URLs, keep only letters/numbers/spaces."""
    if not isinstance(text, str):
        return ""

    # Make everything lowercase
    text = text.lower()

    # Replace URLs with a simple token
    text = re.sub(r"http\S+|www\.\S+", " url ", text)

    # Remove characters that are not letters, numbers or spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_and_split_data(
    data_raw_dir: str,
    true_file: str = "True.csv",
    fake_file: str = "Fake.csv",
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
):
    """
    Load data from True.csv and Fake.csv, label them, clean,
    and split into train/val/test.

    Returns: train_df, val_df, test_df
    with columns: ["text", "label"]
    """
    true_path = os.path.join(data_raw_dir, true_file)
    fake_path = os.path.join(data_raw_dir, fake_file)

    if not os.path.exists(true_path):
        raise FileNotFoundError(f"True file not found at: {true_path}")
    if not os.path.exists(fake_path):
        raise FileNotFoundError(f"Fake file not found at: {fake_path}")

    df_true = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)

    # Add label: 1 = real, 0 = fake (you can flip if you prefer)
    df_true["label"] = 1
    df_fake["label"] = 0

    # Combine title + text into one field
    for df in [df_true, df_fake]:
        if "title" in df.columns and "text" in df.columns:
            df["text_full"] = (df["title"].astype(str) + " " + df["text"].astype(str))
        elif "text" in df.columns:
            df["text_full"] = df["text"].astype(str)
        else:
            raise ValueError(
                "Expected at least a 'text' column in the CSV files."
            )

        # Clean the combined text
        df["text_full"] = df["text_full"].apply(clean_text)

    # Merge true and fake into one dataframe
    df = pd.concat([df_true, df_fake], ignore_index=True)
    df = df[["text_full", "label"]].rename(columns={"text_full": "text"})

    # First split into train + temp (val+test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_size + test_size),
        stratify=df["label"],
        random_state=random_state,
    )

    # Now split temp into val and test according to ratios
    relative_val_size = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - relative_val_size),
        stratify=temp_df["label"],
        random_state=random_state,
    )

    # Reset indexes to keep things clean
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )