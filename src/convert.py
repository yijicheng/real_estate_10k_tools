import subprocess
import sys
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import torch
from jaxtyping import Float, Int, UInt8
from torch import Tensor
from tqdm import tqdm
import json

INPUT_IMAGE_DIR = Path("/workspace/real_estate_10k_tools/datasets/DL3DV-10K-960P")
OUTPUT_DIR = Path("/workspace/real_estate_10k_tools/datasets/DL3DV-10K-960P/dl3dv_pt")

# Target 100 MB per chunk.
TARGET_BYTES_PER_CHUNK = int(5e9)


def get_example_keys(stage: Literal["test", "train"]) -> list[str]:
    sequence_keys = set(
        example.name
        for example in tqdm((INPUT_IMAGE_DIR / stage).iterdir(), desc="Indexing sequences")
    )

    print(f"Found {len(sequence_keys)} keys.")
    return sequence_keys


def get_size(path: Path) -> int:
    """Get file or folder size in bytes."""
    return int(subprocess.check_output(["du", "-b", path]).split()[0].decode("utf-8"))


def load_raw(path: Path) -> UInt8[Tensor, " length"]:
    return torch.tensor(np.memmap(path, dtype="uint8", mode="r"))


def load_images(example_path: Path) -> dict[int, UInt8[Tensor, "..."]]:
    """Load JPG images as raw bytes (do not decode)."""

    return {path.stem: load_raw(path) for path in example_path.iterdir()}


class Metadata(TypedDict):
    url: str
    timestamps: Int[Tensor, " camera"]
    cameras: Float[Tensor, "camera entry"]


class Example(Metadata):
    key: str
    images: list[UInt8[Tensor, "..."]]


def load_metadata(example_path: Path) -> Metadata:
    with example_path.open("r") as f:
        meta = json.load(f)

    return meta


if __name__ == "__main__":
    for stage in ("1K","2K","3K","4K",):
    # for stage in ("1K",):
        keys = get_example_keys(stage)

        chunk_size = 0
        chunk_index = 0
        chunk: list[Example] = []

        def save_chunk():
            global chunk_size
            global chunk_index
            global chunk

            chunk_key = f"{chunk_index:0>6}"
            print(
                f"Saving chunk {chunk_key} of {len(keys)} ({chunk_size / 1e6:.2f} MB)."
            )
            dir = OUTPUT_DIR / stage
            dir.mkdir(exist_ok=True, parents=True)
            torch.save(chunk, dir / f"{chunk_key}.torch")

            # Reset the chunk.
            chunk_size = 0
            chunk_index += 1
            chunk = []
        
        for key in keys:
            image_dir = INPUT_IMAGE_DIR / stage / key / "images_4"
            metadata_dir = INPUT_IMAGE_DIR / stage / key / "transforms.json"
            num_bytes = get_size(image_dir)

            # Read images and metadata.
            try:
                images = load_images(image_dir)
                example = load_metadata(metadata_dir)
            except:
                print(key)
                continue

            # Merge the images into the example.
            example["images"] = images
            # assert len(images.keys()) == len(example["frames"])

            # Add the key to the example.
            example["key"] = key

            print(f"    Added {key} to chunk ({num_bytes / 1e6:.2f} MB).")
            chunk.append(example)
            chunk_size += num_bytes

            if chunk_size >= TARGET_BYTES_PER_CHUNK:
                save_chunk()

        if chunk_size > 0:
            save_chunk()
