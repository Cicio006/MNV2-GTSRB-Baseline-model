"""Run one GTSRB image through MobileNetV2 on a CUDA cluster node."""

import csv
from pathlib import Path

import cupy as cp
import numpy as np

from src.backends.cupy import CuPyBackend
from src.models.Mobile_net import MobileNetV2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "src" / "data" / "gtsrb"
IMAGE_ROOT = DATA_ROOT / "GTSRB" / "Final_Test" / "Images"
LABEL_FILE = DATA_ROOT / "GT-final_test.csv"


def _read_ppm_token(handle) -> bytes:
    token = bytearray()

    while True:
        char = handle.read(1)
        if not char:
            raise ValueError("unexpected end of PPM header")
        if char == b"#":
            handle.readline()
            continue
        if not char.isspace():
            token.extend(char)
            break

    while True:
        char = handle.read(1)
        if not char or char.isspace():
            break
        token.extend(char)

    return bytes(token)


def read_ppm(path: Path) -> np.ndarray:
    """Read a binary P6 PPM image as an RGB NumPy array."""
    with path.open("rb") as image_file:
        magic = _read_ppm_token(image_file)
        if magic != b"P6":
            raise ValueError(f"unsupported PPM format {magic!r}")

        width = int(_read_ppm_token(image_file))
        height = int(_read_ppm_token(image_file))
        max_value = int(_read_ppm_token(image_file))

        if max_value != 255:
            raise ValueError("only 8-bit PPM images are supported")

        data = np.frombuffer(
            image_file.read(width * height * 3),
            dtype=np.uint8,
        )

    if data.size != width * height * 3:
        raise ValueError(f"incomplete PPM image: {path}")

    return data.reshape(height, width, 3)


def resize_nearest(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize HWC image with nearest-neighbor sampling."""
    target_height, target_width = size
    source_height, source_width = image.shape[:2]
    y_indices = np.linspace(
        0,
        source_height - 1,
        target_height,
    ).astype(np.int64)
    x_indices = np.linspace(
        0,
        source_width - 1,
        target_width,
    ).astype(np.int64)
    return image[y_indices][:, x_indices]


def load_first_sample() -> tuple[cp.ndarray, Path, int]:
    last_error: Exception | None = None

    with LABEL_FILE.open(newline="", encoding="utf-8") as label_file:
        for row in csv.DictReader(label_file, delimiter=";"):
            image_path = IMAGE_ROOT / row["Filename"]

            try:
                image = resize_nearest(read_ppm(image_path), (64, 64))
            except (OSError, ValueError) as exc:
                last_error = exc
                print(f"skipping unreadable image {image_path}: {exc}")
                continue

            array = image.astype(np.float32) / 255.0
            inputs = cp.asarray(np.transpose(array, (2, 0, 1))[None, ...])
            return inputs, image_path, int(row["ClassId"])

    raise RuntimeError(
        f"no readable GTSRB test image found below {IMAGE_ROOT}"
    ) from last_error


def main() -> None:
    cp.random.seed(0)
    inputs, image_path, label = load_first_sample()
    model = MobileNetV2(CuPyBackend())
    logits = model(inputs)

    print(f"image: {image_path}")
    print(f"label: {label}")
    print(f"input shape: {inputs.shape}")
    print(f"final feature shape: {model.output_shape}")
    print(f"logits shape: {logits.shape}")
    print(f"finite logits: {bool(cp.isfinite(logits).all())}")
    print(f"random-weight prediction: {int(cp.argmax(logits, axis=1)[0])}")


if __name__ == "__main__":
    main()
