"""Plot Monet vocabulary/image/latent joint PCA with an optional trajectory.

Edit the global variables below and run this file directly. There is no CLI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Global configuration -- edit values here; no command-line arguments are used
# ---------------------------------------------------------------------------

PCA_RESULT_PATH = r"E:\fsdownload\joint_pca_3d.npz"
LATENT_TRAJECTORY_PATH = r"E:\fsdownload\latent_trajectories.npz"

SHOW_TRAJECTORY = True
TRAJECTORY_SAMPLE_ORDINAL = 0

# Set to None to draw every vocabulary/image point. Latent points are never
# downsampled for display.
MAX_DISPLAY_POINTS_PER_KIND: int | None = 50_000
DISPLAY_RANDOM_SEED = 0

FIGURE_SIZE = (12, 10)
SCATTER_SIZE = 2.0
SCATTER_ALPHA = 0.15
TRAJECTORY_LINE_WIDTH = 2.0
TRAJECTORY_MARKER_SIZE = 3.0

COLOR_MAP = {
    0: "yellow",
    1: "orange",
    2: "black",
}
LABEL_MAP = {
    0: "vocabulary_embedding",
    1: "image_feature",
    2: "latent",
}


def _validate_plot_configuration() -> None:
    if MAX_DISPLAY_POINTS_PER_KIND is not None:
        if MAX_DISPLAY_POINTS_PER_KIND <= 0:
            raise ValueError("MAX_DISPLAY_POINTS_PER_KIND must be positive or None.")
    if TRAJECTORY_SAMPLE_ORDINAL < 0:
        raise ValueError("TRAJECTORY_SAMPLE_ORDINAL must be non-negative.")


def _display_indices(kind_codes: np.ndarray, kind_code: int) -> np.ndarray:
    indices = np.flatnonzero(kind_codes == kind_code)
    limit = MAX_DISPLAY_POINTS_PER_KIND
    if kind_code == 2 or limit is None or len(indices) <= limit:
        return indices
    rng = np.random.default_rng(DISPLAY_RANDOM_SEED + kind_code)
    selected = rng.choice(len(indices), size=limit, replace=False)
    return indices[np.sort(selected)]


def _plot_selected_trajectory(ax, trajectory_path: Path) -> None:
    with np.load(trajectory_path, allow_pickle=False) as trajectory:
        coordinates = trajectory["coordinates"]
        sample_offsets = trajectory["sample_offsets"]
        dataset_indices = trajectory["dataset_indices"]
        sample_count = len(sample_offsets) - 1
        if TRAJECTORY_SAMPLE_ORDINAL >= sample_count:
            raise IndexError(
                "TRAJECTORY_SAMPLE_ORDINAL is outside the saved sample range: "
                f"{TRAJECTORY_SAMPLE_ORDINAL} >= {sample_count}"
            )
        start = int(sample_offsets[TRAJECTORY_SAMPLE_ORDINAL])
        end = int(sample_offsets[TRAJECTORY_SAMPLE_ORDINAL + 1])
        if start == end:
            print(
                "Selected sample has no latent trajectory: "
                f"sample_ordinal={TRAJECTORY_SAMPLE_ORDINAL}, "
                f"dataset_index={dataset_indices[TRAJECTORY_SAMPLE_ORDINAL]}"
            )
            return
        selected = coordinates[start:end]
        dataset_index = str(dataset_indices[TRAJECTORY_SAMPLE_ORDINAL])

    ax.plot(
        selected[:, 0],
        selected[:, 1],
        selected[:, 2],
        color="red",
        linewidth=TRAJECTORY_LINE_WIDTH,
        marker="o",
        markersize=TRAJECTORY_MARKER_SIZE,
        label=(
            f"latent trajectory sample={TRAJECTORY_SAMPLE_ORDINAL}, "
            f"dataset={dataset_index}"
        ),
    )


def create_figure(
    pca_path: str | Path = PCA_RESULT_PATH,
    trajectory_path: str | Path = LATENT_TRAJECTORY_PATH,
):
    _validate_plot_configuration()
    pca_path = Path(pca_path).expanduser()
    trajectory_path = Path(trajectory_path).expanduser()
    if not pca_path.is_file():
        raise FileNotFoundError(f"Joint PCA file does not exist: {pca_path}")
    if SHOW_TRAJECTORY and not trajectory_path.is_file():
        raise FileNotFoundError(
            f"Latent trajectory file does not exist: {trajectory_path}"
        )

    with np.load(pca_path, allow_pickle=False) as data:
        coordinates = data["coordinates"]
        kind_codes = data["kind_codes"]
        kind_names = data["kind_names"]
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError(
                f"Expected PCA coordinates with shape [N, 3], got {coordinates.shape}."
            )
        if len(kind_codes) != len(coordinates):
            raise ValueError("kind_codes and coordinates have different lengths.")
        kind_name_map = {
            index: str(name) for index, name in enumerate(kind_names.tolist())
        }
        print(f"PCA kind mapping: {kind_name_map}")

        figure = plt.figure(figsize=FIGURE_SIZE)
        axis = figure.add_subplot(111, projection="3d")
        for kind_code in sorted(np.unique(kind_codes).tolist()):
            indices = _display_indices(kind_codes, int(kind_code))
            axis.scatter(
                coordinates[indices, 0],
                coordinates[indices, 1],
                coordinates[indices, 2],
                s=SCATTER_SIZE,
                alpha=SCATTER_ALPHA,
                c=COLOR_MAP.get(int(kind_code), "gray"),
                label=LABEL_MAP.get(int(kind_code), f"class_{int(kind_code)}"),
                depthshade=False,
            )

    if SHOW_TRAJECTORY:
        _plot_selected_trajectory(axis, trajectory_path)

    axis.set_title("3D Joint PCA: Vocabulary / Image Features / Latents")
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.set_zlabel("PC3")
    axis.legend(markerscale=4)
    figure.tight_layout()
    return figure, axis


def main() -> None:
    create_figure()
    plt.show()


if __name__ == "__main__":
    main()
