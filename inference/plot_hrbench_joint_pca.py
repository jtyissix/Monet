"""
Plot Monet vocabulary / image / latent joint PCA with an optional latent trajectory.

Edit the global variables below and run this file directly.
There is no CLI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Global configuration
# =============================================================================

PCA_RESULT_PATH = r"E:\fsdownload\nnjoint_pca_3d.npz"
LATENT_TRAJECTORY_PATH = r"E:\fsdownload\nnlatent_trajectories.npz"


# -----------------------------------------------------------------------------
# Trajectory
# -----------------------------------------------------------------------------

# Whether to draw one latent trajectory
SHOW_TRAJECTORY = True

# Which sample trajectory to draw.
# This is sample_ordinal, NOT dataset index.
TRAJECTORY_SAMPLE_ORDINAL = 1


# -----------------------------------------------------------------------------
# Point display
# -----------------------------------------------------------------------------

# Maximum number of vocabulary/image points displayed.
#
# None:
#   Draw every point.
#
# int:
#   Randomly subsample vocabulary/image points for visualization.
#
# Latent points are NEVER downsampled.
MAX_DISPLAY_POINTS_PER_KIND: int | None = 10_000

DISPLAY_RANDOM_SEED = 0


# -----------------------------------------------------------------------------
# Figure appearance
# -----------------------------------------------------------------------------

FIGURE_SIZE = (12, 10)

SCATTER_SIZE = 2.0
SCATTER_ALPHA = 0.15

TRAJECTORY_LINE_WIDTH = 2.0
TRAJECTORY_MARKER_SIZE = 4.0

# Colors are mapped by the actual kind name stored inside the NPZ.
#
# This avoids relying on a manually assumed 0/1/2 correspondence.
COLOR_MAP = {
    "vocabulary_embedding": "blue",
    "image_feature": "red",
    "latent": "green",
}

# Keep trajectory different from every point category.
TRAJECTORY_COLOR = "grey"


# -----------------------------------------------------------------------------
# Optional figure saving
# -----------------------------------------------------------------------------

SAVE_FIGURE = False

FIGURE_OUTPUT_PATH = r"E:\fsdownload\nnjoint_pca_3d.png"

SAVE_DPI = 300


# =============================================================================
# Helpers
# =============================================================================


def _validate_plot_configuration() -> None:
    """Validate global plotting options."""

    if MAX_DISPLAY_POINTS_PER_KIND is not None:
        if MAX_DISPLAY_POINTS_PER_KIND <= 0:
            raise ValueError(
                "MAX_DISPLAY_POINTS_PER_KIND must be positive or None."
            )

    if TRAJECTORY_SAMPLE_ORDINAL < 0:
        raise ValueError(
            "TRAJECTORY_SAMPLE_ORDINAL must be non-negative."
        )


def _display_indices(
    kind_codes: np.ndarray,
    kind_code: int,
) -> np.ndarray:
    """
    Return indices that will actually be drawn.

    vocabulary_embedding / image_feature:
        May be downsampled.

    latent:
        Never downsampled.
    """

    indices = np.flatnonzero(kind_codes == kind_code)

    limit = MAX_DISPLAY_POINTS_PER_KIND

    # Latent is always fully displayed.
    if kind_code == 2:
        return indices

    if limit is None or len(indices) <= limit:
        return indices

    rng = np.random.default_rng(
        DISPLAY_RANDOM_SEED + kind_code
    )

    selected_positions = rng.choice(
        len(indices),
        size=limit,
        replace=False,
    )

    selected_positions = np.sort(selected_positions)

    return indices[selected_positions]


def _load_kind_name_map(
    kind_names: np.ndarray,
) -> dict[int, str]:
    """
    Convert the NPZ kind_names array into:

        {
            0: "vocabulary_embedding",
            1: "image_feature",
            2: "latent",
        }
    """

    return {
        index: str(name)
        for index, name in enumerate(kind_names.tolist())
    }


def _make_legend_label(
    kind_name: str,
    total_count: int,
    displayed_count: int,
) -> str:
    """
    Construct legend label containing point counts.
    """

    if displayed_count < total_count:
        return (
            f"{kind_name} "
            f"(N={total_count:,}, shown={displayed_count:,})"
        )

    return f"{kind_name} (N={total_count:,})"


def _plot_selected_trajectory(
    ax,
    trajectory_path: Path,
) -> None:
    """
    Plot one latent trajectory stored in latent_trajectories.npz.
    """

    with np.load(
        trajectory_path,
        allow_pickle=False,
    ) as trajectory:

        coordinates = trajectory["coordinates"]
        sample_offsets = trajectory["sample_offsets"]
        dataset_indices = trajectory["dataset_indices"]

        # -------------------------------------------------------------
        # Basic validation
        # -------------------------------------------------------------

        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError(
                "Expected trajectory coordinates with shape [N, 3], "
                f"got {coordinates.shape}."
            )

        if sample_offsets.ndim != 1:
            raise ValueError(
                "sample_offsets must be a 1D array."
            )

        sample_count = len(sample_offsets) - 1

        if TRAJECTORY_SAMPLE_ORDINAL >= sample_count:
            raise IndexError(
                "TRAJECTORY_SAMPLE_ORDINAL is outside the saved "
                "sample range: "
                f"{TRAJECTORY_SAMPLE_ORDINAL} >= {sample_count}"
            )

        # -------------------------------------------------------------
        # Select trajectory
        # -------------------------------------------------------------

        start = int(
            sample_offsets[TRAJECTORY_SAMPLE_ORDINAL]
        )

        end = int(
            sample_offsets[TRAJECTORY_SAMPLE_ORDINAL + 1]
        )

        dataset_index = str(
            dataset_indices[TRAJECTORY_SAMPLE_ORDINAL]
        )

        if start == end:
            print(
                "[Trajectory] Selected sample has no latent trajectory:"
            )
            print(
                f"  sample_ordinal = "
                f"{TRAJECTORY_SAMPLE_ORDINAL}"
            )
            print(
                f"  dataset_index  = {dataset_index}"
            )
            return

        selected = coordinates[start:end]

        # -------------------------------------------------------------
        # Diagnostics
        # -------------------------------------------------------------

        print()
        print("=" * 80)
        print("Selected latent trajectory")
        print("=" * 80)

        print(
            f"sample_ordinal : "
            f"{TRAJECTORY_SAMPLE_ORDINAL}"
        )

        print(
            f"dataset_index  : {dataset_index}"
        )

        print(
            f"trajectory_len : {len(selected):,}"
        )

        print(
            f"coordinate rows: [{start:,}, {end:,})"
        )

        # -------------------------------------------------------------
        # Plot
        # -------------------------------------------------------------

        ax.plot(
            selected[:, 0],
            selected[:, 1],
            selected[:, 2],
            color=TRAJECTORY_COLOR,
            linewidth=TRAJECTORY_LINE_WIDTH,
            marker="o",
            markersize=TRAJECTORY_MARKER_SIZE,
            label=(
                "latent trajectory "
                f"(sample={TRAJECTORY_SAMPLE_ORDINAL}, "
                f"dataset={dataset_index}, "
                f"steps={len(selected):,})"
            ),
            zorder=10,
        )


# =============================================================================
# Main plotting function
# =============================================================================


def create_figure(
    pca_path: str | Path = PCA_RESULT_PATH,
    trajectory_path: str | Path = LATENT_TRAJECTORY_PATH,
):
    """
    Create the joint 3D PCA figure.

    Returns
    -------
    figure
        Matplotlib Figure.

    axis
        Matplotlib 3D Axes.
    """

    _validate_plot_configuration()

    pca_path = Path(pca_path).expanduser()
    trajectory_path = Path(trajectory_path).expanduser()

    # -------------------------------------------------------------------------
    # Validate files
    # -------------------------------------------------------------------------

    if not pca_path.is_file():
        raise FileNotFoundError(
            f"Joint PCA file does not exist: {pca_path}"
        )

    if SHOW_TRAJECTORY and not trajectory_path.is_file():
        raise FileNotFoundError(
            f"Latent trajectory file does not exist: "
            f"{trajectory_path}"
        )

    # -------------------------------------------------------------------------
    # Load joint PCA
    # -------------------------------------------------------------------------

    with np.load(
        pca_path,
        allow_pickle=False,
    ) as data:

        required_keys = {
            "coordinates",
            "kind_codes",
            "kind_names",
        }

        missing_keys = required_keys.difference(data.files)

        if missing_keys:
            raise KeyError(
                "Joint PCA NPZ is missing keys: "
                + ", ".join(sorted(missing_keys))
            )

        coordinates = data["coordinates"]
        kind_codes = data["kind_codes"]
        kind_names = data["kind_names"]

        # PCA explained variance is available in your generated NPZ.
        explained_variance_ratio = (
            data["explained_variance_ratio"]
            if "explained_variance_ratio" in data.files
            else None
        )

        # ---------------------------------------------------------------------
        # Validate arrays
        # ---------------------------------------------------------------------

        if coordinates.ndim != 2:
            raise ValueError(
                "coordinates must be a 2D array, "
                f"got shape={coordinates.shape}"
            )

        if coordinates.shape[1] != 3:
            raise ValueError(
                "Expected PCA coordinates with shape [N, 3], "
                f"got {coordinates.shape}."
            )

        if kind_codes.ndim != 1:
            raise ValueError(
                "kind_codes must be a 1D array."
            )

        if len(kind_codes) != len(coordinates):
            raise ValueError(
                "kind_codes and coordinates have different lengths: "
                f"{len(kind_codes)} != {len(coordinates)}"
            )

        # ---------------------------------------------------------------------
        # Build real kind mapping from NPZ
        # ---------------------------------------------------------------------

        kind_name_map = _load_kind_name_map(
            kind_names
        )

        print()
        print("=" * 80)
        print("PCA kind mapping")
        print("=" * 80)

        for kind_code, kind_name in kind_name_map.items():
            print(
                f"{kind_code}: {kind_name}"
            )

        # ---------------------------------------------------------------------
        # Create figure
        # ---------------------------------------------------------------------

        figure = plt.figure(
            figsize=FIGURE_SIZE
        )

        axis = figure.add_subplot(
            111,
            projection="3d",
        )

        # ---------------------------------------------------------------------
        # Plot each kind
        # ---------------------------------------------------------------------

        print()
        print("=" * 80)
        print("Point statistics")
        print("=" * 80)

        unique_kind_codes = sorted(
            int(code)
            for code in np.unique(kind_codes)
        )

        total_all = 0
        displayed_all = 0

        for kind_code in unique_kind_codes:

            # -------------------------------------------------------------
            # Obtain actual semantic name from NPZ
            # -------------------------------------------------------------

            kind_name = kind_name_map.get(
                kind_code,
                f"class_{kind_code}",
            )

            # -------------------------------------------------------------
            # Count ALL points
            # -------------------------------------------------------------

            all_indices = np.flatnonzero(
                kind_codes == kind_code
            )

            total_count = len(all_indices)

            # -------------------------------------------------------------
            # Determine points actually displayed
            # -------------------------------------------------------------

            display_indices = _display_indices(
                kind_codes,
                kind_code,
            )

            displayed_count = len(display_indices)

            total_all += total_count
            displayed_all += displayed_count

            # -------------------------------------------------------------
            # Print statistics
            # -------------------------------------------------------------

            print(
                f"{kind_name:<24} "
                f"total={total_count:>10,}   "
                f"displayed={displayed_count:>10,}"
            )

            # -------------------------------------------------------------
            # Build legend
            # -------------------------------------------------------------

            legend_label = _make_legend_label(
                kind_name,
                total_count,
                displayed_count,
            )

            # -------------------------------------------------------------
            # Get color according to actual semantic name
            # -------------------------------------------------------------

            color = COLOR_MAP.get(
                kind_name,
                "gray",
            )

            # -------------------------------------------------------------
            # Scatter
            # -------------------------------------------------------------

            axis.scatter(
                coordinates[display_indices, 0],
                coordinates[display_indices, 1],
                coordinates[display_indices, 2],
                s=SCATTER_SIZE,
                alpha=SCATTER_ALPHA,
                c=color,
                label=legend_label,
                depthshade=False,
            )

        print("-" * 80)

        print(
            f"{'TOTAL':<24} "
            f"total={total_all:>10,}   "
            f"displayed={displayed_all:>10,}"
        )

    # -------------------------------------------------------------------------
    # Plot trajectory
    # -------------------------------------------------------------------------

    if SHOW_TRAJECTORY:
        _plot_selected_trajectory(
            axis,
            trajectory_path,
        )

    # -------------------------------------------------------------------------
    # Axis title / labels
    # -------------------------------------------------------------------------

    axis.set_title(
        "3D Joint PCA: Vocabulary / Image Features / Latents",
        pad=20,
    )

    # Use explained variance percentage if saved in NPZ.
    if (
        explained_variance_ratio is not None
        and len(explained_variance_ratio) >= 3
    ):
        pc1_ratio = explained_variance_ratio[0] * 100
        pc2_ratio = explained_variance_ratio[1] * 100
        pc3_ratio = explained_variance_ratio[2] * 100

        axis.set_xlabel(
            f"PC1 ({pc1_ratio:.2f}%)"
        )

        axis.set_ylabel(
            f"PC2 ({pc2_ratio:.2f}%)"
        )

        axis.set_zlabel(
            f"PC3 ({pc3_ratio:.2f}%)"
        )

        print()
        print("=" * 80)
        print("PCA explained variance")
        print("=" * 80)

        print(f"PC1: {pc1_ratio:.4f}%")
        print(f"PC2: {pc2_ratio:.4f}%")
        print(f"PC3: {pc3_ratio:.4f}%")

        print(
            "PC1+PC2+PC3: "
            f"{pc1_ratio + pc2_ratio + pc3_ratio:.4f}%"
        )

    else:
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        axis.set_zlabel("PC3")

    # -------------------------------------------------------------------------
    # Legend
    # -------------------------------------------------------------------------

    legend = axis.legend(
        loc="best",
        markerscale=4,
    )

    # Scatter alpha is only 0.15, which can make legend markers look nearly
    # invisible. Force legend markers to full opacity.
    #
    # Different Matplotlib versions expose the handles under different names.
    legend_handles = getattr(
        legend,
        "legend_handles",
        None,
    )

    if legend_handles is None:
        legend_handles = getattr(
            legend,
            "legendHandles",
            [],
        )

    for handle in legend_handles:
        try:
            handle.set_alpha(1.0)
        except Exception:
            pass

    figure.tight_layout()

    return figure, axis


# =============================================================================
# Entry
# =============================================================================


def main() -> None:

    figure, _ = create_figure()

    if SAVE_FIGURE:
        output_path = Path(
            FIGURE_OUTPUT_PATH
        ).expanduser()

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        figure.savefig(
            output_path,
            dpi=SAVE_DPI,
            bbox_inches="tight",
        )

        print()
        print(
            f"Figure saved to: {output_path}"
        )

    plt.show()


if __name__ == "__main__":
    main()