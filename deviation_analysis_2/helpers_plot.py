import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerBase
from typing import Optional, Tuple, Union

class NumberedCircleHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        cx = xdescent + 0.5 * width / 6.0
        cy = ydescent + height / 2.0
        size_factor = getattr(orig_handle, "_size", 0.45)
        r = size_factor * height

        circle = mpatches.Circle((cx, cy), radius=r,
                                 transform=trans,
                                 facecolor=orig_handle.get_facecolor(),
                                 edgecolor='none', linewidth=0)
        num_text = plt.Text(cx, cy, str(getattr(orig_handle, "_num", "")),
                            transform=trans, color="white",
                            fontsize=fontsize * 0.9, ha="center", va="center")
        return [circle, num_text]


class SymbolCircleHandler(HandlerBase):
    """
    Legend handler for a colored circle with a fixed white symbol (e.g. ✓, ✗, !).
    """
    def __init__(self, symbol: str, fontsize_factor: float = 0.9, **kwargs):
        super().__init__(**kwargs)
        self.symbol = symbol
        self.fontsize_factor = fontsize_factor

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        cx = xdescent + 0.5 * width / 6.0
        cy = ydescent + height / 2.0
        size_factor = getattr(orig_handle, "_size", 0.45)
        r = size_factor * height

        circle = mpatches.Circle((cx, cy), radius=r,
                                 transform=trans,
                                 facecolor=orig_handle.get_facecolor(),
                                 edgecolor='none', linewidth=0)
        symbol_text = plt.Text(cx, cy, self.symbol,
                               transform=trans, color="white",
                               fontsize=fontsize * self.fontsize_factor, ha="center", va="center")
        return [circle, symbol_text]


def plot_planned_and_unplanned_areas_with_numbered_legend(
    section_df: pd.DataFrame,
    *,
    figsize: Tuple[int, int] = (15, 5),
    itr1: Union[str, int] = "1",        # used for legend label (e.g. "A")
    itr2: Union[str, int] = "2",        # used for legend label (e.g. "A'")
    section_name: str = "",             # e.g. "A_A'" -> placed as endpoint labels
    dpi: int = 150,
    downsample: Optional[int] = None,
    xlabel: str = "Distance (m)",
    ylabel: str = "Elevation (m)",
    save_path: Optional[str] = None,
    show: bool = True,
    planned_color: str = "#2AC92A",   # green
    unplanned_color: str = "#D42525", # red
    planned_alpha: float = 0.5,
    unplanned_alpha: float = 0.5,
    legend_circle_size: float = 0.75,
    legend_handletextpad: float = 0.35,
    legend_labelspacing: float = 0.4,
    legend_borderpad: float = 0.4,
    legend_handlelength: float = 1.0,
    legend_columnspacing: float = 0.6,
) -> Tuple[plt.Figure, plt.Axes]:
    
    """
    Plots elevation profiles (using columns 'z_itr1' and 'z_itr2') and highlights
    planned/unplanned blocks. `itr1` and `itr2` are used for label text in the legend.
    If section_name contains an underscore (e.g. "A_A'"), the two parts are placed
    near the left and right endpoints of the plotted profile.
    """

    # REQUIREMENT CHECK (unchanged from your original code)
    req = {"chainage", "z_itr1", "z_itr2", "line_name", "area_name"}
    if not req.issubset(section_df.columns):
        raise ValueError(f"section_df missing required columns: {req - set(section_df.columns)}")

    # Prepare dataframe (sort + optional downsample)
    df_full = section_df.sort_values("chainage").reset_index(drop=True)
    df_plot = df_full.iloc[::downsample].reset_index(drop=True) if downsample else df_full

    # Base profiles (plot actual columns z_itr1 and z_itr2)
    color_itr1, color_itr2 = "#76c3ec", "#e99930"
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(df_plot["chainage"], df_plot["z_itr1"], color=color_itr1, linewidth=1.4, label=f"ITR-{itr1}")
    ax.plot(df_plot["chainage"], df_plot["z_itr2"], color=color_itr2, linewidth=1.4, label=f"ITR-{itr2}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    title = f"Elevation Profile Section {section_name.replace('_', ' ')}"
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.65, alpha=0.6)

    # Categories (exact-match sets)
    planned_set = {"planned_and_done_excavation", "planned_and_done_dump"}
    unplanned_set = {"unplanned_and_done_excavation", "unplanned_and_done_dump"}

    n = len(df_full)

    # Collect handles for side legends (numbered circles)
    planned_handles, planned_labels = [], []
    unplanned_handles, unplanned_labels = [], []

    # global z-range for offsetting markers
    zmax = df_full[["z_itr1", "z_itr2"]].max().max()
    zmin = df_full[["z_itr1", "z_itr2"]].min().min()
    z_range = zmax - zmin if zmax != zmin else 1.0

    i = 0
    planned_num, unplanned_num = 1, 1

    # track min/max marker to expand y-limits later if needed
    marker_min = float("inf")
    marker_max = float("-inf")

    # track max deviation among unplanned blocks
    max_unplanned_dev = 0.0
    has_unplanned = False

    while i < n:
        lbl = df_full.loc[i, "line_name"]
        if pd.isna(lbl) or (lbl not in planned_set and lbl not in unplanned_set):
            i += 1
            continue

        start_i = i
        current_area = df_full.loc[i, "area_name"]
        # prefer label_name if available, else fall back to area_name
        label_candidate = None
        if "label_name" in df_full.columns:
            label_candidate = df_full.loc[start_i, "label_name"]
        if pd.isna(label_candidate) or label_candidate is None:
            label_candidate = current_area

        i += 1
        while i < n and df_full.loc[i, "line_name"] == lbl:
            i += 1
        end_i = i - 1

        x_block = df_full.loc[start_i:end_i, "chainage"].to_numpy()
        z1_block = df_full.loc[start_i:end_i, "z_itr1"].to_numpy()
        z2_block = df_full.loc[start_i:end_i, "z_itr2"].to_numpy()
        if x_block.size == 0:
            continue

        start_chain, end_chain = float(x_block[0]), float(x_block[-1])
        mid_chain = (start_chain + end_chain) / 2.0

        # planned (green) markers above block
        if lbl in planned_set:
            elev_marker = max(np.max(z1_block), np.max(z2_block)) + 0.02 * z_range
            ax.fill_between(x_block, z1_block, z2_block, color=planned_color, alpha=planned_alpha, linewidth=0)
            ax.scatter([mid_chain], [elev_marker], s=220, color=planned_color, zorder=6, edgecolors='none')
            ax.text(mid_chain, elev_marker, str(planned_num), ha="center", va="center", fontsize=8, color="white", zorder=7)

            legend_circle = mpatches.Circle((0, 0), radius=0.5, facecolor=planned_color, edgecolor='none')
            legend_circle._num = planned_num
            legend_circle._size = legend_circle_size
            planned_handles.append(legend_circle)

            planned_labels.append(str(label_candidate) if label_candidate is not None else "")

            planned_num += 1
            marker_max = max(marker_max, elev_marker)

        # unplanned (red) markers below block
        elif lbl in unplanned_set:
            deviation = float(np.max(np.abs(z1_block - z2_block)))
            if deviation > max_unplanned_dev:
                max_unplanned_dev = deviation
            has_unplanned = True

            elev_marker = min(np.min(z1_block), np.min(z2_block)) - 0.02 * z_range
            ax.fill_between(x_block, z1_block, z2_block, color=unplanned_color, alpha=unplanned_alpha, linewidth=0)
            ax.scatter([mid_chain], [elev_marker], s=220, color=unplanned_color, zorder=6, edgecolors='none')
            ax.text(mid_chain, elev_marker, str(unplanned_num), ha="center", va="center", fontsize=8, color="white", zorder=7)

            legend_circle = mpatches.Circle((0, 0), radius=0.5, facecolor=unplanned_color, edgecolor='none')
            legend_circle._num = unplanned_num
            legend_circle._size = legend_circle_size
            unplanned_handles.append(legend_circle)

            label_text = (str(label_candidate) if label_candidate is not None else "")
            unplanned_labels.append(f"{label_text} ({deviation:.2f} m)")

            unplanned_num += 1
            marker_min = min(marker_min, elev_marker)

    # Side legends for numbered planned/unplanned areas
    if planned_handles:
        handler_map = {h: NumberedCircleHandler() for h in planned_handles}
        leg_planned = ax.legend(handles=planned_handles, labels=planned_labels,
                                loc="lower left", bbox_to_anchor=(0.02, 0.02),
                                frameon=True, fontsize="small", handler_map=handler_map,
                                handletextpad=legend_handletextpad,
                                labelspacing=legend_labelspacing,
                                borderpad=legend_borderpad,
                                handlelength=legend_handlelength)
        ax.add_artist(leg_planned)

    if unplanned_handles:
        handler_map = {h: NumberedCircleHandler() for h in unplanned_handles}
        ax.legend(handles=unplanned_handles, labels=unplanned_labels,
                  loc="lower right", bbox_to_anchor=(0.98, 0.02),
                  frameon=True, fontsize="small", handler_map=handler_map,
                  handletextpad=legend_handletextpad,
                  labelspacing=legend_labelspacing,
                  borderpad=legend_borderpad,
                  handlelength=legend_handlelength)

    # Combine into one single-line legend in the requested order:
    # Planned, Unplanned, ITR-itr1, ITR-itr2, Deviation
    planned_symbol = mpatches.Circle((0, 0), radius=0.5, facecolor=planned_color, edgecolor='none')
    planned_symbol._size = legend_circle_size
    unplanned_symbol = mpatches.Circle((0, 0), radius=0.5, facecolor=unplanned_color, edgecolor='none')
    unplanned_symbol._size = legend_circle_size

    itr1_handle = Line2D([0], [0], color=color_itr1, lw=2)
    itr2_handle = Line2D([0], [0], color=color_itr2, lw=2)
    proxy_dev = Line2D([], [], linestyle="", marker=None, color="none")

    dev_label = f"Deviation detected: {max_unplanned_dev:.2f} m" if has_unplanned else "No deviation detected"

    all_handles = [planned_symbol, unplanned_symbol, itr1_handle, itr2_handle, proxy_dev]
    all_labels  = ["Planned Area", "Unplanned Area", f" ITR {itr1}", f" ITR {itr2}", dev_label]

    fig.legend(
        handles=all_handles,
        labels=all_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=len(all_handles),
        frameon=False,
        fontsize="small",
        handler_map={
            planned_symbol: SymbolCircleHandler("✓"),
            unplanned_symbol: SymbolCircleHandler("✗")
        },
        handletextpad=0.0000005,
        #columnspacing=legend_columnspacing
    )

    # --- Add section endpoint labels if section_name contains an underscore like "A_A'" ---
    if section_name and "_" in section_name:
        left_label, right_label = section_name.split("_", 1)

        # endpoints from the plotted (possibly downsampled) dataframe
        x_start = float(df_plot["chainage"].iloc[0])
        x_end = float(df_plot["chainage"].iloc[-1])

        # take the z-values at endpoints (mean of the two series) so label sits near profile
        z_start_vals = np.array([df_plot["z_itr1"].iloc[0], df_plot["z_itr2"].iloc[0]])
        z_end_vals   = np.array([df_plot["z_itr1"].iloc[-1], df_plot["z_itr2"].iloc[-1]])
        y_start = np.nanmean(z_start_vals)
        y_end   = np.nanmean(z_end_vals)

        # offset labels slightly above the profile to avoid overlapping with markers
        offset = 0.03 * z_range
        ax.text(x_start, y_start + offset, left_label, ha="left", va="bottom",
                fontsize=10, fontweight="bold")
        ax.text(x_end,   y_end   + offset, right_label, ha="right", va="bottom",
                fontsize=10, fontweight="bold")

    # Expand y-limits so no markers are clipped
    if marker_min != float("inf") or marker_max != float("-inf"):
        cur_ylim = ax.get_ylim()
        margin = 0.03 * z_range
        new_ymin = min(cur_ylim[0], marker_min - margin) if marker_min != float("inf") else cur_ylim[0]
        new_ymax = max(cur_ylim[1], marker_max + margin) if marker_max != float("-inf") else cur_ylim[1]
        ax.set_ylim(new_ymin, new_ymax)

    plt.subplots_adjust(bottom=0.18)
    if save_path:
        if os.path.isdir(save_path) or save_path.endswith(os.sep):
            # Build default filename from section_name
            safe_section = section_name.replace("_", " ")
            filename = f"elevation_profile_{safe_section}.png"
            save_path = os.path.join(save_path, filename)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax
