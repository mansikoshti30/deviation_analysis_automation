from pathlib import Path
import math, re
import pandas as pd
import ezdxf
from ezdxf import units, colors as ezcolors
from ezdxf.lldxf import const as ezconst  # alignment enums





def export_elevation_profiles_to_dxf_try_8(
    csv_path: str,
    out_path: str = "elevation_profile.dxf",
    *,
    # line styles
    iter1_color_hex: str = "#1f77b4",
    iter2_color_hex: str = "#ff7f0e",
    layer_iter1: str = "ITR_2024_Profile",
    layer_iter2: str = "ITR_2025_Profile",
    # text / axes
    text_layer: str = "Text",
    text_height: float = 2.0,
    show_axes: bool = True,
    axis_layer: str = "Axes",
    axis_color_hex: str = "#000000",
    x_label: str = "Chainage (m)",
    y_label: str = "Elevation (m)",
    target_tick_count: int = 7,
    tick_len_factor: float = 0.6,
    label_gap_factor: float = 0.5,
    # extra left shove for Y tick labels (in text-heights)
    y_tick_label_side_factor: float = 2.5,
    # transforms
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    # title
    title: str | None = "Elevation Profile (m)",
    # columns
    chain_col: str | None = None,
    elev1_col: str | None = None,
    elev2_col: str | None = None,
    line_name_col: str = "line_name",
    label_name_col: str = "label_name",   # NEW
    # fills
    layer_fill_planned: str = "Fill_Planned",
    layer_fill_unplanned: str = "Fill_Unplanned",
    planned_hex: str = "#27ae60",
    unplanned_hex: str = "#c0392b",
    # badges (numbers)
    add_zone_badges: bool = True,
    label_layer: str = "Zone_Badges",
    badge_radius_factor: float = 0.9,
    badge_position: str = "auto",       # "top" | "bottom" | "auto"
    badge_offset_factor: float = 1.6,   # distance from patch edge in TEXT heights
    # section end labels
    section_name: str | None = None,    # e.g. "K_K'"
    section_label_layer: str = "Section_Labels",
    section_offset_factor: float = 2.0, # vertical offset in TEXT heights
    section_x_pad_factor: float = 1.5,  # horizontal x-padding from plot edges in TEXT heights
    # LEGENDS (NEW)
    show_legends: bool = True,
    legend_layer: str = "Legend",
    legend_pad_factor: float = 1.0,         # padding from axes in text-heights
    legend_row_gap_factor: float = 0.5,    # between rows
    legend_bullet_radius_factor: float = 0.5,
    legend_col_gap_factor: float = 0.7,     # bullet -> text gap
    bottom_legend_gap_factor: float = 5,  # distance above x-axis for the bottom legend
    bottom_legend_col_gap_factor: float = 10.0,
    # NEW: highlight only unplanned areas with deviation > this threshold (meters)
    deviation_threshold: float = 0,
) -> str:
    """DXF elevation profile with numbered badges, per-area legends, and section-end labels.
       Unplanned areas are only highlighted (red hatch + legend entry + red badge) when
       their max deviation exceeds `deviation_threshold`.
    """

    # ---------- helpers
    def _hex_to_rgb(h: str) -> tuple[int, int, int]:
        h = h.strip().lstrip("#")
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        if len(h) != 6:
            raise ValueError(f"Bad hex color: {h}")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    def _find(df: pd.DataFrame, pats: list[str]) -> str | None:
        cols = {c.lower(): c for c in df.columns}
        for p in pats:
            rx = re.compile(p, re.I)
            for lo, orig in cols.items():
                if rx.fullmatch(lo) or rx.search(lo):
                    return orig
        return None

    def _guess_columns(df: pd.DataFrame):
        ch = _find(df, [r"^(chain(age)?|ch|sta(te)?)$"])
        i1 = _find(df, [r"^(elev(ation)?|z)(_?(1|it(era)?t?1|iter_?1|v1))?$"])
        i2 = _find(df, [r"^(elev(ation)?|z)(_?(2|it(era)?t?2|iter_?2|v2))?$"])
        if not (i1 and i2):
            elev_like = [c for c in df.columns if re.search(r"^(elev|z)", c, re.I)]
            if elev_like and not i1:
                i1 = elev_like[0]
            if len(elev_like) >= 2 and not i2:
                i2 = elev_like[1]
        return ch, i1, i2

    def _nice_step(span: float, target: int) -> float:
        if span <= 0:
            return 1.0
        step = span / max(1, target)
        mag = 10 ** math.floor(math.log10(step))
        norm = step / mag
        return (1 if norm < 1.5 else 2 if norm < 3 else 5 if norm < 7 else 10) * mag

    def _format_tick(v: float) -> str:
        a = abs(v)
        if a >= 1000 or a == 0:
            return f"{v:.0f}"
        if a >= 100:
            return f"{v:.1f}"
        if a >= 10:
            return f"{v:.2f}"
        return f"{v:.3f}"

    def _add_profile(msp, df: pd.DataFrame, ccol: str, zcol: str, layer: str, rgb: tuple[int, int, int]):
        d = df[[ccol, zcol]].dropna().sort_values(ccol)
        if d.empty:
            return
        pts = [
            (float(r[ccol]) * scale_x + offset_x, float(r[zcol]) * scale_y + offset_y)
            for _, r in d.iterrows()
        ]
        msp.add_lwpolyline(pts, dxfattribs={"layer": layer, "true_color": ezcolors.rgb2int(rgb)})

    def _add_text_centered(s, x, y, rgb=None, h=None, layer=None):
        t = msp.add_text(
            s,
            dxfattribs={
                "layer": layer or text_layer,
                "height": h or text_height,
                "halign": ezconst.CENTER,
                "valign": ezconst.MIDDLE,
                **({"true_color": ezcolors.rgb2int(rgb)} if rgb else {}),
            },
        )
        t.dxf.insert = (x, y)
        t.dxf.align_point = (x, y)
        return t

    def _add_text_left(s, x, y, h=None, layer=None):
        t = msp.add_text(
            s,
            dxfattribs={
                "layer": layer or text_layer,
                "height": h or text_height,
                "halign": ezconst.LEFT,
                "valign": ezconst.MIDDLE,
            },
        )
        t.dxf.insert = (x, y)
        t.dxf.align_point = (x, y)
        return t

    def _first_non_empty(series, fallback: str) -> str:
        """Return first non-empty/NaN string from a Series; else fallback."""
        if series is None:
            return fallback
        for v in series:
            if pd.notna(v):
                s = str(v).strip()
                if s and s.lower() != "nan":
                    return s
        return fallback

    # ---------- load
    df = pd.read_csv(csv_path, low_memory=False)
    if chain_col is None or (elev1_col is None and elev2_col is None):
        g_chain, g_e1, g_e2 = _guess_columns(df)
        chain_col = chain_col or g_chain
        elev1_col = elev1_col or g_e1
        elev2_col = elev2_col or g_e2
    if chain_col is None:
        raise ValueError("No chainage column found.")
    if elev1_col is None or elev2_col is None:
        raise ValueError("Need two elevation columns.")

    # ---------- dxf doc
    doc = ezdxf.new("R2018", units=units.M)
    doc.header["$INSUNITS"] = units.M
    for layer in (
        layer_iter1,
        layer_iter2,
        text_layer,
        axis_layer if show_axes else None,
        layer_fill_planned,
        layer_fill_unplanned,
        label_layer if add_zone_badges else None,
        section_label_layer if section_name else None,
        legend_layer if show_legends else None,
    ):
        if layer and layer not in doc.layers:
            doc.layers.add(layer)
    msp = doc.modelspace()

    # profiles
    _add_profile(msp, df, chain_col, elev1_col, layer_iter1, _hex_to_rgb(iter1_color_hex))
    _add_profile(msp, df, chain_col, elev2_col, layer_iter2, _hex_to_rgb(iter2_color_hex))

    # title
    if title:
        t = msp.add_text(title, dxfattribs={"layer": text_layer, "height": text_height})
        t.dxf.insert = (offset_x, offset_y + text_height * 4)

    # keep for legend placement
    plot_x_min = plot_x_max = plot_y_min = plot_y_max = None

    # ---------- axes
    if show_axes:
        chains = df[chain_col].dropna().astype(float)
        elevs = pd.concat(
            [df[elev1_col].dropna().astype(float), df[elev2_col].dropna().astype(float)]
        )
        ch_min, ch_max = float(chains.min()), float(chains.max())
        z_min, z_max = float(elevs.min()), float(elevs.max())
        xs, ys = _nice_step(ch_max - ch_min, target_tick_count), _nice_step(
            z_max - z_min, target_tick_count
        )

        x0 = math.floor(ch_min / xs) * xs
        x1 = math.ceil(ch_max / xs) * xs
        y0 = math.floor(z_min / ys) * ys
        y1 = math.ceil(z_max / ys) * ys

        ox = x0 * scale_x + offset_x
        oy = y0 * scale_y + offset_y
        axis_rgb = _hex_to_rgb(axis_color_hex)
        tick_len = tick_len_factor * text_height
        gap = label_gap_factor * text_height

        # axis bbox for legends
        plot_x_min = ox
        plot_x_max = x1 * scale_x + offset_x
        plot_y_min = oy
        plot_y_max = y1 * scale_y + offset_y

        # axis lines
        msp.add_line(
            (ox, oy), (x1 * scale_x + offset_x, oy),
            dxfattribs={"layer": axis_layer, "true_color": ezcolors.rgb2int(axis_rgb)},
        )
        msp.add_line(
            (ox, oy), (ox, y1 * scale_y + offset_y),
            dxfattribs={"layer": axis_layer, "true_color": ezcolors.rgb2int(axis_rgb)},
        )

        # X ticks & labels
        xt = x0
        while xt <= x1 + 1e-9:
            x_pos = xt * scale_x + offset_x
            msp.add_line(
                (x_pos, oy), (x_pos, oy - tick_len),
                dxfattribs={"layer": axis_layer, "true_color": ezcolors.rgb2int(axis_rgb)},
            )
            tx = msp.add_text(_format_tick(xt), dxfattribs={"layer": text_layer, "height": text_height})
            tx.dxf.insert = (x_pos, oy - tick_len - gap)
            xt += xs

        # Y ticks & labels
        yt = y0
        while yt <= y1 + 1e-9:
            y_pos = yt * scale_y + offset_y
            msp.add_line(
                (ox, y_pos), (ox - tick_len, y_pos),
                dxfattribs={"layer": axis_layer, "true_color": ezcolors.rgb2int(axis_rgb)},
            )
            ty = msp.add_text(
                _format_tick(yt),
                dxfattribs={
                    "layer": text_layer,
                    "height": text_height,
                    "halign": ezconst.RIGHT,
                    "valign": ezconst.MIDDLE,
                },
            )
            extra = y_tick_label_side_factor * text_height
            anchor_x = ox - tick_len - gap - extra
            ty.dxf.insert = (anchor_x, y_pos)
            ty.dxf.align_point = (anchor_x, y_pos)
            yt += ys

        # axis labels
        xtxt = msp.add_text(x_label, dxfattribs={"layer": text_layer, "height": text_height})
        xtxt.dxf.insert = ((ox + (x1 * scale_x + offset_x)) / 2, oy - 3 * text_height)

        ytxt = msp.add_text(
            y_label, dxfattribs={"layer": text_layer, "height": text_height, "rotation": 90},
        )
        ytxt.dxf.insert = (ox - 4 * text_height, (oy + (y1 * scale_y + offset_y)) / 2)

    # ---------- fills + badges + collect legend items
    legend_items_planned = []   # (idx, label)
    legend_items_unplanned = [] # (idx, label, deviation)
    max_unplanned_dev = 0.0

    if line_name_col in df.columns:
        valid = {
            "planned_and_done_excavation",
            "unplanned_and_done_excavation",
            "planned_and_done_dump",
            "unplanned_and_done_dump",
        }
        ln = df[line_name_col].where(df[line_name_col].isin(valid))
        grp = (ln.ne(ln.shift()) & ln.notna()).cumsum()
        segs = df.loc[ln.notna()].assign(_grp=grp[ln.notna()].values)

        badge_r = badge_radius_factor * text_height
        planned_idx = 1
        unplanned_idx = 1

        grouped = list(segs.groupby("_grp"))
        grouped.sort(key=lambda g: float(g[1][chain_col].astype(float).min()))

        for _, seg in grouped:
            name = seg[line_name_col].iloc[0]

            # geometry sub (safe even if label_name has NaNs)
            sub = (
                seg[[chain_col, elev1_col, elev2_col]]
                .dropna(subset=[chain_col, elev1_col, elev2_col])
                .sort_values(chain_col)
            )

            # get robust label from whole contiguous seg
            label_text = _first_non_empty(seg.get(label_name_col), name)

            # if too short to draw polygon, still reserve legend slot (but only if planned or dev > threshold)
            if len(sub) < 2:
                if name.startswith("planned"):
                    legend_items_planned.append((planned_idx, label_text))
                    planned_idx += 1
                else:
                    # cannot compute deviation for <2 pts; treat as non-highlighted (no hatch, no legend)
                    pass
                continue

            is_planned = name.startswith("planned")
            hatch_layer = layer_fill_planned if is_planned else layer_fill_unplanned
            rgb = _hex_to_rgb(planned_hex if is_planned else unplanned_hex)

            # patch boundary
            p_top = [
                (float(r[chain_col]) * scale_x + offset_x, float(r[elev1_col]) * scale_y + offset_y)
                for _, r in sub.iterrows()
            ]
            p_bot = [
                (float(r[chain_col]) * scale_x + offset_x, float(r[elev2_col]) * scale_y + offset_y)
                for _, r in sub.iloc[::-1].iterrows()
            ]
            boundary = p_top + p_bot

            # compute max deviation for this unplanned seg
            dev = float((sub[elev1_col].astype(float) - sub[elev2_col].astype(float)).abs().max())
            # always track overall max for the summary
            if not is_planned:
                max_unplanned_dev = max(max_unplanned_dev, dev)

            # For planned: always hatch + badge + legend
            if is_planned:
                hatch = msp.add_hatch(
                    dxfattribs={"layer": hatch_layer, "true_color": ezcolors.rgb2int(rgb)}
                )
                hatch.set_solid_fill()
                hatch.paths.add_polyline_path(boundary, is_closed=True)

                # mid-point (for badge)
                mid_i = len(sub) // 2
                ch_mid = float(sub.iloc[mid_i][chain_col])
                z1_mid = float(sub.iloc[mid_i][elev1_col])
                z2_mid = float(sub.iloc[mid_i][elev2_col])
                top_z = max(z1_mid, z2_mid)
                bot_z = min(z1_mid, z2_mid)

                pos = badge_position.lower()
                if pos == "auto":
                    pos = "top"
                if pos == "top":
                    z_label = top_z + badge_offset_factor * (text_height / max(1e-9, scale_y))
                else:
                    z_label = bot_z - badge_offset_factor * (text_height / max(1e-9, scale_y))

                bx = ch_mid * scale_x + offset_x
                by = z_label * scale_y + offset_y
                idx = planned_idx

                if add_zone_badges:
                    msp.add_circle((bx, by), badge_r,
                                   dxfattribs={"layer": label_layer, "true_color": ezcolors.rgb2int(rgb)})
                    tt = msp.add_text(
                        str(idx),
                        dxfattribs={
                            "layer": label_layer,
                            "height": text_height,
                            "true_color": ezcolors.rgb2int(rgb),
                            "halign": ezconst.CENTER,
                            "valign": ezconst.MIDDLE,
                        },
                    )
                    tt.dxf.insert = (bx, by)
                    tt.dxf.align_point = (bx, by)

                # legend capture
                legend_items_planned.append((idx, label_text))
                planned_idx += 1

            # For unplanned: only hatch/add badge/legend if dev > deviation_threshold
            else:
                if dev > deviation_threshold:
                    # draw hatch
                    hatch = msp.add_hatch(
                        dxfattribs={"layer": hatch_layer, "true_color": ezcolors.rgb2int(rgb)}
                    )
                    hatch.set_solid_fill()
                    hatch.paths.add_polyline_path(boundary, is_closed=True)

                    # mid-point (for badge)
                    mid_i = len(sub) // 2
                    ch_mid = float(sub.iloc[mid_i][chain_col])
                    z1_mid = float(sub.iloc[mid_i][elev1_col])
                    z2_mid = float(sub.iloc[mid_i][elev2_col])
                    top_z = max(z1_mid, z2_mid)
                    bot_z = min(z1_mid, z2_mid)

                    pos = badge_position.lower()
                    if pos == "auto":
                        pos = "bottom"
                    if pos == "top":
                        z_label = top_z + badge_offset_factor * (text_height / max(1e-9, scale_y))
                    else:
                        z_label = bot_z - badge_offset_factor * (text_height / max(1e-9, scale_y))

                    bx = ch_mid * scale_x + offset_x
                    by = z_label * scale_y + offset_y
                    idx = unplanned_idx

                    if add_zone_badges:
                        msp.add_circle((bx, by), badge_r,
                                       dxfattribs={"layer": label_layer, "true_color": ezcolors.rgb2int(rgb)})
                        tt = msp.add_text(
                            str(idx),
                            dxfattribs={
                                "layer": label_layer,
                                "height": text_height,
                                "true_color": ezcolors.rgb2int(rgb),
                                "halign": ezconst.CENTER,
                                "valign": ezconst.MIDDLE,
                            },
                        )
                        tt.dxf.insert = (bx, by)
                        tt.dxf.align_point = (bx, by)

                    # legend capture + deviation
                    legend_items_unplanned.append((idx, label_text, dev))
                    unplanned_idx += 1
                else:
                    # dev <= threshold -> do not hatch, do not badge, do not add to legend
                    pass

    # ---------- section end labels
    if section_name:
        try:
            left_tag, right_tag = [s.strip() for s in section_name.split("_", 1)]
        except ValueError:
            left_tag = section_name.strip()
            right_tag = ""

        d2 = df[[chain_col, elev1_col, elev2_col]].dropna().astype(float).sort_values(chain_col)
        if len(d2) >= 1:
            ch_min = float(d2[chain_col].min())
            ch_max = float(d2[chain_col].max())

            pad_ch = section_x_pad_factor * (text_height / max(1e-9, scale_x))

            i_left = (d2[chain_col] - ch_min).abs().idxmin()
            i_right = (d2[chain_col] - ch_max).abs().idxmin()
            z_left = max(float(d2.loc[i_left, elev1_col]), float(d2.loc[i_left, elev2_col]))
            z_right = max(float(d2.loc[i_right, elev1_col]), float(d2.loc[i_right, elev2_col]))

            x_l = (ch_min + pad_ch) * scale_x + offset_x
            x_r = (ch_max - pad_ch) * scale_x + offset_x
            y_l = z_left * scale_y + offset_y + section_offset_factor * text_height
            y_r = z_right * scale_y + offset_y + section_offset_factor * text_height

            if left_tag:
                tl = msp.add_text(left_tag, dxfattribs={"layer": section_label_layer, "height": text_height})
                tl.dxf.insert = (x_l, y_l)

            if right_tag:
                tr = msp.add_text(right_tag, dxfattribs={"layer": section_label_layer, "height": text_height})
                tr.dxf.insert = (x_r, y_r)

    # ---------- legends render
    if show_legends and (plot_x_min is not None):
        rgb_plan = _hex_to_rgb(planned_hex)
        rgb_unpl = _hex_to_rgb(unplanned_hex)
        rgb_itr1 = _hex_to_rgb(iter1_color_hex)
        rgb_itr2 = _hex_to_rgb(iter2_color_hex)

        pad = legend_pad_factor * text_height
        row_gap = legend_row_gap_factor * text_height
        bullet_r = legend_bullet_radius_factor * text_height
        col_gap = legend_col_gap_factor * text_height

        # LEFT list: planned (1️⃣ at top)
        xL = plot_x_min + pad
        total_rows = len(legend_items_planned)
        y_start = plot_y_min + pad + (total_rows - 1) * (text_height + row_gap)
        for idx, lbl in sorted(legend_items_planned, key=lambda t: t[0]):
            msp.add_circle((xL, y_start), bullet_r,
                        dxfattribs={"layer": legend_layer, "true_color": ezcolors.rgb2int(rgb_plan)})
            _add_text_centered(str(idx), xL, y_start, rgb=rgb_plan, layer=legend_layer)
            _add_text_left(str(lbl), xL + bullet_r + col_gap, y_start, layer=legend_layer)
            y_start -= text_height + row_gap

        # RIGHT list: unplanned (1️⃣ at top)
        xR = plot_x_max - pad
        total_rows = len(legend_items_unplanned)
        y_start = plot_y_min + pad + (total_rows - 1) * (text_height + row_gap)
        for idx, lbl, dev in sorted(legend_items_unplanned, key=lambda t: t[0]):
            msp.add_circle((xR, y_start), bullet_r,
                        dxfattribs={"layer": legend_layer, "true_color": ezcolors.rgb2int(rgb_unpl)})
            _add_text_centered(str(idx), xR, y_start, rgb=rgb_unpl, layer=legend_layer)
            lab = f"{lbl} ({dev:.2f} m)"
            t = msp.add_text(
                lab, dxfattribs={"layer": legend_layer, "height": text_height,
                                "halign": ezconst.RIGHT, "valign": ezconst.MIDDLE}
            )
            tx = xR - (bullet_r + col_gap)
            t.dxf.insert = (tx, y_start); t.dxf.align_point = (tx, y_start)
            y_start -= text_height + row_gap

        # BOTTOM center strip (✓, ✗, ITR lines, deviation)
        yB = plot_y_min - bottom_legend_gap_factor * text_height  # ↓ below instead of ↑ above
        cx = (plot_x_min + plot_x_max) / 2.0
        dx = bottom_legend_col_gap_factor * text_height

        # ✓ planned
        x = cx - 2 * dx
        msp.add_circle((x, yB), bullet_r,
                       dxfattribs={"layer": legend_layer, "true_color": ezcolors.rgb2int(rgb_plan)})
        _add_text_centered("✓", x, yB, rgb=rgb_plan, h=text_height*0.95, layer=legend_layer)
        _add_text_left("Planned Area", x + bullet_r + col_gap, yB, layer=legend_layer)

        # ✗ unplanned
        x = cx - dx
        msp.add_circle((x, yB), bullet_r,
                       dxfattribs={"layer": legend_layer, "true_color": ezcolors.rgb2int(rgb_unpl)})
        _add_text_centered("✗", x, yB, rgb=rgb_unpl, h=text_height*0.95, layer=legend_layer)
        _add_text_left("Unplanned Area", x + bullet_r + col_gap, yB, layer=legend_layer)

        # ITR 1 sample line
        x = cx + 0
        msp.add_line((x - 1.8*text_height, yB), (x + 1.8*text_height, yB),
                     dxfattribs={"layer": legend_layer, "true_color": ezcolors.rgb2int(rgb_itr1)})
        _add_text_left("ITR 2024", x + 2.0*text_height, yB, layer=legend_layer)

        # ITR 2 sample line
        x = cx + dx
        msp.add_line((x - 1.8*text_height, yB), (x + 1.8*text_height, yB),
                     dxfattribs={"layer": legend_layer, "true_color": ezcolors.rgb2int(rgb_itr2)})
        _add_text_left("ITR 2025", x + 2.0*text_height, yB, layer=legend_layer)

        # deviation summary
        x = cx + 2 * dx
        dev_lbl = (
            f"Deviation detected: {max_unplanned_dev:.2f} m"
            if legend_items_unplanned else "No deviation detected"
        )
        _add_text_left(dev_lbl, x, yB, layer=legend_layer)

    # ---------- save
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(out_p)
    return str(out_p.resolve())


# export_elevation_profiles_to_dxf_try_8(
#     out_csv,
#     r"D:/2_Analytics/6_plan_vs_actual/13_oct_output_1/sectional_deviation_analysis/K K'/section_data/elevation_profile_try_16.dxf",
#     chain_col="chainage",
#     elev1_col="z_itr1",
#     elev2_col="z_itr2",
#     line_name_col="line_name",
#     deviation_threshold=7,
# )
