import os
import re
import traceback
from typing import Callable, List, Dict, Optional




def process_all_sections(
    output_dir: str,
    section_line_path: str,
    dtm_itr1_path: str,
    dtm_itr2_path: str,
    done_exv: Optional[str] = None,
    unplanned_exv: Optional[str] = None,
    done_dump: Optional[str] = None,
    unplanned_dump: Optional[str] = None,
    threshold: float = 0.0,
    interval: float = 0.1,
    itr_1: str = "ITR1",
    itr_2: str = "ITR2",
    # dependency-injection of custom helpers
    sample_func: Callable = None,
    intersect_func: Callable = None,
    add_planned_func: Callable = None,
    add_unplanned_func: Callable = None,
    plot_func: Callable = None,
    # folder names
    analysis_subfolder: str = "sectional_deviation_analysis",
    intersecting_lines_sub: str = "intersecting_lines",
    elevation_profile_sub: str = "elevation_profile",
    section_data_sub: str = "section_data",
    dxf_output_sub: str = "dxf_output",  # 🆕 DXF folder
    sanitize_keep_spaces_apostrophe: Optional[Callable[[str], str]] = None,
    export_dxf_func: Callable = None,  # 🆕 expects export_elevation_profiles_to_dxf_try_8
    deviation_threshold: float = 0.0,  # 🆕 for DXF coloring
) -> List[Dict]:
    """
    Process all sections and save outputs (CSV, PNG, DXF) per section.

    Adds support for DXF export via `export_dxf_func`.
    """

    # fallback sanitizer
    def _default_sanitize(s: str) -> str:
        if s is None:
            return "section"
        s = str(s)
        s = s.replace("_", " ").strip()
        s = re.sub(r'[<>:"/\\|?*\n\r\t]+', "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s or "section"

    sanitize = sanitize_keep_spaces_apostrophe or _default_sanitize

    # validate injected funcs
    if sample_func is None or intersect_func is None or add_planned_func is None \
       or add_unplanned_func is None or plot_func is None:
        raise ValueError("You must provide sample_func, intersect_func, add_planned_func, add_unplanned_func, and plot_func.")

    if export_dxf_func is None:
        raise ValueError("You must also provide export_dxf_func (e.g. export_elevation_profiles_to_dxf_try_8).")

    # create main analysis folder
    output_folder_sectional_analysis = os.path.join(output_dir, analysis_subfolder)
    os.makedirs(output_folder_sectional_analysis, exist_ok=True)

    # read all section lines
    import geopandas as gpd
    section_gdf = gpd.read_file(section_line_path)
    n_sections = len(section_gdf)
    print(f"Found {n_sections} sections in {section_line_path}")

    results: List[Dict] = []

    for idx in range(n_sections):
        print(f"\n--- Processing section index {idx} ---")
        try:
            row = section_gdf.iloc[idx]
            section_name_from_row = f"{row.get('start_text', '')}_{row.get('end_text', '')}"
            readable_section_name = sanitize(section_name_from_row)
            print(f"Section: {section_name_from_row}  → Folder: {readable_section_name}")

            # Create per-section folders
            base_section_folder = os.path.join(output_folder_sectional_analysis, readable_section_name)
            intersecting_lines_folder = os.path.join(base_section_folder, intersecting_lines_sub)
            elevation_profile_folder = os.path.join(base_section_folder, elevation_profile_sub)
            section_data_folder = os.path.join(base_section_folder, section_data_sub)
            dxf_output_folder = os.path.join(base_section_folder, dxf_output_sub)  # 🆕 DXF folder

            os.makedirs(intersecting_lines_folder, exist_ok=True)
            os.makedirs(elevation_profile_folder, exist_ok=True)
            os.makedirs(section_data_folder, exist_ok=True)
            os.makedirs(dxf_output_folder, exist_ok=True)

            # Sample elevations
            df = sample_func(
                dtm_path1=dtm_itr1_path,
                dtm_path2=dtm_itr2_path,
                section_gdf=section_gdf,
                section_number=idx,
                interval=interval
            )

            # Intersect polygons
            intersect_res = intersect_func(
                line_gdf=section_gdf.iloc[[idx]],
                output_folder_path=intersecting_lines_folder,
                planned_and_done_excavation_path=done_exv,
                unplanned_and_done_excavation_path=unplanned_exv,
                planned_and_used_dump_path=done_dump,
                unplanned_and_used_dump_path=unplanned_dump
            )

            lines_planned_and_done_excavation = lines_unplanned_and_done_excavation = None
            lines_planned_and_used_dump = lines_unplanned_and_used_dump = None
            returned_section_name = None

            if isinstance(intersect_res, tuple):
                if len(intersect_res) == 2:
                    lines_tuple, returned_section_name = intersect_res
                elif len(intersect_res) == 3:
                    lines_tuple, returned_section_name, _ = intersect_res
                else:
                    lines_tuple = intersect_res[0]
                try:
                    (lines_planned_and_done_excavation,
                     lines_unplanned_and_done_excavation,
                     lines_planned_and_used_dump,
                     lines_unplanned_and_used_dump) = lines_tuple
                except Exception:
                    print("⚠️ Couldn't unpack lines tuple from intersect result — continuing with None line_gdfs.")
            else:
                print(f"⚠️ Unexpected type from intersect_func: {type(intersect_res)}")

            section_name_for_title = returned_section_name or section_name_from_row

            # Save CSV path
            csv_path = os.path.join(section_data_folder, "section_data.csv")

            # Planned & unplanned labeling
            df_1 = add_planned_func(df, lines_planned_and_done_excavation, "planned_and_done_excavation", csv_path)
            df_2 = add_planned_func(df_1, lines_planned_and_used_dump, "planned_and_used_dump", csv_path)
            df_3 = add_unplanned_func(df_2, lines_unplanned_and_done_excavation, "unplanned_and_done_excavation", threshold, csv_path)
            df_4 = add_unplanned_func(df_3, lines_unplanned_and_used_dump, "unplanned_and_used_dump", threshold, csv_path)

            # Plot PNG
            png_filename = f"elevation_profile {sanitize(section_name_for_title)}.png"
            save_file_path = os.path.join(elevation_profile_folder, re.sub(r'[<>:"/\\|?*\n\r\t]+', "", png_filename).strip())

            plot_func(df_4, itr1=itr_1, itr2=itr_2, section_name=section_name_for_title, save_path=save_file_path, show=False)

            # --- 🆕 DXF Export ---
            dxf_filename = f"elevation_profile_{sanitize(section_name_for_title)}.dxf"
            dxf_path = os.path.join(dxf_output_folder, dxf_filename)

            export_dxf_func(
                csv_path,  # pass CSV (since your DXF func reads CSV)
                out_path=dxf_path,
                chain_col="chainage",
                elev1_col="z_itr1",
                elev2_col="z_itr2",
                line_name_col="line_name",
                section_name=section_name_for_title,
                deviation_threshold=deviation_threshold,  # highlight only above threshold
            )

            print(f"✅ Saved for section '{readable_section_name}':")
            print(" - Shapefiles ->", intersecting_lines_folder)
            print(" - CSV ->", csv_path)
            print(" - PNG ->", save_file_path)
            print(" - DXF ->", dxf_path)

            results.append({
                "section_index": idx,
                "section_name": readable_section_name,
                "csv_path": csv_path,
                "png_path": save_file_path,
                "dxf_path": dxf_path,
                "success": True,
            })

        except Exception as e:
            print(f"❌ Error processing section {idx}: {e}")
            traceback.print_exc()
            results.append({
                "section_index": idx,
                "section_name": readable_section_name if 'readable_section_name' in locals() else f"section_{idx}",
                "success": False,
                "error": str(e)
            })

    return results