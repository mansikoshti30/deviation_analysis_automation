import ezdxf
import subprocess
import ezdxf
import os
import glob
from collections import defaultdict, Counter
import geopandas as gpd
from shapely.geometry import LineString, Point
from scipy.spatial import cKDTree

## convert dwg to dxf
def ODA_convertor(input_folder, output_folder, oda_exe):
    os.makedirs(output_folder, exist_ok=True)

    cmd = [
        oda_exe,
        input_folder,
        output_folder,
        "ACAD2013",  # safer than 2010
        "DXF",
        "0",
        "*.dwg"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    dxf_files = glob.glob(os.path.join(output_folder, "*.dxf"))

    if not dxf_files:  # if empty list
        return "process not completed"

    # Return the first DXF file path
    return dxf_files[0]


## set section line dxf file  
def get_layers_from_dxf(input_dxf, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    out_file = os.path.join(output_folder, "section_lines.dxf")

    # === RULES ===
    # Only these types are allowed in the target "section lines" layer:
    ALLOWED_TYPES = {"LINE", "LWPOLYLINE", "TEXT", "MTEXT"}
    # Explicitly disallow common polygonal/fill types:
    DISALLOWED_TYPES = {"HATCH", "POLYLINE"}  # add more if needed (e.g., "SPLINE", "CIRCLE", ...)

    def is_lwpolyline_closed(lp) -> bool:
        """Return True if an LWPOLYLINE is closed."""
        closed_attr = getattr(lp, "closed", None)
        if isinstance(closed_attr, bool):
            return closed_attr
        # Fallback: check DXF flags bit 1
        try:
            flags = int(lp.dxf.flags)
            return bool(flags & 1)
        except Exception:
            # If we can't determine, be conservative and treat as closed
            return True

    def layer_matches_section_criteria(entities):
        """Check that a layer contains ONLY allowed types, with at least one text and one line-ish entity,
        and that NO LWPOLYLINE is closed (i.e., no polygons)."""
        if not entities:
            return False

        type_counts = Counter(e.dxftype() for e in entities)

        # Hard disallow first
        if any(t in DISALLOWED_TYPES for t in type_counts):
            return False

        # Must be subset of allowed types
        if not set(type_counts).issubset(ALLOWED_TYPES):
            return False

        # Must have some text (section letters)
        text_count = type_counts.get("TEXT", 0) + type_counts.get("MTEXT", 0)
        if text_count == 0:
            return False

        # Must have some lines (LINE or LWPOLYLINE)
        lineish_count = type_counts.get("LINE", 0) + type_counts.get("LWPOLYLINE", 0)
        if lineish_count == 0:
            return False

        # Ensure no LWPOLYLINE is closed (to avoid polygons)
        for e in entities:
            if e.dxftype() == "LWPOLYLINE" and is_lwpolyline_closed(e):
                return False

        return True

    def save_entities_as_dxf(entities, filename):
        """Save given entities to a fresh DXF."""
        new_doc = ezdxf.new(setup=True)
        new_msp = new_doc.modelspace()
        for e in entities:
            try:
                new_msp.add_foreign_entity(e)
            except Exception as ex:
                print(f"  Skipped {e.dxftype()}: {ex}")
        new_doc.saveas(filename)

    # === LOAD & EVALUATE ===
    try:
        doc = ezdxf.readfile(input_dxf)
    except Exception as e:
        print(f" Failed to read DXF: {e}")
        return "process not completed"

    msp = doc.modelspace()

    # Group entities by layer
    layer_entities = defaultdict(list)
    for ent in msp:
        layer_entities[ent.dxf.layer].append(ent)

    # Find candidate layers that satisfy criteria
    candidates = []
    for layer, ents in layer_entities.items():
        if layer_matches_section_criteria(ents):
            # Score by text count (prefer layer with most section letters)
            tcount = sum(1 for e in ents if e.dxftype() in ("TEXT", "MTEXT"))
            total = len(ents)
            candidates.append((layer, tcount, total, ents))

    if not candidates:
        print(" No layer matched the 'section lines + letters' criteria.")
        return "no section lines found"
    else:
        # Pick the best candidate: highest text count, then most entities as tiebreaker
        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
        best_layer, best_texts, best_total, best_entities = candidates[0]

        # Save as section_lines.dxf (overwrite intentionally)
        save_entities_as_dxf(best_entities, out_file)

        # Console report
        print(f" Selected layer: '{best_layer}'")
        print(f"   - Text count: {best_texts}")
        print(f"   - Entity total: {best_total}")
        # Type breakdown for visibility
        type_counts = Counter(e.dxftype() for e in best_entities)
        for t, c in sorted(type_counts.items()):
            print(f"   - {t}: {c}")
        print(f" Saved as: {out_file}")

        return out_file




import os
from collections import Counter
import ezdxf

def clean_section_lines_dxf(section_lines_dxf, output_folder, prec=6):
    """
    Deduplicate LINE, LWPOLYLINE, TEXT, and MTEXT entities in a DXF
    and save a cleaned version.
    
    Args:
        section_lines_dxf (str): Input DXF path
        output_folder (str): Folder to save cleaned DXF
        prec (int): Rounding precision for geometry comparison (default=6)
    
    Returns:
        str: Path of the cleaned DXF file
    """
    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, "section_lines_clean.dxf")

    # ---------- Helpers ----------
    def qr(val, p=prec):
        return round(float(val), p)

    def qpt3(v, p=prec):
        """Convert a point or tuple into rounded (x,y,z)."""
        if hasattr(v, "x"):
            return (qr(v.x, p), qr(v.y, p), qr(getattr(v, "z", 0.0), p))
        if len(v) == 2:
            return (qr(v[0], p), qr(v[1], p), 0.0)
        return (qr(v[0], p), qr(v[1], p), qr(v[2], p))

    # ---------- Read source DXF ----------
    doc_in = ezdxf.readfile(section_lines_dxf)
    msp_in = doc_in.modelspace()

    # ---------- Deduplicate ----------
    line_set = set()
    text_geo_to_entity = {}
    mtext_geo_to_entity = {}

    # LINE entities
    for line in msp_in.query("LINE"):
        start = qpt3(line.dxf.start)
        end = qpt3(line.dxf.end)
        key = tuple(sorted([start, end]))
        line_set.add(key)

    # LWPOLYLINE entities (as segments)
    for pl in msp_in.query("LWPOLYLINE"):
        pts = list(pl.get_points("xy"))
        if len(pts) >= 2:
            for i in range(len(pts) - 1):
                a = (qr(pts[i][0]), qr(pts[i][1]), 0.0)
                b = (qr(pts[i + 1][0]), qr(pts[i + 1][1]), 0.0)
                key = tuple(sorted([a, b]))
                line_set.add(key)
            if pl.closed and len(pts) > 2:
                a = (qr(pts[-1][0]), qr(pts[-1][1]), 0.0)
                b = (qr(pts[0][0]), qr(pts[0][1]), 0.0)
                key = tuple(sorted([a, b]))
                line_set.add(key)

    # TEXT entities
    for tx in msp_in.query("TEXT"):
        ins = qpt3(tx.dxf.insert)
        h = qr(tx.dxf.height) if tx.dxf.hasattr("height") else 0.0
        rot = qr(tx.dxf.rotation) if tx.dxf.hasattr("rotation") else 0.0
        key = ("TEXT", ins, h, rot)
        if key not in text_geo_to_entity:
            text_geo_to_entity[key] = tx

    # MTEXT entities
    for mt in msp_in.query("MTEXT"):
        ins = qpt3(mt.dxf.insert)
        h = qr(mt.dxf.char_height) if mt.dxf.hasattr("char_height") else 0.0
        w = qr(mt.dxf.width) if mt.dxf.hasattr("width") else 0.0
        rot = qr(mt.dxf.rotation) if mt.dxf.hasattr("rotation") else 0.0
        key = ("MTEXT", ins, h, w, rot)
        if key not in mtext_geo_to_entity:
            mtext_geo_to_entity[key] = mt

    # ---------- Write new DXF ----------
    doc_out = ezdxf.new(setup=True)
    msp_out = doc_out.modelspace()

    # Lines
    for start, end in line_set:
        msp_out.add_line(start, end)

    # TEXT
    for key, tx in text_geo_to_entity.items():
        _, ins, h, rot = key
        new_tx = msp_out.add_text(tx.dxf.text or "", dxfattribs={
            "height": h,
            "rotation": rot,
            "layer": tx.dxf.layer if tx.dxf.hasattr("layer") else "0",
        })
        new_tx.set_pos(ins)

    # MTEXT
    for key, mt in mtext_geo_to_entity.items():
        _, ins, h, w, rot = key
        new_mt = msp_out.add_mtext(mt.text or "", dxfattribs={
            "char_height": h,
            "width": w,
            "rotation": rot,
            "layer": mt.dxf.layer if mt.dxf.hasattr("layer") else "0",
        })
        new_mt.set_location(ins)

    doc_out.saveas(out_path)
    return out_path


def section_lines_to_shp(input_dxf, work_dir, output_name="LINES_with_start_end_text.shp", crs="EPSG:32644"):
    """
    Split a cleaned DXF into line and text DXFs, convert to GeoDataFrames,
    attach nearest text to each line start/end, and save a final shapefile.

    Args:
        input_dxf (str): Path to cleaned input DXF (section lines).
        work_dir (str): Folder where intermediate files and final shapefile will be written.
        output_name (str): Filename for the final shapefile (default: 'LINES_with_start_end_text.shp').
        crs (str): Coordinate reference system for output GeoDataFrames (default: 'EPSG:32644').

    Returns:
        str: Full path to the saved shapefile.

    Raises:
        FileNotFoundError: when input_dxf does not exist.
        ValueError: when no lines or no texts are found in the DXF.
    """
    os.makedirs(work_dir, exist_ok=True)

    lines_dxf = os.path.join(work_dir, "LINES.dxf")
    texts_dxf = os.path.join(work_dir, "TEXTS.dxf")
    output_shp = os.path.join(work_dir, output_name)

    # ---------- Step 1: Split DXF into lines and texts ----------
    if not os.path.exists(input_dxf):
        raise FileNotFoundError(f"Input DXF not found: {input_dxf}")

    doc = ezdxf.readfile(input_dxf)
    msp = doc.modelspace()

    line_entities, text_entities = [], []
    for e in msp:
        etype = e.dxftype()
        if etype in ["LINE", "LWPOLYLINE"]:
            line_entities.append(e)
        elif etype in ["TEXT", "MTEXT"]:
            text_entities.append(e)

    if not line_entities:
        raise ValueError("No line entities found in input DXF.")
    if not text_entities:
        raise ValueError("No text entities found in input DXF.")

    # Save lines DXF
    doc_lines = ezdxf.new(setup=True)
    msp_lines = doc_lines.modelspace()
    for e in line_entities:
        try:
            msp_lines.add_foreign_entity(e)
        except Exception as ex:
            # non-fatal: skip entity but notify
            print(f"Skipped line entity: {ex}")
    doc_lines.saveas(lines_dxf)

    # Save texts DXF
    doc_texts = ezdxf.new(setup=True)
    msp_texts = doc_texts.modelspace()
    for e in text_entities:
        try:
            msp_texts.add_foreign_entity(e)
        except Exception as ex:
            print(f"Skipped text entity: {ex}")
    doc_texts.saveas(texts_dxf)

    # ---------- Step 2: Convert Lines DXF to GeoDataFrame ----------
    doc_lines_r = ezdxf.readfile(lines_dxf)
    msp_lines_r = doc_lines_r.modelspace()

    lines = []
    for line in msp_lines_r.query("LINE"):
        start = (float(line.dxf.start.x), float(line.dxf.start.y))
        end = (float(line.dxf.end.x), float(line.dxf.end.y))
        lines.append({"geometry": LineString([start, end]), "start": start, "end": end})

    for pline in msp_lines_r.query("LWPOLYLINE"):
        pts = list(pline.get_points("xy"))
        if len(pts) > 1:
            for i in range(len(pts) - 1):
                start, end = (float(pts[i][0]), float(pts[i][1])), (float(pts[i+1][0]), float(pts[i+1][1]))
                lines.append({"geometry": LineString([start, end]), "start": start, "end": end})
            if pline.closed:
                start, end = (float(pts[-1][0]), float(pts[-1][1])), (float(pts[0][0]), float(pts[0][1]))
                lines.append({"geometry": LineString([start, end]), "start": start, "end": end})

    if not lines:
        raise ValueError("No line geometries extracted from lines DXF.")

    gdf_lines = gpd.GeoDataFrame(lines, crs=crs)
    # drop exact geometry duplicates
    gdf_lines = gdf_lines.drop_duplicates(subset=["geometry"]).reset_index(drop=True)

    # ---------- Step 3: Convert Texts DXF to GeoDataFrame ----------
    doc_texts_r = ezdxf.readfile(texts_dxf)
    msp_texts_r = doc_texts_r.modelspace()

    texts = []
    for t in msp_texts_r.query("TEXT MTEXT"):
        # TEXT: .dxf.text, MTEXT: .text
        content = None
        if t.dxftype() == "TEXT":
            content = getattr(t.dxf, "text", "") or ""
        else:  # MTEXT
            content = getattr(t, "text", "") or ""
        content = content.strip()
        insert_x = float(t.dxf.insert.x)
        insert_y = float(t.dxf.insert.y)
        texts.append({"geometry": Point((insert_x, insert_y)), "text": content})

    if not texts:
        raise ValueError("No text geometries extracted from texts DXF.")

    gdf_texts = gpd.GeoDataFrame(texts, crs=crs)
    gdf_texts = gdf_texts.drop_duplicates(subset=["geometry", "text"]).reset_index(drop=True)

    # ---------- Step 4: Nearest text for start & end ----------
    # Build KD-tree of text coordinates
    text_coords = [(p.x, p.y) for p in gdf_texts.geometry]
    tree = cKDTree(text_coords)

    def nearest_text_for_point(pt):
        dist, idx = tree.query([pt.x, pt.y], k=1)
        return gdf_texts.iloc[int(idx)]["text"]

    start_labels, end_labels = [], []
    for _, row in gdf_lines.iterrows():
        start_point = Point(row["start"])
        end_point = Point(row["end"])
        start_labels.append(nearest_text_for_point(start_point))
        end_labels.append(nearest_text_for_point(end_point))

    gdf_lines["start_text"] = start_labels
    gdf_lines["end_text"] = end_labels

    # ---------- Step 5: Save Shapefile ----------
    # Ensure .shp driver writes strings properly by converting to wider dtypes if necessary
    # geopandas will usually handle this; if the text fields are long, consider .gpkg instead.
    gdf_lines.to_file(output_shp)
    print(f"Final line shapefile with start/end texts saved: {output_shp}")

    return output_shp


## get section lines from dwg to shapefile 
def get_section_lines(input_folder, section_line_folder, oda_exe, crs):
    dxf_file  = ODA_convertor(input_folder, section_line_folder, oda_exe)
    section_lines_dxf = get_layers_from_dxf(dxf_file, section_line_folder)
    clean_section_lines_path = clean_section_lines_dxf(section_lines_dxf, section_line_folder, prec=6)
    section_lines_shp = section_lines_to_shp(clean_section_lines_path, section_line_folder, output_name="LINES_with_start_end_text.shp", crs=crs)
  
    return section_lines_shp