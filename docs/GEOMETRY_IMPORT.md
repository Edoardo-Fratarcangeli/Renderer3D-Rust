# Universal Geometry Import

Renderer3D-Rust is a complete 3D renderer that can display **very large
numbers of geometries fast**, importing them from many different worlds:
plain geometry strings, CSV tables, Excel workbooks, JSON documents,
generic text / XYZ point files, and ML data blocks (see
[ML_VISUALIZER.md](ML_VISUALIZER.md) for the latter).

Open the **📦 Geometry** window from the toolbar: paste data directly into
the text box or import a file. Every import becomes a **layer** that can be
toggled, focused (🎯 moves the camera to its centroid) or removed. Each new
layer gets a distinct default color from the palette; records can override
it individually.

## Why it is fast

- Records are grouped per primitive shape into **instanced batches**: a
  million geometries are still at most three instanced draw calls
  (cube / sphere / plane).
- GPU instance buffers are rebuilt **only when a layer changes** (dirty
  flag), never per frame.
- Sphere batches above 2 000 instances automatically switch to a
  **low-poly LOD mesh** (12×8 instead of 32×32 segments), keeping huge
  point clouds interactive.
- File parsing runs on a **background thread** — the UI never blocks.

## 1. Geometry strings (DSL)

One record per line (or `;`-separated). Comments start with `#` or `//`.

```text
<shape> <x> <y> <z> [size | sx sy sz] [options...]
```

| Element | Meaning |
|---------|---------|
| `shape` | `cube`/`box`, `sphere`/`ball`, `plane`/`quad`, `point`/`dot`/`vertex` |
| bare numbers after position | one value = uniform scale, three = per-axis |
| `#rrggbb` / `#rgb` / `r,g,b` | color (components 0–1 or 0–255) |
| `rot=rx,ry,rz` | Euler rotation in degrees |
| `size=v` / `radius=v` / `scale=sx,sy,sz` | explicit scale |
| `color=...` / `label=...` / `name=...` | keyed options |
| other bare words | record label |

Example:

```text
# a small scene
cube   0 0 0  2        #ff8800  base
sphere 0 0 2  0.5      color=0,1,0 label=marker
plane  0 0 -1 4 4 1    rot=0,0,45  floor
point  1 1 1; ball 5 5 5 radius=2
```

## 2. CSV tables (`.csv`)

Header-mapped: column order is free, names are case-insensitive.

| Purpose | Accepted headers |
|---------|------------------|
| shape | `shape`, `type`, `geometry`, `geom`, `kind` — *optional*: rows without it become points |
| position | `x`/`y`/`z`, `px`/`py`/`pz`, `pos_x`/`pos_y`/`pos_z` |
| uniform scale | `size`, `radius`, `scale` |
| per-axis scale | `sx`/`sy`/`sz`, `scale_x`/`scale_y`/`scale_z` |
| rotation (degrees) | `rx`/`ry`/`rz`, `rot_x`/`rot_y`/`rot_z` |
| color | `color` (hex or `"r,g,b"`) **or** `r`/`g`/`b` (`red`/`green`/`blue`) columns |
| label | `label`, `name`, `id`, `tag` |

```csv
shape,x,y,z,size,color,name
cube,0,0,0,2,#ff0000,base
sphere,1,2,3,0.5,"0,255,0",ball
```

## 3. Excel workbooks (`.xlsx`, `.xlsm`, `.xls`, `.ods`)

The **first sheet** is read with the same header mapping as CSV — build
your geometry table in Excel/LibreOffice and import it directly. Numeric
cells are handled natively (no string formatting needed).

## 4. JSON documents (`.json`)

A top-level array, or an object with a `geometries` / `objects` / `shapes`
/ `items` array. Field spellings are tolerant:

```json
{
  "geometries": [
    { "shape": "cube", "pos": [1, 2, 3], "size": 2,
      "rotation": [0, 45, 0], "color": "#ff8800", "label": "box" },
    { "type": "plane", "x": 0, "y": 0, "z": -1,
      "scale": [4, 4, 1], "color": [255, 128, 0], "name": "floor" },
    { "x": 9, "y": 9, "z": 9 }
  ]
}
```

Entries without a shape become points; colors accept `"#hex"` or `[r,g,b]`
(0–1 or 0–255).

## 5. Text & point files (`.txt`, `.geo`, `.dsl`, `.xyz`)

- `.xyz` — `x y z [size]` per line (whitespace or commas), rendered as
  point spheres.
- `.txt` / `.geo` / `.dsl` — **auto-detected**: lines starting with numbers
  are treated as XYZ points, otherwise the DSL grammar applies.

Pasted text follows the same auto-detection, with JSON recognized by a
leading `[` or `{`.

## 6. ML data blocks

NPY / NPZ / CSV / Parquet / IDX datasets go through the **📊 Dataset**
window instead, which adds label color-mapping, PCA projection, filters,
search and export — see [ML_VISUALIZER.md](ML_VISUALIZER.md).

## Error reporting

Every parser reports the offending **line / record number** and what was
wrong (`line 2: unknown shape 'spherex'`), surfaced in the window's status
area. Unknown file extensions list the supported ones.

## Tests

- `tests/geometry_import/` — real files for every format (a genuine
  `.xlsx` is written via `rust_xlsxwriter` and read back), auto-detection,
  error paths, batch grouping, and an ignored 500k-record benchmark.
- `tests/ui/geometry_panel_tests.rs` — headless egui coverage of the
  window: paste flow, worker-thread file import (success and failure),
  layer visibility → batch rebuild.
- In-module unit tests cover the DSL/XYZ/JSON/table parsers and the
  instancing math.
