# Solids & Geometry Import

Renderer3D-Rust can display **very large numbers of geometries fast** and
import them from many different worlds: **3D solid models** (STL/OBJ/glTF),
plain geometry strings (DSL), JSON documents, generic text / XYZ point
files. Tabular **CSV/Excel data** and the other ML formats go through the
Dataset window instead (see [ML_VISUALIZER.md](ML_VISUALIZER.md)).

Open the **🧊 Solids** window from the toolbar: import a 3D model file, paste
geometry data into the text box, or import a DSL/JSON/XYZ file. Pasted/file
geometry becomes a **layer** that can be toggled, focused (🎯 moves the camera
to its centroid) or removed; each new layer gets a distinct default color.
Imported 3D models become **scene objects** (auto-scaled and centered) listed
in the shared bottom object list, like manually created primitives.

> **Measure tool** — the toolbar's 📏 button toggles a measurement mode:
> click two surface points (on models or primitives) to read the straight-line
> distance, drawn as a labelled segment.

## Why it is fast

- Records are grouped per primitive shape into **instanced batches**: a
  million geometries are still at most three instanced draw calls
  (cube / sphere / plane).
- Imported 3D models are uploaded once as indexed meshes (32-bit indices)
  and drawn with one call per model; GPU buffers are cached by object id.
- GPU instance buffers are rebuilt **only when a layer changes** (dirty
  flag), never per frame.
- Sphere batches above 2 000 instances automatically switch to a
  **low-poly LOD mesh** (12×8 instead of 32×32 segments), keeping huge
  point clouds interactive.
- File parsing / model loading runs on a **background thread** — the UI
  never blocks.

## 1. 3D solid models (`.stl`, `.obj`, `.gltf`, `.glb`)

Type a model path into the Solids window's *Import a 3D model* field and
import it. The mesh is loaded on a worker thread, normals are synthesized
when a file omits them, and the model is added to the scene as a selected,
auto-scaled, origin-centered object that can be transformed, recolored,
measured and picked like any primitive.

- **STL** — binary/ASCII triangle soup (per-face normals).
- **OBJ** — Wavefront meshes (smooth normals computed if absent).
- **glTF / GLB** — positions, normals and indices from all primitives.
- **STEP** (`.step`/`.stp`) — recognised but **not yet tessellated**; it
  returns a clear "not supported yet" error.

## 2. Geometry strings (DSL)

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

## 3. CSV / Excel geometry tables

> Generic **tabular CSV/Excel _data_** import now lives in the **📊 Dataset**
> window. The schema below describes the *geometry-record* table format still
> understood by the geometry loader (one primitive per row); it is kept for
> programmatic/file imports and is no longer advertised in the Solids window UI.

Header-mapped: column order is free, names are case-insensitive. The same
mapping applies to the first sheet of an Excel workbook
(`.xlsx`/`.xlsm`/`.xls`/`.ods`, via `calamine`).

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

NPY / NPZ / CSV / Excel / Parquet / IDX datasets go through the **📊 Dataset**
window instead, which adds label color-mapping, configurable 1D/2D/3D
projection (PCA or chosen columns), filters, search and export — see
[ML_VISUALIZER.md](ML_VISUALIZER.md).

## Error reporting

Every parser reports the offending **line / record number** and what was
wrong (`line 2: unknown shape 'spherex'`), surfaced in the window's status
area. Unknown file extensions list the supported ones.

## Tests

- `tests/import_tests.rs` — real 3D models under `tests/import/` load
  through `MeshData::load` (STL/OBJ), with STEP and unknown-extension error
  paths.
- `tests/geometry_import/` — real files for every geometry-record format (a
  genuine `.xlsx` is written via `rust_xlsxwriter` and read back),
  auto-detection, error paths, batch grouping, and an ignored 500k-record
  benchmark.
- `tests/ui/geometry_panel_tests.rs` — headless egui coverage of the
  window: paste flow, worker-thread file import (success and failure),
  layer visibility → batch rebuild.
- In-module unit tests cover the DSL/XYZ/JSON/table parsers, the mesh loader
  (AABB, synthesized normals, ray-pick) and the instancing math.
