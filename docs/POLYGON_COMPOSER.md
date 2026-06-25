# Polygon Composer — Sketch 2D → Superfici → Solidi 3D (B-rep)

Roadmap di design per la creazione di superfici da profili poligonali (segmenti
retti o curvi, chiusi o aperti) e per il loro assemblaggio in solidi 3D tramite
lati in comune.

Questo documento è **design/architettura**, non implementazione: definisce i
moduli, le strutture dati, le fasi e le regole anti-duplicazione. È coerente con
lo stile di `docs/GEOMETRY_IMPORT.md`.

## Decisioni di progetto

- **Triangolazione**: ear clipping interno (nessuna dipendenza esterna; poligoni
  semplici concavi supportati subito, holes rimandati alla Fase C).
- **Curve**: archi di cerchio + Bézier cubiche, con un unico flattening adattivo.
- **Output unico**: ogni superficie/solido generato produce un
  `crate::mesh::MeshData` (indici u32) e viaggia nella pipeline esistente.

## Principi architetturali

1. **Logica pura separata da UI/GPU.** I nuovi moduli `sketch/` e `brep/` non
   dipendono da egui/wgpu (come `geometry/` e `dataset`). Testabili headless.
2. **Una sola rappresentazione di output:** `mesh::MeshData`. Nessun nuovo tipo
   mesh. (Nota: esiste già una duplicazione storica `primitives::MeshData` u16
   vs `mesh::MeshData` u32 — non aggiungerne una terza.)
3. **Un solo flattening curve→polilinea**, riusato dai segmenti retti (caso
   degenere a 2 punti) e da archi/Bézier.
4. **Un solo welding per posizione** (merge vertici entro ε): usato sia dalla
   triangolazione sia dal compositore. Definito una volta in `brep::weld`.
5. **Modello dati incrementale**: il solido è un B-rep leggero (vertici / lati /
   facce). Le superfici della Fase A sono già "facce", agganciate in Fase B
   senza riconversioni.
6. **Nessuna regressione di selezione/undo**: tutto passa dallo stesso percorso
   di `State::add_mesh_object`, quindi ray-pick e undo funzionano gratis.

## Punti di estensione del codebase (riuso, no-dup)

| Punto | Uso | Riuso |
|---|---|---|
| `model::Vertex` | vertice universale pos/color/normal | pipeline unica |
| `mesh::MeshData` | output di sketch e solidi | `build`, `smooth_normals`, `aabb`, `ray_hit` |
| `State::add_mesh_object` | da rifattorizzare in `insert_custom_mesh()` | upload GPU, undo, selezione, camera |
| `GeometryType::Line` | polilinee aperte | path linea esistente |
| `GeometryType::Mesh` + `CustomMesh` | superfici/solidi generati | selezione + ray-pick |
| `ui/geometry_panel.rs` | template per i nuovi pannelli | stile, status, i18n `t!` |

### Refactor abilitante (Fase A5)

Estrarre da `State::add_mesh_object` un helper privato:

```rust
fn insert_custom_mesh(&mut self, mesh: mesh::MeshData, label: String) -> usize
```

che esegue upload buffer GPU, creazione `CustomMesh`, inserimento `SceneObject`
(`GeometryType::Mesh`), selezione e `UndoCommand::Add`. Sia l'import file sia la
generazione sketch/solidi lo riusano: nessuna duplicazione del codice GPU/scene.

## Layout dei moduli

```
src/
  sketch/                 # FASE A — profili 2D (logica pura)
    mod.rs                # Sketch, Plane, tipi pubblici
    segment.rs           # Segment: Line | Arc | Bezier  → flatten()
    profile.rs           # Profile: Vec<Segment>, aperto/chiuso, validazione
    tessellate.rs        # loop chiuso → triangoli (ear clipping)
  brep/                   # FASE B — B-rep + compositore (logica pura)
    mod.rs
    topology.rs          # Vertex/Edge/Face
    weld.rs              # merge per posizione (ε) — UNICA implementazione
    solid.rs             # assembla facce, rileva lati comuni
    validate.rs          # manifold, orientazione normali coerente
    to_mesh.rs           # Solid/Face → mesh::MeshData (riusa mesh::build)
```

UI: `src/ui/sketch_panel.rs` e poi `src/ui/composer_panel.rs`, sul modello di
`geometry_panel.rs`. Nessuna logica geometrica nei pannelli.

## Modello dati

### Fase A — Sketch

```rust
enum Segment {                 // segment.rs
    Line   { a: Vec2, b: Vec2 },
    Arc    { center: Vec2, radius: f32, start: f32, end: f32 },
    Bezier { p0: Vec2, c1: Vec2, c2: Vec2, p1: Vec2 },
}
impl Segment {
    /// UNICO entry-point curve→polilinea. Retta = 2 punti; arco/Bézier =
    /// suddivisione adattiva sulla tolleranza di sagitta.
    fn flatten(&self, tol: f32) -> Vec<Vec2>;
}

struct Profile { segments: Vec<Segment>, closed: bool }
struct Sketch  { plane: Plane, profile: Profile }   // plane = origine + normale + basi
```

- **Profilo chiuso** → superficie piena (triangolata).
- **Polilinea aperta** → resa come `GeometryType::Line` (strip), nessuna fill.

### Fase B — B-rep

```rust
struct VertexId(u32); struct EdgeId(u32); struct FaceId(u32);
struct Edge { v0: VertexId, v1: VertexId }          // non orientato per il match
struct Face { loop_edges: Vec<EdgeId>, normal: Vec3, source: Sketch }
struct Solid {
    verts: Vec<Vec3>,
    edges: Vec<Edge>,                                // dedup: una entry per lato fisico
    faces: Vec<Face>,
    edge_use: HashMap<EdgeId, SmallVec<[FaceId; 2]>>,// 2 facce ⇒ lato interno (manifold)
}
```

Il "lato in comune" è un `EdgeId` referenziato da ≥2 facce dopo il `weld`.

## Fasi e milestone

### Fase A — Superfici da profilo

- **A1** `sketch::segment` — `Segment` + `flatten(tol)` adattivo.
  Test: retta→2 punti, monotonia, cerchio chiuso, tolleranza rispettata.
- **A2** `sketch::profile` — concatenazione, continuità seg *i* → seg *i+1*,
  chiusura, area con segno (orientazione). Errori in stile `GeometryError`.
- **A3** `sketch::tessellate` — ear clipping su poligono semplice.
  Test: area triangolata ≈ area poligono, indici in range, winding CCW.
- **A4** `sketch → mesh` — proiezione punti 2D nel piano 3D + `mesh::build()`
  per le normali. Riuso totale di `build`/`smooth_normals`.
- **A5** Integrazione State+UI — refactor `insert_custom_mesh()` +
  `State::add_sketch_surface(Sketch)` + `sketch_panel` minimale (aggiungi
  segmenti, chiudi profilo, "Crea superficie").

### Fase B — Compositore di solidi

- **B1** `brep::weld` — merge vertici per posizione (ε) con griglia spaziale.
  Unica implementazione, usata anche da `to_mesh`.
- **B2** `brep::topology` + `solid` — inserimento facce, lati deduplicati,
  `edge_use`; rilevamento lati comuni (`edge_use.len() == 2`).
- **B3** `brep::validate` — manifold check (ogni lato 1–2 facce), orientazione
  normali coerente (propagazione via lati condivisi, flip dove discordi).
- **B4** `brep::to_mesh` — solido → singolo `mesh::MeshData` (vertici welded;
  normali smussate o per-faccia secondo flag). Ray-pick gratis via `CustomMesh`.
- **B5** UI compositore — `composer_panel`: superfici disponibili, aggancio per
  lato comune, anteprima, "Genera solido".

### Fase C — Rifiniture (post-MVP)

- Holes nei profili (loop interni) via bridge nell'ear clipping.
- Estrusione/loft di un profilo (caso particolare del compositore).
- Pick dedicato di vertici/lati (oggi il pick è triangolo→oggetto).
- Persistenza di sketch e solidi nel formato progetto.

## Strategia di test

- Unit test per ogni modulo puro (come `primitives`/`mesh`/`geometry`):
  proprietà geometriche (aree, chiusura, manifold, indici in range, normali
  unitarie), non solo smoke.
- Integrazione in `tests/` (`sketch_tests.rs`, `brep_tests.rs`): es. cubo da 6
  quad → dopo weld 8 vertici, 12 lati, ogni lato 2 facce, mesh chiusa.
- UI test minimale sul modello di `tests/ui/geometry_panel_tests.rs`.
