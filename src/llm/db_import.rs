//! Import of relational databases into the 3D network visualizer.
//!
//! Sources:
//! - **SQL dumps** (`.sql`) — MySQL / MariaDB, PostgreSQL (including
//!   `COPY … FROM stdin` data blocks), SQLite and standard-SQL dumps:
//!   `CREATE TABLE`, `INSERT INTO`, `ALTER TABLE … ADD FOREIGN KEY`.
//! - **SQLite database files** (`.db` / `.db3` / `.sqlite` / `.sqlite3`) —
//!   native read-only parser of the SQLite file format (b-tree walk, record
//!   decoding, overflow chains); no external library.
//!
//! The relational content becomes a 3D entity network:
//! - every **table** is its own [`Layer`] with a distinct color (golden-angle
//!   palette) and a distinct primitive **shape** (sphere / cube / plane);
//! - every **row** is a node, positioned inside its table's spatial cluster;
//! - every **foreign-key** value is an edge from the row to the referenced
//!   row (resolved through the referenced table's key column);
//! - node **size** grows with how many relations touch the row;
//! - every column value is preserved as node internal data for the inspector.
//!
//! DDL-only dumps (schema without rows) fall back to a schema graph: one
//! node per table, sized by column count, with FK edges between tables.

use std::collections::HashMap;

use crate::llm::graph_import::{fibonacci_sphere, MAX_GRAPH_EDGES, MAX_GRAPH_NODES};
use crate::llm::network::{Edge, Layer, LayerKind, NetworkGraph, Node};
use crate::scene::GeometryType;

/// Bounding radius of the whole laid-out database network in world units
/// (matches the graph importer's layout radius).
const LAYOUT_RADIUS: f32 = 8.0;

/// Shapes cycled per table, mirroring
/// [`crate::visualization::geometry_assigner`]'s per-label order.
const TABLE_SHAPES: [GeometryType; 3] =
    [GeometryType::Sphere, GeometryType::Cube, GeometryType::Plane];

/// Column names that make a good human label for a row, checked in order.
const LABEL_COLUMNS: [&str; 12] = [
    "name", "title", "label", "username", "full_name", "first_name", "nome",
    "titolo", "email", "code", "slug", "description",
];

/// File extensions routed to this module.
pub fn is_db_extension(ext: &str) -> bool {
    matches!(ext, "sql" | "sqlite" | "sqlite3" | "db" | "db3")
}

/// Parse a database source (SQL dump text or SQLite binary file) into a
/// relational 3D network. The content is sniffed so a `.db` holding a dump
/// (or a `.sql` renamed SQLite file) still loads.
pub fn load_db(bytes: &[u8], ext: &str, name: &str) -> Result<NetworkGraph, String> {
    let tables = if bytes.starts_with(SQLITE_MAGIC) {
        parse_sqlite(bytes)?
    } else if ext == "sql" {
        parse_sql_dump(as_utf8(bytes)?)?
    } else {
        // Binary extension without the SQLite magic: accept a text dump,
        // otherwise fail with a format hint.
        let text = std::str::from_utf8(bytes)
            .map_err(|_| format!(".{ext} file is neither a SQLite database nor a SQL dump"))?;
        parse_sql_dump(text)?
    };
    tables_to_network(tables, name)
}

fn as_utf8(bytes: &[u8]) -> Result<&str, String> {
    std::str::from_utf8(bytes).map_err(|_| "SQL dump is not UTF-8".to_string())
}

// ---------------------------------------------------------------------------
// Relational model shared by both parsers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DbColumn {
    pub name: String,
    /// Declared SQL type (may be empty for dump-inferred columns).
    pub ty: String,
}

#[derive(Debug, Clone)]
pub struct DbForeignKey {
    /// Local column index.
    pub column: usize,
    /// Referenced table name (as written, unquoted).
    pub ref_table: String,
    /// Referenced column name; None means the referenced table's key.
    pub ref_column: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct DbTable {
    pub name: String,
    pub columns: Vec<DbColumn>,
    /// Single-column primary key index, when declared.
    pub pk: Option<usize>,
    pub fks: Vec<DbForeignKey>,
    /// Row values as strings; None is SQL NULL.
    pub rows: Vec<Vec<Option<String>>>,
}

impl DbTable {
    fn col_index(&self, name: &str) -> Option<usize> {
        self.columns
            .iter()
            .position(|c| c.name.eq_ignore_ascii_case(name))
    }

    /// Column used to resolve incoming references: the declared PK, else a
    /// column literally named "id", else the first column.
    fn key_column(&self) -> usize {
        self.pk
            .or_else(|| self.col_index("id"))
            .unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// SQL dump parsing
// ---------------------------------------------------------------------------

/// One lexical token of a SQL statement (comments already stripped).
#[derive(Debug, Clone, PartialEq)]
enum Tok {
    /// Bare word: identifier, keyword or number literal.
    Word(String),
    /// Quoted identifier (`"x"`, `` `x` ``, `[x]`) — never a keyword.
    Quoted(String),
    /// Single-quoted string literal, unescaped.
    Str(String),
    Punct(char),
}

impl Tok {
    fn is_word(&self, kw: &str) -> bool {
        matches!(self, Tok::Word(w) if w.eq_ignore_ascii_case(kw))
    }

    /// Identifier text of a Word/Quoted token.
    fn ident(&self) -> Option<&str> {
        match self {
            Tok::Word(w) => Some(w),
            Tok::Quoted(q) => Some(q),
            _ => None,
        }
    }
}

/// Split dump text into `;`-terminated statements, honoring string literals
/// (`'…'` with both `''` and `\'` escapes), quoted identifiers, `$tag$…$tag$`
/// dollar quoting and `--` / `#` / `/* */` comments (which are stripped).
fn split_statements(text: &str) -> Vec<String> {
    let b = text.as_bytes();
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i] as char;
        match c {
            '-' if b.get(i + 1) == Some(&b'-') => {
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
            }
            '#' => {
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
            }
            '/' if b.get(i + 1) == Some(&b'*') => {
                i += 2;
                while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                    i += 1;
                }
                i = (i + 2).min(b.len());
            }
            '\'' => {
                cur.push(c);
                i += 1;
                while i < b.len() {
                    if b[i] == b'\\' && i + 1 < b.len() {
                        cur.push(b[i] as char);
                        cur.push(b[i + 1] as char);
                        i += 2;
                        continue;
                    }
                    if b[i] == b'\'' {
                        // '' is an escaped quote inside the literal.
                        if b.get(i + 1) == Some(&b'\'') {
                            cur.push_str("''");
                            i += 2;
                            continue;
                        }
                        break;
                    }
                    cur.push(b[i] as char);
                    i += 1;
                }
                if i < b.len() {
                    cur.push('\'');
                    i += 1;
                }
            }
            '"' | '`' => {
                let q = b[i];
                cur.push(c);
                i += 1;
                while i < b.len() && b[i] != q {
                    cur.push(b[i] as char);
                    i += 1;
                }
                if i < b.len() {
                    cur.push(b[i] as char);
                    i += 1;
                }
            }
            '$' => {
                // Dollar-quoted string: $tag$ … $tag$ (PostgreSQL).
                let tag_end = b[i + 1..]
                    .iter()
                    .position(|&ch| ch == b'$')
                    .map(|rel| i + 1 + rel);
                let is_tag = tag_end.is_some_and(|e| {
                    b[i + 1..e]
                        .iter()
                        .all(|&ch| ch.is_ascii_alphanumeric() || ch == b'_')
                });
                if let (Some(e), true) = (tag_end, is_tag) {
                    let tag = &text[i..=e];
                    let body_start = e + 1;
                    let close = text[body_start..].find(tag).map(|rel| body_start + rel);
                    let end = close.map_or(b.len(), |c| c + tag.len());
                    cur.push_str(&text[i..end]);
                    i = end;
                } else {
                    cur.push(c);
                    i += 1;
                }
            }
            ';' => {
                let stmt = cur.trim().to_string();
                if !stmt.is_empty() {
                    out.push(stmt);
                }
                cur.clear();
                i += 1;
            }
            _ => {
                cur.push(c);
                i += 1;
            }
        }
    }
    let stmt = cur.trim().to_string();
    if !stmt.is_empty() {
        out.push(stmt);
    }
    out
}

/// Tokenize one comment-free SQL statement.
fn tokenize(stmt: &str) -> Vec<Tok> {
    let b = stmt.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i] as char;
        if c.is_whitespace() {
            i += 1;
        } else if c == '\'' {
            let mut s = String::new();
            i += 1;
            while i < b.len() {
                if b[i] == b'\\' && i + 1 < b.len() {
                    // MySQL backslash escapes; keep the escaped char.
                    s.push(match b[i + 1] {
                        b'n' => '\n',
                        b't' => '\t',
                        b'r' => '\r',
                        other => other as char,
                    });
                    i += 2;
                    continue;
                }
                if b[i] == b'\'' {
                    if b.get(i + 1) == Some(&b'\'') {
                        s.push('\'');
                        i += 2;
                        continue;
                    }
                    i += 1;
                    break;
                }
                s.push(b[i] as char);
                i += 1;
            }
            out.push(Tok::Str(s));
        } else if c == '"' || c == '`' || c == '[' {
            let close = if c == '[' { b']' } else { b[i] };
            let start = i + 1;
            let mut j = start;
            while j < b.len() && b[j] != close {
                j += 1;
            }
            out.push(Tok::Quoted(stmt[start..j].to_string()));
            i = (j + 1).min(b.len());
        } else if c.is_ascii_alphanumeric() || c == '_' || c == '$' {
            let start = i;
            while i < b.len() {
                let ch = b[i] as char;
                if ch.is_ascii_alphanumeric() || ch == '_' || ch == '$' {
                    i += 1;
                } else if (ch == '.' || ch == '+' || ch == '-')
                    && b[start].is_ascii_digit()
                    && (ch != '+' && ch != '-'
                        || matches!(b[i - 1], b'e' | b'E'))
                {
                    // Keep decimal points and exponent signs inside numbers.
                    i += 1;
                } else {
                    break;
                }
            }
            out.push(Tok::Word(stmt[start..i].to_string()));
        } else {
            out.push(Tok::Punct(c));
            i += 1;
        }
    }
    out
}

/// Strip a schema prefix: the last dot-separated segment of a path already
/// split into tokens is produced by [`tokenize`] as separate tokens, so this
/// only needs to advance the cursor over `schema . name` sequences.
fn read_table_name(toks: &[Tok], i: &mut usize) -> Option<String> {
    let mut name = toks.get(*i)?.ident()?.to_string();
    *i += 1;
    while toks.get(*i) == Some(&Tok::Punct('.')) {
        name = toks.get(*i + 1)?.ident()?.to_string();
        *i += 2;
    }
    Some(name)
}

/// Consume a parenthesized identifier list: `( a, b, … )`.
fn read_ident_list(toks: &[Tok], i: &mut usize) -> Vec<String> {
    let mut out = Vec::new();
    if toks.get(*i) != Some(&Tok::Punct('(')) {
        return out;
    }
    *i += 1;
    while let Some(t) = toks.get(*i) {
        match t {
            Tok::Punct(')') => {
                *i += 1;
                break;
            }
            Tok::Punct(',') => *i += 1,
            _ => {
                if let Some(id) = t.ident() {
                    out.push(id.to_string());
                }
                *i += 1;
            }
        }
    }
    out
}

/// `REFERENCES table [(col)]` starting after the REFERENCES keyword.
fn read_reference(toks: &[Tok], i: &mut usize) -> Option<(String, Option<String>)> {
    let table = read_table_name(toks, i)?;
    let cols = read_ident_list(toks, i);
    Some((table, cols.into_iter().next()))
}

/// Parse `CREATE TABLE` (tokens after the table name, starting at `(`).
fn parse_table_body(toks: &[Tok], i: &mut usize, table: &mut DbTable) {
    if toks.get(*i) != Some(&Tok::Punct('(')) {
        return;
    }
    *i += 1;
    // Split the body into top-level comma-separated items.
    let mut depth = 0usize;
    let mut item: Vec<Tok> = Vec::new();
    let mut items: Vec<Vec<Tok>> = Vec::new();
    while let Some(t) = toks.get(*i) {
        match t {
            Tok::Punct('(') => {
                depth += 1;
                item.push(t.clone());
            }
            Tok::Punct(')') => {
                if depth == 0 {
                    *i += 1;
                    break;
                }
                depth -= 1;
                item.push(t.clone());
            }
            Tok::Punct(',') if depth == 0 => items.push(std::mem::take(&mut item)),
            _ => item.push(t.clone()),
        }
        *i += 1;
    }
    if !item.is_empty() {
        items.push(item);
    }

    for item in items {
        parse_table_item(&item, table);
    }
}

/// One item of a CREATE TABLE body: a column definition or a table constraint.
fn parse_table_item(item: &[Tok], table: &mut DbTable) {
    let mut j = 0;
    // `CONSTRAINT name …` prefixes are transparent.
    if item.first().is_some_and(|t| t.is_word("CONSTRAINT")) {
        j = 2;
    }
    let Some(head) = item.get(j) else { return };

    // Table-level constraints.
    if head.is_word("PRIMARY") {
        let mut k = j + 2; // skip PRIMARY KEY
        let cols = read_ident_list(item, &mut k);
        if cols.len() == 1 {
            table.pk = table.col_index(&cols[0]);
        }
        return;
    }
    if head.is_word("FOREIGN") {
        let mut k = j + 2; // skip FOREIGN KEY
        let cols = read_ident_list(item, &mut k);
        if item.get(k).is_some_and(|t| t.is_word("REFERENCES")) {
            k += 1;
            if let (Some(col), Some((rt, rc))) =
                (cols.first(), read_reference(item, &mut k))
            {
                if let Some(ci) = table.col_index(col) {
                    table.fks.push(DbForeignKey {
                        column: ci,
                        ref_table: rt,
                        ref_column: rc,
                    });
                }
            }
        }
        return;
    }
    if matches!(head, Tok::Word(w) if ["UNIQUE", "KEY", "INDEX", "CHECK", "EXCLUDE", "FULLTEXT", "SPATIAL", "LIKE"]
        .iter()
        .any(|kw| w.eq_ignore_ascii_case(kw)))
    {
        return;
    }

    // Column definition: name, type tokens, inline constraints.
    let Some(name) = head.ident() else { return };
    let col_idx = table.columns.len();
    let mut ty = String::new();
    let mut k = j + 1;
    // Type = words/parens up to the first constraint keyword.
    while let Some(t) = item.get(k) {
        let is_kw = matches!(t, Tok::Word(w) if [
            "PRIMARY", "REFERENCES", "NOT", "NULL", "DEFAULT", "UNIQUE", "CHECK",
            "COLLATE", "AUTO_INCREMENT", "AUTOINCREMENT", "GENERATED", "CONSTRAINT",
            "COMMENT", "ON",
        ]
        .iter()
        .any(|kw| w.eq_ignore_ascii_case(kw)));
        if is_kw {
            break;
        }
        match t {
            Tok::Word(w) => {
                if !ty.is_empty() {
                    ty.push(' ');
                }
                ty.push_str(w);
            }
            Tok::Punct('(') => {
                // Copy the precision block verbatim: (10,2).
                let mut depth = 1;
                ty.push('(');
                k += 1;
                while let Some(t2) = item.get(k) {
                    match t2 {
                        Tok::Punct('(') => depth += 1,
                        Tok::Punct(')') => {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        _ => {}
                    }
                    match t2 {
                        Tok::Word(w) => ty.push_str(w),
                        Tok::Punct(c) => ty.push(*c),
                        _ => {}
                    }
                    k += 1;
                }
                ty.push(')');
            }
            _ => break,
        }
        k += 1;
    }
    table.columns.push(DbColumn {
        name: name.to_string(),
        ty,
    });

    // Inline constraints on the remainder.
    while let Some(t) = item.get(k) {
        if t.is_word("PRIMARY") && item.get(k + 1).is_some_and(|t| t.is_word("KEY")) {
            table.pk = Some(col_idx);
            k += 2;
        } else if t.is_word("REFERENCES") {
            k += 1;
            if let Some((rt, rc)) = read_reference(item, &mut k) {
                table.fks.push(DbForeignKey {
                    column: col_idx,
                    ref_table: rt,
                    ref_column: rc,
                });
            }
        } else {
            k += 1;
        }
    }
}

/// Parse one SQL literal from the token stream into an optional string value.
fn read_value(toks: &[Tok], i: &mut usize) -> Option<Option<String>> {
    let t = toks.get(*i)?;
    match t {
        Tok::Str(s) => {
            // `x'ABCD'` never reaches here (x is a separate Word) — plain string.
            *i += 1;
            Some(Some(s.clone()))
        }
        Tok::Word(w) if w.eq_ignore_ascii_case("NULL") || w.eq_ignore_ascii_case("DEFAULT") => {
            *i += 1;
            Some(None)
        }
        Tok::Word(w)
            if (w.eq_ignore_ascii_case("x") || w.eq_ignore_ascii_case("b"))
                && matches!(toks.get(*i + 1), Some(Tok::Str(_))) =>
        {
            // Hex / bit literal: x'1F'.
            let Some(Tok::Str(s)) = toks.get(*i + 1) else { unreachable!() };
            *i += 2;
            Some(Some(format!("0x{s}")))
        }
        Tok::Word(w) => {
            let mut j = *i + 1;
            let value = if toks.get(j) == Some(&Tok::Punct('(')) {
                // Function call (NOW(), uuid(), …): skip the balanced call,
                // keep the name as an opaque marker.
                let mut depth = 1;
                j += 1;
                while let Some(t2) = toks.get(j) {
                    match t2 {
                        Tok::Punct('(') => depth += 1,
                        Tok::Punct(')') => {
                            depth -= 1;
                            if depth == 0 {
                                j += 1;
                                break;
                            }
                        }
                        _ => {}
                    }
                    j += 1;
                }
                None
            } else {
                Some(w.clone())
            };
            *i = j;
            Some(value)
        }
        Tok::Punct('-') | Tok::Punct('+') => {
            let sign = if *t == Tok::Punct('-') { "-" } else { "" };
            if let Some(Tok::Word(w)) = toks.get(*i + 1) {
                *i += 2;
                Some(Some(format!("{sign}{w}")))
            } else {
                *i += 1;
                Some(None)
            }
        }
        Tok::Punct('(') => {
            // Nested expression: skip balanced, value unknown.
            let mut depth = 1;
            *i += 1;
            while let Some(t2) = toks.get(*i) {
                match t2 {
                    Tok::Punct('(') => depth += 1,
                    Tok::Punct(')') => {
                        depth -= 1;
                        if depth == 0 {
                            *i += 1;
                            break;
                        }
                    }
                    _ => {}
                }
                *i += 1;
            }
            Some(None)
        }
        _ => {
            *i += 1;
            Some(None)
        }
    }
}

/// Case-insensitive table lookup / creation.
fn table_entry<'a>(
    tables: &'a mut Vec<DbTable>,
    index: &mut HashMap<String, usize>,
    name: &str,
) -> &'a mut DbTable {
    let key = name.to_ascii_lowercase();
    let idx = *index.entry(key).or_insert_with(|| {
        tables.push(DbTable {
            name: name.to_string(),
            ..Default::default()
        });
        tables.len() - 1
    });
    &mut tables[idx]
}

/// Extract PostgreSQL `COPY table (cols) FROM stdin; … \.` blocks before
/// statement splitting. Returns the text with the blocks removed plus each
/// block's (header line, data lines).
fn extract_copy_blocks(text: &str) -> (String, Vec<(String, Vec<String>)>) {
    let mut rest = String::with_capacity(text.len());
    let mut blocks = Vec::new();
    let mut lines = text.lines();
    while let Some(line) = lines.next() {
        let up = line.trim_start().to_ascii_uppercase();
        if up.starts_with("COPY ") && up.contains(" FROM STDIN") {
            let mut data = Vec::new();
            for data_line in lines.by_ref() {
                if data_line.trim_end() == "\\." {
                    break;
                }
                data.push(data_line.to_string());
            }
            blocks.push((line.trim().trim_end_matches(';').to_string(), data));
        } else {
            rest.push_str(line);
            rest.push('\n');
        }
    }
    (rest, blocks)
}

/// Unescape one PostgreSQL COPY text-format field; `\N` (caught by the
/// caller) is NULL.
fn copy_unescape(field: &str) -> String {
    let mut out = String::with_capacity(field.len());
    let mut chars = field.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some(other) => out.push(other),
            None => out.push('\\'),
        }
    }
    out
}

/// Parse a SQL dump into tables with columns, keys and rows.
pub fn parse_sql_dump(text: &str) -> Result<Vec<DbTable>, String> {
    let (rest, copy_blocks) = extract_copy_blocks(text);
    let mut tables: Vec<DbTable> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();

    for stmt in split_statements(&rest) {
        let toks = tokenize(&stmt);
        let mut i;
        let Some(first) = toks.first() else { continue };

        if first.is_word("CREATE") {
            // CREATE [TEMPORARY|UNLOGGED] TABLE [IF NOT EXISTS] name (…)
            i = 1;
            while toks.get(i).is_some_and(|t| {
                matches!(t, Tok::Word(w) if ["TEMPORARY", "TEMP", "UNLOGGED", "GLOBAL", "LOCAL"]
                    .iter()
                    .any(|kw| w.eq_ignore_ascii_case(kw)))
            }) {
                i += 1;
            }
            if !toks.get(i).is_some_and(|t| t.is_word("TABLE")) {
                continue; // CREATE INDEX / VIEW / TRIGGER / …
            }
            i += 1;
            if toks.get(i).is_some_and(|t| t.is_word("IF")) {
                i += 3; // IF NOT EXISTS
            }
            let Some(name) = read_table_name(&toks, &mut i) else { continue };
            let entry = table_entry(&mut tables, &mut index, &name);
            let mut fresh = DbTable {
                name: entry.name.clone(),
                ..Default::default()
            };
            parse_table_body(&toks, &mut i, &mut fresh);
            // Keep rows already inserted before the DDL appeared (rare but legal).
            fresh.rows = std::mem::take(&mut entry.rows);
            *entry = fresh;
        } else if first.is_word("INSERT") || first.is_word("REPLACE") {
            // INSERT [IGNORE] INTO name [(cols)] VALUES (…), (…)
            i = 1;
            while toks
                .get(i)
                .is_some_and(|t| t.is_word("IGNORE") || t.is_word("INTO") || t.is_word("OR"))
                || toks.get(i).is_some_and(|t| t.is_word("REPLACE"))
            {
                i += 1;
            }
            let Some(name) = read_table_name(&toks, &mut i) else { continue };
            let cols = read_ident_list(&toks, &mut i);
            if !toks
                .get(i)
                .is_some_and(|t| t.is_word("VALUES") || t.is_word("VALUE"))
            {
                continue; // INSERT … SELECT and friends.
            }
            i += 1;

            // Ensure the table and its columns exist (dumps without DDL).
            {
                let entry = table_entry(&mut tables, &mut index, &name);
                for col in &cols {
                    if entry.col_index(col).is_none() {
                        entry.columns.push(DbColumn {
                            name: col.clone(),
                            ty: String::new(),
                        });
                    }
                }
            }

            let mut tuples: Vec<Vec<Option<String>>> = Vec::new();
            while toks.get(i) == Some(&Tok::Punct('(')) {
                i += 1;
                let mut tuple = Vec::new();
                loop {
                    match toks.get(i) {
                        Some(Tok::Punct(')')) => {
                            i += 1;
                            break;
                        }
                        Some(Tok::Punct(',')) => i += 1,
                        Some(_) => match read_value(&toks, &mut i) {
                            Some(v) => tuple.push(v),
                            None => break,
                        },
                        None => break,
                    }
                }
                tuples.push(tuple);
                if toks.get(i) == Some(&Tok::Punct(',')) {
                    i += 1;
                }
            }

            let entry = table_entry(&mut tables, &mut index, &name);
            for tuple in tuples {
                let row = if cols.is_empty() {
                    // Positional insert: extend the column set if the dump
                    // never declared the table.
                    while entry.columns.len() < tuple.len() {
                        entry.columns.push(DbColumn {
                            name: format!("c{}", entry.columns.len() + 1),
                            ty: String::new(),
                        });
                    }
                    let mut row = tuple;
                    row.resize(entry.columns.len(), None);
                    row
                } else {
                    let mut row = vec![None; entry.columns.len()];
                    for (col, value) in cols.iter().zip(tuple) {
                        if let Some(ci) = entry.col_index(col) {
                            row[ci] = value;
                        }
                    }
                    row
                };
                entry.rows.push(row);
            }
        } else if first.is_word("ALTER") {
            // ALTER TABLE [ONLY] name ADD [CONSTRAINT x] FOREIGN KEY (c) REFERENCES t (c)
            i = 1;
            if !toks.get(i).is_some_and(|t| t.is_word("TABLE")) {
                continue;
            }
            i += 1;
            if toks.get(i).is_some_and(|t| t.is_word("ONLY")) {
                i += 1;
            }
            if toks.get(i).is_some_and(|t| t.is_word("IF")) {
                i += 2; // IF EXISTS
            }
            let Some(name) = read_table_name(&toks, &mut i) else { continue };
            while i < toks.len() {
                if toks[i].is_word("FOREIGN") && toks.get(i + 1).is_some_and(|t| t.is_word("KEY")) {
                    let mut k = i + 2;
                    let cols = read_ident_list(&toks, &mut k);
                    if toks.get(k).is_some_and(|t| t.is_word("REFERENCES")) {
                        k += 1;
                        if let Some((rt, rc)) = read_reference(&toks, &mut k) {
                            let entry = table_entry(&mut tables, &mut index, &name);
                            if let Some(ci) = cols.first().and_then(|c| entry.col_index(c)) {
                                entry.fks.push(DbForeignKey {
                                    column: ci,
                                    ref_table: rt,
                                    ref_column: rc,
                                });
                            }
                        }
                    }
                    i = k;
                } else {
                    i += 1;
                }
            }
        }
    }

    // Apply COPY data blocks.
    for (header, data) in copy_blocks {
        let toks = tokenize(&header);
        let mut i = 1; // skip COPY
        let Some(name) = read_table_name(&toks, &mut i) else { continue };
        let cols = read_ident_list(&toks, &mut i);
        let entry = table_entry(&mut tables, &mut index, &name);
        for col in &cols {
            if entry.col_index(col).is_none() {
                entry.columns.push(DbColumn {
                    name: col.clone(),
                    ty: String::new(),
                });
            }
        }
        for line in data {
            let fields: Vec<Option<String>> = line
                .split('\t')
                .map(|f| {
                    if f == "\\N" {
                        None
                    } else {
                        Some(copy_unescape(f))
                    }
                })
                .collect();
            let row = if cols.is_empty() {
                while entry.columns.len() < fields.len() {
                    entry.columns.push(DbColumn {
                        name: format!("c{}", entry.columns.len() + 1),
                        ty: String::new(),
                    });
                }
                let mut row = fields;
                row.resize(entry.columns.len(), None);
                row
            } else {
                let mut row = vec![None; entry.columns.len()];
                for (col, value) in cols.iter().zip(fields) {
                    if let Some(ci) = entry.col_index(col) {
                        row[ci] = value;
                    }
                }
                row
            };
            entry.rows.push(row);
        }
    }

    if tables.is_empty() {
        return Err("SQL dump contains no CREATE TABLE / INSERT statements".to_string());
    }
    Ok(tables)
}

// ---------------------------------------------------------------------------
// SQLite binary file parsing
// ---------------------------------------------------------------------------

const SQLITE_MAGIC: &[u8] = b"SQLite format 3\0";

/// Decoded SQLite value.
enum SqlVal {
    Null,
    Int(i64),
    Real(f64),
    Text(String),
    Blob(usize),
}

impl SqlVal {
    fn into_string(self) -> Option<String> {
        match self {
            SqlVal::Null => None,
            SqlVal::Int(i) => Some(i.to_string()),
            SqlVal::Real(f) => Some(f.to_string()),
            SqlVal::Text(s) => Some(s),
            SqlVal::Blob(n) => Some(format!("blob({n} bytes)")),
        }
    }

    fn as_int(&self) -> Option<i64> {
        match self {
            SqlVal::Int(i) => Some(*i),
            _ => None,
        }
    }

    fn as_text(&self) -> Option<&str> {
        match self {
            SqlVal::Text(s) => Some(s),
            _ => None,
        }
    }
}

struct SqliteFile<'a> {
    data: &'a [u8],
    page_size: usize,
    usable: usize,
}

/// SQLite varint: 1–9 bytes, big-endian 7-bit groups (9th byte carries 8 bits).
fn read_varint(data: &[u8], pos: &mut usize) -> Option<i64> {
    let mut result: i64 = 0;
    for k in 0..9 {
        let byte = *data.get(*pos)?;
        *pos += 1;
        if k == 8 {
            result = (result << 8) | byte as i64;
            return Some(result);
        }
        result = (result << 7) | (byte & 0x7F) as i64;
        if byte & 0x80 == 0 {
            return Some(result);
        }
    }
    Some(result)
}

impl<'a> SqliteFile<'a> {
    fn open(data: &'a [u8]) -> Result<Self, String> {
        if !data.starts_with(SQLITE_MAGIC) || data.len() < 100 {
            return Err("not a SQLite database (bad magic)".to_string());
        }
        let raw = u16::from_be_bytes([data[16], data[17]]) as usize;
        let page_size = if raw == 1 { 65536 } else { raw };
        if !(512..=65536).contains(&page_size) || !page_size.is_power_of_two() {
            return Err(format!("invalid SQLite page size {raw}"));
        }
        let reserved = data[20] as usize;
        let usable = page_size
            .checked_sub(reserved)
            .filter(|&u| u >= 480)
            .ok_or("invalid SQLite reserved-space byte")?;
        let encoding = u32::from_be_bytes([data[56], data[57], data[58], data[59]]);
        // 0 appears in empty databases before the first table is created.
        if encoding > 1 {
            return Err("only UTF-8 SQLite databases are supported".to_string());
        }
        Ok(SqliteFile {
            data,
            page_size,
            usable,
        })
    }

    /// Whole page (1-based numbering, as in the file format).
    fn page(&self, n: u32) -> Result<&'a [u8], String> {
        let n = n as usize;
        if n == 0 {
            return Err("page number 0 in b-tree".to_string());
        }
        let start = (n - 1) * self.page_size;
        self.data
            .get(start..start + self.page_size)
            .ok_or_else(|| format!("page {n} beyond end of file"))
    }

    /// Payload bytes stored locally in a table b-tree leaf cell.
    fn local_payload(&self, payload: usize) -> usize {
        let u = self.usable;
        let max_local = u - 35;
        let min_local = (u - 12) * 32 / 255 - 23;
        if payload <= max_local {
            payload
        } else {
            let k = min_local + (payload - min_local) % (u - 4);
            if k <= max_local {
                k
            } else {
                min_local
            }
        }
    }

    /// Assemble a cell payload, following the overflow chain when needed.
    fn read_payload(
        &self,
        page: &'a [u8],
        mut pos: usize,
        payload_len: usize,
    ) -> Result<Vec<u8>, String> {
        let local = self.local_payload(payload_len);
        let local_bytes = page
            .get(pos..pos + local)
            .ok_or("cell payload beyond page end")?;
        if local == payload_len {
            return Ok(local_bytes.to_vec());
        }
        let mut out = Vec::with_capacity(payload_len);
        out.extend_from_slice(local_bytes);
        pos += local;
        let ptr = page
            .get(pos..pos + 4)
            .ok_or("overflow pointer beyond page end")?;
        let mut next = u32::from_be_bytes([ptr[0], ptr[1], ptr[2], ptr[3]]);
        let mut guard = 0usize;
        while next != 0 && out.len() < payload_len {
            guard += 1;
            if guard > self.data.len() / self.page_size + 2 {
                return Err("overflow chain loop".to_string());
            }
            let opage = self.page(next)?;
            next = u32::from_be_bytes([opage[0], opage[1], opage[2], opage[3]]);
            let take = (payload_len - out.len()).min(self.usable - 4);
            out.extend_from_slice(
                opage.get(4..4 + take).ok_or("overflow page too short")?,
            );
        }
        if out.len() < payload_len {
            return Err("truncated overflow chain".to_string());
        }
        Ok(out)
    }

    /// Walk a table b-tree, calling `on_row` for every (rowid, record).
    /// Stops early when `on_row` returns false.
    fn walk_table(
        &self,
        root: u32,
        on_row: &mut dyn FnMut(i64, Vec<SqlVal>) -> bool,
    ) -> Result<(), String> {
        let mut stack = vec![root];
        let mut visited = 0usize;
        while let Some(page_no) = stack.pop() {
            visited += 1;
            if visited > self.data.len() / self.page_size + 2 {
                return Err("b-tree page loop".to_string());
            }
            let page = self.page(page_no)?;
            // Page 1 hosts the 100-byte database header before the b-tree header.
            let hdr = if page_no == 1 { 100 } else { 0 };
            let page_type = page[hdr];
            let cell_count = u16::from_be_bytes([page[hdr + 3], page[hdr + 4]]) as usize;
            match page_type {
                5 => {
                    // Interior table page: 12-byte header, right-most pointer at +8.
                    let right =
                        u32::from_be_bytes([page[hdr + 8], page[hdr + 9], page[hdr + 10], page[hdr + 11]]);
                    stack.push(right);
                    let ptrs = hdr + 12;
                    for c in 0..cell_count {
                        let po = ptrs + c * 2;
                        let off = u16::from_be_bytes([page[po], page[po + 1]]) as usize;
                        let child = page
                            .get(off..off + 4)
                            .ok_or("interior cell beyond page end")?;
                        stack.push(u32::from_be_bytes([child[0], child[1], child[2], child[3]]));
                    }
                }
                13 => {
                    let ptrs = hdr + 8;
                    for c in 0..cell_count {
                        let po = ptrs + c * 2;
                        let (b0, b1) = (
                            *page.get(po).ok_or("cell pointer beyond page")?,
                            *page.get(po + 1).ok_or("cell pointer beyond page")?,
                        );
                        let mut pos = u16::from_be_bytes([b0, b1]) as usize;
                        let payload_len = read_varint(page, &mut pos)
                            .ok_or("bad payload varint")? as usize;
                        let rowid =
                            read_varint(page, &mut pos).ok_or("bad rowid varint")?;
                        let payload = self.read_payload(page, pos, payload_len)?;
                        let values = decode_record(&payload)?;
                        if !on_row(rowid, values) {
                            return Ok(());
                        }
                    }
                }
                2 | 10 => {
                    // Index b-tree (WITHOUT ROWID storage): not walked.
                    return Err("index b-tree where a table was expected".to_string());
                }
                other => return Err(format!("unknown b-tree page type {other}")),
            }
        }
        Ok(())
    }
}

/// Decode one SQLite record (header of serial types + body).
fn decode_record(payload: &[u8]) -> Result<Vec<SqlVal>, String> {
    let mut pos = 0usize;
    let header_len = read_varint(payload, &mut pos).ok_or("bad record header varint")? as usize;
    let mut serials = Vec::new();
    while pos < header_len.min(payload.len()) {
        serials.push(read_varint(payload, &mut pos).ok_or("bad serial type varint")?);
    }
    let mut body = header_len;
    let mut out = Vec::with_capacity(serials.len());
    for st in serials {
        let take = |n: usize, body: &mut usize| -> Result<&[u8], String> {
            let s = payload
                .get(*body..*body + n)
                .ok_or("record body beyond payload")?;
            *body += n;
            Ok(s)
        };
        let val = match st {
            0 => SqlVal::Null,
            1..=6 => {
                let n = match st {
                    1 => 1,
                    2 => 2,
                    3 => 3,
                    4 => 4,
                    5 => 6,
                    _ => 8,
                };
                let bytes = take(n, &mut body)?;
                // Sign-extended big-endian two's complement.
                let mut v: i64 = if bytes[0] & 0x80 != 0 { -1 } else { 0 };
                for &b in bytes {
                    v = (v << 8) | b as i64;
                }
                SqlVal::Int(v)
            }
            7 => {
                let bytes = take(8, &mut body)?;
                let mut arr = [0u8; 8];
                arr.copy_from_slice(bytes);
                SqlVal::Real(f64::from_be_bytes(arr))
            }
            8 => SqlVal::Int(0),
            9 => SqlVal::Int(1),
            st if st >= 13 && st % 2 == 1 => {
                let n = ((st - 13) / 2) as usize;
                let bytes = take(n, &mut body)?;
                SqlVal::Text(String::from_utf8_lossy(bytes).into_owned())
            }
            st if st >= 12 => {
                let n = ((st - 12) / 2) as usize;
                take(n, &mut body)?;
                SqlVal::Blob(n)
            }
            other => return Err(format!("reserved serial type {other}")),
        };
        out.push(val);
    }
    Ok(out)
}

/// Parse a SQLite database file into tables with schema and rows.
pub fn parse_sqlite(bytes: &[u8]) -> Result<Vec<DbTable>, String> {
    let file = SqliteFile::open(bytes)?;

    // sqlite_master: (type, name, tbl_name, rootpage, sql), rooted at page 1.
    struct MasterRow {
        name: String,
        rootpage: u32,
        sql: String,
    }
    let mut master = Vec::new();
    file.walk_table(1, &mut |_rowid, values| {
        let ty = values.first().and_then(|v| v.as_text()).unwrap_or("");
        if ty == "table" {
            let name = values.get(1).and_then(|v| v.as_text()).unwrap_or("");
            let rootpage = values.get(3).and_then(|v| v.as_int()).unwrap_or(0);
            let sql = values.get(4).and_then(|v| v.as_text()).unwrap_or("");
            if !name.is_empty() && !name.starts_with("sqlite_") && rootpage > 0 {
                master.push(MasterRow {
                    name: name.to_string(),
                    rootpage: rootpage as u32,
                    sql: sql.to_string(),
                });
            }
        }
        true
    })?;
    if master.is_empty() {
        return Err("SQLite database contains no user tables".to_string());
    }

    let mut tables = Vec::new();
    for m in master {
        // Reuse the dump DDL parser on the stored CREATE TABLE statement.
        let mut table = parse_sql_dump(&m.sql)
            .ok()
            .and_then(|mut ts| {
                ts.iter()
                    .position(|t| t.name.eq_ignore_ascii_case(&m.name))
                    .map(|i| ts.swap_remove(i))
            })
            .unwrap_or_else(|| DbTable {
                name: m.name.clone(),
                ..Default::default()
            });
        table.name = m.name.clone();

        // WITHOUT ROWID tables use index b-trees; keep the schema, skip rows.
        if m.sql.to_ascii_uppercase().contains("WITHOUT ROWID") {
            tables.push(table);
            continue;
        }

        // `INTEGER PRIMARY KEY` is a rowid alias: its record slot is NULL and
        // the true value is the rowid.
        let rowid_alias = table
            .pk
            .filter(|&pk| table.columns[pk].ty.eq_ignore_ascii_case("INTEGER"));

        let mut rows: Vec<Vec<Option<String>>> = Vec::new();
        file.walk_table(m.rootpage, &mut |rowid, values| {
            let width = table.columns.len().max(values.len());
            let mut row: Vec<Option<String>> = Vec::with_capacity(width);
            for (ci, val) in values.into_iter().enumerate() {
                if Some(ci) == rowid_alias && matches!(val, SqlVal::Null) {
                    row.push(Some(rowid.to_string()));
                } else {
                    row.push(val.into_string());
                }
            }
            row.resize(width, None);
            rows.push(row);
            rows.len() < MAX_GRAPH_NODES
        })?;

        // Tables inserted without stored DDL columns: name them positionally.
        let width = rows.iter().map(Vec::len).max().unwrap_or(0);
        while table.columns.len() < width {
            table.columns.push(DbColumn {
                name: format!("c{}", table.columns.len() + 1),
                ty: String::new(),
            });
        }
        table.rows = rows;
        tables.push(table);
    }
    Ok(tables)
}

// ---------------------------------------------------------------------------
// Relational model → 3D network
// ---------------------------------------------------------------------------

/// Distinct per-table color: golden-angle hue rotation over a fixed
/// saturation/value, so any number of tables stays distinguishable.
pub fn table_color(index: usize) -> [f32; 3] {
    let h = (index as f32 * 137.508) % 360.0;
    hsv_to_rgb(h, 0.62, 0.92)
}

/// Shape assigned to a table's nodes (cycled).
pub fn table_shape(index: usize) -> GeometryType {
    TABLE_SHAPES[index % TABLE_SHAPES.len()]
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let c = v * s;
    let hp = h / 60.0;
    let x = c * (1.0 - (hp % 2.0 - 1.0).abs());
    let (r, g, b) = match hp as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    [r + m, g + m, b + m]
}

/// Center all positions on the origin and scale the bounding radius to
/// [`LAYOUT_RADIUS`] (same convention as the graph importer).
fn normalize_positions(layers: &mut [Layer]) {
    let mut c = [0.0f64; 3];
    let mut n = 0usize;
    for layer in layers.iter() {
        for node in &layer.nodes {
            for (acc, p) in c.iter_mut().zip(node.position) {
                *acc += p as f64;
            }
            n += 1;
        }
    }
    if n == 0 {
        return;
    }
    for a in &mut c {
        *a /= n as f64;
    }
    let mut r_max = 0.0f32;
    for layer in layers.iter_mut() {
        for node in &mut layer.nodes {
            for (p, off) in node.position.iter_mut().zip(c) {
                *p -= off as f32;
            }
            let p = node.position;
            r_max = r_max.max((p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt());
        }
    }
    let s = LAYOUT_RADIUS / r_max.max(1e-6);
    for layer in layers.iter_mut() {
        for node in &mut layer.nodes {
            for a in node.position.iter_mut() {
                *a *= s;
            }
        }
    }
}

/// Pick the label column of a table, if any.
fn label_column(table: &DbTable) -> Option<usize> {
    LABEL_COLUMNS
        .iter()
        .find_map(|name| table.col_index(name))
}

/// Build the entity network from parsed tables.
pub fn tables_to_network(mut tables: Vec<DbTable>, name: &str) -> Result<NetworkGraph, String> {
    tables.retain(|t| !t.columns.is_empty() || !t.rows.is_empty());
    if tables.is_empty() {
        return Err("no tables found in the database source".to_string());
    }
    let total_rows: usize = tables.iter().map(|t| t.rows.len()).sum();
    if total_rows == 0 {
        return Ok(schema_network(&tables, name));
    }

    // Proportional cap: huge databases are sampled instead of refused.
    if total_rows > MAX_GRAPH_NODES {
        let scale = MAX_GRAPH_NODES as f64 / total_rows as f64;
        for t in &mut tables {
            let keep = ((t.rows.len() as f64 * scale) as usize).clamp(1, t.rows.len());
            t.rows.truncate(keep);
        }
    }

    // Cluster geometry: table centers on a sphere, rows in a ball around
    // their center whose radius grows with the row count.
    let nt = tables.len();
    let centers: Vec<[f32; 3]> = if nt == 1 {
        vec![[0.0; 3]]
    } else {
        fibonacci_sphere(nt)
            .into_iter()
            .map(|p| [p[0] * LAYOUT_RADIUS, p[1] * LAYOUT_RADIUS, p[2] * LAYOUT_RADIUS])
            .collect()
    };
    let max_rows = tables.iter().map(|t| t.rows.len()).max().unwrap_or(1).max(1);

    let mut layers: Vec<Layer> = Vec::with_capacity(nt);
    for (ti, table) in tables.iter().enumerate() {
        let color = table_color(ti);
        let label_col = label_column(table);
        let key_col = table.key_column();
        let n = table.rows.len();
        let cluster_r = if nt == 1 {
            LAYOUT_RADIUS * 0.9
        } else {
            (LAYOUT_RADIUS * 0.55)
                * (0.35 + 0.65 * (n as f32 / max_rows as f32).cbrt())
        };
        let dirs = fibonacci_sphere(n.max(1));
        let nodes: Vec<Node> = table
            .rows
            .iter()
            .enumerate()
            .map(|(ri, row)| {
                // Ball distribution: push successive points outwards so the
                // cluster fills a volume instead of a shell.
                let t = ((ri + 1) as f32 / n.max(1) as f32).cbrt();
                let d = dirs[ri.min(dirs.len() - 1)];
                let c = centers[ti];
                let key = row
                    .get(key_col)
                    .and_then(|v| v.clone())
                    .unwrap_or_else(|| (ri + 1).to_string());
                let label = label_col
                    .and_then(|ci| row.get(ci).and_then(|v| v.clone()))
                    .unwrap_or_else(|| format!("{} #{key}", table.name));
                let attrs: Vec<(String, String)> = table
                    .columns
                    .iter()
                    .zip(row.iter())
                    .filter_map(|(col, v)| {
                        v.as_ref().map(|v| (col.name.clone(), v.clone()))
                    })
                    .collect();
                Node {
                    position: [
                        c[0] + d[0] * cluster_r * t,
                        c[1] + d[1] * cluster_r * t,
                        c[2] + d[2] * cluster_r * t,
                    ],
                    weight_magnitude: 0.5, // refined below from the FK degree
                    color: Some(color),
                    label: Some(label),
                    attrs,
                }
            })
            .collect();
        layers.push(Layer {
            name: table.name.clone(),
            kind: LayerKind::Graph,
            nodes,
            shape: Some(table_shape(ti)),
        });
    }

    // Foreign-key edges: resolve each FK value through the referenced
    // table's key (or explicitly referenced) column.
    let table_index: HashMap<String, usize> = tables
        .iter()
        .enumerate()
        .map(|(i, t)| (t.name.to_ascii_lowercase(), i))
        .collect();
    // Lazily-built value → row maps, keyed by (table, column).
    let mut value_maps: HashMap<(usize, usize), HashMap<String, usize>> = HashMap::new();
    let mut edges: Vec<Edge> = Vec::new();
    for (ti, table) in tables.iter().enumerate() {
        for fk in &table.fks {
            let Some(&rt) = table_index.get(&fk.ref_table.to_ascii_lowercase()) else {
                continue;
            };
            let ref_col = fk
                .ref_column
                .as_ref()
                .and_then(|c| tables[rt].col_index(c))
                .unwrap_or_else(|| tables[rt].key_column());
            let map = value_maps.entry((rt, ref_col)).or_insert_with(|| {
                tables[rt]
                    .rows
                    .iter()
                    .enumerate()
                    .filter_map(|(ri, row)| {
                        row.get(ref_col)
                            .and_then(|v| v.clone())
                            .map(|v| (v, ri))
                    })
                    .collect()
            });
            for (ri, row) in table.rows.iter().enumerate() {
                let Some(value) = row.get(fk.column).and_then(|v| v.as_ref()) else {
                    continue;
                };
                if let Some(&target) = map.get(value) {
                    edges.push(Edge {
                        from_layer: ti,
                        from_node: ri,
                        to_layer: rt,
                        to_node: target,
                        importance: 1.0,
                    });
                }
            }
        }
    }
    edges.truncate(MAX_GRAPH_EDGES);

    // Node size from relation degree (rows touched by more FKs read bigger).
    let mut degree: HashMap<(usize, usize), usize> = HashMap::new();
    for e in &edges {
        *degree.entry((e.from_layer, e.from_node)).or_default() += 1;
        *degree.entry((e.to_layer, e.to_node)).or_default() += 1;
    }
    let max_deg = degree.values().copied().max().unwrap_or(0).max(1) as f32;
    for (li, layer) in layers.iter_mut().enumerate() {
        for (ni, node) in layer.nodes.iter_mut().enumerate() {
            let d = degree.get(&(li, ni)).copied().unwrap_or(0) as f32;
            node.weight_magnitude = if max_deg > 0.0 && !degree.is_empty() {
                (0.2 + 0.8 * (d / max_deg).sqrt()).clamp(0.05, 1.0)
            } else {
                0.5
            };
        }
    }

    normalize_positions(&mut layers);
    Ok(NetworkGraph {
        name: name.to_string(),
        layers,
        edges,
        estimated_vram_gb: None,
        moe_config: None,
        data_graph: true,
    })
}

/// Schema-only fallback (DDL without rows): one node per table, sized by
/// column count, FK edges between tables.
fn schema_network(tables: &[DbTable], name: &str) -> NetworkGraph {
    let nt = tables.len();
    let centers: Vec<[f32; 3]> = if nt == 1 {
        vec![[0.0; 3]]
    } else {
        fibonacci_sphere(nt)
            .into_iter()
            .map(|p| [p[0] * LAYOUT_RADIUS, p[1] * LAYOUT_RADIUS, p[2] * LAYOUT_RADIUS])
            .collect()
    };
    let max_cols = tables.iter().map(|t| t.columns.len()).max().unwrap_or(1).max(1);
    let table_index: HashMap<String, usize> = tables
        .iter()
        .enumerate()
        .map(|(i, t)| (t.name.to_ascii_lowercase(), i))
        .collect();

    let layers: Vec<Layer> = tables
        .iter()
        .enumerate()
        .map(|(ti, t)| {
            let mut attrs: Vec<(String, String)> = t
                .columns
                .iter()
                .map(|c| {
                    let ty = if c.ty.is_empty() { "?".to_string() } else { c.ty.clone() };
                    (c.name.clone(), ty)
                })
                .collect();
            if let Some(pk) = t.pk {
                attrs.push(("primary key".to_string(), t.columns[pk].name.clone()));
            }
            Layer {
                name: t.name.clone(),
                kind: LayerKind::Graph,
                nodes: vec![Node {
                    position: centers[ti],
                    weight_magnitude: (0.3
                        + 0.7 * (t.columns.len() as f32 / max_cols as f32).sqrt())
                    .clamp(0.05, 1.0),
                    color: Some(table_color(ti)),
                    label: Some(t.name.clone()),
                    attrs,
                }],
                shape: Some(table_shape(ti)),
            }
        })
        .collect();

    let edges: Vec<Edge> = tables
        .iter()
        .enumerate()
        .flat_map(|(ti, t)| {
            let table_index = &table_index;
            t.fks.iter().filter_map(move |fk| {
                table_index
                    .get(&fk.ref_table.to_ascii_lowercase())
                    .map(|&rt| Edge {
                        from_layer: ti,
                        from_node: 0,
                        to_layer: rt,
                        to_node: 0,
                        importance: 1.0,
                    })
            })
        })
        .collect();

    NetworkGraph {
        name: name.to_string(),
        layers,
        edges,
        estimated_vram_gb: None,
        moe_config: None,
        data_graph: true,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── SQL dump parsing ────────────────────────────────────────────────────

    const MYSQL_DUMP: &str = r#"
-- MySQL dump
/*!40101 SET NAMES utf8 */;
CREATE TABLE `users` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `name` VARCHAR(64) DEFAULT NULL,
  `bio` TEXT,
  PRIMARY KEY (`id`)
);
CREATE TABLE `orders` (
  `id` INT NOT NULL,
  `user_id` INT,
  `item` VARCHAR(64),
  `price` DECIMAL(10,2),
  PRIMARY KEY (`id`),
  CONSTRAINT `fk_user` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`)
);
INSERT INTO `users` (`id`, `name`, `bio`) VALUES
  (1, 'Alice', 'likes; semicolons'),
  (2, 'Bob', NULL);
INSERT INTO `orders` (`id`, `user_id`, `item`, `price`) VALUES
  (10, 1, 'Book', 12.50),
  (11, 1, 'Pen', 1.20),
  (12, 2, 'Lamp', 30.00);
"#;

    #[test]
    fn mysql_dump_parses_tables_keys_and_rows() {
        let tables = parse_sql_dump(MYSQL_DUMP).unwrap();
        assert_eq!(tables.len(), 2);
        let users = tables.iter().find(|t| t.name == "users").unwrap();
        assert_eq!(users.pk, Some(0));
        assert_eq!(users.rows.len(), 2);
        assert_eq!(users.rows[0][1].as_deref(), Some("Alice"));
        assert_eq!(
            users.rows[0][2].as_deref(),
            Some("likes; semicolons"),
            "semicolon inside a string literal must not split the statement"
        );
        assert_eq!(users.rows[1][2], None, "NULL survives as None");
        let orders = tables.iter().find(|t| t.name == "orders").unwrap();
        assert_eq!(orders.fks.len(), 1);
        assert_eq!(orders.fks[0].column, 1);
        assert_eq!(orders.fks[0].ref_table, "users");
        assert_eq!(orders.fks[0].ref_column.as_deref(), Some("id"));
        assert_eq!(orders.rows.len(), 3);
    }

    #[test]
    fn mysql_dump_becomes_a_clustered_network() {
        let tables = parse_sql_dump(MYSQL_DUMP).unwrap();
        let g = tables_to_network(tables, "shop").unwrap();
        assert!(g.data_graph);
        assert_eq!(g.layers.len(), 2, "one layer per table");
        assert_eq!(g.node_count(), 5);
        // Every FK value resolves: 3 order → user edges.
        assert_eq!(g.edges.len(), 3);

        let users_li = g.layers.iter().position(|l| l.name == "users").unwrap();
        let orders_li = g.layers.iter().position(|l| l.name == "orders").unwrap();
        // Distinct per-table shape and color.
        assert_ne!(g.layers[users_li].shape, g.layers[orders_li].shape);
        let cu = g.layers[users_li].nodes[0].color.unwrap();
        let co = g.layers[orders_li].nodes[0].color.unwrap();
        assert!(cu.iter().zip(co).any(|(a, b)| (a - b).abs() > 0.05));

        // Rows are labeled by their name column (users) or "table #pk".
        let labels: Vec<&str> = g.layers[users_li]
            .nodes
            .iter()
            .filter_map(|n| n.label.as_deref())
            .collect();
        assert!(labels.contains(&"Alice") && labels.contains(&"Bob"));
        assert!(g.layers[orders_li].nodes[0]
            .label
            .as_deref()
            .unwrap()
            .starts_with("orders #"));

        // Column values are preserved as internal data.
        let alice = g.layers[users_li]
            .nodes
            .iter()
            .find(|n| n.label.as_deref() == Some("Alice"))
            .unwrap();
        assert!(alice
            .attrs
            .iter()
            .any(|(k, v)| k == "bio" && v == "likes; semicolons"));

        // Alice (2 incoming orders) reads bigger than Bob (1 order).
        let bob = g.layers[users_li]
            .nodes
            .iter()
            .find(|n| n.label.as_deref() == Some("Bob"))
            .unwrap();
        assert!(alice.weight_magnitude > bob.weight_magnitude);

        // Table clusters occupy distinct regions of space.
        let centroid = |li: usize| -> [f32; 3] {
            let ns = &g.layers[li].nodes;
            let mut c = [0.0f32; 3];
            for n in ns {
                for (acc, p) in c.iter_mut().zip(n.position) {
                    *acc += p / ns.len() as f32;
                }
            }
            c
        };
        let (cu, co) = (centroid(users_li), centroid(orders_li));
        let d: f32 = (0..3).map(|a| (cu[a] - co[a]).powi(2)).sum::<f32>().sqrt();
        assert!(d > 1.0, "cluster centers are separated (distance {d})");
    }

    #[test]
    fn postgres_copy_blocks_and_alter_table_fk() {
        let dump = r#"
CREATE TABLE public."authors" (
    id integer PRIMARY KEY,
    full_name character varying(80)
);
CREATE TABLE public.books (
    id integer PRIMARY KEY,
    author_id integer,
    title text
);
COPY public.authors (id, full_name) FROM stdin;
1	Umberto Eco
2	Italo Calvino
\.
COPY public.books (id, author_id, title) FROM stdin;
1	1	Il nome della rosa
2	2	Il barone rampante
3	2	\N
\.
ALTER TABLE ONLY public.books
    ADD CONSTRAINT books_author_fk FOREIGN KEY (author_id) REFERENCES public.authors(id);
"#;
        let tables = parse_sql_dump(dump).unwrap();
        let authors = tables.iter().find(|t| t.name == "authors").unwrap();
        assert_eq!(authors.rows.len(), 2);
        assert_eq!(authors.rows[0][1].as_deref(), Some("Umberto Eco"));
        let books = tables.iter().find(|t| t.name == "books").unwrap();
        assert_eq!(books.rows.len(), 3);
        assert_eq!(books.rows[2][2], None, "\\N is NULL");
        assert_eq!(books.fks.len(), 1);

        let g = tables_to_network(tables, "library").unwrap();
        assert_eq!(g.edges.len(), 3);
        // Labels honor the full_name / title label columns.
        let all_labels: Vec<String> = g
            .layers
            .iter()
            .flat_map(|l| l.nodes.iter().filter_map(|n| n.label.clone()))
            .collect();
        assert!(all_labels.iter().any(|l| l == "Umberto Eco"));
        assert!(all_labels.iter().any(|l| l == "Il nome della rosa"));
    }

    #[test]
    fn schema_only_dump_falls_back_to_table_graph() {
        let ddl = r#"
CREATE TABLE a (id INT PRIMARY KEY, name TEXT);
CREATE TABLE b (id INT PRIMARY KEY, a_id INT REFERENCES a(id));
CREATE TABLE c (id INT PRIMARY KEY, b_id INT, FOREIGN KEY (b_id) REFERENCES b(id));
"#;
        let g = tables_to_network(parse_sql_dump(ddl).unwrap(), "schema").unwrap();
        assert_eq!(g.layers.len(), 3);
        assert!(g.layers.iter().all(|l| l.nodes.len() == 1));
        assert_eq!(g.edges.len(), 2, "one edge per FK relation");
        // Node internal data documents the columns.
        let a = g.layers.iter().find(|l| l.name == "a").unwrap();
        assert!(a.nodes[0].attrs.iter().any(|(k, v)| k == "name" && v == "TEXT"));
        assert!(a.nodes[0]
            .attrs
            .iter()
            .any(|(k, v)| k == "primary key" && v == "id"));
    }

    #[test]
    fn insert_without_ddl_infers_table_and_columns() {
        let dump = "INSERT INTO pets VALUES (1, 'Fido'), (2, 'Rex');";
        let tables = parse_sql_dump(dump).unwrap();
        assert_eq!(tables.len(), 1);
        assert_eq!(tables[0].columns.len(), 2);
        assert_eq!(tables[0].rows.len(), 2);
        let g = tables_to_network(tables, "pets").unwrap();
        assert_eq!(g.node_count(), 2);
    }

    #[test]
    fn splitter_survives_comments_dollar_quotes_and_functions() {
        let dump = r#"
-- comment with ; semicolon
/* block ; comment */
CREATE TABLE t (id INT PRIMARY KEY, v TEXT, ts TIMESTAMP);
INSERT INTO t (id, v, ts) VALUES (1, 'it''s; fine', NOW());
CREATE FUNCTION noise() RETURNS text AS $$ SELECT 'a;b'; $$ LANGUAGE sql;
INSERT INTO t (id, v) VALUES (-2, 'neg');
"#;
        let tables = parse_sql_dump(dump).unwrap();
        let t = tables.iter().find(|t| t.name == "t").unwrap();
        assert_eq!(t.rows.len(), 2);
        assert_eq!(t.rows[0][1].as_deref(), Some("it's; fine"));
        assert_eq!(t.rows[0][2], None, "function call value is unknown → NULL");
        assert_eq!(t.rows[1][0].as_deref(), Some("-2"));
    }

    #[test]
    fn per_table_palette_and_shapes_cycle() {
        // Colors stay distinct for a realistic table count.
        for i in 0..8 {
            for j in (i + 1)..8 {
                let (a, b) = (table_color(i), table_color(j));
                let d: f32 = (0..3).map(|k| (a[k] - b[k]).abs()).sum();
                assert!(d > 0.05, "colors {i} and {j} too close");
            }
        }
        assert_eq!(table_shape(0), GeometryType::Sphere);
        assert_eq!(table_shape(1), GeometryType::Cube);
        assert_eq!(table_shape(2), GeometryType::Plane);
        assert_eq!(table_shape(3), GeometryType::Sphere);
    }

    // ── SQLite binary fixture ───────────────────────────────────────────────

    /// Minimal SQLite writer for test fixtures: one table-leaf page per
    /// table, small records, no overflow.
    mod sqlite_writer {
        pub enum TV<'a> {
            Null,
            I(i64),
            T(&'a str),
        }

        pub fn varint(v: u64) -> Vec<u8> {
            if v <= 0x7F {
                return vec![v as u8];
            }
            let mut groups = Vec::new();
            let mut rest = v;
            while rest > 0 {
                groups.push((rest & 0x7F) as u8);
                rest >>= 7;
            }
            let last = groups.len() - 1;
            groups
                .iter()
                .rev()
                .enumerate()
                .map(|(i, g)| if i == last { *g } else { g | 0x80 })
                .collect()
        }

        pub fn record(vals: &[TV]) -> Vec<u8> {
            let mut types = Vec::new();
            let mut body = Vec::new();
            for v in vals {
                match v {
                    TV::Null => types.extend(varint(0)),
                    TV::I(i) => {
                        types.extend(varint(4));
                        body.extend_from_slice(&(*i as i32).to_be_bytes());
                    }
                    TV::T(s) => {
                        types.extend(varint(13 + 2 * s.len() as u64));
                        body.extend_from_slice(s.as_bytes());
                    }
                }
            }
            // Header length varint includes itself.
            let mut hl = types.len() + 1;
            if varint(hl as u64).len() > 1 {
                hl = types.len() + varint((types.len() + 2) as u64).len();
            }
            let mut out = varint(hl as u64);
            out.extend(types);
            out.extend(body);
            out
        }

        pub fn leaf_cell(rowid: u64, rec: &[u8]) -> Vec<u8> {
            let mut cell = varint(rec.len() as u64);
            cell.extend(varint(rowid));
            cell.extend_from_slice(rec);
            cell
        }

        pub fn leaf_page(page_size: usize, hdr: usize, cells: &[Vec<u8>]) -> Vec<u8> {
            let mut page = vec![0u8; page_size];
            page[hdr] = 13;
            page[hdr + 3..hdr + 5].copy_from_slice(&(cells.len() as u16).to_be_bytes());
            let mut content = page_size;
            let mut ptrs = Vec::new();
            for cell in cells {
                content -= cell.len();
                page[content..content + cell.len()].copy_from_slice(cell);
                ptrs.push(content as u16);
            }
            page[hdr + 5..hdr + 7].copy_from_slice(&(content as u16).to_be_bytes());
            for (i, p) in ptrs.iter().enumerate() {
                let off = hdr + 8 + i * 2;
                page[off..off + 2].copy_from_slice(&p.to_be_bytes());
            }
            page
        }

        /// Assemble a database: `tables` = (name, sql, rows).
        pub fn build(page_size: usize, tables: &[(&str, &str, Vec<Vec<TV>>)]) -> Vec<u8> {
            let master_cells: Vec<Vec<u8>> = tables
                .iter()
                .enumerate()
                .map(|(i, (name, sql, _))| {
                    let rec = record(&[
                        TV::T("table"),
                        TV::T(name),
                        TV::T(name),
                        TV::I(i as i64 + 2),
                        TV::T(sql),
                    ]);
                    leaf_cell(i as u64 + 1, &rec)
                })
                .collect();
            let mut page1 = leaf_page(page_size, 100, &master_cells);
            page1[..16].copy_from_slice(b"SQLite format 3\0");
            page1[16..18].copy_from_slice(&(page_size as u16).to_be_bytes());
            page1[18] = 1; // write version
            page1[19] = 1; // read version
            page1[21] = 64;
            page1[22] = 32;
            page1[23] = 32;
            let n_pages = 1 + tables.len() as u32;
            page1[28..32].copy_from_slice(&n_pages.to_be_bytes());
            page1[44..48].copy_from_slice(&1u32.to_be_bytes()); // schema format
            page1[56..60].copy_from_slice(&1u32.to_be_bytes()); // UTF-8

            let mut out = page1;
            for (_, _, rows) in tables {
                let cells: Vec<Vec<u8>> = rows
                    .iter()
                    .enumerate()
                    .map(|(ri, vals)| leaf_cell(ri as u64 + 1, &record(vals)))
                    .collect();
                out.extend(leaf_page(page_size, 0, &cells));
            }
            out
        }
    }

    #[test]
    fn sqlite_binary_file_roundtrips_schema_rows_and_relations() {
        use sqlite_writer::TV;
        let long_bio = "b".repeat(200); // 2-byte serial-type varint in the record
        let db = sqlite_writer::build(
            4096,
            &[
                (
                    "users",
                    "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, bio TEXT)",
                    vec![
                        vec![TV::Null, TV::T("Alice"), TV::T(&long_bio)],
                        vec![TV::Null, TV::T("Bob"), TV::Null],
                    ],
                ),
                (
                    "orders",
                    "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INT REFERENCES users(id), item TEXT)",
                    vec![
                        vec![TV::Null, TV::I(1), TV::T("Book")],
                        vec![TV::Null, TV::I(1), TV::T("Pen")],
                        vec![TV::Null, TV::I(2), TV::T("Lamp")],
                    ],
                ),
            ],
        );

        let tables = parse_sqlite(&db).unwrap();
        assert_eq!(tables.len(), 2);
        let users = tables.iter().find(|t| t.name == "users").unwrap();
        assert_eq!(users.columns.len(), 3);
        assert_eq!(users.rows.len(), 2);
        // INTEGER PRIMARY KEY is a rowid alias: NULL slot becomes the rowid.
        assert_eq!(users.rows[0][0].as_deref(), Some("1"));
        assert_eq!(users.rows[1][0].as_deref(), Some("2"));
        assert_eq!(users.rows[0][2].as_deref(), Some(long_bio.as_str()));
        assert_eq!(users.rows[1][2], None);

        let g = load_db(&db, "db", "shop").unwrap();
        assert_eq!(g.layers.len(), 2);
        assert_eq!(g.node_count(), 5);
        assert_eq!(g.edges.len(), 3, "FK values resolve through the rowid alias");
        assert_ne!(g.layers[0].shape, g.layers[1].shape);
    }

    #[test]
    fn sqlite_magic_wins_over_extension_and_bad_files_error() {
        use sqlite_writer::TV;
        let db = sqlite_writer::build(
            512,
            &[(
                "t",
                "CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)",
                vec![vec![TV::Null, TV::T("x")]],
            )],
        );
        // A SQLite file behind a .sql extension still loads (magic sniffing).
        assert!(load_db(&db, "sql", "n").is_ok());
        // Garbage with a db extension fails with a readable error.
        assert!(load_db(&[0u8; 64], "db", "n").is_err());
        assert!(load_db(b"not really sql", "sql", "n").is_err());
    }

    #[test]
    fn db_extensions_are_routed() {
        for ext in ["sql", "sqlite", "sqlite3", "db", "db3"] {
            assert!(is_db_extension(ext));
        }
        assert!(!is_db_extension("json"));
        assert!(!is_db_extension("graphml"));
    }

    #[test]
    fn huge_row_counts_are_sampled_not_refused() {
        let mut t = DbTable {
            name: "big".into(),
            columns: vec![DbColumn { name: "id".into(), ty: "INT".into() }],
            pk: Some(0),
            ..Default::default()
        };
        t.rows = (0..MAX_GRAPH_NODES + 500)
            .map(|i| vec![Some(i.to_string())])
            .collect();
        let g = tables_to_network(vec![t], "big").unwrap();
        assert!(g.node_count() <= MAX_GRAPH_NODES);
        assert!(g.node_count() > MAX_GRAPH_NODES / 2);
    }
}
