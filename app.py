import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
import tempfile
import os
import gc

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Dynamic SLA Calculator & Comparator", page_icon="🚚", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 2rem; max-width: 95%; }
    div[data-testid="metric-container"] {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 12px 16px;
    }
    .streamlit-expanderHeader { font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)

# ── Status config ─────────────────────────────────────────────────────────────
STATUS_META = {
    "Degraded" : {"bg": "#fef2f2", "icon": "▲", "color": "#dc2626"},
    "Improved" : {"bg": "#f0fdf4", "icon": "▼", "color": "#16a34a"},
    "Same"     : {"bg": "#f8fafc", "icon": "●", "color": "#6b7280"},
    "New"      : {"bg": "#eff6ff", "icon": "✦", "color": "#2563eb"},
    "Removed"  : {"bg": "#fff7ed", "icon": "✖", "color": "#ea580c"},
}
STATUS_ORDER = ["Degraded", "Improved", "Same", "New", "Removed"]

# ── State Callbacks ───────────────────────────────────────────────────────────
def reset_computation():
    for key in ["results_path", "status_counts", "total_rows", "key_cols", "val_col", "grp_col", "higher_is", "has_city"]:
        if key in st.session_state:
            del st.session_state[key]

# ── Safe Disk Writer (Zero RAM Spike) ─────────────────────────────────────────
def write_upload_to_temp(uploaded_file) -> str:
    """Safely streams an uploaded file to disk in tiny chunks to prevent RAM crashes."""
    _, ext = os.path.splitext(uploaded_file.name)
    fd, temp_path = tempfile.mkstemp(suffix=ext.lower())
    uploaded_file.seek(0)
    with os.fdopen(fd, 'wb') as f:
        while chunk := uploaded_file.read(8192):
            f.write(chunk)
    return temp_path

# ── UI Preview Helpers (Pandas) ───────────────────────────────────────────────
def clean_preview_df(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().lower() for c in df.columns]
    df.dropna(how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    return df

def get_preview_data(file, file_name: str) -> pd.DataFrame:
    """Loads a 1500-row preview using Pandas for the UI dropdowns."""
    file.seek(0)
    if file_name.lower().endswith('.parquet'): df = pd.read_parquet(file).head(1500)
    elif file_name.lower().endswith('.xlsx'): df = pd.read_excel(file, dtype=str, keep_default_na=False, nrows=1500)
    else: df = pd.read_csv(file, dtype=str, keep_default_na=False, skipinitialspace=True, encoding_errors="replace", nrows=1500)
    return clean_preview_df(df)

def match_columns(cols_a: list, cols_b: list) -> dict:
    return {c: c for c in cols_a if c in cols_b}   

def sniff_numeric(df: pd.DataFrame, col: str) -> bool:
    vals = df[col].dropna().head(1000)
    if vals.empty: return False
    return pd.to_numeric(vals, errors="coerce").notna().mean() > 0.6

# ── Core Processing Engine (POLARS - Rust Powered) ────────────────────────────
def process_with_polars(path_a, path_b, path_c, comp_mode, key_cols, val_col, grp_col, higher_is, metric_strat, sla_col, f2f_col, status_text):
    
    def load_pl(path):
        if path.endswith(".parquet"): return pl.read_parquet(path)
        return pl.read_csv(path, ignore_errors=True, infer_schema_length=10000).select(pl.all().cast(pl.Utf8))

    status_text.markdown("**⏳ Loading files into Polars Engine...**")
    df_a = load_pl(path_a)
    df_b = load_pl(path_b)
    
    # Standardize column names to lowercase
    df_a = df_a.rename({c: c.strip().lower() for c in df_a.columns})
    df_b = df_b.rename({c: c.strip().lower() for c in df_b.columns})
    
    has_city = False

    # 1. OPTIONAL FILE 3: CITY MAPPING
    if path_c:
        status_text.markdown("**⏳ Mapping Cities to Data...**")
        df_c = load_pl(path_c).rename({c: c.strip().lower() for c in load_pl(path_c).columns})
        if "city" in df_c.columns and "ph_name" in df_c.columns:
            has_city = True
            df_c = df_c.select(["ph_name", "city"]).unique("ph_name")
            
            # Left join city onto A and B where ph_name matches
            if "ph_name" in df_a.columns: df_a = df_a.join(df_c, on="ph_name", how="left")
            if "ph_name" in df_b.columns: df_b = df_b.join(df_c, on="ph_name", how="left")
            
            # Ensure city is treated as a context column to be coalesced later
            if "city" not in key_cols:
                if "city" not in df_a.columns: df_a = df_a.with_columns(pl.lit(None).alias("city"))
                if "city" not in df_b.columns: df_b = df_b.with_columns(pl.lit(None).alias("city"))

    # 2. COMPUTED SLA
    if "Compute" in metric_strat:
        status_text.markdown("**⏳ Calculating Derived SLAs...**")
        df_a = df_a.with_columns(((pl.col(sla_col).cast(pl.Float64, strict=False) - pl.col(f2f_col).cast(pl.Float64, strict=False)) / 24).round(0).alias(val_col))
        df_b = df_b.with_columns(((pl.col(sla_col).cast(pl.Float64, strict=False) - pl.col(f2f_col).cast(pl.Float64, strict=False)) / 24).round(0).alias(val_col))
    else:
        df_a = df_a.with_columns(pl.col(val_col).cast(pl.Float64, strict=False))
        df_b = df_b.with_columns(pl.col(val_col).cast(pl.Float64, strict=False))

    # 3. COMPOSITE KEYS & DEDUPLICATION
    status_text.markdown("**⏳ Generating Keys & Deduplicating...**")
    df_a = df_a.with_columns(pl.concat_str(key_cols, separator="-").alias("__key__"))
    df_b = df_b.with_columns(pl.concat_str(key_cols, separator="-").alias("__key__"))

    if "1-to-1" in comp_mode:
        df_a = df_a.unique(subset=["__key__"], keep="first")
        df_b = df_b.unique(subset=["__key__"], keep="first")

    # 4. THE OUTER JOIN (Polars handles 2M rows instantly)
    status_text.markdown("**⏳ Performing Full Comparison Join...**")
    merged = df_a.join(df_b, on="__key__", how="full", suffix="_B")

    # 5. METRICS & STATUS
    status_text.markdown("**⏳ Calculating Statuses...**")
    vA = pl.col(val_col)
    vB = pl.col(f"{val_col}_B")
    
    merged = merged.with_columns([
        (vB - vA).alias("Δ Change")
    ])

    deg_cond = vB > vA if higher_is == "higher_is_worse" else vB < vA
    imp_cond = vB < vA if higher_is == "higher_is_worse" else vB > vA

    merged = merged.with_columns(
        pl.when(vA.is_null() & vB.is_not_null()).then(pl.lit("New"))
        .when(vA.is_not_null() & vB.is_null()).then(pl.lit("Removed"))
        .when(deg_cond).then(pl.lit("Degraded"))
        .when(imp_cond).then(pl.lit("Improved"))
        .otherwise(pl.lit("Same")).alias("Status")
    )

    # 6. COALESCE CONTEXT COLUMNS
    status_text.markdown("**⏳ Finalizing Dataset...**")
    base_cols = ["__key__", val_col, f"{val_col}_B", "Δ Change", "Status"]
    if grp_col and grp_col != "(none)":
        merged = merged.with_columns(pl.coalesce([pl.col(grp_col), pl.col(f"{grp_col}_B")]).fill_null("Unknown").alias("Group"))
        base_cols.append("Group")
    else:
        merged = merged.with_columns(pl.lit("All").alias("Group"))
        base_cols.append("Group")
        
    if has_city:
        merged = merged.with_columns(pl.coalesce([pl.col("city"), pl.col("city_B")]).alias("city"))
        base_cols.append("city")

    # Rename for output
    key_display = "-".join(key_cols)
    merged = merged.rename({
        "__key__": key_display,
        val_col: f"{val_col} (File A)",
        f"{val_col}_B": f"{val_col} (File B)"
    })
    
    final_cols = [key_display, f"{val_col} (File A)", f"{val_col} (File B)", "Δ Change", "Status", "Group"]
    if has_city: final_cols.append("city")

    merged = merged.select([c for c in final_cols if c in merged.columns])

    # 7. SAVE TO DISK
    out_parquet = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet").name
    merged.write_parquet(out_parquet)
    
    status_counts = merged["Status"].value_counts().to_pandas().set_index("Status")["count"].to_dict()
    total_rows = len(merged)
    
    return out_parquet, status_counts, total_rows, has_city

# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════
st.title("🚚 Dynamic SLA Calculator & Comparator (Big Data Edition)")
st.caption("Upload files up to 2 Million rows. Powered by Polars.")

with st.container(border=True):
    st.markdown("#### 1. Upload Datasets")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**📁 File A — Baseline / Previous**")
        up_a = st.file_uploader("File A", type=["csv", "xlsx", "parquet"], key="fa", label_visibility="collapsed", on_change=reset_computation)
    with c2:
        st.markdown("**📁 File B — Current / New**")
        up_b = st.file_uploader("File B", type=["csv", "xlsx", "parquet"], key="fb", label_visibility="collapsed", on_change=reset_computation)
    
    st.markdown("---")
    st.markdown("**📁 File 3 (Optional) — City Mapping**")
    st.caption("Upload a file with `ph_name` and `city` columns to append City data for post-analysis filtering.")
    up_c = st.file_uploader("File 3", type=["csv", "xlsx", "parquet"], key="fc", label_visibility="collapsed", on_change=reset_computation)

if not (up_a and up_b):
    st.info("⬆ Upload File A and File B to begin.", icon="ℹ️")
    st.stop()

with st.spinner("Extracting headers..."):
    df_a_preview = get_preview_data(up_a, up_a.name)
    df_b_preview = get_preview_data(up_b, up_b.name)

col_map = match_columns(list(df_a_preview.columns), list(df_b_preview.columns))
common = list(col_map.keys())

if not common:
    st.error("No matching columns found between File A and File B.")
    st.stop()

with st.container(border=True):
    st.markdown("#### 2. Configure Metric & Logic")
    metric_strat = st.radio("Choose how you want to evaluate the SLA:", ["Compare an existing column", "Compute Derived SLA (Days) -> Formula: Round((Total SLA - F2F) / 24, 0)"], on_change=reset_computation)
    
    numeric_cols = [c for c in common if sniff_numeric(df_a_preview, c)]
    
    if "Compute" in metric_strat:
        mc1, mc2, mc3 = st.columns(3)
        sla_hrs_col = mc1.selectbox("⏱️ Total SLA Hours Col", options=numeric_cols or common, on_change=reset_computation)
        f2f_hrs_col = mc2.selectbox("🛑 F2F / Buffer Hours Col", options=numeric_cols or common, index=min(1, len(numeric_cols)-1), on_change=reset_computation)
        val_col = mc3.text_input("✏️ Name for Computed Column", value="computed_sla_days", on_change=reset_computation).lower()
    else:
        val_col = st.selectbox("📐 Metric to Compare", options=numeric_cols or common, on_change=reset_computation)

    st.markdown("---")
    mode_c1, mode_c2 = st.columns(2)
    comp_mode = mode_c1.radio("⚙️ Match Architecture", ["Strict 1-to-1 (Deduplicate Both)", "1-to-Many (Broadcast granular rows)"], index=0, on_change=reset_computation)
    
    st.markdown("---")
    cfg1, cfg2 = st.columns(2)
    with cfg1:
        key_cols = st.multiselect("🔑 Unique Identifier(s)", options=common, default=[common[0]] if common else [], on_change=reset_computation)
    with cfg2:
        grp_sel = st.selectbox("🗂 Group By (Optional)", options=["(none)"] + [c for c in common if c != val_col], on_change=reset_computation)
        grp_col = None if grp_sel == "(none)" else grp_sel

    higher_is = st.radio("📈 Value direction meaning", options=["higher_is_worse", "higher_is_better"], format_func=lambda x: "⬆ Higher = Worse" if x == "higher_is_worse" else "⬆ Higher = Better", horizontal=True, on_change=reset_computation)

    run_disabled = not key_cols or (metric_strat.startswith("Compute") and (not sla_hrs_col or not f2f_hrs_col or not val_col))
    run = st.button("🚀 Run Full Analysis", type="primary", width="stretch", disabled=run_disabled)

if not run and "results_path" not in st.session_state:
    st.stop()

if run:
    st.markdown("---")
    status_text = st.empty()
    
    # Write to disk to protect RAM
    status_text.markdown("**⏳ Securing files to disk to prevent RAM overflow...**")
    path_a = write_upload_to_temp(up_a)
    path_b = write_upload_to_temp(up_b)
    path_c = write_upload_to_temp(up_c) if up_c else None

    # Run Polars Engine
    out_path, counts, total, has_city = process_with_polars(path_a, path_b, path_c, comp_mode, key_cols, val_col, grp_col, higher_is, metric_strat, sla_hrs_col if "Compute" in metric_strat else None, f2f_hrs_col if "Compute" in metric_strat else None, status_text)
        
    st.session_state.update({"results_path": out_path, "status_counts": counts, "total_rows": total, "key_cols": key_cols, "val_col": val_col, "grp_col": grp_col, "higher_is": higher_is, "has_city": has_city})
    
    # Cleanup temp raw files
    os.remove(path_a); os.remove(path_b)
    if path_c: os.remove(path_c)
    gc.collect()
    status_text.empty()

results_path, status_counts, total_rows, key_cols, val_col, grp_col, higher_is, has_city = [st.session_state[k] for k in ["results_path", "status_counts", "total_rows", "key_cols", "val_col", "grp_col", "higher_is", "has_city"]]

cols_m = st.columns(6)
cols_m[0].metric("Total Rows Evaluated", f"{total_rows:,}")
for i, s in enumerate(STATUS_ORDER):
    cols_m[i+1].metric(f"{STATUS_META[s]['icon']} {s}", f"{status_counts.get(s, 0):,}")

st.markdown("")
tab_data, tab_export = st.tabs(["📋 Viewer (Preview Mode)", "💾 Filter & Download CSV"])

with tab_data:
    st.info("💡 **Big Data Mode:** Showing a 1,000-row preview to prevent browser freezing.", icon="ℹ️")
    try:
        view = pd.read_parquet(results_path).head(1000)
        st.dataframe(view, width="stretch")
    except Exception as e:
        st.error("Error loading preview.")

with tab_export:
    st.markdown("#### Extract & Download Data")
    
    try:
        # Load the massive dataset lazily to prevent RAM spike during export prep
        lazy_df = pl.scan_parquet(results_path)
        
        # City Filter UI (Only shows if File 3 was uploaded and matched successfully)
        selected_cities = []
        if has_city:
            unique_cities = lazy_df.select("city").drop_nulls().unique().collect().to_series().to_list()
            unique_cities = sorted([str(c) for c in unique_cities])
            
            st.markdown("**🏙️ Filter Output by City**")
            selected_cities = st.multiselect("Select Cities to include (Leave empty to download ALL rows):", options=unique_cities)
        
        # Generate the final CSV string dynamically based on the filter
        if st.button("🔄 Generate CSV File for Download"):
            with st.spinner("Preparing your CSV..."):
                if selected_cities:
                    final_export = lazy_df.filter(pl.col("city").is_in(selected_cities)).collect()
                else:
                    final_export = lazy_df.collect()
                    
                csv_bytes = final_export.write_csv().encode('utf-8')
                
                st.download_button(
                    label=f"⬇️ Download Output ({len(final_export):,} rows)", 
                    data=csv_bytes, 
                    file_name=f"SLA_Report_Filtered.csv", 
                    mime="text/csv", 
                    type="primary"
                )
                
    except Exception as e:
        st.error(f"Error accessing output: {e}")
