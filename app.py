import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json
import urllib.request
from datetime import date
from collections import defaultdict

try:
    from rapidfuzz import process, fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

st.set_page_config(page_title="Fundglos Analytics Dashboard", layout="wide")

def safe_extract(df, col_name, default=np.nan):
    """Safely extract a column from a DataFrame, returning a default-filled Series if missing."""
    return df[col_name] if col_name in df.columns else pd.Series([default] * len(df), dtype='object')

def has_scheme_in_ids(row, target_scheme):
    """Check if an organization has a specific identifier scheme in primary or additional identifiers."""
    primary_scheme = row.get('recipient.identifier.scheme')
    if pd.notna(primary_scheme) and str(primary_scheme).strip() == target_scheme:
        return True
    add_ids = row.get('recipient.additionalIdentifiers')
    if isinstance(add_ids, list):
        for item in add_ids:
            if isinstance(item, dict) and str(item.get('scheme', '')).strip() == target_scheme:
                return True
    return False

def get_id_schemes(row):
    """Extract all identifier schemes present for a recipient."""
    schemes = set()
    primary_scheme = row.get('recipient.identifier.scheme')
    if pd.notna(primary_scheme):
        schemes.add(str(primary_scheme).strip())
    add_ids = row.get('recipient.additionalIdentifiers')
    if isinstance(add_ids, list):
        for item in add_ids:
            if isinstance(item, dict) and pd.notna(item.get('scheme')):
                schemes.add(str(item['scheme']).strip())
    return ', '.join(sorted(list(schemes))) if schemes else 'None'

def extract_chc_identifier(row):
    """Extract raw GB-CHC identifier values from primary and additional identifiers."""
    ids = []
    if pd.notna(row.get('recipient.identifier.scheme')) and str(row['recipient.identifier.scheme']).strip() == 'GB-CHC':
        ids.append(str(row.get('recipient.identifier.identifier', '')).strip())
    add_ids = row.get('recipient.additionalIdentifiers')
    if isinstance(add_ids, list):
        for item in add_ids:
            if isinstance(item, dict) and str(item.get('scheme', '')).strip() == 'GB-CHC':
                ids.append(str(item.get('identifier', '')).strip())
    return list(set(ids)) if ids else []

def format_gb_chc_id(raw_id: str) -> str:
    """Normalize an identifier to ensure it is strictly prefixed with 'GB-CHC-'."""
    if not raw_id:
        return ""
    clean = str(raw_id).strip().upper()
    clean = clean.replace("GB-CHC-", "").replace("GB-CHC:", "").replace("GB-CHC", "").strip(" -:")
    return f"GB-CHC-{clean}" if clean else ""

@st.cache_data(ttl=3600)
def fetch_live_data(url):
    """Fetch raw text from a URL."""
    with urllib.request.urlopen(url) as response:
        return response.read().decode('utf-8')

def parse_jsonl_text(raw_text):
    """Parse raw JSONL text into a list of dictionaries."""
    data, parse_errors = [], []
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    for i, line in enumerate(lines, 1):
        try: data.append(json.loads(line))
        except json.JSONDecodeError as e: parse_errors.append(f"Line {i}: {str(e)}")
    return data, parse_errors

@st.cache_data
def load_and_process_data(json_list):
    """Normalize JSON list and extract schema-aligned columns with robust cleaning."""
    df = pd.json_normalize(json_list, sep='.')
    
    # 📅 ROBUST DATE PARSING
    clean_dates = df['date'].astype(str).str.strip()
    clean_dates = clean_dates.replace({'': np.nan, 'null': np.nan, 'None': np.nan, 'TBD': np.nan, 'Unknown': np.nan})
    df['date_parsed'] = pd.to_datetime(clean_dates, utc=True, errors='coerce')
    df['date_parsed'] = df['date_parsed'].dt.tz_convert(None).dt.date
    
    # 💰 ROBUST AMOUNT PARSING
    raw_amounts = safe_extract(df, 'value.amount').astype(str).str.replace(r'[^\d.\-]', '', regex=True)
    raw_amounts = raw_amounts.replace({'': np.nan, 'null': np.nan, 'None': np.nan})
    df['value_amount'] = pd.to_numeric(raw_amounts, errors='coerce')
    
    df['value_currency'] = safe_extract(df, 'value.currency').fillna('GBP')
    
    # 🏛️ ORGANISATION EXTRACTION
    df['funder_name'] = safe_extract(df, 'funder.name')
    df['funder_scheme'] = safe_extract(df, 'funder.identifier.scheme').fillna('Unknown') # NEW: Extract funder scheme
    df['recipient_name'] = safe_extract(df, 'recipient.name')
    
    df['programme_title'] = safe_extract(df, 'programme.title').fillna('Unspecified Programme')
    df['activity_title'] = safe_extract(df, 'activity.title').fillna('Unspecified Activity')
    
    if 'recipient.classifications' in df.columns:
        df['recipient_classifications_raw'] = df['recipient.classifications'].apply(lambda x: x if isinstance(x, list) else [])
    else:
        df['recipient_classifications_raw'] = [[] for _ in range(len(df))]
        
    loc_mapping = {'country': 'ctrynm', 'region': 'rgnnm', 'utla': 'utlanm', 'lad': 'ladnm'}
    for lvl, suffix in loc_mapping.items():
        df[f'act_loc_{lvl}'] = safe_extract(df, f'activity.location.{suffix}').fillna('Unknown')
        df[f'rec_loc_{lvl}'] = safe_extract(df, f'recipient.location.{suffix}').fillna('Unknown')
            
    df['type'] = safe_extract(df, 'type').fillna('Unknown')
    return df

def main():
    st.title("💷 Fundglos Funding Analytics Dashboard")
    st.caption("Interactive analysis powered by the Fundglos Funding Opportunity Schema")

    # 📁 Data Input & Source Selection
    with st.sidebar:
        st.header("📂 Data Source")
        data_source = st.radio("Choose data source:", ["Live Data (GitHub)", "Upload Local JSONL"])
        
        data = []
        parse_errors = []
        
        if data_source == "Live Data (GitHub)":
            url = "https://raw.githubusercontent.com/OpenDataServices/fundglos-pipeline/refs/heads/main/funding-data.jsonl"
            with st.spinner("Fetching live data from GitHub..."):
                try:
                    raw_text = fetch_live_data(url)
                    data, parse_errors = parse_jsonl_text(raw_text)
                    if parse_errors: st.warning(f"⚠️ Skipped {len(parse_errors)} malformed lines in live data.")
                    if not data: st.error("No valid JSON objects found in live data."); return
                    st.success(f"✅ Loaded `{len(data):,}` funding opportunities from live source.")
                except Exception as e:
                    st.error(f"Failed to fetch live data: {e}")
                    return
        else:
            uploaded = st.file_uploader("Upload JSONL dataset", type=["jsonl", "json"])
            if not uploaded:
                st.info("Upload a `.jsonl` file to begin.")
                return
                
            raw_text = uploaded.read().decode("utf-8")
            data, parse_errors = parse_jsonl_text(raw_text)
            if parse_errors: st.warning(f"⚠️ Skipped {len(parse_errors)} malformed lines.")
            if not data: st.error("No valid JSON objects found."); return
            st.success(f"✅ Loaded `{len(data):,}` funding opportunities from uploaded file.")
        
        df_parsed = load_and_process_data(data)
        
        date_valid = df_parsed['date_parsed'].notna()
        amount_valid = df_parsed['value_amount'].notna()
        
        strict_mode = st.sidebar.checkbox(
            "Strict Mode: Exclude records with missing/invalid dates or amounts", 
            value=True,
            help="When checked, records missing core financial or date data are dropped to ensure chart accuracy."
        )
        
        if strict_mode:
            df = df_parsed.dropna(subset=['date_parsed', 'value_amount']).reset_index(drop=True)
        else:
            df = df_parsed.copy()
            df['date_parsed'] = df['date_parsed'].fillna(pd.to_datetime('1900-01-01').date())
            df['value_amount'] = df['value_amount'].fillna(0)

        with st.sidebar.expander("📊 Data Quality Breakdown"):
            st.write(f"**Total lines uploaded**: {len(df_parsed):,}")
            st.write(f"**Currently in analysis**: {len(df):,}")
            st.write(f"**Excluded**: {len(df_parsed) - len(df):,}")
            st.write("---")
            st.write("**Reasons for exclusion (if Strict Mode is ON):**")
            st.write(f"- Missing/invalid `date`: {(~date_valid).sum():,}")
            st.write(f"- Missing/invalid `value.amount`: {(~amount_valid).sum():,}")
            st.write(f"- Missing both: {(~date_valid & ~amount_valid).sum():,}")

    # 🔍 Sidebar Global Filters
    st.sidebar.header("🔎 Global Filters")
    valid_dates = df['date_parsed'].dropna()
    if valid_dates.empty: st.error("No valid dates found."); return
    min_dt, max_dt = valid_dates.min(), valid_dates.max()
    
    date_range = st.sidebar.date_input("Award Date Range", value=(min_dt, max_dt), min_value=min_dt, max_value=max_dt)
    start_date, end_date = (date_range, date_range) if isinstance(date_range, date) else date_range

    selected_types = st.sidebar.multiselect("Funding Type", options=df['type'].unique(), default=df['type'].unique())
    
    # 🔑 NEW: Funder Identifier Scheme Filter
    selected_funder_schemes = st.sidebar.multiselect(
        "Funder ID Scheme", 
        options=sorted(df['funder_scheme'].dropna().unique()), 
        default=[],
        help="Filter by the identifier scheme used by the funder (e.g., GB-CHC, GB-COH)"
    )
    
    selected_funders = st.sidebar.multiselect("Funder", options=sorted(df['funder_name'].dropna().unique()), default=[])
    selected_currencies = st.sidebar.multiselect("Currency", options=df['value_currency'].unique(), default=[])

    # 🔒 Apply Global Filters
    mask = (
        (df['date_parsed'] >= start_date) & (df['date_parsed'] <= end_date) &
        (df['type'].isin(selected_types))
    )
    if selected_funder_schemes: mask &= df['funder_scheme'].isin(selected_funder_schemes)
    if selected_funders: mask &= df['funder_name'].isin(selected_funders)
    if selected_currencies: mask &= df['value_currency'].isin(selected_currencies)
        
    df_filtered = df.loc[mask].copy()

    # 📊 KEY METRICS AT TOP OF PAGE
    st.subheader("📈 Key Metrics")
    if df_filtered.empty:
        st.warning("No records match your global filters. Try broadening your selection.")
        return
        
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Opportunities", f"{len(df_filtered):,}")
    c2.metric("Total Funding Value", f"£{df_filtered['value_amount'].sum():,.0f}")
    c3.metric("Median Award", f"£{df_filtered['value_amount'].median():,.0f}")
    c4.metric("Unique Funders", f"{df_filtered['funder_name'].nunique():,}")

    # 📑 Analysis Tabs
    tab_temporal, tab_financial, tab_funders, tab_recipients, tab_thematic, tab_geography, tab_dq = st.tabs([
        "📅 Temporal", "💰 Financial", "🏛️ Funders", "🎁 Recipients", "🏷️ Thematic", "🗺️ Geography", "🔍 Data Quality"
    ])

    with tab_temporal:
        st.subheader("Funding Volume Over Time")
        df_temp = df_filtered.copy()
        df_temp['date_ts'] = pd.to_datetime(df_temp['date_parsed'])
        monthly = df_temp.set_index('date_ts').resample('ME').agg(count=('id', 'count'), total=('value_amount', 'sum')).reset_index()
        monthly.columns = ['Month', 'Opportunities', 'Total Value (£)']
        col_t1, col_t2 = st.columns(2)
        col_t1.plotly_chart(px.line(monthly, x='Month', y='Total Value (£)', markers=True), use_container_width=True)
        col_t2.plotly_chart(px.bar(monthly, x='Month', y='Opportunities'), use_container_width=True)

    with tab_financial:
        st.subheader("💰 Financial Insights")
        st.caption("Understand funding distribution, concentration, and type differences.")
        
        st.write("### 📊 Award Value Bands")
        bins = [0, 1_000, 10_000, 50_000, 100_000, 500_000, np.inf]
        labels = ['<£1k', '£1k-10k', '£10k-50k', '£50k-100k', '£100k-500k', '£500k+']
        df_bands = df_filtered.copy()
        df_bands['value_band'] = pd.cut(df_bands['value_amount'], bins=bins, labels=labels, right=False)
        band_stats = df_bands.groupby('value_band', observed=True).agg(count=('value_amount', 'count'), total_value=('value_amount', 'sum')).reset_index()
        fig_bands = go.Figure()
        fig_bands.add_trace(go.Bar(x=band_stats['value_band'], y=band_stats['count'], name='Number of Awards', marker_color='#636EFA', yaxis='y1'))
        fig_bands.add_trace(go.Scatter(x=band_stats['value_band'], y=band_stats['total_value'], name='Total Value (£)', marker_color='#EF553B', mode='lines+markers', yaxis='y2'))
        fig_bands.update_layout(title='Awards by Value Band: Count vs Total Value', xaxis_title='Value Band', yaxis_title='Number of Awards', yaxis2=dict(title='Total Value (£)', overlaying='y', side='right', type='log'), legend=dict(x=0, y=1.1, orientation='h'), height=400)
        st.plotly_chart(fig_bands, use_container_width=True)

        st.write("### 📈 Funding Concentration (Pareto Analysis)")
        sorted_df = df_filtered.sort_values('value_amount', ascending=False).reset_index(drop=True)
        sorted_df['cumulative_pct'] = sorted_df['value_amount'].cumsum() / sorted_df['value_amount'].sum() * 100
        sorted_df['award_rank_pct'] = (sorted_df.index + 1) / len(sorted_df) * 100
        fig_pareto = px.line(sorted_df, x='award_rank_pct', y='cumulative_pct', labels={'award_rank_pct': '% of Awards (largest first)', 'cumulative_pct': '% of Total Funding'})
        fig_pareto.add_hline(y=80, line_dash="dot", line_color="gray", annotation_text="80% of funding")
        fig_pareto.add_vline(x=20, line_dash="dot", line_color="gray", annotation_text="Top 20% of awards")
        fig_pareto.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode='lines', line=dict(dash='dash', color='gray'), name='Equal Distribution', showlegend=True))
        fig_pareto.update_layout(height=400)
        st.plotly_chart(fig_pareto, use_container_width=True)

        if df_filtered['type'].nunique() > 1:
            st.write("### ⚖️ Grant vs Procurement Distribution")
            df_pos = df_filtered[df_filtered['value_amount'] > 0].copy()
            fig_type = px.box(df_pos, x='type', y='value_amount', color='type', labels={'value_amount': 'Award Amount (£)', 'type': 'Type'}, title='Award Size Distribution by Funding Type', color_discrete_map={'grant': '#19D3F3', 'procurement': '#FF4B4B'})
            fig_type.update_yaxes(type="log", title='Award Amount (£) [log scale]')
            st.plotly_chart(fig_type, use_container_width=True)

    with tab_funders:
        st.subheader("🏛️ Funder Analysis")
        c1, c2 = st.columns(2)
        with c1:
            top_f = df_filtered.groupby('funder_name')['value_amount'].sum().nlargest(20).reset_index()
            st.plotly_chart(px.bar(top_f, x='funder_name', y='value_amount', text_auto='.2s', labels={'value_amount': 'Total Funding (£)'}, color_discrete_sequence=['#636EFA']), use_container_width=True)
        with c2:
            top_f_count = df_filtered.groupby('funder_name').size().nlargest(20).reset_index(name='count')
            st.plotly_chart(px.bar(top_f_count, x='funder_name', y='count', text_auto=True, labels={'count': 'Opportunities'}, color_discrete_sequence=['#EF553B']), use_container_width=True)
        st.divider()
        st.subheader("📊 Detailed Funder Breakdown")
        funder_stats = df_filtered.groupby('funder_name').agg(total_amount=('value_amount', 'sum'), total_opportunities=('value_amount', 'count'), mean_value=('value_amount', 'mean'), median_value=('value_amount', 'median')).reset_index().sort_values('total_amount', ascending=False).head(20)
        st.dataframe(funder_stats.style.format({'total_amount': '£{:,.0f}', 'total_opportunities': '{:,.0f}', 'mean_value': '£{:,.0f}', 'median_value': '£{:,.0f}'}), use_container_width=True, hide_index=True, column_config={"funder_name": "Funder Name", "total_amount": "Total Funding (£)", "total_opportunities": "Opportunities", "mean_value": "Mean Award (£)", "median_value": "Median Award (£)"})

    with tab_recipients:
        st.subheader("🎁 Recipient Analysis")
        c1, c2 = st.columns(2)
        with c1:
            top_r = df_filtered.groupby('recipient_name')['value_amount'].sum().nlargest(20).reset_index()
            st.plotly_chart(px.bar(top_r, x='recipient_name', y='value_amount', text_auto='.2s', labels={'value_amount': 'Total Funding (£)'}, color_discrete_sequence=['#19D3F3']), use_container_width=True)
        with c2:
            top_r_count = df_filtered.groupby('recipient_name').size().nlargest(20).reset_index(name='count')
            st.plotly_chart(px.bar(top_r_count, x='recipient_name', y='count', text_auto=True, labels={'count': 'Opportunities'}, color_discrete_sequence=['#FF4B4B']), use_container_width=True)
        st.divider()
        st.subheader("📊 Detailed Recipient Breakdown")
        recipient_stats = df_filtered.groupby('recipient_name').agg(total_amount=('value_amount', 'sum'), total_opportunities=('value_amount', 'count'), mean_value=('value_amount', 'mean'), median_value=('value_amount', 'median')).reset_index().sort_values('total_amount', ascending=False).head(20)
        st.dataframe(recipient_stats.style.format({'total_amount': '£{:,.0f}', 'total_opportunities': '{:,.0f}', 'mean_value': '£{:,.0f}', 'median_value': '£{:,.0f}'}), use_container_width=True, hide_index=True, column_config={"recipient_name": "Recipient Name", "total_amount": "Total Funding (£)", "total_opportunities": "Opportunities", "mean_value": "Mean Award (£)", "median_value": "Median Award (£)"})

    # 🏷️ THEMATIC TAB
    with tab_thematic:
        st.subheader("🏷️ Recipient Classifications by Scheme")
        st.caption("Breakdown of funding by recipient classification scheme, including total funding value.")
        
        exploded = df_filtered.explode('recipient_classifications_raw')
        valid_mask = exploded['recipient_classifications_raw'].apply(lambda x: isinstance(x, dict))
        valid_exploded = exploded[valid_mask].copy()
        
        if not valid_exploded.empty:
            class_details = pd.json_normalize(valid_exploded['recipient_classifications_raw'])
            scheme_df = pd.concat([
                valid_exploded[['value_amount']].reset_index(drop=True),
                class_details.reset_index(drop=True)
            ], axis=1)
            
            if 'scheme' not in scheme_df.columns: scheme_df['scheme'] = 'Unknown'
            if 'value' not in scheme_df.columns: scheme_df['value'] = 'Unknown'
                
            scheme_df['scheme'] = scheme_df['scheme'].fillna('Unknown')
            scheme_df['value'] = scheme_df['value'].fillna('Unknown')
            
            schemes = scheme_df['scheme'].value_counts().index.tolist()
            
            for scheme in schemes:
                with st.expander(f"📋 {scheme}", expanded=(scheme == schemes[0])):
                    sc_data = scheme_df[scheme_df['scheme'] == scheme].copy()
                    sc_data['description'] = sc_data['description'].fillna('')
                    sc_data['uri'] = sc_data['uri'].fillna('')
                    
                    has_descriptions = (sc_data['description'] != '').any()
                    has_uris = (sc_data['uri'] != '').any()
                    
                    vc = sc_data.groupby(['value', 'description', 'uri'], dropna=False).agg(
                        frequency=('value_amount', 'count'),
                        total_value=('value_amount', 'sum')
                    ).reset_index().sort_values('frequency', ascending=False).head(50)
                    
                    if not vc.empty:
                        if has_uris:
                            vc['uri_link'] = vc.apply(lambda r: f"[🔗]({r['uri']})" if str(r['uri']).strip() != '' else "", axis=1)
                        
                        cols = ['frequency', 'total_value', 'value']
                        if has_descriptions: cols.append('description')
                        if has_uris: cols.append('uri_link')
                        
                        rename_map = {
                            'frequency': 'Opportunities', 
                            'total_value': 'Total Funding (£)',
                            'value': 'Classification Value', 
                            'description': 'Description', 
                            'uri_link': 'Reference'
                        }
                        
                        st.dataframe(
                            vc[cols].rename(columns=rename_map).style.format({
                                'Opportunities': '{:,.0f}',
                                'Total Funding (£)': '£{:,.0f}'
                            }), 
                            use_container_width=True, 
                            hide_index=True
                        )
                    else:
                        st.info(f"No data available for scheme: {scheme}")
        else:
            st.info("No recipient classifications found in this filtered dataset.")

    # 🗺️ GEOGRAPHY TAB
    with tab_geography:
        st.subheader("🗺️ Geographic Analysis")
        st.caption("Filters below are dynamically limited to locations present in your globally filtered dataset.")
        
        for key in ['act_region', 'act_utla', 'act_lad', 'rec_region', 'rec_utla', 'rec_lad']: 
            st.session_state.setdefault(key, [])
            
        act_regions = sorted(df_filtered[df_filtered['act_loc_region'] != 'Unknown']['act_loc_region'].unique())
        act_utlas = sorted(df_filtered[df_filtered['act_loc_utla'] != 'Unknown']['act_loc_utla'].unique())
        act_lads = sorted(df_filtered[df_filtered['act_loc_lad'] != 'Unknown']['act_loc_lad'].unique())
        rec_regions = sorted(df_filtered[df_filtered['rec_loc_region'] != 'Unknown']['rec_loc_region'].unique())
        rec_utlas = sorted(df_filtered[df_filtered['rec_loc_utla'] != 'Unknown']['rec_loc_utla'].unique())
        rec_lads = sorted(df_filtered[df_filtered['rec_loc_lad'] != 'Unknown']['rec_loc_lad'].unique())

        has_any_geo = any(len(x) > 0 for x in [act_regions, act_utlas, act_lads, rec_regions, rec_utlas, rec_lads])
        
        if not has_any_geo:
            st.info("📭 No geographic location data is available in your currently filtered dataset.")
        else:
            with st.expander("📍 Filter by Activity Location", expanded=False):
                st.session_state.act_region = st.multiselect("Region", options=act_regions, default=[x for x in st.session_state.act_region if x in act_regions], key="act_region_w")
                st.session_state.act_utla = st.multiselect("County/Unitary", options=act_utlas, default=[x for x in st.session_state.act_utla if x in act_utlas], key="act_utla_w")
                st.session_state.act_lad = st.multiselect("District", options=act_lads, default=[x for x in st.session_state.act_lad if x in act_lads], key="act_lad_w")

            with st.expander("🏢 Filter by Recipient Location", expanded=False):
                st.session_state.rec_region = st.multiselect("Region", options=rec_regions, default=[x for x in st.session_state.rec_region if x in rec_regions], key="rec_region_w")
                st.session_state.rec_utla = st.multiselect("County/Unitary", options=rec_utlas, default=[x for x in st.session_state.rec_utla if x in rec_utlas], key="rec_utla_w")
                st.session_state.rec_lad = st.multiselect("District", options=rec_lads, default=[x for x in st.session_state.rec_lad if x in rec_lads], key="rec_lad_w")

            df_geo = df_filtered.copy()
            if st.session_state.act_region: df_geo = df_geo[df_geo['act_loc_region'].isin(st.session_state.act_region)]
            if st.session_state.act_utla: df_geo = df_geo[df_geo['act_loc_utla'].isin(st.session_state.act_utla)]
            if st.session_state.act_lad: df_geo = df_geo[df_geo['act_loc_lad'].isin(st.session_state.act_lad)]
            if st.session_state.rec_region: df_geo = df_geo[df_geo['rec_loc_region'].isin(st.session_state.rec_region)]
            if st.session_state.rec_utla: df_geo = df_geo[df_geo['rec_loc_utla'].isin(st.session_state.rec_utla)]
            if st.session_state.rec_lad: df_geo = df_geo[df_geo['rec_loc_lad'].isin(st.session_state.rec_lad)]
            
            if df_geo.empty:
                st.warning("No records match your location filters.")
            else:
                st.write("### Activity Location Breakdown")
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    st.write("**By Country**")
                    country_stats = df_geo[df_geo['act_loc_country'] != 'Unknown'].groupby('act_loc_country').size().reset_index(name='count')
                    if not country_stats.empty: st.dataframe(country_stats.sort_values('count', ascending=False), use_container_width=True)
                    else: st.caption("No data available")
                with col_g2:
                    st.write("**By Region**")
                    region_stats = df_geo[df_geo['act_loc_region'] != 'Unknown'].groupby('act_loc_region').size().reset_index(name='count')
                    if not region_stats.empty: st.dataframe(region_stats.sort_values('count', ascending=False).head(10), use_container_width=True)
                    else: st.caption("No data available")
                
                st.write("### Interactive Map (Ward Level)")
                map_df = df_geo[df_geo['activity.location.latitude'].notna() & df_geo['activity.location.longitude'].notna()].copy()
                if not map_df.empty:
                    map_df['lat'] = pd.to_numeric(map_df['activity.location.latitude'], errors='coerce')
                    map_df['lon'] = pd.to_numeric(map_df['activity.location.longitude'], errors='coerce')
                    map_df = map_df.dropna(subset=['lat', 'lon'])
                    if not map_df.empty:
                        fig_map = px.scatter_mapbox(
                            map_df, lat='lat', lon='lon', hover_name='activity_title', 
                            hover_data=['recipient_name', 'value_amount', 'type'],
                            color='type', size='value_amount', size_max=20, zoom=5, 
                            center={"lat": 54.0, "lon": -2.0}, mapbox_style="open-street-map"
                        )
                        fig_map.update_layout(height=500)
                        st.plotly_chart(fig_map, use_container_width=True)
                    else: 
                        st.info("⚠️ Coordinates parsed but no valid lat/lon pairs found.")
                else: 
                    st.info("ℹ️ Ward-level latitude/longitude coordinates not present in this dataset.")
                        
                st.write("### Local Authority Districts")
                lad_stats = df_geo[df_geo['act_loc_lad'] != 'Unknown'].groupby('act_loc_lad').size().reset_index(name='count')
                if not lad_stats.empty: 
                    st.dataframe(lad_stats.sort_values('count', ascending=False).head(15), use_container_width=True)
                else: 
                    st.caption("No district-level data available")

    # 🔍 DATA QUALITY TAB
    with tab_dq:
        st.subheader("🔍 Data Quality: Curate GB-CHC Identifier Mappings")
        
        if 'mappings' not in st.session_state:
            st.session_state.mappings = []
        if 'added_mappings' not in st.session_state:
            st.session_state.added_mappings = set()

        st.write("### 📋 Recipients Missing GB-CHC Identifiers")
        check_df = df.copy()
        check_df['has_gb_chc'] = check_df.apply(lambda row: has_scheme_in_ids(row, 'GB-CHC'), axis=1)
        check_df['id_schemes_present'] = check_df.apply(get_id_schemes, axis=1)
        
        missing_gb = check_df[~check_df['has_gb_chc']].copy()
        
        if not missing_gb.empty:
            gb_missing_grouped = missing_gb.groupby('recipient_name').agg(
                opportunities=('value_amount', 'count'),
                total_value=('value_amount', 'sum'),
                schemes_present=('id_schemes_present', 'first'),
                example_id=('recipient.identifier.identifier', 'first')
            ).reset_index().sort_values('total_value', ascending=False)

            st.dataframe(
                gb_missing_grouped.style.format({'total_value': '£{:,.0f}', 'opportunities': '{:,.0f}'}),
                use_container_width=True, hide_index=True,
                column_config={
                    "recipient_name": "Recipient Name", "opportunities": "Opportunities",
                    "total_value": "Total Value (£)", "schemes_present": "Other ID Schemes Found", "example_id": "Sample Identifier ID"
                }
            )

        st.divider()
        st.write("### 🛠️ Custom Mappings Workspace")
        chc_refs = []
        has_gb = check_df[check_df['has_gb_chc']].copy()
        for _, row in has_gb.iterrows():
            name = row['recipient_name']
            ids = extract_chc_identifier(row)
            for chc_id in ids:
                chc_refs.append({'name': name, 'chc_id': chc_id})

        name_to_ids = defaultdict(list)
        for ref in chc_refs:
            clean_id = format_gb_chc_id(ref['chc_id'])
            if clean_id and clean_id not in name_to_ids[ref['name']]:
                name_to_ids[ref['name']].append(clean_id)

        chc_ref_list = [{'name': name, 'chc_ids': ids} for name, ids in name_to_ids.items()]
        chc_names = [ref['name'] for ref in chc_ref_list]
        missing_names = sorted(gb_missing_grouped['recipient_name'].unique())

        if missing_names and RAPIDFUZZ_AVAILABLE:
            target_name = st.selectbox("1. Select a recipient missing a GB-CHC identifier:", missing_names)
            if target_name:
                st.write(f"**2. Potential Matches for '{target_name}':**")
                matches = process.extract(target_name, chc_names, scorer=fuzz.ratio, limit=10)
                
                seen_map_keys = set()
                for i, (matched_name, score, _) in enumerate(matches):
                    ids = next((ref['chc_ids'] for ref in chc_ref_list if ref['name'] == matched_name), [])
                    unique_ids = list(dict.fromkeys(ids))
                    for chc_id in unique_ids:
                        map_key = f"{target_name}|||{chc_id}"
                        if map_key in seen_map_keys: continue
                        seen_map_keys.add(map_key)
                        
                        already_added = map_key in st.session_state.added_mappings
                        if already_added:
                            col_chk, col_rem = st.columns([4, 1])
                            with col_chk: st.markdown(f"✅ **{matched_name}** (Score: {score}%) → `{chc_id}`")
                            with col_rem:
                                if st.button("Remove", key=f"rem_{map_key}"):
                                    st.session_state.added_mappings.discard(map_key)
                                    st.session_state.mappings = [m for m in st.session_state.mappings if not (m['name'] == target_name and m['org_id'] == chc_id)]
                                    st.rerun()
                        else:
                            if st.checkbox(f"Add mapping: **{matched_name}** (Score: {score}%) → `{chc_id}`", key=f"chk_{map_key}"):
                                st.session_state.added_mappings.add(map_key)
                                st.session_state.mappings.append({"name": target_name, "org_id": chc_id})
                                st.success(f"Added: {target_name} → {chc_id}")
                                st.rerun()
                
                st.markdown("---")
                st.write("**3. Manual Entry:**")
                col_man1, col_man2 = st.columns([3, 1])
                with col_man1:
                    manual_id = st.text_input("Enter GB-CHC Identifier manually:", placeholder="e.g., 1234567", label_visibility="collapsed")
                with col_man2:
                    if st.button("Add Manual Mapping", type="primary"):
                        if manual_id.strip():
                            clean_id = format_gb_chc_id(manual_id)
                            map_key = f"{target_name}|||{clean_id}"
                            if map_key not in st.session_state.added_mappings:
                                st.session_state.added_mappings.add(map_key)
                                st.session_state.mappings.append({"name": target_name, "org_id": clean_id})
                                st.success(f"Added manual mapping: {target_name} → {clean_id}")
                                st.rerun()

        st.divider()
        st.subheader("📋 Your Curated Mappings")
        if st.session_state.mappings:
            df_mappings = pd.DataFrame(st.session_state.mappings)[['name', 'org_id']].drop_duplicates()
            st.dataframe(df_mappings, use_container_width=True, hide_index=True, column_config={"name": "Organization Name", "org_id": "GB-CHC Identifier"})
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                st.download_button(label="📥 Download Mappings (CSV)", data=df_mappings.to_csv(index=False).encode('utf-8'), file_name="gb_chc_custom_mappings.csv", mime="text/csv")
            with col_btn2:
                if st.button("🗑️ Clear All Mappings"):
                    st.session_state.mappings = []
                    st.session_state.added_mappings = set()
                    st.rerun()
        else:
            st.info("No custom mappings added yet.")

        st.divider()
        st.subheader("🚫 Excluded Records Analysis")
        excluded_mask = df_parsed['date_parsed'].isna() | df_parsed['value_amount'].isna()
        df_excluded = df_parsed[excluded_mask].copy()
        
        if not df_excluded.empty:
            def get_exclusion_reason(row):
                date_bad = pd.isna(row['date_parsed'])
                amount_bad = pd.isna(row['value_amount'])
                if date_bad and amount_bad: return "Missing/Invalid Date & Amount"
                elif date_bad: return "Missing/Invalid Date"
                else: return "Missing/Invalid Amount"
                
            df_excluded['Exclusion Reason'] = df_excluded.apply(get_exclusion_reason, axis=1)
            df_excluded['Raw Date Value (Debug)'] = df_excluded['date'].apply(lambda x: repr(x) if pd.notna(x) else str(x))
            
            display_cols = ['id', 'recipient_name', 'funder_name', 'Raw Date Value (Debug)', 'value.amount', 'Exclusion Reason']
            df_display = df_excluded[display_cols].rename(columns={
                'id': 'Opportunity ID', 'recipient_name': 'Recipient', 'funder_name': 'Funder',
                'value.amount': 'Raw Amount Value', 'Exclusion Reason': 'Reason for Exclusion'
            })
            
            st.write(f"**Total Excluded Records:** {len(df_excluded):,}")
            st.dataframe(df_display.head(100), use_container_width=True, hide_index=True)
            
            if len(df_excluded) > 100:
                st.info(f"Showing first 100 of {len(df_excluded):,} excluded records. Download the CSV below for the full list.")
                
            st.download_button(
                label="📥 Download All Excluded Records (CSV)",
                data=df_display.to_csv(index=False).encode('utf-8'),
                file_name="excluded_records_data_quality.csv",
                mime="text/csv"
            )
        else:
            st.success("✅ No records were excluded due to parsing errors.")

    # 📉 Data Completeness Footer
    st.divider()
    unique_recs = df_filtered['recipient_name'].nunique()
    recs_with_class = df_filtered[df_filtered['recipient_classifications_raw'].apply(len) > 0]['recipient_name'].nunique()
    pct_recs_with_class = (recs_with_class / unique_recs * 100) if unique_recs > 0 else 0
    
    st.caption("🔍 Data Quality: "
               f"Activity Geo {(df_filtered['act_loc_region'] != 'Unknown').mean()*100:.1f}% | "
               f"Recipient Geo {(df_filtered['rec_loc_region'] != 'Unknown').mean()*100:.1f}% | "
               f"Recipients with Classifications: {pct_recs_with_class:.1f}%")

if __name__ == "__main__":
    main()
