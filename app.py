import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json
from datetime import date

st.set_page_config(page_title="Fundglos Analytics Dashboard", layout="wide")

def safe_extract(df, col_name, default=np.nan):
    """Safely extract a column from a DataFrame, returning a default-filled Series if missing."""
    return df[col_name] if col_name in df.columns else pd.Series([default] * len(df), dtype='object')

@st.cache_data
def load_and_process_data(json_list):
    """Normalize JSON list and extract schema-aligned columns."""
    df = pd.json_normalize(json_list, sep='.')
    
    # 📅 Date parsing
    df['date_parsed'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
    df['date_parsed'] = df['date_parsed'].dt.tz_localize(None).dt.date
    
    # 💰 Value extraction (required)
    df['value_amount'] = pd.to_numeric(safe_extract(df, 'value.amount'), errors='coerce')
    df['value_currency'] = safe_extract(df, 'value.currency').fillna('GBP')
    
    # 🏛️ Organisation names (required)
    df['funder_name'] = safe_extract(df, 'funder.name')
    df['recipient_name'] = safe_extract(df, 'recipient.name')
    
    # 📋 Optional fields
    df['programme_title'] = safe_extract(df, 'programme.title').fillna('Unspecified Programme')
    df['activity_title'] = safe_extract(df, 'activity.title').fillna('Unspecified Activity')
    
    # 🏷️ RECIPIENT CLASSIFICATIONS
    if 'recipient.classifications' in df.columns:
        df['recipient_classifications_raw'] = df['recipient.classifications'].apply(
            lambda x: x if isinstance(x, list) else []
        )
    else:
        df['recipient_classifications_raw'] = [[] for _ in range(len(df))]
        
    # 🗺️ LOCATION EXTRACTION (Activity & Recipient)
    loc_mapping = {'country': 'ctrynm', 'region': 'rgnnm', 'utla': 'utlanm', 'lad': 'ladnm'}
    for lvl, suffix in loc_mapping.items():
        df[f'act_loc_{lvl}'] = safe_extract(df, f'activity.location.{suffix}').fillna('Unknown')
        df[f'rec_loc_{lvl}'] = safe_extract(df, f'recipient.location.{suffix}').fillna('Unknown')
            
    df['type'] = safe_extract(df, 'type').fillna('Unknown')
    return df.dropna(subset=['date_parsed', 'value_amount']).reset_index(drop=True)

def main():
    st.title("💷 Fundglos Funding Analytics Dashboard")
    st.caption("Interactive analysis powered by the Fundglos Funding Opportunity Schema")

    # 📁 Data Input
    with st.sidebar:
        st.header("📂 Data Source")
        uploaded = st.file_uploader("Upload JSONL dataset", type=["jsonl", "json"])
        if not uploaded:
            st.info("Upload a `.jsonl` file to begin.")
            return

        raw_text = uploaded.read().decode("utf-8")
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        data, parse_errors = [], []
        for i, line in enumerate(lines, 1):
            try: data.append(json.loads(line))
            except json.JSONDecodeError as e: parse_errors.append(f"Line {i}: {str(e)}")

        if parse_errors: st.warning(f"⚠️ Skipped {len(parse_errors)} malformed lines.")
        if not data: st.error("No valid JSON objects found."); return

        st.success(f"✅ Loaded `{len(data):,}` funding opportunities.")
        df = load_and_process_data(data)

    # 🔍 Sidebar Global Filters
    st.sidebar.header("🔎 Global Filters")
    valid_dates = df['date_parsed'].dropna()
    if valid_dates.empty: st.error("No valid dates found."); return
    min_dt, max_dt = valid_dates.min(), valid_dates.max()
    
    date_range = st.sidebar.date_input("Award Date Range", value=(min_dt, max_dt), min_value=min_dt, max_value=max_dt)
    start_date, end_date = (date_range, date_range) if isinstance(date_range, date) else date_range

    selected_types = st.sidebar.multiselect("Funding Type", options=df['type'].unique(), default=df['type'].unique())
    selected_funders = st.sidebar.multiselect("Funder", options=sorted(df['funder_name'].dropna().unique()), default=[])
    selected_currencies = st.sidebar.multiselect("Currency", options=df['value_currency'].unique(), default=[])

    # 🔒 Apply Global Filters
    mask = (
        (df['date_parsed'] >= start_date) & (df['date_parsed'] <= end_date) &
        (df['type'].isin(selected_types))
    )
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
    tab_temporal, tab_financial, tab_funders, tab_recipients, tab_thematic, tab_geography = st.tabs([
        "📅 Temporal", "💰 Financial", "🏛️ Funders", "🎁 Recipients", "🏷️ Thematic", "🗺️ Geography"
    ])

    # 📅 Temporal
    with tab_temporal:
        st.subheader("Funding Volume Over Time")
        df_temp = df_filtered.copy()
        df_temp['date_ts'] = pd.to_datetime(df_temp['date_parsed'])
        monthly = df_temp.set_index('date_ts').resample('ME').agg(count=('id', 'count'), total=('value_amount', 'sum')).reset_index()
        monthly.columns = ['Month', 'Opportunities', 'Total Value (£)']
        col_t1, col_t2 = st.columns(2)
        col_t1.plotly_chart(px.line(monthly, x='Month', y='Total Value (£)', markers=True), use_container_width=True)
        col_t2.plotly_chart(px.bar(monthly, x='Month', y='Opportunities'), use_container_width=True)

    # 💰 FINANCIAL TAB (ENHANCED)
    with tab_financial:
        st.subheader("💰 Financial Insights")
        st.caption("Understand funding distribution, concentration, and type differences.")
        
        # --- 1. Value Bands Analysis ---
        st.write("### 📊 Award Value Bands")
        st.caption("Most awards are small, but large awards drive most of the total value.")
        
        # Create value bands
        bins = [0, 1_000, 10_000, 50_000, 100_000, 500_000, np.inf]
        labels = ['<£1k', '£1k-10k', '£10k-50k', '£50k-100k', '£100k-500k', '£500k+']
        df_bands = df_filtered.copy()
        df_bands['value_band'] = pd.cut(df_bands['value_amount'], bins=bins, labels=labels, right=False)
        
        band_stats = df_bands.groupby('value_band', observed=True).agg(
            count=('value_amount', 'count'),
            total_value=('value_amount', 'sum')
        ).reset_index()
        
        # Dual-axis chart
        fig_bands = go.Figure()
        fig_bands.add_trace(go.Bar(
            x=band_stats['value_band'], y=band_stats['count'],
            name='Number of Awards', marker_color='#636EFA',
            yaxis='y1'
        ))
        fig_bands.add_trace(go.Scatter(
            x=band_stats['value_band'], y=band_stats['total_value'],
            name='Total Value (£)', marker_color='#EF553B', mode='lines+markers',
            yaxis='y2'
        ))
        fig_bands.update_layout(
            title='Awards by Value Band: Count vs Total Value',
            xaxis_title='Value Band',
            yaxis_title='Number of Awards',
            yaxis2=dict(title='Total Value (£)', overlaying='y', side='right', type='log'),
            legend=dict(x=0, y=1.1, orientation='h'),
            height=400
        )
        st.plotly_chart(fig_bands, use_container_width=True)
        
        # --- 2. Pareto Cumulative Curve ---
        st.write("### 📈 Funding Concentration (Pareto Analysis)")
        st.caption("What % of awards account for what % of total funding?")
        
        sorted_df = df_filtered.sort_values('value_amount', ascending=False).reset_index(drop=True)
        sorted_df['cumulative_value'] = sorted_df['value_amount'].cumsum()
        sorted_df['cumulative_pct'] = sorted_df['cumulative_value'] / sorted_df['value_amount'].sum() * 100
        sorted_df['award_rank_pct'] = (sorted_df.index + 1) / len(sorted_df) * 100
        
        fig_pareto = px.line(
            sorted_df, x='award_rank_pct', y='cumulative_pct',
            labels={'award_rank_pct': '% of Awards (largest first)', 'cumulative_pct': '% of Total Funding'},
            title='Cumulative Funding: Top X% of Awards Account for Y% of Total Value'
        )
        fig_pareto.add_hline(y=80, line_dash="dot", line_color="gray", annotation_text="80% of funding")
        fig_pareto.add_vline(x=20, line_dash="dot", line_color="gray", annotation_text="Top 20% of awards")
        fig_pareto.add_trace(go.Scatter(
            x=[0, 100], y=[0, 100], mode='lines', line=dict(dash='dash', color='gray'),
            name='Equal Distribution', showlegend=True
        ))
        fig_pareto.update_layout(height=400)
        st.plotly_chart(fig_pareto, use_container_width=True)
        
        # --- 3. Funding Type Comparison ---
        if df_filtered['type'].nunique() > 1:
            st.write("### ⚖️ Grant vs Procurement Distribution")
            st.caption("Compare award size distributions by funding type (log scale).")
            
            df_pos = df_filtered[df_filtered['value_amount'] > 0].copy()
            fig_type = px.box(
                df_pos, x='type', y='value_amount', color='type',
                labels={'value_amount': 'Award Amount (£)', 'type': 'Type'},
                title='Award Size Distribution by Funding Type',
                color_discrete_map={'grant': '#19D3F3', 'procurement': '#FF4B4B'}
            )
            fig_type.update_yaxes(type="log", title='Award Amount (£) [log scale]')
            st.plotly_chart(fig_type, use_container_width=True)
            
            # Summary stats table
            st.write("**Summary Statistics by Type**")
            type_stats = df_filtered.groupby('type')['value_amount'].describe()[['count','mean','median','min','max','std']].round(0)
            st.dataframe(
                type_stats.style.format({
                    'mean': '£{:,.0f}', 'median': '£{:,.0f}', 'min': '£{:,.0f}', 'max': '£{:,.0f}', 'std': '£{:,.0f}'
                }),
                use_container_width=True
            )

    # 🏛️ FUNDERS TAB
    with tab_funders:
        st.subheader("🏛️ Funder Analysis")
        st.caption("Charts and tables reflect your globally applied filters.")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Top 20 Funders by Funding Amount")
            top_f_amount = df_filtered.groupby('funder_name')['value_amount'].sum().nlargest(20).reset_index()
            st.plotly_chart(
                px.bar(top_f_amount, x='funder_name', y='value_amount', text_auto='.2s',
                       labels={'value_amount': 'Total Funding Awarded (£)'},
                       color_discrete_sequence=['#636EFA']), 
                use_container_width=True
            )
        with c2:
            st.subheader("Top 20 Funders by Opportunity Count")
            top_f_count = df_filtered.groupby('funder_name').size().nlargest(20).reset_index(name='count')
            st.plotly_chart(
                px.bar(top_f_count, x='funder_name', y='count', text_auto=True,
                       labels={'count': 'Number of Opportunities'},
                       color_discrete_sequence=['#EF553B']), 
                use_container_width=True
            )
            
        st.divider()
        st.subheader("📊 Detailed Funder Breakdown")
        funder_stats = df_filtered.groupby('funder_name').agg(
            total_amount=('value_amount', 'sum'),
            total_opportunities=('value_amount', 'count'),
            mean_value=('value_amount', 'mean'),
            median_value=('value_amount', 'median')
        ).reset_index()
        
        funder_stats = funder_stats.sort_values('total_amount', ascending=False).head(20)
        
        st.dataframe(
            funder_stats.style.format({
                'total_amount': '£{:,.0f}',
                'total_opportunities': '{:,.0f}',
                'mean_value': '£{:,.0f}',
                'median_value': '£{:,.0f}'
            }),
            use_container_width=True,
            hide_index=True,
            column_config={
                "funder_name": "Funder Name",
                "total_amount": "Total Funding Awarded (£)",
                "total_opportunities": "Opportunities",
                "mean_value": "Mean Award (£)",
                "median_value": "Median Award (£)"
            }
        )

    # 🎁 RECIPIENTS TAB
    with tab_recipients:
        st.subheader("🎁 Recipient Analysis")
        st.caption("Charts and tables reflect your globally applied filters.")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Top 20 Recipients by Funding Amount")
            top_r_amount = df_filtered.groupby('recipient_name')['value_amount'].sum().nlargest(20).reset_index()
            st.plotly_chart(
                px.bar(top_r_amount, x='recipient_name', y='value_amount', text_auto='.2s',
                       labels={'value_amount': 'Total Funding Received (£)'},
                       color_discrete_sequence=['#19D3F3']), 
                use_container_width=True
            )
        with c2:
            st.subheader("Top 20 Recipients by Award Count")
            top_r_count = df_filtered.groupby('recipient_name').size().nlargest(20).reset_index(name='count')
            st.plotly_chart(
                px.bar(top_r_count, x='recipient_name', y='count', text_auto=True,
                       labels={'count': 'Number of Opportunities'},
                       color_discrete_sequence=['#FF4B4B']), 
                use_container_width=True
            )
            
        st.divider()
        st.subheader("📊 Detailed Recipient Breakdown")
        recipient_stats = df_filtered.groupby('recipient_name').agg(
            total_amount=('value_amount', 'sum'),
            total_opportunities=('value_amount', 'count'),
            mean_value=('value_amount', 'mean'),
            median_value=('value_amount', 'median')
        ).reset_index()
        
        recipient_stats = recipient_stats.sort_values('total_amount', ascending=False).head(20)
        
        st.dataframe(
            recipient_stats.style.format({
                'total_amount': '£{:,.0f}',
                'total_opportunities': '{:,.0f}',
                'mean_value': '£{:,.0f}',
                'median_value': '£{:,.0f}'
            }),
            use_container_width=True,
            hide_index=True,
            column_config={
                "recipient_name": "Recipient Name",
                "total_amount": "Total Funding Received (£)",
                "total_opportunities": "Opportunities",
                "mean_value": "Mean Award (£)",
                "median_value": "Median Award (£)"
            }
        )

    # 🏷️ Thematic
    with tab_thematic:
        st.subheader("🏷️ Recipient Classifications by Scheme")
        exploded = df_filtered.explode('recipient_classifications_raw')
        if exploded['recipient_classifications_raw'].notna().any():
            scheme_df = pd.DataFrame([
                {'scheme': c.get('scheme', 'Unknown'), 'value': c.get('value', 'Unknown'), 
                 'description': c.get('description'), 'uri': c.get('uri')}
                for c in exploded['recipient_classifications_raw'].dropna() if isinstance(c, dict)
            ])
            if not scheme_df.empty:
                schemes = scheme_df['scheme'].value_counts().index.tolist()
                for scheme in schemes[:min(5, len(schemes))]:
                    with st.expander(f"📋 {scheme}", expanded=(scheme == schemes[0])):
                        sc_data = scheme_df[scheme_df['scheme'] == scheme].copy()
                        vc = sc_data.groupby(['value', 'description', 'uri']).size().reset_index(name='frequency').sort_values('frequency', ascending=False).head(50)
                        if not vc.empty:
                            if 'uri' in vc.columns and vc['uri'].notna().any():
                                vc['uri_link'] = vc.apply(lambda r: f"[🔗]({r['uri']})" if pd.notna(r['uri']) else "", axis=1)
                            cols = ['frequency', 'value']
                            if 'description' in vc.columns and vc['description'].notna().any(): cols.append('description')
                            if 'uri_link' in vc.columns: cols.append('uri_link')
                            rename_map = {'frequency': 'Count', 'value': 'Classification Value', 'description': 'Description', 'uri_link': 'Reference'}
                            st.dataframe(vc[cols].rename(columns=rename_map).style.format({'Count': '{:,.0f}'}), use_container_width=True, hide_index=True)

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
                    if not country_stats.empty:
                        st.dataframe(country_stats.sort_values('count', ascending=False), use_container_width=True)
                    else:
                        st.caption("No data available")
                with col_g2:
                    st.write("**By Region**")
                    region_stats = df_geo[df_geo['act_loc_region'] != 'Unknown'].groupby('act_loc_region').size().reset_index(name='count')
                    if not region_stats.empty:
                        st.dataframe(region_stats.sort_values('count', ascending=False).head(10), use_container_width=True)
                    else:
                        st.caption("No data available")
                
                st.write("### Interactive Map (Ward Level)")
                map_df = df_geo[df_geo['activity.location.latitude'].notna() & df_geo['activity.location.longitude'].notna()].copy()
                if not map_df.empty:
                    map_df['lat'] = pd.to_numeric(map_df['activity.location.latitude'], errors='coerce')
                    map_df['lon'] = pd.to_numeric(map_df['activity.location.longitude'], errors='coerce')
                    map_df = map_df.dropna(subset=['lat', 'lon'])
                    if not map_df.empty:
                        fig_map = px.scatter_mapbox(map_df, lat='lat', lon='lon', hover_name='activity_title', 
                                                    hover_data=['recipient_name', 'value_amount', 'type'],
                                                    color='type', size='value_amount', size_max=20, zoom=5, 
                                                    center={"lat": 54.0, "lon": -2.0}, mapbox_style="open-street-map")
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
