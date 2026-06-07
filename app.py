"""
Bolo Safety Cloud — Main Application
Run locally:  streamlit run app.py
Deploy:       Streamlit Community Cloud (connect GitHub repo)

Architecture:
  Phone (ASR app) → Cloud Storage (Supabase bucket)
  Admin presses ⚡ Process → Whisper + GPT → Supabase DB
  Admin: full dashboard + process button
  Viewer: dashboard only
"""

import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime
from pipeline import run_pipeline

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bolo Safety",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# SECRETS  (.streamlit/secrets.toml locally, or Streamlit Cloud settings)
# ─────────────────────────────────────────────────────────────
SUPABASE_URL     = st.secrets["SUPABASE_URL"]
SUPABASE_KEY     = st.secrets["SUPABASE_KEY"]
OPENAI_API_KEY   = st.secrets["OPENAI_API_KEY"]
ADMIN_PASSWORD   = st.secrets["ADMIN_PASSWORD"]
VIEWER_PASSWORD  = st.secrets["VIEWER_PASSWORD"]
DROPBOX_APP_KEY       = st.secrets["DROPBOX_APP_KEY"]
DROPBOX_APP_SECRET    = st.secrets["DROPBOX_APP_SECRET"]
DROPBOX_REFRESH_TOKEN = st.secrets["DROPBOX_REFRESH_TOKEN"]
DROPBOX_FOLDER        = st.secrets.get("DROPBOX_FOLDER", "/ASR Recordings")

# ─────────────────────────────────────────────────────────────
# CACHED RESOURCES  (loaded once, reused across reruns)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_supabase():
    from supabase import create_client
    return create_client(SUPABASE_URL, SUPABASE_KEY)

@st.cache_resource
def get_whisper():
    import whisper
    return whisper.load_model("medium")

@st.cache_resource
def get_openai():
    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY)

@st.cache_resource
def get_dropbox():
    import dropbox
    return dropbox.Dropbox(
        app_key=DROPBOX_APP_KEY,
        app_secret=DROPBOX_APP_SECRET,
        oauth2_refresh_token=DROPBOX_REFRESH_TOKEN,
    )

# ─────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────
for key, default in [
    ("role",              None),
    ("run_pipeline_flag", False),
    ("pipeline_log",      []),
    ("pipeline_done",     False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────────────────────
# COLOURS
# ─────────────────────────────────────────────────────────────
CATEGORY_COLORS = {
    "Unsafe Act":       "#FFD700",
    "Unsafe Condition": "#FFA07A",
    "Near Miss":        "#87CEEB",
    "Incident":         "#FF6347",
    "Unclassified":     "#D3D3D3",
}
SEVERITY_COLORS = {
    "High":   "#FF4C4C",
    "Medium": "#FFA500",
    "Low":    "#90EE90",
}

# ─────────────────────────────────────────────────────────────
# LOGIN PAGE
# ─────────────────────────────────────────────────────────────
def show_login():
    col_l, col_m, col_r = st.columns([1, 1.2, 1])
    with col_m:
        st.image("https://img.icons8.com/color/96/hard-hat.png", width=80)
        st.title("🦺 Bolo Safety")
        st.caption("HSE Observation Intelligence Platform · Engro Corporation")
        st.divider()
        pwd = st.text_input("Enter your password", type="password")
        if st.button("Login", type="primary", use_container_width=True):
            if pwd == ADMIN_PASSWORD:
                st.session_state.role = "admin"
                st.rerun()
            elif pwd == VIEWER_PASSWORD:
                st.session_state.role = "viewer"
                st.rerun()
            else:
                st.error("Incorrect password. Please contact your HSE officer.")

# ─────────────────────────────────────────────────────────────
# DATA LOADING FROM SUPABASE DB
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_observations():
    try:
        resp = get_supabase().table("observations").select("*").order(
            "time_of_reporting", desc=False
        ).execute()
        if not resp.data:
            return pd.DataFrame()
        df = pd.DataFrame(resp.data)
        df.rename(columns={
            "time_of_reporting":   "Time of Reporting",
            "name":                "Name",
            "urdu_observation":    "Urdu Observation",
            "english_translation": "English Translation",
            "hse_categorization":  "HSE Categorization",
            "severity":            "Severity",
            "location":            "Location",
            "ai_reasoning":        "AI Reasoning",
            "audio_file":          "Audio File",
        }, inplace=True)
        df["Time of Reporting"] = pd.to_datetime(df["Time of Reporting"], errors="coerce")
        df["Date"] = df["Time of Reporting"].dt.date
        df["Hour"] = df["Time of Reporting"].dt.hour
        for col in ["HSE Categorization", "Severity", "Location", "Name"]:
            if col not in df.columns:
                df[col] = "Not specified"
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()

# ─────────────────────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────────────────────
def show_dashboard():

    # ── SIDEBAR ──────────────────────────────────────────────
    with st.sidebar:
        st.title("🦺 Bolo Safety")
        role_label = "👑 Admin" if st.session_state.role == "admin" else "👁️ Viewer"
        st.caption(f"Logged in as **{role_label}**")
        st.divider()

        # Admin-only: process button
        if st.session_state.role == "admin":
            st.markdown("### 🎙️ New Voice Recordings?")
            if st.button("⚡ Process & Refresh", type="primary", use_container_width=True):
                st.session_state.run_pipeline_flag = True
                st.session_state.pipeline_log      = []
                st.session_state.pipeline_done     = False
                st.rerun()

            if st.session_state.pipeline_log:
                if st.button("🗑️ Clear log", use_container_width=True):
                    st.session_state.pipeline_log  = []
                    st.session_state.pipeline_done = False
                    st.rerun()

            st.divider()

        # ── Filters ──────────────────────────────────────────
        df_raw = load_observations()

        if df_raw.empty:
            if st.session_state.role == "admin":
                st.info("No observations yet. Once voice notes are uploaded to the cloud bucket, press ⚡ Process & Refresh.")
            else:
                st.info("No observations available yet.")
            st.divider()
            if st.button("🔓 Logout", use_container_width=True):
                st.session_state.role = None
                st.rerun()
            st.stop()

        min_date   = df_raw["Date"].min()
        max_date   = df_raw["Date"].max()
        date_range = st.date_input("📅 Date range", value=(min_date, max_date),
                                   min_value=min_date, max_value=max_date)

        all_cats  = sorted(df_raw["HSE Categorization"].dropna().unique().tolist())
        sel_cats  = st.multiselect("🗂 Category", all_cats, default=all_cats)

        all_sevs  = ["High", "Medium", "Low", "Not specified"]
        sel_sevs  = st.multiselect("⚠️ Severity", all_sevs, default=all_sevs)

        all_names = sorted(df_raw["Name"].dropna().unique().tolist())
        sel_names = st.multiselect("👤 Reporter", all_names, default=all_names)

        st.divider()
        if st.button("🔄 Refresh charts", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.caption(f"Last loaded: {datetime.now().strftime('%H:%M:%S')}")
        st.divider()
        if st.button("🔓 Logout", use_container_width=True):
            st.session_state.role = None
            st.rerun()

    # ── FILTERING ─────────────────────────────────────────────
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range[0]

    df = df_raw[
        (df_raw["Date"] >= start_date) &
        (df_raw["Date"] <= end_date) &
        (df_raw["HSE Categorization"].isin(sel_cats)) &
        (df_raw["Severity"].isin(sel_sevs)) &
        (df_raw["Name"].isin(sel_names))
    ].copy()

    # ── HEADER ────────────────────────────────────────────────
    st.title("🦺 Bolo Safety — HSE Observation Dashboard")
    st.caption(f"Showing **{len(df)}** observations  |  {start_date} → {end_date}")

    # ── PIPELINE LOG (admin only, persists across reruns) ─────
    if st.session_state.role == "admin":
        if st.session_state.run_pipeline_flag:
            st.session_state.run_pipeline_flag = False
            with st.expander("📋 Processing Log", expanded=True):
                run_pipeline(
                    ui                    = st,
                    supabase_client       = get_supabase(),
                    whisper_model         = get_whisper(),
                    openai_client         = get_openai(),
                    dropbox_app_key       = DROPBOX_APP_KEY,
                    dropbox_app_secret    = DROPBOX_APP_SECRET,
                    dropbox_refresh_token = DROPBOX_REFRESH_TOKEN,
                    dropbox_folder        = DROPBOX_FOLDER,
                )
                st.session_state.pipeline_done = True
            st.cache_data.clear()
            st.rerun()

        if st.session_state.pipeline_log:
            with st.expander("📋 Last Processing Log", expanded=st.session_state.pipeline_done):
                for kind, msg in st.session_state.pipeline_log:
                    if kind == "error":  st.error(msg)
                    elif kind == "warn": st.warning(msg)
                    elif kind == "ok":   st.success(msg)
                    elif kind == "info": st.info(msg)
                    else:                st.write(msg)

    st.divider()

    # ── KPI CARDS ─────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("📋 Total",         len(df))
    k2.metric("🔴 Incidents",     int((df["HSE Categorization"] == "Incident").sum()))
    k3.metric("🟡 Near Misses",   int((df["HSE Categorization"] == "Near Miss").sum()))
    k4.metric("🟠 Unsafe Acts",   int((df["HSE Categorization"] == "Unsafe Act").sum()))
    k5.metric("🚨 High Severity", int((df["Severity"] == "High").sum()))
    st.divider()

    # ── ROW 1: Trend + Donut ──────────────────────────────────
    col_trend, col_donut = st.columns([3, 2])

    with col_trend:
        st.subheader("📈 Observations Over Time")
        if not df.empty:
            trend = df.groupby(["Date", "HSE Categorization"]).size().reset_index(name="Count")
            fig   = px.bar(trend, x="Date", y="Count", color="HSE Categorization",
                           color_discrete_map=CATEGORY_COLORS, barmode="stack",
                           labels={"Date": "Date", "Count": "Observations"})
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              legend_title_text="Category", margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data in selected range.")

    with col_donut:
        st.subheader("🗂 Category Breakdown")
        if not df.empty:
            cat_counts = df["HSE Categorization"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
            fig2 = px.pie(cat_counts, names="Category", values="Count", hole=0.55,
                          color="Category", color_discrete_map=CATEGORY_COLORS)
            fig2.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                               legend=dict(orientation="h", yanchor="bottom", y=-0.3))
            fig2.update_traces(textposition="outside", textinfo="percent+label")
            st.plotly_chart(fig2, use_container_width=True)

    # ── ROW 2: Severity + Location ────────────────────────────
    col_sev, col_loc = st.columns(2)

    with col_sev:
        st.subheader("⚠️ Severity Distribution")
        if not df.empty:
            sev_counts = (df["Severity"].value_counts()
                          .reindex(["High", "Medium", "Low", "Not specified"], fill_value=0)
                          .reset_index())
            sev_counts.columns = ["Severity", "Count"]
            fig3 = px.bar(sev_counts, x="Severity", y="Count", color="Severity",
                          color_discrete_map=SEVERITY_COLORS, text="Count")
            fig3.update_traces(textposition="outside")
            fig3.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
                               paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig3, use_container_width=True)

    with col_loc:
        st.subheader("📍 Top Locations Reported")
        if not df.empty:
            loc_df = df[df["Location"].notna() & (df["Location"] != "Not specified")]
            if not loc_df.empty:
                loc_counts = loc_df["Location"].value_counts().head(10).reset_index()
                loc_counts.columns = ["Location", "Count"]
                fig4 = px.bar(loc_counts, x="Count", y="Location", orientation="h",
                              color="Count", color_continuous_scale="Reds", text="Count")
                fig4.update_layout(coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)",
                                   paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=10, b=0),
                                   yaxis=dict(autorange="reversed"))
                fig4.update_traces(textposition="outside")
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.info("No location data available.")

    # ── ROW 3: Hour chart + Reporter leaderboard ──────────────
    col_hour, col_rep = st.columns(2)

    with col_hour:
        st.subheader("🕐 Reporting Activity by Hour")
        if not df.empty and df["Hour"].notna().any():
            hour_counts = df.groupby("Hour").size().reset_index(name="Count")
            fig5 = px.bar(hour_counts, x="Hour", y="Count", color="Count",
                          color_continuous_scale="Blues",
                          labels={"Hour": "Hour of Day (24h)", "Count": "Observations"})
            fig5.update_layout(coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)",
                               paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=10, b=0),
                               xaxis=dict(tickmode="linear", tick0=0, dtick=1))
            st.plotly_chart(fig5, use_container_width=True)

    with col_rep:
        st.subheader("🏆 Most Active Reporters")
        if not df.empty:
            rep_df = df[df["Name"].notna() & (df["Name"] != "Not available")]
            if not rep_df.empty:
                rep_counts = rep_df["Name"].value_counts().head(10).reset_index()
                rep_counts.columns = ["Reporter", "Observations"]
                rep_counts["Observations"] = rep_counts["Observations"].astype(int)
                rep_counts.index = range(1, len(rep_counts) + 1)
                st.dataframe(
                    rep_counts, use_container_width=True,
                    column_config={
                        "Reporter":     st.column_config.TextColumn("👤 Reporter"),
                        "Observations": st.column_config.ProgressColumn(
                            "📊 Observations", min_value=0,
                            max_value=int(rep_counts["Observations"].max())
                        ),
                    },
                )

    # ── OBSERVATION LOG TABLE ─────────────────────────────────
    st.divider()
    st.subheader("📄 Observation Log")

    search     = st.text_input("🔍 Search observations (English or Urdu)", "")
    display_df = df.copy()
    if search:
        mask = (
            display_df.get("English Translation", pd.Series(dtype=str)).str.contains(search, case=False, na=False) |
            display_df.get("Urdu Observation",    pd.Series(dtype=str)).str.contains(search, case=False, na=False) |
            display_df["Name"].str.contains(search, case=False, na=False)
        )
        display_df = display_df[mask]

    show_cols = [c for c in [
        "Time of Reporting", "Name", "HSE Categorization", "Severity",
        "Location", "Urdu Observation", "English Translation", "AI Reasoning"
    ] if c in display_df.columns]

    st.dataframe(
        display_df[show_cols].reset_index(drop=True),
        use_container_width=True, height=420,
        column_config={
            "Time of Reporting":   st.column_config.DatetimeColumn("🕐 Time", format="DD MMM YYYY, HH:mm"),
            "Name":                st.column_config.TextColumn("👤 Reporter",  width="small"),
            "HSE Categorization":  st.column_config.TextColumn("🗂 Category",  width="medium"),
            "Severity":            st.column_config.TextColumn("⚠️ Severity",  width="small"),
            "Location":            st.column_config.TextColumn("📍 Location",  width="medium"),
            "Urdu Observation":    st.column_config.TextColumn("🗣 Urdu",      width="large"),
            "English Translation": st.column_config.TextColumn("🌐 English",   width="large"),
            "AI Reasoning":        st.column_config.TextColumn("🤖 Reasoning", width="large"),
        },
    )

    csv = display_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ Download filtered data as CSV", data=csv,
                       file_name=f"bolo_safety_{datetime.today().strftime('%Y%m%d')}.csv",
                       mime="text/csv")

    st.divider()
    st.caption("Bolo Safety · Powered by OpenAI Whisper + GPT · Engro Corporation Limited")


# ─────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────
if st.session_state.role is None:
    show_login()
else:
    show_dashboard()
