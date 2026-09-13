"""
VayuSurya AI — Interactive Dashboard v2.0

Upgraded for Phase 2: Metrics, Cluster Aggregation, Model Evaluation Page
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

from vayusurya_model_ import VayuSuryaForecaster, REGION_PROFILES, CLUSTER_GROUPS

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VayuSurya AI | KREDL/KSPDCL",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Inter:wght@400;500&display=swap');
html, body, .stApp { background:#060d1f !important; color:#e2e8f0; font-family:'Inter',sans-serif; }
h1,h2,h3 { color:#38bdf8 !important; font-family:'Rajdhani',sans-serif !important; }
.kpi-card { background:linear-gradient(135deg,#0f1f3d,#0a1628); border:1px solid #1e3a5f;
    border-top:3px solid #38bdf8; border-radius:10px; padding:16px 12px; text-align:center; }
.kpi-val  { font-size:1.8rem; font-weight:700; color:#38bdf8; font-family:'Rajdhani',sans-serif; }
.kpi-lbl  { font-size:0.75rem; color:#64748b; margin-top:3px; text-transform:uppercase; letter-spacing:.05em; }
.metric-card { background:#0a1628; border:1px solid #1e3a5f; border-left:4px solid #10b981;
    border-radius:8px; padding:12px 16px; margin:4px 0; }
.metric-val { font-size:1.5rem; font-weight:700; color:#10b981; }
.metric-lbl { font-size:0.75rem; color:#64748b; text-transform:uppercase; }
.alert-warn { background:#1c1000; border-left:4px solid #f59e0b; padding:10px 14px;
    border-radius:0 8px 8px 0; margin:5px 0; color:#fcd34d; font-size:.88rem; }
.alert-ok   { background:#001c0e; border-left:4px solid #10b981; padding:10px 14px;
    border-radius:0 8px 8px 0; margin:5px 0; color:#6ee7b7; font-size:.88rem; }
.alert-info { background:#001233; border-left:4px solid #38bdf8; padding:10px 14px;
    border-radius:0 8px 8px 0; margin:5px 0; color:#bae6fd; font-size:.88rem; }
div[data-testid="stSidebar"] { background:#070e20 !important; border-right:1px solid #1e3a5f; }
.stButton>button { background:linear-gradient(90deg,#0ea5e9,#2563eb); color:white;
    border:none; border-radius:8px; font-weight:600; padding:10px; }
.stTabs [data-baseweb="tab"] { color:#64748b; }
.stTabs [aria-selected="true"] { color:#38bdf8 !important; border-bottom:2px solid #38bdf8; }
</style>
""", unsafe_allow_html=True)


# ─── Cache Model ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_forecaster():
    return VayuSuryaForecaster()


# ─── Header ──────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.markdown("<div style='font-size:3.2rem;text-align:center;padding-top:8px;'>🌤️</div>", unsafe_allow_html=True)
with col_title:
    st.markdown("""
    <h1 style='font-size:2.3rem;margin-bottom:0;'>VayuSurya AI <span style='font-size:1rem;color:#475569;font-weight:400;'>v2.0</span></h1>
    <p style='color:#475569;margin-top:2px;font-size:0.92rem;'>
        Renewable Generation Forecasting &nbsp;|&nbsp; KREDL / KSPDCL Karnataka
        &nbsp;|&nbsp; <span style='color:#38bdf8;'>AI for Bharat — Theme 10</span>
    </p>
    """, unsafe_allow_html=True)

st.markdown("<hr style='border-color:#1e3a5f;margin:8px 0 12px;'>", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    view_mode = st.radio("📋 View Mode", ["Single Plant", "Cluster Aggregation"], index=0)

    if view_mode == "Single Plant":
        region = st.selectbox("📍 Region / Cluster", list(REGION_PROFILES.keys()))
        selected_regions = [region]
    else:
        cluster_name = st.selectbox("🏭 Cluster Group", list(CLUSTER_GROUPS.keys()))
        selected_regions = CLUSTER_GROUPS[cluster_name]
        st.markdown(f"<p style='color:#64748b;font-size:0.78rem;'>Includes {len(selected_regions)} plants</p>", unsafe_allow_html=True)
        region = cluster_name

    horizon = st.selectbox("⏱️ Forecast Horizon", [
        "Day-Ahead (24 hrs)", "Intra-Day (6 hrs)", "Hourly (1 hr)"
    ])
    horizon_key = {"Day-Ahead (24 hrs)":"day-ahead","Intra-Day (6 hrs)":"intra-day","Hourly (1 hr)":"hourly"}[horizon]
    confidence  = st.select_slider("📊 Confidence Band", options=[80,85,90,95,99], value=90)
    forecast_date = st.date_input("📅 Forecast Date", datetime.today()+timedelta(days=1))

    st.markdown("---")
    st.markdown("### 🌦️ Weather Override")
    use_custom = st.checkbox("Custom Weather Input")
    custom = {}
    if use_custom:
        custom["irradiance"]  = st.slider("☀️ Irradiance (W/m²)", 0, 1000, 600)
        custom["cloud_cover"] = st.slider("☁️ Cloud Cover (%)", 0, 100, 25)
        custom["temperature"] = st.slider("🌡️ Temperature (°C)", 10, 45, 28)
        custom["wind_speed"]  = st.slider("💨 Wind Speed (m/s)", 0, 25, 8)

    st.markdown("---")
    st.markdown(f"<p style='color:#334155;font-size:0.72rem;text-align:center;'>VayuSurya AI v2.0 &nbsp;|&nbsp; {datetime.now().strftime('%d %b %Y')}</p>", unsafe_allow_html=True)


# ─── Run Model ───────────────────────────────────────────────────────────────
forecaster = get_forecaster()

with st.spinner(f"🔄 Running VayuSurya AI forecast for {region}..."):
    if view_mode == "Single Plant":
        forecaster.train(selected_regions[0])
        result = forecaster.forecast(selected_regions[0], horizon=horizon_key)
        if use_custom and custom:
            for k, v in custom.items():
                if k in result["weather"].columns:
                    result["weather"][k] = v
        p50  = result["forecast_p50"]
        p10  = result["forecast_p10"]
        p90  = result["forecast_p90"]
        base = result["baseline"]
        unc  = result["uncertainty_pct"]
        act  = result["actual_simulated"]
        cap  = result["capacity_mw"]
        metrics = result["metrics"]
        wx   = result["weather"]
        shap = result["shap_importance"]
    else:
        result = forecaster.forecast_cluster(selected_regions, horizon=horizon_key)
        p50  = result["forecast_p50"]
        p10  = result["forecast_p10"]
        p90  = result["forecast_p90"]
        base = result["baseline"]
        unc  = result["uncertainty_pct"]
        act  = result["actual_simulated"]
        cap  = result["total_capacity"]
        metrics = result["metrics"]
        wx   = None
        shap = None

hours = result["hours"]
ts    = [f"{forecast_date} {h:02d}:00" for h in range(hours)]

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Forecast Dashboard",
    "📊 Model Evaluation",
    "🏭 Cluster View",
    "📥 Data Export"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — FORECAST DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    # KPI Row
    st.markdown("### 📊 Forecast Summary")
    k1,k2,k3,k4,k5 = st.columns(5)
    kpis = [
        (f"{p50.sum():.0f} MWh", "Total Generation"),
        (f"{p50.max():.0f} MW",  f"Peak @ Hour {int(np.argmax(p50))}"),
        (f"±{unc.mean():.1f}%",  "Avg Uncertainty"),
        (f"{confidence}%",        "Confidence Level"),
        (f"{cap} MW",             "Total Capacity"),
    ]
    for col, (val,lbl) in zip([k1,k2,k3,k4,k5], kpis):
        col.markdown(f'<div class="kpi-card"><div class="kpi-val">{val}</div><div class="kpi-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Main Chart + SHAP
    col_main, col_imp = st.columns([3, 1])

    with col_main:
        st.markdown("### 📈 Generation Forecast with Confidence Intervals")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ts+ts[::-1], y=list(p90)+list(p10)[::-1],
            fill="toself", fillcolor="rgba(56,189,248,0.10)",
            line=dict(color="rgba(0,0,0,0)"), name=f"{confidence}% Confidence Band",
        ))
        fig.add_trace(go.Scatter(x=ts, y=p90, mode="lines", line=dict(color="#38bdf8",width=1,dash="dot"), name="P90 (Upper)", opacity=0.6))
        fig.add_trace(go.Scatter(x=ts, y=p10, mode="lines", line=dict(color="#38bdf8",width=1,dash="dot"), name="P10 (Lower)", opacity=0.6))
        fig.add_trace(go.Scatter(x=ts, y=p50, mode="lines+markers", line=dict(color="#0ea5e9",width=2.8), marker=dict(size=5), name="P50 Forecast (MW)"))
        fig.add_trace(go.Scatter(x=ts, y=act, mode="lines", line=dict(color="#10b981",width=1.5,dash="dot"), name="Simulated Actual", opacity=0.8))
        fig.add_trace(go.Scatter(x=ts, y=base, mode="lines", line=dict(color="#f472b6",width=1.5,dash="dash"), name="Persistence Baseline"))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,22,48,0.8)", height=380,
            legend=dict(orientation="h", y=1.04, font=dict(size=10)),
            xaxis=dict(title="Time", gridcolor="#0f2040"),
            yaxis=dict(title="Generation (MW)", gridcolor="#0f2040"),
            margin=dict(l=50,r=20,t=30,b=50),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_imp:
        st.markdown("### 🔑 Feature Drivers")
        if shap:
            label_map = {
                "irradiance":"☀️ Irradiance","cloud_cover":"☁️ Cloud Cover",
                "wind_speed":"💨 Wind Speed","wind_cube":"💨 Wind Power",
                "temperature":"🌡️ Temperature","irr_cloud_interaction":"🌤️ Irr×Cloud",
                "temp_derating":"🌡️ Temp Derate","hour_sin":"🕐 Hour Sin",
                "hour_cos":"🕐 Hour Cos","wind_dir":"🧭 Wind Dir",
                "wind_dir_cos":"🧭 Wind Dir Cos","humidity":"💧 Humidity",
            }
            for feat, imp in list(shap.items())[:7]:
                name  = label_map.get(feat, feat)
                color = "#0ea5e9" if imp>15 else "#f59e0b" if imp>8 else "#64748b"
                st.markdown(f"""
                <div style="margin:5px 0;">
                  <div style="display:flex;justify-content:space-between;color:{color};font-size:0.78rem;">
                    <span>{name}</span><span>{imp:.0f}%</span>
                  </div>
                  <div style="background:#0f1f3d;border-radius:3px;height:5px;margin-top:2px;">
                    <div style="background:{color};width:{min(imp,100)}%;height:100%;border-radius:3px;"></div>
                  </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#64748b;font-size:0.85rem;'>SHAP available in Single Plant mode</p>", unsafe_allow_html=True)

    st.markdown("---")

    # Weather + Uncertainty
    if wx is not None:
        col_w, col_u = st.columns(2)
        with col_w:
            st.markdown("### 🌦️ Weather Inputs")
            fig2 = make_subplots(rows=2,cols=2,
                subplot_titles=["Irradiance (W/m²)","Cloud Cover (%)","Temperature (°C)","Wind Speed (m/s)"],
                vertical_spacing=0.18)
            for (col_,row_,key,color) in [
                (1,1,"irradiance","#fbbf24"),(2,1,"cloud_cover","#94a3b8"),
                (1,2,"temperature","#f87171"),(2,2,"wind_speed","#34d399"),
            ]:
                fig2.add_trace(go.Scatter(
                    x=list(range(hours)), y=wx[key].values,
                    mode="lines+markers", line=dict(color=color,width=2),
                    marker=dict(size=4), showlegend=False,
                ), row=row_, col=col_)
            fig2.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(10,22,48,0.8)",height=300,margin=dict(l=30,r=20,t=40,b=20))
            st.plotly_chart(fig2, use_container_width=True)

        with col_u:
            st.markdown("### 📉 Hourly Uncertainty")
            colors = ["#10b981" if u<15 else "#f59e0b" if u<25 else "#ef4444" for u in unc]
            fig3 = go.Figure(go.Bar(
                x=list(range(hours)), y=unc, marker_color=colors,
                text=[f"{u:.0f}%" for u in unc], textposition="outside",
            ))
            fig3.add_hline(y=20, line=dict(color="#f59e0b",dash="dash",width=1), annotation_text="20% Alert")
            fig3.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(10,22,48,0.8)",height=300,
                xaxis=dict(title="Hour"),yaxis=dict(title="Uncertainty (%)"),
                margin=dict(l=30,r=20,t=20,b=40))
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.markdown("### 📉 Uncertainty by Hour")
        colors = ["#10b981" if u<15 else "#f59e0b" if u<25 else "#ef4444" for u in unc]
        fig3 = go.Figure(go.Bar(
            x=list(range(hours)), y=unc, marker_color=colors,
            text=[f"{u:.0f}%" for u in unc], textposition="outside",
        ))
        fig3.add_hline(y=20, line=dict(color="#f59e0b",dash="dash",width=1), annotation_text="20% Alert")
        fig3.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,22,48,0.8)",height=280,
            xaxis=dict(title="Hour"),yaxis=dict(title="Uncertainty (%)"),
            margin=dict(l=30,r=20,t=20,b=40))
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # SHAP Full Chart
    if shap:
        st.markdown("### 🔍 Explainability — Feature Impact")
        shap_vals = {label_map.get(k,k):v for k,v in list(shap.items())[:8]}
        fig4 = go.Figure(go.Bar(
            x=list(shap_vals.values()), y=list(shap_vals.keys()),
            orientation="h",
            marker_color=["#0ea5e9" if v>0 else "#f87171" for v in shap_vals.values()],
            text=[f"{v:.1f}%" for v in shap_vals.values()], textposition="outside",
        ))
        fig4.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,22,48,0.8)",height=260,
            xaxis=dict(title="Relative Importance (%)"),
            margin=dict(l=160,r=60,t=10,b=40))
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("---")

    # Alerts
    st.markdown("### ⚠️ Operational Alerts")
    high_unc  = [i for i,u in enumerate(unc) if u>20]
    low_gen   = [i for i,f in enumerate(p50) if f < p50.mean()*0.35]
    peak_hrs  = [i for i,f in enumerate(p50) if f > p50.max()*0.9]

    if high_unc:
        st.markdown(f'<div class="alert-warn">⚠️ <strong>High Uncertainty at Hours:</strong> {high_unc[:8]} — Maintain backup reserves.</div>', unsafe_allow_html=True)
    if low_gen:
        st.markdown(f'<div class="alert-warn">🔋 <strong>Low Generation at Hours:</strong> {low_gen[:8]} — Grid scheduling adjustment recommended.</div>', unsafe_allow_html=True)
    if peak_hrs:
        st.markdown(f'<div class="alert-info">⚡ <strong>Peak Generation at Hours:</strong> {peak_hrs} — Optimal dispatch window.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="alert-ok">✅ <strong>VayuSurya AI Status:</strong> Operating normally. MAE: {metrics["MAE (MW)"]} MW | RMSE: {metrics["RMSE (MW)"]} MW | Skill Score: {metrics["Skill Score (%)"]}% improvement over baseline. Last run: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Model Performance Metrics")
    st.markdown("<p style='color:#64748b;font-size:0.85rem;'>Evaluated against simulated actual generation using persistence baseline as reference.</p>", unsafe_allow_html=True)

    # Metric cards
    m1,m2,m3,m4,m5 = st.columns(5)
    metric_colors = {
        "MAE (MW)":        ("#10b981","Lower is better"),
        "RMSE (MW)":       ("#38bdf8","Lower is better"),
        "MAPE (%)":        ("#f59e0b","Lower is better"),
        "nRMSE (%)":       ("#a78bfa","Lower is better"),
        "Skill Score (%)": ("#10b981","Higher is better"),
    }
    for col, (key, (color, hint)) in zip([m1,m2,m3,m4,m5], metric_colors.items()):
        val = metrics[key]
        col.markdown(f"""
        <div class="kpi-card" style="border-top-color:{color};">
            <div class="kpi-val" style="color:{color};">{val}</div>
            <div class="kpi-lbl">{key}</div>
            <div style="font-size:0.65rem;color:#334155;margin-top:4px;">{hint}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Actual vs Forecast Chart
    st.markdown("### 📈 Forecast vs Actual (Simulated)")
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=ts+ts[::-1], y=list(p90)+list(p10)[::-1],
        fill="toself", fillcolor="rgba(56,189,248,0.08)",
        line=dict(color="rgba(0,0,0,0)"), name="Confidence Band",
    ))
    fig5.add_trace(go.Scatter(x=ts, y=p50, mode="lines", line=dict(color="#0ea5e9",width=2.5), name="VayuSurya Forecast (P50)"))
    fig5.add_trace(go.Scatter(x=ts, y=act, mode="lines+markers", line=dict(color="#10b981",width=2), marker=dict(size=5,symbol="circle"), name="Simulated Actual"))
    fig5.add_trace(go.Scatter(x=ts, y=base, mode="lines", line=dict(color="#f472b6",width=1.5,dash="dash"), name="Persistence Baseline"))
    fig5.update_layout(
        template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,22,48,0.8)",height=350,
        legend=dict(orientation="h",y=1.04,font=dict(size=11)),
        xaxis=dict(title="Time",gridcolor="#0f2040"),
        yaxis=dict(title="Generation (MW)",gridcolor="#0f2040"),
        margin=dict(l=50,r=20,t=30,b=50),
    )
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("---")

    # Error Distribution
    col_err, col_res = st.columns(2)
    errors = act - p50

    with col_err:
        st.markdown("### 📉 Forecast Error Distribution")
        fig6 = go.Figure(go.Histogram(
            x=errors, nbinsx=15,
            marker_color="#0ea5e9", opacity=0.8,
            name="Forecast Error (MW)"
        ))
        fig6.add_vline(x=0, line=dict(color="#10b981",width=2,dash="dash"), annotation_text="Zero Error")
        fig6.update_layout(
            template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,22,48,0.8)",height=280,
            xaxis=dict(title="Error (MW)"),yaxis=dict(title="Frequency"),
            margin=dict(l=40,r=20,t=20,b=40)
        )
        st.plotly_chart(fig6, use_container_width=True)

    with col_res:
        st.markdown("### 📊 Hourly Absolute Error")
        abs_err = np.abs(errors)
        err_colors = ["#10b981" if e<5 else "#f59e0b" if e<15 else "#ef4444" for e in abs_err]
        fig7 = go.Figure(go.Bar(
            x=list(range(hours)), y=abs_err,
            marker_color=err_colors,
            text=[f"{e:.1f}" for e in abs_err], textposition="outside",
        ))
        fig7.update_layout(
            template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,22,48,0.8)",height=280,
            xaxis=dict(title="Hour"),yaxis=dict(title="|Error| (MW)"),
            margin=dict(l=30,r=20,t=20,b=40)
        )
        st.plotly_chart(fig7, use_container_width=True)

    st.markdown("---")

    # All regions benchmark
    st.markdown("### 🏆 Performance Across All Regions")
    bench_data = []
    f2 = get_forecaster()
    for reg in REGION_PROFILES.keys():
        f2.train(reg)
        r2 = f2.forecast(reg, "day-ahead")
        m2 = r2["metrics"]
        bench_data.append({
            "Region": reg,
            "Type": REGION_PROFILES[reg]["type"].capitalize(),
            "MAE (MW)": m2["MAE (MW)"],
            "RMSE (MW)": m2["RMSE (MW)"],
            "MAPE (%)": m2["MAPE (%)"],
            "Skill Score (%)": m2["Skill Score (%)"],
        })
    df_bench = pd.DataFrame(bench_data)
    st.dataframe(df_bench, use_container_width=True, hide_index=True)

    fig8 = go.Figure()
    fig8.add_trace(go.Bar(
        name="RMSE (MW)", x=df_bench["Region"], y=df_bench["RMSE (MW)"],
        marker_color="#0ea5e9",
    ))
    fig8.add_trace(go.Bar(
        name="MAE (MW)", x=df_bench["Region"], y=df_bench["MAE (MW)"],
        marker_color="#10b981",
    ))
    fig8.update_layout(
        template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,22,48,0.8)",height=300,barmode="group",
        legend=dict(orientation="h",y=1.04),
        xaxis=dict(tickangle=-20),yaxis=dict(title="Error (MW)"),
        margin=dict(l=40,r=20,t=30,b=80)
    )
    st.plotly_chart(fig8, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CLUSTER VIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🏭 Karnataka-Wide Cluster Forecast")
    st.markdown("<p style='color:#64748b;font-size:0.85rem;'>Aggregated generation forecast across all 6 Karnataka solar and wind plants — 1,000 MW total capacity.</p>", unsafe_allow_html=True)

    f3 = get_forecaster()
    with st.spinner("Aggregating all 6 Karnataka plants..."):
        cluster_result = f3.forecast_cluster(list(REGION_PROFILES.keys()), horizon_key)

    cp50 = cluster_result["forecast_p50"]
    cp10 = cluster_result["forecast_p10"]
    cp90 = cluster_result["forecast_p90"]
    cact = cluster_result["actual_simulated"]
    cbase= cluster_result["baseline"]
    cmet = cluster_result["metrics"]
    cts  = [f"{forecast_date} {h:02d}:00" for h in range(cluster_result["hours"])]

    # KPIs
    ck1,ck2,ck3,ck4 = st.columns(4)
    ckpis = [
        (f"{cp50.sum():.0f} MWh", "Total Karnataka Generation"),
        (f"{cp50.max():.0f} MW",  f"Peak @ Hour {int(np.argmax(cp50))}"),
        (f"{cluster_result['total_capacity']} MW", "Total Installed Capacity"),
        (f"{cmet['Skill Score (%)']}%", "Skill vs Baseline"),
    ]
    for col,(val,lbl) in zip([ck1,ck2,ck3,ck4], ckpis):
        col.markdown(f'<div class="kpi-card"><div class="kpi-val">{val}</div><div class="kpi-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Cluster chart
    figc = go.Figure()
    figc.add_trace(go.Scatter(
        x=cts+cts[::-1], y=list(cp90)+list(cp10)[::-1],
        fill="toself", fillcolor="rgba(56,189,248,0.10)",
        line=dict(color="rgba(0,0,0,0)"), name="Confidence Band",
    ))
    figc.add_trace(go.Scatter(x=cts,y=cp50,mode="lines+markers",line=dict(color="#0ea5e9",width=2.8),marker=dict(size=5),name="Total Forecast (MW)"))
    figc.add_trace(go.Scatter(x=cts,y=cact,mode="lines",line=dict(color="#10b981",width=1.5,dash="dot"),name="Simulated Actual"))
    figc.add_trace(go.Scatter(x=cts,y=cbase,mode="lines",line=dict(color="#f472b6",width=1.5,dash="dash"),name="Persistence Baseline"))
    figc.update_layout(
        template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,22,48,0.8)",height=380,
        legend=dict(orientation="h",y=1.04,font=dict(size=11)),
        xaxis=dict(title="Time",gridcolor="#0f2040"),
        yaxis=dict(title="Generation (MW)",gridcolor="#0f2040"),
        margin=dict(l=50,r=20,t=30,b=50),
    )
    st.plotly_chart(figc, use_container_width=True)

    st.markdown("---")

    # Per-plant contribution
    st.markdown("### 🌐 Per-Plant Contribution")
    contrib_data = []
    for reg in REGION_PROFILES.keys():
        f3.train(reg)
        r3 = f3.forecast(reg, horizon_key)
        contrib_data.append({
            "Plant": reg,
            "Type": REGION_PROFILES[reg]["type"].capitalize(),
            "Capacity (MW)": REGION_PROFILES[reg]["cap"],
            "Forecast Total (MWh)": round(r3["forecast_p50"].sum(), 1),
            "Peak (MW)": round(r3["forecast_p50"].max(), 1),
            "Avg Uncertainty (%)": round(r3["uncertainty_pct"].mean(), 1),
        })
    df_contrib = pd.DataFrame(contrib_data)
    st.dataframe(df_contrib, use_container_width=True, hide_index=True)

    # Pie chart
    col_pie1, col_pie2 = st.columns(2)
    with col_pie1:
        fig_pie = go.Figure(go.Pie(
            labels=df_contrib["Plant"],
            values=df_contrib["Forecast Total (MWh)"],
            hole=0.4,
            marker=dict(colors=["#0ea5e9","#06b6d4","#38bdf8","#10b981","#34d399","#6ee7b7"]),
        ))
        fig_pie.update_layout(
            template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
            title="Generation Share by Plant",height=320,
            margin=dict(l=20,r=20,t=40,b=20)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_pie2:
        solar_total = df_contrib[df_contrib["Type"]=="Solar"]["Forecast Total (MWh)"].sum()
        wind_total  = df_contrib[df_contrib["Type"]=="Wind"]["Forecast Total (MWh)"].sum()
        fig_pie2 = go.Figure(go.Pie(
            labels=["Solar","Wind"],
            values=[solar_total, wind_total],
            hole=0.4,
            marker=dict(colors=["#fbbf24","#0ea5e9"]),
        ))
        fig_pie2.update_layout(
            template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
            title="Solar vs Wind Mix",height=320,
            margin=dict(l=20,r=20,t=40,b=20)
        )
        st.plotly_chart(fig_pie2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DATA EXPORT
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📥 Forecast Data Export")

    base_cols = {
        "Timestamp":       ts,
        "Forecast_P50_MW": p50.round(2),
        "Forecast_P10_MW": p10.round(2),
        "Forecast_P90_MW": p90.round(2),
        "Actual_MW":       act.round(2),
        "Uncertainty_pct": unc.round(2),
        "Error_MW":        (act-p50).round(2),
    }
    if wx is not None:
        base_cols.update({
            "Irradiance_Wm2":  wx["irradiance"].values.round(1),
            "Cloud_Cover_pct": wx["cloud_cover"].values.round(1),
            "Temperature_C":   wx["temperature"].values.round(1),
            "Wind_Speed_ms":   wx["wind_speed"].values.round(2),
        })

    df_out = pd.DataFrame(base_cols)
    st.dataframe(df_out, use_container_width=True, height=300)

    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.download_button("⬇️ Download Forecast CSV", df_out.to_csv(index=False),
            file_name=f"VayuSurya_{region.replace(' ','_')}_{forecast_date}.csv",
            mime="text/csv", use_container_width=True)
    with col_d2:
        summary = f"""VayuSurya AI v2.0 — Forecast Report
=======================================
Region/Cluster : {region}
Horizon        : {horizon}
Date           : {forecast_date}
Confidence     : {confidence}%
---------------------------------------
Total Generation : {p50.sum():.1f} MWh
Peak Generation  : {p50.max():.1f} MW @ Hour {int(np.argmax(p50))}
Avg Uncertainty  : ±{unc.mean():.1f}%
---------------------------------------
Model Metrics:
  MAE        : {metrics['MAE (MW)']} MW
  RMSE       : {metrics['RMSE (MW)']} MW
  MAPE       : {metrics['MAPE (%)']}%
  nRMSE      : {metrics['nRMSE (%)']}%
  Skill Score: {metrics['Skill Score (%)']}%
---------------------------------------
Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
System    : VayuSurya AI | KREDL/KSPDCL
"""
        st.download_button("📄 Download Report TXT", summary,
            file_name=f"VayuSurya_Report_{region.replace(' ','_')}_{forecast_date}.txt",
            mime="text/plain", use_container_width=True)
    with col_d3:
        # Metrics CSV
        df_metrics = pd.DataFrame([metrics])
        df_metrics.insert(0, "Region", region)
        df_metrics.insert(1, "Date", str(forecast_date))
        st.download_button("📊 Download Metrics CSV", df_metrics.to_csv(index=False),
            file_name=f"VayuSurya_Metrics_{region.replace(' ','_')}_{forecast_date}.csv",
            mime="text/csv", use_container_width=True)
