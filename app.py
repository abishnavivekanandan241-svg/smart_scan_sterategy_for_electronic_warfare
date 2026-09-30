import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.preprocessing.tsrd_adapter import TSRDAdapter
from src.preprocessing.pdw_processor import PDWPreprocessor
from src.state.band_state import BandState
from src.schedulers.apss import APSScheduler
from src.schedulers.round_robin import RoundRobinScheduler
from src.schedulers.random_scheduler import RandomScheduler
import time

st.set_page_config(page_title="APSS Dashboard", layout="wide")

st.title("APSS — Adaptive Predictive Spectrum Scheduler")
st.markdown("SIH Problem Statement: 26055 — Smart Scan Strategy for Electronic Warfare")

@st.cache_data
def load_data():
    adapter = TSRDAdapter()
    pdws = adapter.convert_to_pdws("data/tsrd/tsrd_50k.csv")
    pdws.sort(key=lambda p: p.toa)
    return pdws[:5000] # Use a 5000-PDW subset for fast dashboard response

pdws = load_data()

num_bands = 100
band_width = 170.0
start_freq = 0.0
window_size = 5000.0

st.sidebar.header("Controls")
view_mode = st.sidebar.radio("Select View", [
    "Overall Dashboard",
    "Spectrum Visualization",
    "APSS Decision",
    "Scheduler Comparison",
    "Feedback & State Update"
])

def run_simulation(scheduler, name):
    processor = PDWPreprocessor(num_bands=num_bands, band_width=band_width, start_freq=start_freq)
    states = [BandState(band_id=i) for i in range(num_bands)]
    
    total_scans = 0
    total_hits = 0
    latency_records = []
    scans_per_band = {i: 0 for i in range(num_bands)}
    pending_pulses = {i: [] for i in range(num_bands)}
    
    decision_history = []
    
    for start_t, end_t, batch_pdws in processor.batch_by_time_window(pdws, window_size=window_size):
        for s in states:
            s.update_time(end_t)
            
        for p in batch_pdws:
            b_id = processor.get_band_id(p.centre_frequency)
            if 0 <= b_id < num_bands:
                pending_pulses[b_id].append(p.toa)
                
        selected_band = scheduler.select_next(states)
        
        if selected_band == -1:
            continue
            
        band_stats = processor.process_batch(batch_pdws)
        stats = band_stats.get(selected_band, {"pulse_count": 0, "activity": 0.0})
        is_hit = stats.get("pulse_count", 0) > 0
        
        if pending_pulses[selected_band]:
            for p_toa in pending_pulses[selected_band]:
                latency_records.append(end_t - p_toa)
            pending_pulses[selected_band].clear()
            
        # Record state for visualization
        sel_state = states[selected_band]
        decision_history.append({
            "window_end": end_t,
            "selected_band": selected_band,
            "activity": sel_state.activity,
            "persistence": sel_state.persistence,
            "novelty": sel_state.novelty,
            "uncertainty": sel_state.uncertainty,
            "coverage_debt": sel_state.coverage_debt,
            "miss_risk": sel_state.miss_risk,
            "scan_cost": sel_state.scan_cost,
            "is_hit": is_hit,
            "pulses_found": stats.get("pulse_count", 0)
        })
        
        states[selected_band].update_from_pdw_stats(end_t, stats)
        
        total_scans += 1
        scans_per_band[selected_band] += 1
        if is_hit:
            total_hits += 1
            
    unique_bands = len([b for b, c in scans_per_band.items() if c > 0])
    coverage = (unique_bands / num_bands) * 100
    detection_rate = (total_hits / total_scans) * 100 if total_scans > 0 else 0
    avg_latency = sum(latency_records) / len(latency_records) if latency_records else 0.0
    
    return {
        "name": name,
        "scans": total_scans,
        "hits": total_hits,
        "detection_rate": detection_rate,
        "coverage": coverage,
        "unique_bands": unique_bands,
        "avg_latency": avg_latency,
        "history": decision_history,
        "scans_per_band": scans_per_band
    }

if view_mode == "Overall Dashboard":
    st.header("Overall APSS Dashboard")
    st.markdown("This dashboard tracks the simulated evaluation of the APSS algorithm against real TSRD observation data.")
    
    with st.spinner("Running APSS..."):
        apss_res = run_simulation(APSScheduler(), "APSS")
        
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Scans", apss_res['scans'])
    col2.metric("Simulated Hits", apss_res['hits'])
    col3.metric("Simulated Detection Rate", f"{apss_res['detection_rate']:.2f}%")
    col4.metric("Coverage", f"{apss_res['coverage']:.1f}%")
    
    st.subheader("Data Overview")
    st.write(f"Using TSRD dataset sample: {len(pdws)} PDWs")
    st.write("Temporal Window Size: 5000 units")
    st.write("Number of Bands: 100")
    
elif view_mode == "Spectrum Visualization":
    st.header("Frequency Band Activity Visualization")
    
    processor = PDWPreprocessor(num_bands=num_bands, band_width=band_width, start_freq=start_freq)
    all_stats = processor.process_batch(pdws)
    
    bands = list(range(num_bands))
    activities = [all_stats.get(b, {}).get('pulse_count', 0) for b in bands]
    
    fig = px.bar(x=bands, y=activities, labels={'x': 'Band ID', 'y': 'Total PDWs (Activity)'},
                 title="Total Ground Truth Activity per Frequency Band")
    st.plotly_chart(fig, use_container_width=True)

elif view_mode == "APSS Decision":
    st.header("APSS Decision Logic")
    with st.spinner("Running APSS..."):
        apss_res = run_simulation(APSScheduler(), "APSS")
    
    history = apss_res["history"]
    if history:
        st.subheader("Decision Audit Log")
        df = pd.DataFrame(history)
        st.dataframe(df)
        
        st.subheader("Decision Breakdown (Last Scan)")
        last = history[-1]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Selected Band", last['selected_band'])
            st.metric("Hit Result", "HIT" if last['is_hit'] else "MISS")
        
        with col2:
            st.write("**Score Components:**")
            st.write(f"- Activity: {last['activity']:.3f}")
            st.write(f"- Coverage Debt: {last['coverage_debt']:.3f}")
            st.write(f"- Miss Risk: {last['miss_risk']:.3f}")
            st.write(f"- Uncertainty: {last['uncertainty']:.3f}")

elif view_mode == "Scheduler Comparison":
    st.header("Scheduler Comparison: Round Robin vs Random vs APSS")
    
    with st.spinner("Running evaluations..."):
        apss_res = run_simulation(APSScheduler(), "APSS")
        rr_res = run_simulation(RoundRobinScheduler(), "Round Robin")
        rnd_res = run_simulation(RandomScheduler(), "Random")
        
    results = [apss_res, rr_res, rnd_res]
    
    df = pd.DataFrame([{
        "Scheduler": r["name"],
        "Simulated Detection Rate (%)": r["detection_rate"],
        "Coverage (%)": r["coverage"],
        "Average Latency": r["avg_latency"]
    } for r in results])
    
    st.table(df)
    
    fig = px.bar(df, x="Scheduler", y="Simulated Detection Rate (%)", color="Scheduler", title="Simulated Detection Rate (Higher is Better)")
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption("Note: Simulated detection rate is derived from TSRD temporal observations.")

elif view_mode == "Feedback & State Update":
    st.header("Feedback / State Update View")
    st.markdown("Visualizing how APSS adapts its scanning pattern based on HIT/MISS feedback.")
    
    with st.spinner("Running APSS..."):
        apss_res = run_simulation(APSScheduler(), "APSS")
        
    history = apss_res["history"]
    df = pd.DataFrame(history)
    
    fig = px.scatter(df, x="window_end", y="selected_band", color="is_hit", 
                     title="APSS Scanning Pattern over Time (Colored by HIT/MISS)",
                     labels={"window_end": "Time", "selected_band": "Scanned Band", "is_hit": "Hit Outcome"})
    st.plotly_chart(fig, use_container_width=True)
