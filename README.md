# APSS — Adaptive Predictive Spectrum Scheduler

## Problem
SIH Problem Statement: 26055 — Smart Scan Strategy for Electronic Warfare.
Modern electronic warfare (EW) receivers are overwhelmed by dense, dynamic RF environments. Traditional Round-Robin or static-priority spectrum scanning mechanisms suffer from high latency in detecting critical threats and waste time scanning empty frequencies. The challenge is to dynamically adapt the scan strategy to minimize detection latency for active signals while maintaining sufficient exploratory coverage.

## Proposed Solution
APSS (Adaptive Predictive Spectrum Scheduler) uses a closed-loop, state-driven approach to determine the optimal frequency band to scan next:
1. **RF Observations:** Pulse Descriptor Words (PDWs) are aggregated.
2. **Per-Band State:** Each band tracks its historical and current RF characteristics.
3. **Adaptive Priority Score:** APSS dynamically weights multiple features to score bands.
4. **Scan Decision:** The scheduler tunes to the highest-scoring band.
5. **HIT/MISS Feedback:** The result of the scan updates the band's state.
6. **State Update & Next Decision:** Continuous learning optimizes future scans.

## Technical Architecture
```
apss/
├── src/
│   ├── environment/   # Simulated RF environment and PDW processing
│   ├── preprocessing/ # 5-D PDW processing into band-level observations
│   ├── state/         # BandState definition
│   ├── schedulers/    # Round Robin, Random, and APSS implementations
│   └── evaluation/    # Simulation and metric evaluation logic
├── experiments/       # Diagnostic isolated experiments
├── tests/             # Unit test suite
└── app.py             # Streamlit Dashboard UI
```

## Key Features
- Multi-factor band-state representation
  - **Activity:** Recent signal density
  - **Persistence:** Long-term signal stability
  - **Novelty:** Newly emerged signals
  - **Uncertainty:** Lack of recent information
  - **Coverage Debt:** Penalty for neglecting bands
  - **Miss Risk:** Penalty for missing pulses in active bands
  - **Scan Cost:** Hardware tuning overhead penalty
- Explainable priority scoring
- Closed-loop feedback

## Technology Stack
- Python
- NumPy
- Pandas
- Streamlit
- Plotly
- TSRD Dataset

## Prototype
The current prototype demonstrates the APSS logic against a simulated environment derived from real TSRD temporal observations.
*Screenshots available in the PPT.*

## Evaluation
Simulated detection rate derived from TSRD temporal observations.
*(Note: This is a simulated detection rate based on a temporal subset of historical data; it is not a physical probability of detection from a live receiver.)*

## Dataset
A representative subset of the TSRD (Time-Series RF Dataset) was used for development and evaluation. 
*Note: Due to file size and sensitivity, the dataset itself is excluded from this repository.*

## Limitations
- Current evaluation uses TSRD-derived temporal observations.
- HIT/MISS is simulated from available observations.
- No physical EW receiver hardware is connected.

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the test suite:
```bash
pytest tests/
```

3. Launch the dashboard:
```bash
streamlit run app.py
```
