# Dayton incident detection pilot — final report

Dayton DPW Traffic Operations + Halden. Pilot ran Nov 2024 – Apr 2025, converted to contract May
2025. Updated with 2026 production numbers in September 2026 for the Series B data room.

## Scope

64 cameras on downtown arterials and the US-35 / I-75 approaches, 23 HX-4 units. Alerts into the
city TMC's existing ATMS via webhook. Operators confirm or dismiss each alert.

## Pilot results (6 months)

- 212 operator-verified incidents (crashes 71, stalled vehicles 118, debris 23)
- **Median time-to-detect, stalled vehicles: 41 s** after the vehicle stopped, vs. a median of
  6 min 30 s for the same incident types in 2023–24 (first 911 call or patrol report, TMC logs)
- Crashes: median 12 s to alert (v2.9 model at the time)
- False alarms: 0.40 per camera-day at start, 0.29 by month 6 after retraining

## 2026 production (Jan–Aug, model v3.1/v3.2)

- 1,038 verified incidents; operator dismiss rate 11%
- Median time-to-alert all incidents: 8.1 s
- TMC reports average lane-blockage clearance 4.6 min faster on Halden-detected stalls (their
  number, not ours; we can't verify the clearance side independently)

The 41 s number is detection after the vehicle *stopped*, including the dwell threshold so we
don't flag normal stops. It's not comparable to the 7.8 s time-to-alert in the benchmark, which
starts from the labeled event onset.
