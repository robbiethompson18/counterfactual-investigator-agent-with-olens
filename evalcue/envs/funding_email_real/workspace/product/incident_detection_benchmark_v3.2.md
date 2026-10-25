# Incident detection benchmark — model v3.2

Ananya Iyer (ML lead), 2026-09-18. Internal. Numbers here are the ones to quote externally.

## Eval set

- 4,812 labeled clips from 9 deployments (US + NL/BE), Jan–Aug 2026, held out from training
- 1,207 positives: 512 crashes, 486 stalled vehicles, 61 wrong-way, 148 debris/obstruction
- Labels: two annotators + traffic-ops confirmation where the agency logged the event
- 31% night, 18% rain/snow, 9% sun glare

## Results (HX-4 edge path)

| Event | Precision | Recall | Median time-to-alert | p95 |
|---|---|---|---|---|
| Crash | 0.94 | 0.91 | 7.6 s | 13.9 s |
| Stalled vehicle | 0.92 | 0.95 | 8.4 s | 15.1 s |
| Wrong-way driver | 0.98 | 0.97 | 6.1 s | 9.8 s |
| Debris / obstruction | 0.81 | 0.77 | 9.2 s | 18.4 s |
| **All incidents** | **0.92** | **0.92** | **7.8 s** | **14.2 s** |

False alarms: 0.29 per camera-day (v3.1: 0.41). Most remaining FPs are parked delivery vans in
the curb lane and shadows on wet pavement.

Cloud path (full-frame upload, A10G): median 4.6 s, but that's not how any customer runs it. The
old "under 5 seconds" line came from the cloud path on v2.9. Edge is the product.

## Conditions

| Condition | Recall (all) |
|---|---|
| Day, dry | 0.95 |
| Night | 0.88 |
| Rain/snow | 0.86 |
| Glare | 0.83 |

## Counting accuracy (vs manual counts, 212 site-hours)

- Vehicles: 97.3% (trucks/buses 95.0%)
- Cyclists: 94.8% overall; 96.1% Eindhoven daytime (dedicated bike lanes help)
- Pedestrians (aggregate crossing counts only): 92.4%

## vs. v3.1

All-incident recall +4 pts, FP rate −29%, median latency −0.9 s. Gains mostly from the temporal
model retrain with Dayton/Boise data.
