# HX-4 Edge Unit — Technical Specification, Rev D

Doc owner: Marcus Oyelaran (CTO) · Hardware: Owen Hartley · Rev D, 2026-08-27
Rev C → D: new enclosure gasket (AeroFab rev D), PSU derating table, LTE module swap.

## Purpose

Cabinet- or pole-mounted unit that runs Halden's detection and counting models on existing
municipal traffic-camera feeds, so cities don't stream full video to the cloud. Outputs are
aggregate traffic data: counts by class and movement, queue length, occupancy, speeds, signal-phase
utilization, and incident events (crash, stalled vehicle, wrong-way driver, debris).

The unit does not store video by default. Event clips (8 s pre / 12 s post) are kept only for
incident events, encrypted, auto-deleted after 30 days unless the agency changes retention. No
plate reading, no face processing; the models don't have those heads.

## Hardware

| | |
|---|---|
| Compute | NVIDIA Jetson Orin NX 16GB (100 TOPS INT8) |
| Camera inputs | Up to 4 × 1080p RTSP/ONVIF streams at 15 fps (6 at 720p) |
| Network | 2 × GbE (1 PoE+ out), LTE Cat-6 (Quectel EG25 → EM06 in rev D), optional SFP fiber |
| Storage | 256 GB industrial NVMe |
| Power | 120/240 VAC via cabinet PSU, 25 W typical, 38 W max |
| Operating temp | −30 °C to +60 °C (derate to 3 streams above +55 °C) |
| Enclosure | Die-cast aluminum, IP67, AeroFab Components rev D, 3.1 kg |
| Mounting | NEMA TS2 / 332 cabinet shelf or pole bracket |
| Signal controller I/O | NTCIP 1202 over Ethernet; SDLC via adapter for older 170 cabinets |
| MTBF (calc.) | 61,000 h |

## Software

- HaldenOS 4.x (Yocto-based), signed OTA updates, A/B partitions
- Detector: Halden-Det v3.2 (INT8, ~11 ms/frame/stream on Orin NX)
- Tracker: ByteTrack variant, per-camera, anonymous track IDs reset every 60 s
- Incident model: v3.2 temporal classifier on top of tracks (see benchmark doc)
- Signal timing agent: runs in cloud, recommends splits/offsets every 5 min, pushes via NTCIP only
  if the agency enables closed-loop mode (8 of 12 municipal/highway customers run advisory-only)
- Uplink: compressed metadata + low-rate keyframes, ~180 kbps per camera average (patent pending
  codec, see legal/patents)
- C-001 deployment runs a separate integrator-managed build; not covered by this spec.

## Latency budget (incident path, median)

| Stage | ms |
|---|---|
| Frame capture + decode | 140 |
| Detection + tracking | 260 |
| Temporal model window fill | 6,100 |
| Rule check + event build | 90 |
| Uplink + cloud alert fan-out | 1,210 |
| **Total (median)** | **~7,800** |

The window fill dominates. v3.3 experiments with a 4 s window drop median to ~5.9 s but recall on
slow stalls falls 3 pts. Not shipping until that's fixed.

## Known issues (rev D)

- Rev C gaskets (Spokane pre-deployment batch, 40 units) showed condensation at −20 °C; all
  replaced with rev D under warranty, no field failures since.
- 170-cabinet SDLC adapter is a third-party part with 9-week lead time.
