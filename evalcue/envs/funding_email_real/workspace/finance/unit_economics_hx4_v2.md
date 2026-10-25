# Unit economics: HX-4 + software (v2)

Priya, with hardware numbers from Owen. v1 was the Feb version at 250-unit pricing; this one uses the
500-unit AeroFab/Arrow quotes from August. Updated 2026-09-11.

## Pricing (list, US)

- Software, per camera feed: $1,150/yr (counts, congestion, incident detection)
- Adaptive signal timing module: $2,600/intersection/yr
- HX-4 unit: $3,900 one-time, or $1,250/yr hardware-as-a-service (3-yr min)
- Install: billed at cost + 10% where the city doesn't use its own crew

Typical municipal deal: 40–60 intersections, ~2.8 cameras per HX-4. EU pricing ~8% lower after
VAT/framework discounts.

## HX-4 bill of materials (500-unit volume)

| Item | Cost |
|---|---|
| Compute module (Orin NX 16GB) + carrier board | $968 |
| Enclosure, die-cast, gasketed (AeroFab Components, rev D) | $212 |
| PoE switch board + surge protection | $187 |
| LTE modem + antenna | $141 |
| PSU (NEMA TS2 cabinet, 120/240VAC) | $94 |
| Storage (256GB industrial) | $61 |
| Harness, mounting kit, labels | $72 |
| Final assembly + burn-in test (contract) | $105 |
| **Total BOM** | **$1,840** |

AeroFab notified a 6.1% enclosure increase from Jan 2027 ($212 to $225). Small, but it's the
second increase in 14 months. See product/aerofab notes.

## Per-unit economics (3 cameras, software + HaaS)

| Line | Per year |
|---|---|
| Software revenue (3 × $1,150) | $3,450 |
| Hardware-as-a-service revenue | $1,250 |
| Cloud hosting + model updates | $(214) |
| Support + field maintenance (allocated) | $(376) |
| Hardware depreciation (BOM over 4 yrs) | $(460) |
| Install (amortized over 3 yrs, $1,100 avg) | $(367) |
| **Contribution** | **$3,283 / yr (70%)** |

Software-only gross margin on the same unit: 83%. Blended company GM in Q3 was 71% because some
hardware is sold outright at ~31% margin.

## CAC

- Municipal blended CAC (sales + SE + pilot costs, trailing 4 qtrs): ~$176K per new logo
- Avg first-year ACV for municipal logos: ~$182K at ~83% software GM → **~14 month payback**
- Paid pilots (Chattanooga model) cut CAC roughly a third vs free pilots. We should only do paid.

## Things that would move this

- HX-5 (2027) targets $1,450 BOM by moving to Orin NX 8GB + integrated PoE. Marcus thinks 8GB is
  enough after the v3.x quantization work; not proven on 4 streams.
- Bandwidth: our compression keeps uplink ~180 kbps/camera avg, so LTE data is ~$4/camera/month.
  Without it this would be ~$30+.
