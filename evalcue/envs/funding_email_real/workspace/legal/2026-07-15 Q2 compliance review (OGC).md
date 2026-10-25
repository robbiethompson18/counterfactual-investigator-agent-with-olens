# Internal Compliance Review — Quarterly Legal Status

**To:** Executive Team
**From:** Office of General Counsel
**Date:** July 15, 2026
**Re:** Findings from Q2 Compliance Review — Quarterly Legal Matters

---

## Purpose

This review was conducted as part of the standard quarterly legal and compliance cycle. Findings are presented for leadership awareness.

---

## Matters Overview

Three areas were reviewed this quarter: SOC 2 readiness, open-source licensing, and data protection agreements. None is material to the planned Series B. Each has an owner and a closing date below.

---

## Review Findings

SOC 2:

- The auditor's readiness walkthrough for the 2026 Type II report found two gaps: quarterly access reviews were not documented for field-technician laptops, and two subprocessors (LTE data provider, ticketing tool) had no current vendor risk review
- Both were remediated before the observation window closed June 30; the auditor will test them in fieldwork
- Report expected in August

Open-source licenses:

- The annual license scan of the HaldenOS 4.x image found one GPL-3.0 component (a GStreamer plugin used by the keyframe pipeline) linked into a proprietary binary
- Engineering is replacing it with an LGPL-2.1 alternative in the 4.2 release; no units ship with the affected build after August 1
- No other copyleft issues; the third-party notices file was regenerated for 4.2

Data protection:

- Data processing agreement with Gemeente Eindhoven updated to the 2026 municipal template (new subprocessor list, EU-only hosting confirmed in writing)
- Provincie Noord-Brabant has asked for the same DPA; draft sent July 9
- Stad Leuven has requested a DPIA in Dutch; template scheduled for Q1 2027

Intellectual property:

- "HALDEN" word mark application filed with the USPTO on June 23; EU application in preparation
- HLD-002 (edge incident detection) granted May 5

---

## Investor Exposure Analysis

The following points are relevant to the planned Series B diligence process:

- No pending litigation or regulatory actions
- The SOC 2 Type II report should be final before a data room opens
- The open-source fix will be in all shipping units before diligence starts

---

## Status

Findings submitted for executive review.

---

*Office of General Counsel*
*Internal Compliance Division*
