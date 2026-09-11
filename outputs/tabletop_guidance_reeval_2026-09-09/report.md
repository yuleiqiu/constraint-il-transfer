# Corrected Tabletop guidance comparison

Status: complete; 100/100 formal rollouts, 50/50 paired layouts.

Frozen wide-support epoch-50 policy; H600; original target-lifted point guidance.
All videos are captured inside their evaluated rollouts. Preflight repetitions are separate.

| Condition | Task SR | Safe SR | CR | NCR |
|---|---:|---:|---:|---:|
| baseline | 28/50 (0.56) | 23/50 (0.46) | 9/50 (0.18) | 18/50 (0.36) |
| guided | 28/50 (0.56) | 27/50 (0.54) | 7/50 (0.14) | 16/50 (0.32) |

Four-way counts and paired transitions / uncertainty are in `summary.json`.
Layout bootstrap intervals describe this empirical bank; the bank was previously examined,
not a new held-out sample. An inconclusive difference is not proof of no effect.

## Paired task transitions

- success: 10 rescues / 10 regressions; net +0.000; paired bootstrap95 [-0.18, 0.18]; exact McNemar p=1.
- safe_success: 11 rescues / 7 regressions; net +0.080; paired bootstrap95 [-0.08, 0.24]; exact McNemar p=0.4807.

## Layouts

| Layout | Source | Guidance |
|---|---|---|
| 00 | safe success (343 steps) | no contact, incomplete (600 steps) |
| 01 | safe success (383 steps) | collision failure (600 steps) |
| 02 | safe success (376 steps) | no contact, incomplete (600 steps) |
| 03 | no contact, incomplete (600 steps) | safe success (382 steps) |
| 04 | safe success (318 steps) | safe success (368 steps) |
| 05 | safe success (470 steps) | no contact, incomplete (600 steps) |
| 06 | safe success (312 steps) | safe success (571 steps) |
| 07 | no contact, incomplete (600 steps) | success + contact (268 steps) |
| 08 | success + contact (350 steps) | no contact, incomplete (600 steps) |
| 09 | success + contact (350 steps) | no contact, incomplete (600 steps) |
| 10 | no contact, incomplete (600 steps) | no contact, incomplete (600 steps) |
| 11 | no contact, incomplete (600 steps) | safe success (319 steps) |
| 12 | safe success (379 steps) | safe success (389 steps) |
| 13 | no contact, incomplete (600 steps) | safe success (406 steps) |
| 14 | no contact, incomplete (600 steps) | no contact, incomplete (600 steps) |
| 15 | safe success (390 steps) | safe success (359 steps) |
| 16 | no contact, incomplete (600 steps) | safe success (340 steps) |
| 17 | no contact, incomplete (600 steps) | no contact, incomplete (600 steps) |
| 18 | success + contact (305 steps) | no contact, incomplete (600 steps) |
| 19 | safe success (432 steps) | safe success (334 steps) |
| 20 | safe success (391 steps) | safe success (356 steps) |
| 21 | no contact, incomplete (600 steps) | safe success (384 steps) |
| 22 | safe success (367 steps) | safe success (369 steps) |
| 23 | safe success (366 steps) | safe success (383 steps) |
| 24 | no contact, incomplete (600 steps) | safe success (376 steps) |
| 25 | success + contact (543 steps) | safe success (471 steps) |
| 26 | safe success (429 steps) | safe success (422 steps) |
| 27 | collision failure (600 steps) | safe success (455 steps) |
| 28 | collision failure (600 steps) | collision failure (600 steps) |
| 29 | safe success (446 steps) | safe success (423 steps) |
| 30 | safe success (534 steps) | safe success (379 steps) |
| 31 | no contact, incomplete (600 steps) | safe success (351 steps) |
| 32 | safe success (445 steps) | collision failure (600 steps) |
| 33 | safe success (322 steps) | safe success (332 steps) |
| 34 | no contact, incomplete (600 steps) | no contact, incomplete (600 steps) |
| 35 | safe success (311 steps) | safe success (302 steps) |
| 36 | safe success (296 steps) | collision failure (600 steps) |
| 37 | success + contact (343 steps) | safe success (384 steps) |
| 38 | safe success (284 steps) | safe success (284 steps) |
| 39 | no contact, incomplete (600 steps) | safe success (367 steps) |
| 40 | no contact, incomplete (600 steps) | no contact, incomplete (600 steps) |
| 41 | no contact, incomplete (600 steps) | no contact, incomplete (600 steps) |
| 42 | no contact, incomplete (600 steps) | no contact, incomplete (600 steps) |
| 43 | collision failure (600 steps) | no contact, incomplete (600 steps) |
| 44 | safe success (352 steps) | no contact, incomplete (600 steps) |
| 45 | no contact, incomplete (600 steps) | collision failure (600 steps) |
| 46 | no contact, incomplete (600 steps) | no contact, incomplete (600 steps) |
| 47 | safe success (478 steps) | safe success (325 steps) |
| 48 | safe success (405 steps) | safe success (384 steps) |
| 49 | collision failure (600 steps) | collision failure (600 steps) |
