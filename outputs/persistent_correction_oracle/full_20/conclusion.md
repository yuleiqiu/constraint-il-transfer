# Persistent-Correction Oracle: Full 20-State Conclusion

Status: complete, 2026-07-30.

## Result

The fixed persistent-correction protocol was evaluated on all 20 held-out
strict D2 pre-collision states. Each state used four deterministic correction
trajectories, at most 20 closed-loop SDP chunks, and an exact frozen-source
continuation from every collision-free chunk boundary.

- state coverage: **7/20** states contain at least one safe recoverable
  handoff;
- trajectory coverage: **27/80** trajectories contain a safe handoff;
- within the seven recoverable states: **27/28** trajectories contain a safe
  handoff;
- within the other thirteen states: **0/52** trajectories contain one;
- temporal boundaries: 311 safe successes, 280 collision failures, 488 NCRs,
  and 20 successes with collision among 1,099 safe-prefix boundaries;
- correction safety: 31/80 correction trajectories eventually collide;
- direct correction completion: 0/80;
- earliest state-level safe handoff: chunks `5, 1, 6, 2, 6, 1, 6`, with
  median 5;
- five of seven recoverable states require more than one correction chunk
  under all four sampled trajectories.

Boundary outcomes are temporally correlated and are not independent trials.
State coverage is the relevant upper-bound statistic.

## Interpretation

The expansion confirms the pilot's mechanism result. Persistent correction
creates safe source-recoverable states that are absent from the rejected
one-chunk candidate set. It is not merely a lucky diffusion sample: six of the
seven recoverable states succeed under all four correction trajectories, while
all 52 trajectories from the thirteen uncovered states fail.

Recoverability is also non-monotonic. A correction trajectory can move from
NCR or collision failure into safe success, leave that window, and later
re-enter it. Task success alone is insufficient because 20 boundaries complete
with collision. A deployable termination model must therefore estimate
**safe source recoverability**, not elapsed correction time, local clearance,
or generic task progress alone.

The same strict state bank has a waypoint-teacher upper bound of 18/20.
Consequently, the current mixed-policy SDP checkpoint's 7/20 oracle coverage
is a real but limited behavior-generator upper bound. A learned handoff model
cannot recover the remaining thirteen states when no positive handoff exists.

## Decision

Retain the recoverability-gated corrective option as the method abstraction,
but do not train its handoff estimator against the current SDP generator yet.
The next bounded experiment should train a **correction-only, temporally
extended option** from the existing verified intervention segments while
keeping the reliable source frozen. Collect additional segments only if the
stored sequences are insufficient for that training interface. Repeat this
identical 20-state oracle evaluation on the new generator. Train the
source-recoverability estimator only after the generator materially expands
state coverage.

Machine-generated details and every temporal window are in `report.md`;
configuration is in `manifest.json`.
