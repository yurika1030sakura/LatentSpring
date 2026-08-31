# PAGE CUT PLAN — plan only, nothing executed

Paper: **What Energy Values Teach De Novo Molecular Flows: Ordering Without Calibration**
Date: 2026-08-14
Status: **NO CUT IN THIS DOCUMENT HAS BEEN MADE.** This is the specification for the separate
compression pass. The manuscript on disk is the uncompressed, internally consistent version.

---

## 1. Where the manuscript stands

| | Pages |
|---|---|
| Body text, §1 through §6 (ends mid p. 14) | **14** |
| References (mid p. 14 through p. 16) | 2.5 — excluded from the limit |
| Appendices A and B | 29 — excluded from the limit |
| Total PDF | 45 |
| **ICLR main-text limit** | **9** |
| **To remove** | **5 pages** |

Measured layout (from `main.aux`, build of 2026-08-14 01:57):

| Section | Starts | Approx. extent |
|---|---|---|
| §1 Introduction | p. 1 | 1.6 pp prose |
| §2 Setting and Related Work | p. 3 | 1.0 pp |
| §3 Energy-Value Supervision (+ §3.1 gradient baseline) | p. 4 | 1.2 pp |
| §4 What Does Group-Wise Supervision Identify? | p. 5 | 1.6 pp |
| §5.1 Setting and metrics | p. 7 | 1.2 pp |
| §5.2 Does value supervision improve local ordering? | p. 8 | 1.0 pp |
| §5.3 What carries the gain? | p. 9 | 0.8 pp |
| §5.4 Ordering is not calibration | p. 10 | 1.5 pp |
| §5.5 What is preserved, and what remains unresolved? | p. 12 | 1.0 pp |
| §6 Limitations and Conclusion | p. 13 | 1.3 pp |
| Floats: Fig. 1 (p. 2), Table 1 (p. 9), Fig. 2 (p. 10), Fig. 3 (p. 11), Fig. 4 (p. 12) | | **≈3.3 pp** including captions |

Content budget: **7,530 prose words + 1,207 caption words** across the five section files, at
roughly 700 page-equivalent words per full page. Floats are **23% of the main text**, which is why
the float work below carries the plan.

---

## 2. Two rules that make the arithmetic easy

1. **The appendix has no page limit.** Almost nothing needs deleting; it needs *moving*. Appendix B
   already carries a full version of most main-text material flagged below, which is why these are
   cheap: the cut is a duplication removal plus a `\Cref` pointer, not a loss of content.
2. **Every cut below is a length change, not a claim change.** Nothing in §5 alters what the paper
   asserts, and the protected list in §6 of this document is not negotiable against the page count.

---

## 3. Tier 1 — do these first (≈2.9 pp, zero cost to the argument)

Deterministic, appendix-duplicated, or purely typographic. None requires a judgement call.

| # | Item | Action | Saving |
|---|---|---|---|
| T1.1 | **§5.5 in full** | The generation-quality discussion and the three "limits remain" clauses duplicate `app:generation` and limitations (iii), (v), (vi) verbatim. Reduce to three sentences with pointers. Keep the resolution range and the stability census sentence, which appear nowhere else in the main text. | **0.7 pp** |
| T1.2 | **All four figure captions** | 1,207 words → ≈500. Named compressible passages, already flagged by the figure agents: Fig. 2's "What the panel adds is the shape of the association…" and the trailing "not a bound on what is attainable"; Fig. 4's `E_m[1−r_m²] ≠ 1−(E_m r_m)²` justification (moves to §5.4 as one clause, where it is load-bearing); Fig. 3's second half of the contrast enumeration, which Table-1-style numbers already carry; Fig. 1's panel-by-panel walkthrough, which the body text repeats. | **0.7 pp** |
| T1.3 | **§6 limitations enumeration** | Eight limitations are enumerated in full in §6 *and* in `A2_details.tex` §B.14. Keep four sentences naming (i), (ii), (v) and (viii) plus one pointer; the rest goes. | **0.5 pp** |
| T1.4 | **§5.1 metrics paragraph** | The Simpson's-paradox justification for per-parent computation is stated twice (§5.1 and `rem:crossbatch`). Main text needs one sentence and a pointer. The `NRV_min` and retrieval definitions can move to `app:scalediag`, keeping only the `NRV` display in the body. | **0.4 pp** |
| T1.5 | **§2 Setting** | The bond-free rationale and the FlowMol3 interpolant recap are both restated in `app:data` and `app:hardware`. Halve. | **0.4 pp** |
| T1.6 | **Run-in heads and repeated caveats** | Several `\textbf{...}` run-in heads are full sentences; make them noun phrases. The completion-conditioning qualification currently appears in §5.3, the Fig. 3 caption, §6 *and* the abstract; keep §5.3 and the caption, make §6 a pointer, keep the abstract (it is the load-bearing one). | **0.2 pp** |

**Tier 1 subtotal: 2.9 pp.** After Tier 1 the manuscript is ≈11.1 pp.

---

## 4. Tier 2 — the float restructuring (≈2.1 pp, low cost, needs generator edits)

These are the largest single savings in the paper and they need a rebuild of the figure PDFs, so
they should be batched.

| # | Item | Action | Saving |
|---|---|---|---|
| T2.1 | **Merge Figure 2 and Figure 4 into one figure** | Both draw the *same* population (93 disjoint parents, reference dropped, GFN2-xTB, four seeds per arm) and both belong to the same message (Fig. 2 is property (i), Fig. 4 is property (ii)). One four-panel figure titled *Ordering improves; calibration does not* carries: (a) per-seed mean `r` with the primary contrast, (b) the per-parent distribution, (c) retrieval, (d) raw NRV against `NRV_min` with the teacher-matching reference. Saves one float's inter-float space, one caption block, and the duplicated "same population, same evaluator, four seeds" preamble that both captions currently carry. | **0.9 pp** |
| T2.2 | **Drop the standardised-scatter panel** (current Fig. 2c) as part of T2.1 | Its summary statistic is redundant with panel (a) by construction. The two things it uniquely shows, that the conditional mean steepens across the whole range rather than in one tail and that the residual scatter stays wide, become one main-text clause in §5.2. It was kept this round on the judgement that the clause was worth a panel; at a 5-page deficit it is not. Three-line change to `fig2_ordering.py`. | **0.4 pp** |
| T2.3 | **Figure 3 panel (b) → inline** | Panel (b) plots six contrasts with Welch SEs that the §5.3 prose already states numerically. Keep panel (a), whose raw seed points and launched/completed/diverged triples are the object a sceptical reader needs, and move the six contrasts into a compact three-line inline list or a two-column mini-table. | **0.5 pp** |
| T2.4 | **Figure 1 height** | Keep the figure; it carries the theory's only claim and panel (d) is the paper's one picture of what is *not* identified. Reduce the panel height by ≈15% and let the caption (T1.2) carry the rest. | **0.25 pp** |

**Tier 2 subtotal: 2.05 pp.** After Tiers 1–2 the manuscript is ≈9.0 pp, **at the limit, with no margin.**

---

## 5. Tier 3 — contingency (≈2.0 pp available, use only what is needed)

Tiers 1–2 land exactly on 9 pages, which is not a safe place to stop: float reflow after a
restructuring of this size routinely moves ±0.5 pp in either direction. Take items from Tier 3 in
order until there is ≈0.5 pp of slack.

| # | Item | Action | Saving |
|---|---|---|---|
| T3.1 | **§1 Introduction, P1 and P4** | P1 makes the three-way ordering/calibration/mass distinction in seven sentences; four suffice. P4 restates six numbers that Table 1 also carries; three suffice. Do **not** touch P2 (the question) or P3 (the identifiability scope and the prior-work cession). | **0.9 pp** |
| T3.2 | **§4 theory** | The multiplicity paragraph after `cor:basins` shrinks to one sentence plus a pointer to §B.4, and the "one further gap" paragraph to one clause, now that the appendix carries both in full. The continuous/empirical corollary split stays in the body: it is the repair, and burying it re-creates the defect. | **0.6 pp** |
| T3.3 | **§5.4** | Keep the fourfold improvement list, the `NRV_min` movement and the offset-invariance argument. Move the `eq:rm` rescaling derivation and the `T_eff` reading to `app:scalediag`, leaving the two numbers and a pointer. | **0.5 pp** |
| T3.4 | **§3.1 gradient-supervision baseline** | Three sentences: what it is, that it is a natural baseline, and that cell E measures it. The rest is in `app:trap`. | **0.3 pp** |
| T3.5 | **Last resort — Table 1 to two arms** | Move the scrambled and teacher-matching-reference rows into the merged figure's caption. **Do this only if T3.1–T3.4 fall short**, because it removes the defect flag on the earlier scrambled arm from the most-read object in the paper, and that flag is a correction this round installed on purpose. | **0.3 pp** |

---

## 6. Protected — do not compress, at any page count

These are the corrections this round installed. Removing them re-creates a defect the manuscript
was just repaired for, which is a worse outcome than being over length.

1. The **three-way separation** of local ordering, local calibration and global mass allocation, and
   the statement that (iii) is not measured.
2. Every **completion-conditioned** qualification on cells D and F, and every
   launched / completed / diverged triple.
3. Every **teacher-matching reference** phrasing. Never revert to "ceiling", "achievable maximum" or
   "attainable ordering", and never quote 39.8% as the gain (the gain is 23.0% gap closure).
4. The **offset-invariance argument** in §5.4: because NRV is invariant to an additive per-group
   constant, no offset term can move it. This is the correction of the paper's most damaging internal
   contradiction.
5. The **ensemble-scope paragraph** at the head of §5.1: local stress test with a broad energy
   window, not a thermal conformer ensemble; `kT = 1.0` eV is a specified numerical target.
6. The **identifiability scoping of connectivity** ("sufficient, and necessary in the identifiability
   sense"), in the abstract and in §4.
7. The disclosure that the reported density is a **position-only flow density** with the discrete
   channels clamped.
8. The **defect flag on the earlier round's scrambled arm** at its point of use.
9. "We train 30,000 optimiser steps, approximately 0.12 data epochs; **we do not establish
   convergence**" — do not compress back into "no model here is converged".
10. Retrieval described as a **secondary within-cloud read-out**, not a conformer-search benchmark.

---

## 7. Recommended execution order

The order matters because float placement makes some sequences much more expensive than others.

1. **T1.1, T1.3, T1.4, T1.5** — the appendix-duplicated prose. Pure deletions plus `\Cref` pointers,
   no float involvement, no rebuild. Recompile once and record the new page count.
2. **T1.2 and T1.6** — caption and run-in-head compression. Captions are the highest words-per-page
   density in the paper and shrinking them frees float area without touching a generator.
3. **T2.1 + T2.2 together**, the Figure 2 / Figure 4 merge. One generator edit, one caption, one
   rebuild, one set of cross-reference updates (`fig:ordering` and `fig:calibration` collapse to one
   label; §5.2, §5.4, §6 and both appendix subsections cite them). Do this as a single commit so a
   revert is clean.
4. **T2.3 and T2.4**, the remaining float work.
5. **Recompile and measure.** Expect to be within ±0.5 pp of 9.
6. **Tier 3, in the listed order**, until there is ≈0.5 pp of slack.
7. **Final typographic pass only**: `\textfloatsep`, `\intextsep`, float placement specifiers, and
   Table 1's column widths. No further prose cut. If the paper is still over after this, the
   remaining lever is T3.5, and its cost is stated above.

---

## 8. Risks specific to this compression

- **The merge in T2.1 collapses two labels into one.** `\Cref{fig:ordering}` and
  `\Cref{fig:calibration}` are cited from §5.2, §5.4, §6, `A1_proofs.tex` and `A2_details.tex`, and
  Figure 3's caption cites `fig:ordering` by label (not by number) to point at the corrected earlier
  checkpoint family. Keep one of the two labels alive as an alias rather than rewriting six call
  sites.
- **Compressing §5.5 must not drop the resolution range.** The +0.382 → +0.037 range across
  smoothing scales is the corrected result from this round's REVIEWER_DISAGREE D1 and appears in the
  main text only there and in limitation (v).
- **`build.sh` runs a literal forbidden-vocabulary gate that cannot see negations.** One phrase was
  already reworded this round for that reason alone. Recompressed sentences should be run through
  the build before being considered done.
- **Do not let the appendix absorb material without a pointer.** Every T1 and T3 move is only a
  saving if the main text keeps a one-clause statement plus a `\Cref`; a silent deletion turns a
  length cut into a content loss and would be caught in review.

---

## 9. Summary

| Tier | Saving | Cumulative main text |
|---|---|---|
| — | — | 14.0 pp |
| Tier 1 (duplication and captions) | 2.9 pp | 11.1 pp |
| Tier 2 (float restructuring) | 2.05 pp | 9.0 pp |
| Tier 3 (contingency, as needed) | up to 2.0 pp | 7.0–9.0 pp |

Tiers 1 and 2 alone reach the limit; Tier 3 exists to buy the margin that float reflow will consume.
Total available is ≈7.0 pp against a 5.0 pp requirement, so the compression pass does not need to
touch anything on the protected list.
