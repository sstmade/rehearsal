## Earlier headline (v1, superseded by Evaluation v3 above; kept for the record)

_v1 ran mostly on Chinese instructions and pre-merge snapshots. Where it differs from v3, v3 wins: the windowsill
and kitchen bullets below are replaced by v3 sections 2–3 (kitchen's "none 20/20" holds for the Chinese
instruction only, and its denominator included the 8 first-try successes)._

- **Reading the log matters where the fix is not guessable.** windowsill_01 (tray at the edge of reach, tall
  vase in the straight path): Muse Spark failed the first try in 15/15 runs. Recovery within 2 revisions,
  pooled over two code snapshots (n = 15 per condition, paired first attempts): **full log 15/15, failed
  predicates only 12/15, "attempt failed" only 5/15** (full vs none: Fisher exact p = 0.0002). On the
  agent-branch snapshot alone (via in the schema, n = 10): 10/10, 9/10, 4/10.
- **…and does not matter where it is.** kitchen_01 (the cup must wait until the kettle has moved): Muse first
  try 8/20, and every feedback condition recovered (full log 20/20, none 20/20, outcome-only 18/20). Reordering
  is a fix the model guesses without evidence. desk_01: Muse solved all 4 instructions first try (20/20), so
  there is nothing to revise.
- **Muse beat the action schema it was given.** Without a `via` field (pre-merge snapshot) it still solved
  windowsill_01 with the full log in 5/5 runs by relaying the mug through an intermediate reachable spot,
  using the reach limits quoted in the IK evidence.
- **Robustness is a curve, not a number.** Final plans succeed 100% at σ ≤ 0.4 cm and fall to 20–40% at
  σ = 2.4 cm (held-out seeds, n = 60). At the stress test's σ = 1.2 cm: desk 87%, kitchen 93%, windowsill 92%.
  The measured synthetic-render error (median 0.54, P90 1.30 cm) sits on the shoulder of the curve.
- **One extra look (held-out seeds 1000–1199, n = 200):** the VoI pick raises success desk 178→191
  (+13 fixed / −0 broken, p = 0.0002), kitchen 192→200, windowsill 191→199 (each +8/−0, p = 0.008), vs.
  ~182/193/193 expected for a random object. The old "35/40 → 40/40" was selected and scored on the same
  seeds; held out it is 89% → 95.5%. **Honest caveat:** a simple heuristic (re-observe the object that appears
  most often in failed predicates) picked the same object in all three scenes; "nearest the carry path" did not.
- **Cost:** the whole Muse Spark matrix (226 paid calls, Contributor mode) cost **$0.23**.
