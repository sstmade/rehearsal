# GoalBench blind hold-out (30 cases)

Written to check the goal-compilation prompt on cases its tuner never saw.

## Blindness statement
Not opened or read: `bench/goals/`, `docs/goalbench.md`, `docs/intent.md`, `docs/litscout.md`, the compile prompts in `rehearsal/agents.py`, any eval results.
Read: `scenes/*.json` (ids, labels, sizes, positions, fragile flags), `rehearsal/predicates.py` (predicate names/arguments only), the top of `README.md`/task brief. No model or Meta API was called.

## Authoring rules
- Instructions typed the way people type: some lowercase, sloppy, 6 in Chinese.
- Only object ids that exist in the case's scene. 10 scenes used.
- Mix: 5 explicit, 5 referential, 7 implicit, 4 ordering, 4 ambiguous, 5 infeasible.
- Scenes carry no liquid flag, so "full of liquid" is stated in the instruction text.

## Labeling conventions
- `required`: user-stated atoms plus implied ones (liquid -> `upright`+`stable`; "don't touch X" -> `undisturbed X`; "nothing on X" -> `clear X`). The automatic implicit `undisturbed` over unmentioned objects is NOT listed.
- Argument order: `on(a,b)` a rests on b; `inside(a,b)`; `stacked(lowest,...,highest)`; `near(a,b)` (symmetric, swap accepted where noted).
- `acceptable_alternatives`: other required-sets that also count as correct. `forbidden`: atoms that must not appear.
- `order_matters`: the list order is the stated execution order.
- `clarify`: `plausible_readings` lists candidate required-sets; the right behaviour is to ask, not guess.
- `refuse`: `refuse_kind` in missing_object, physically_impossible, out_of_reach, unsafe.
- Ordering cases are judged on both atoms and sequence; out_of_reach judgements assume a tabletop arm.
