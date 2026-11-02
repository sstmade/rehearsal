"""GoalBench cases: instruction -> hand-labelled reference predicates. Source of truth; `python -m bench.goals.cases`
writes cases.json.

Atom syntax: "on(a,b)"; "on(a,b)|inside(a,b)" = either satisfies. stacked(...) is written as its chain of on() atoms.
Fields per case:
  expect    "goals" (compile these), "flag" (should ask / refuse; any goal set is wrong),
            "either" (ambiguous: flagging is best, but a goal set matching one of `refs` is accepted)
  refs      list of acceptable readings, each {required, implicit, optional, forbidden}
            required  atoms the instruction states
            implicit  atoms the user means but does not say (full cup -> upright; "leave X alone" -> undisturbed(X))
            optional  extra atoms that are harmless. upright(x)/stable(x) of any object the reading moves are always
                      harmless and need not be listed.
            forbidden atoms that contradict the user (scored even if other atoms are right)
  keep      extra ids that must end undisturbed. By default every object the reading does not mention must stay
            undisturbed (the product adds this automatically for objects the compiled goals do not mention, so a
            model that needlessly mentions an object strips its protection).
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path


def chain(*ids):
    """stacked(bottom, ..., top) as on() atoms."""
    return [f"on({u},{l})" for l, u in zip(ids, ids[1:])]


def C(id, scene, lang, category, instruction, probes, expect="goals", refs=None, note="", pair=None, **ref):
    if refs is None:
        refs = [ref] if ref else []
    refs = [{k: list(r.get(k, [])) for k in ("required", "implicit", "optional", "forbidden", "keep")} for r in refs]
    return {"id": id, "pair": pair or id.rstrip("z").rstrip("_"), "scene": scene, "lang": lang,
            "category": category, "instruction": instruction, "probes": probes, "expect": expect,
            "refs": refs, "note": note}


BOOKS = ("book_large", "book_medium", "book_small")
BOXES = ("box_large", "box_medium", "box_small")

CASES = [
    # ---------------------------------------------------------------- explicit
    C("e01", "desk_01", "en", "explicit", "Put the mug in the bowl.", ["wrong_relation"],
      required=["inside(mug_1,bowl_1)"]),
    C("e01z", "desk_01", "zh", "explicit", "把杯子放进碗里。", ["wrong_relation"],
      required=["inside(mug_1,bowl_1)"]),
    C("e02", "kitchen_01", "en", "explicit", "Put the cup on the coaster and the kettle on the trivet.", ["multi_goal"],
      required=["on(cup_1,coaster_1)", "on(kettle_1,trivet_1)"]),
    C("e02z", "kitchen_01", "zh", "explicit", "把杯子放到杯垫上，再把水壶放到隔热垫上。", ["multi_goal"],
      required=["on(cup_1,coaster_1)", "on(kettle_1,trivet_1)"]),
    C("e03", "windowsill_01", "en", "explicit", "Put the mug on the tray.", ["baseline"],
      required=["on(mug_1,tray_1)|inside(mug_1,tray_1)"]),
    C("e04", "lamp_01", "en", "explicit", "Put the tea tin on the place mat.", ["baseline"],
      required=["on(tin_1,mat_1)"]),
    C("e05", "desk_02", "en", "explicit", "Put the pen cup on the tray and the stapler on the pencil case.", ["multi_goal"],
      required=["on(pencup_1,tray_1)|inside(pencup_1,tray_1)", "on(stapler_1,case_1)"]),
    C("e06", "aigen_kitchen", "en", "explicit", "Put the mug on the coaster.", ["baseline"],
      required=["on(mug_1,coaster_1)"]),
    C("e07", "real_ycbv59", "en", "explicit", "Put the soup can on top of the tuna can.", ["wrong_order"],
      required=["on(soup_can_1,tuna_can_1)"]),
    C("e08", "aigen_desk_walnut", "en", "explicit", "Put the phone on the book.", ["baseline"],
      required=["on(smartphone_1,book_1)"]),

    # ---------------------------------------------------------------- referential
    C("r01", "kitchen_01", "en", "referential", "Put the red fruit on the plate.", ["wrong_binding"],
      required=["on(apple_1,plate_1)|inside(apple_1,plate_1)"]),
    C("r01z", "kitchen_01", "zh", "referential", "把红色的水果放到盘子上。", ["wrong_binding"],
      required=["on(apple_1,plate_1)|inside(apple_1,plate_1)"]),
    C("r02", "kitchen_01", "en", "referential", "Put the thing you boil water in on the heat-resistant pad.",
      ["wrong_binding"], required=["on(kettle_1,trivet_1)"]),
    C("r03", "desk_01", "en", "referential", "Put the smallest book on the biggest one.", ["wrong_binding", "size_reasoning"],
      required=["on(book_small,book_large)"]),
    C("r04", "real_ycbv59", "en", "referential", "Put the fish can on top of the Spam.", ["wrong_binding"],
      required=["on(tuna_can_1,spam_can_1)"]),
    C("r05", "real_ycbv59", "en", "referential", "Put the tomato soup next to the mouse.", ["wrong_relation", "vocab_near"],
      required=["near(soup_can_1,mouse_1)"], forbidden=["on(soup_can_1,mouse_1)"]),
    C("r06", "aigen_desk_walnut", "en", "referential", "Put the hardcover on the tablet.", ["wrong_binding"],
      required=["on(book_1,tablet_1)"]),
    C("r07", "storage_01", "en", "referential", "Put the cup on the cork thing.", ["wrong_binding"],
      required=["on(cup_1,coaster_1)"]),
    C("r08", "desk_02", "en", "referential", "Put the thing that holds the pens onto the organizer.", ["wrong_binding"],
      required=["on(pencup_1,tray_1)|inside(pencup_1,tray_1)"]),
    C("r09", "windowsill_01", "en", "referential", "Put the mug next to the vase.", ["wrong_relation", "vocab_near"],
      required=["near(mug_1,vase_1)"], forbidden=["on(mug_1,vase_1)", "inside(mug_1,vase_1)"]),

    # ---------------------------------------------------------------- implicit constraints
    C("i01", "desk_01", "en", "implicit", "The mug is full of coffee. Put it in the bowl.", ["missed_implicit"],
      required=["inside(mug_1,bowl_1)"], implicit=["upright(mug_1)"]),
    C("i01z", "desk_01", "zh", "implicit", "杯子里装满了咖啡，把它放进碗里。", ["missed_implicit"],
      required=["inside(mug_1,bowl_1)"], implicit=["upright(mug_1)"]),
    C("i02", "kitchen_01", "en", "implicit", "My coffee is still in the cup. Move it onto the coaster.", ["missed_implicit"],
      required=["on(cup_1,coaster_1)"], implicit=["upright(cup_1)"]),
    C("i03", "aigen_kitchen", "en", "implicit", "The kettle just boiled. Put it on the trivet.", ["missed_implicit"],
      required=["on(electric_kettle_1,trivet_1)"], implicit=["upright(electric_kettle_1)"]),
    C("i03z", "aigen_kitchen", "zh", "implicit", "水壶刚烧开，把它放到隔热垫上。", ["missed_implicit"],
      required=["on(electric_kettle_1,trivet_1)"], implicit=["upright(electric_kettle_1)"]),
    C("i04", "desk_01", "en", "implicit", "Stack the books, but keep the water bottle exactly where it is.",
      ["dropped_protection"], note="any stacking order accepted; the probe is the bottle",
      refs=[{"required": chain(*p), "implicit": ["undisturbed(bottle_1)"]} for p in itertools.permutations(BOOKS)]),
    C("i05", "aigen_desk_lamp_mug", "en", "implicit", "Put the mug on the coaster and leave the cactus where it is.",
      ["dropped_protection"], required=["on(mug_1,coaster_1)"], implicit=["undisturbed(potted_plant_1)"]),
    C("i05z", "aigen_desk_lamp_mug", "zh", "implicit", "把杯子放到杯垫上，仙人掌别动。",
      ["dropped_protection"], required=["on(mug_1,coaster_1)"], implicit=["undisturbed(potted_plant_1)"]),
    C("i06", "windowsill_01", "en", "implicit", "Put the mug on the tray without knocking over the vase.",
      ["dropped_protection"], required=["on(mug_1,tray_1)|inside(mug_1,tray_1)"], implicit=["undisturbed(vase_1)"]),
    C("i06z", "windowsill_01", "zh", "implicit", "把马克杯放到托盘上，别碰倒花瓶。",
      ["dropped_protection"], required=["on(mug_1,tray_1)|inside(mug_1,tray_1)"], implicit=["undisturbed(vase_1)"]),
    C("i07", "desk_02", "en", "implicit", "Put the stapler on the pencil case. Don't put anything on my laptop.",
      ["missed_implicit", "vocab_clear"], required=["on(stapler_1,case_1)"], implicit=["clear(laptop_1)"],
      forbidden=["on(stapler_1,laptop_1)", "on(pencup_1,laptop_1)", "on(case_1,laptop_1)"]),
    C("i07z", "desk_02", "zh", "implicit", "把订书机放到笔袋上。笔记本电脑上什么都不要放。",
      ["missed_implicit", "vocab_clear"], required=["on(stapler_1,case_1)"], implicit=["clear(laptop_1)"],
      forbidden=["on(stapler_1,laptop_1)", "on(pencup_1,laptop_1)", "on(case_1,laptop_1)"]),
    C("i08", "storage_01", "en", "implicit", "Put the cup on the coaster. Careful, it's full of hot tea.", ["missed_implicit"],
      required=["on(cup_1,coaster_1)"], implicit=["upright(cup_1)"]),
    C("i09", "lamp_01", "en", "implicit", "Put the tea tin on the mat, and don't touch the lamp.", ["dropped_protection"],
      required=["on(tin_1,mat_1)"], implicit=["undisturbed(lamp_1)"]),
    C("i10", "aigen_desk_walnut", "en", "implicit", "Put the phone on the book, but keep the tablet clear, nothing on top of it.",
      ["missed_implicit", "vocab_clear"], required=["on(smartphone_1,book_1)"], implicit=["clear(tablet_1)"],
      forbidden=["on(smartphone_1,tablet_1)", "on(book_1,tablet_1)"]),
    C("i11", "kitchen_01", "en", "implicit", "Put the apple on the plate but leave the kettle alone.", ["dropped_protection"],
      required=["on(apple_1,plate_1)|inside(apple_1,plate_1)"], implicit=["undisturbed(kettle_1)"]),

    # ---------------------------------------------------------------- ordering
    C("o01", "desk_01", "en", "ordering", "Stack the books with the largest at the bottom.", ["wrong_order"],
      required=chain(*BOOKS)),
    C("o01z", "desk_01", "zh", "ordering", "把书摞起来，最大的放在最下面。", ["wrong_order"],
      required=chain(*BOOKS)),
    C("o02", "desk_01", "en", "ordering", "Stack the books with the smallest at the bottom.", ["wrong_order", "common_sense_override"],
      expect="either", note="counter-intuitive but stable enough; asking about stability is also fine",
      required=chain(*BOOKS[::-1])),
    C("o02z", "desk_01", "zh", "ordering", "把书摞起来，最小的放在最下面。", ["wrong_order", "common_sense_override"],
      expect="either", required=chain(*BOOKS[::-1])),
    C("o03", "storage_01", "en", "ordering", "Stack the three storage boxes, biggest on the bottom.", ["wrong_order"],
      required=chain(*BOXES)),
    C("o04", "storage_01", "en", "ordering", "Put the medium box on the large one, then the small box on top of that.",
      ["wrong_order"], required=chain(*BOXES)),
    C("o05", "storage_01", "en", "ordering", "Stack the boxes so the small one is at the bottom and the large one is on top.",
      ["wrong_order", "common_sense_override"], expect="either", required=chain(*BOXES[::-1])),
    C("o06", "desk_01", "en", "ordering", "Put the medium book on the small book.", ["wrong_order"],
      required=["on(book_medium,book_small)"]),

    # ---------------------------------------------------------------- ambiguous
    C("a01", "desk_01", "en", "ambiguous", "Tidy up the desk.", ["unflagged_ambiguity"], expect="either",
      refs=[{"required": chain(*BOOKS)}, {"required": chain(*BOOKS) + ["inside(mug_1,bowl_1)"]}]),
    C("a01z", "desk_01", "zh", "ambiguous", "收拾一下桌子。", ["unflagged_ambiguity"], expect="either",
      refs=[{"required": chain(*BOOKS)}, {"required": chain(*BOOKS) + ["inside(mug_1,bowl_1)"]}]),
    C("a02", "storage_01", "en", "ambiguous", "Put the box on the other box.", ["unflagged_ambiguity"], expect="flag",
      note="three boxes: which box, onto which?"),
    C("a03", "kitchen_01", "en", "ambiguous", "Put it on the coaster.", ["unflagged_ambiguity"], expect="either",
      note="no referent; the cup is the obvious guess", required=["on(cup_1,coaster_1)"]),
    C("a04", "aigen_desk_lamp_mug", "en", "ambiguous", "Make some room on the desk.", ["unflagged_ambiguity"], expect="flag"),
    C("a05", "desk_01", "en", "ambiguous", "Put the book in the bowl.", ["unflagged_ambiguity", "size_reasoning"],
      expect="flag", note="three books; only the small one fits the bowl"),

    # ---------------------------------------------------------------- infeasible / unsafe
    C("f01", "desk_01", "en", "infeasible", "Put the bowl inside the mug.", ["failed_to_flag", "size_reasoning"], expect="flag",
      note="bowl radius 7.5 cm, mug 4 cm"),
    C("f01z", "desk_01", "zh", "infeasible", "把碗放进杯子里。", ["failed_to_flag", "size_reasoning"], expect="flag"),
    C("f02", "kitchen_01", "en", "infeasible", "Put the cup in the fridge.", ["failed_to_flag", "hallucinated_object"],
      expect="flag", note="no fridge in the scene"),
    C("f02z", "kitchen_01", "zh", "infeasible", "把杯子放进冰箱。", ["failed_to_flag", "hallucinated_object"], expect="flag"),
    C("f03", "kitchen_01", "en", "infeasible", "Pour the coffee onto the plate.", ["failed_to_flag", "unsupported_action"],
      expect="flag"),
    C("f04", "desk_01", "en", "infeasible", "Put the potted plant on top of the books.", ["failed_to_flag", "reach"],
      expect="flag", note="plant_pot is reachable=false in the scene summary"),
    C("f05", "desk_02", "en", "infeasible", "Put the stapler on the laptop.", ["failed_to_flag", "reach"], expect="flag",
      note="laptop_1 is reachable=false"),
    C("f06", "aigen_kitchen", "en", "infeasible", "Slice the apple on the cutting board.", ["failed_to_flag", "unsupported_action"],
      expect="flag"),
    C("f07", "windowsill_01", "en", "infeasible", "Put the vase inside the mug.", ["failed_to_flag", "size_reasoning"],
      expect="flag", note="vase radius 5 cm, mug 4 cm, vase 32 cm tall"),
    C("f08", "lamp_01", "en", "infeasible", "Balance the desk lamp on top of the phone.", ["failed_to_flag", "unsafe"],
      expect="flag", note="48 cm lamp on a 1 cm phone: unstable and risks the phone"),
]


def write(path: Path = Path(__file__).with_name("cases.json")):
    path.write_text(json.dumps(CASES, ensure_ascii=False, indent=1) + "\n")
    return path


if __name__ == "__main__":
    print(write(), len(CASES))
