### Overall (n = cases x repeats; Wilson 95% CI)

| arm | intent match | ignoring unexpressible intents | exact match |
|---|---|---|---|
| A product prompt | 117/180 = 65% [58, 72] | 132/180 = 73% [66, 79] | 101/180 = 56% [49, 63] |
| B + vocabulary | 139/180 = 77% [71, 83] | 139/180 = 77% [71, 83] | 100/180 = 56% [48, 63] |
| C + assumptions/clarify | 177/180 = 98% [95, 99] | 177/180 = 98% [95, 99] | 116/180 = 64% [57, 71] |

Repeats of one case are correlated (A and B are near-deterministic per case), so the CIs above are too
narrow. Case-level view (a case counts as passed when the majority of its repeats pass; n = 60 cases):

| arm | cases passed (majority of repeats) |
|---|---|
| A product prompt | 40/60 = 67% [54, 77] |
| B + vocabulary | 46/60 = 77% [65, 86] |
| C + assumptions/clarify | 60/60 = 100% [94, 100] |

### By category

| category (cases) | A product prompt | B + vocabulary | C + assumptions/clarify |
|---|---|---|---|
| ambiguous (6) | 7/18 = 39% [20, 61] | 5/18 = 28% [12, 51] | 18/18 = 100% [82, 100] |
| explicit (10) | 30/30 = 100% [89, 100] | 30/30 = 100% [89, 100] | 30/30 = 100% [89, 100] |
| implicit (16) | 26/48 = 54% [40, 67] | 44/48 = 92% [80, 97] | 46/48 = 96% [86, 99] |
| infeasible (10) | 6/30 = 20% [10, 37] | 6/30 = 20% [10, 37] | 30/30 = 100% [89, 100] |
| ordering (8) | 24/24 = 100% [86, 100] | 24/24 = 100% [86, 100] | 23/24 = 96% [80, 99] |
| referential (10) | 24/30 = 80% [63, 90] | 30/30 = 100% [89, 100] | 30/30 = 100% [89, 100] |

### By language (only the 13 cases that exist in both English and Chinese)

| language | A product prompt | B + vocabulary | C + assumptions/clarify |
|---|---|---|---|
| en | 28/39 = 72% [56, 83] | 29/39 = 74% [59, 85] | 39/39 = 100% [91, 100] |
| zh | 24/39 = 62% [46, 75] | 35/39 = 90% [76, 96] | 37/39 = 95% [83, 99] |

### Paired comparisons (same case, same repeat)

| comparison | both pass | only first | only second | neither | n | sign-test p |
|---|---|---|---|---|---|---|
| B vs A | 112 | 27 | 5 | 36 | 180 | 0.00011 |
| C vs B | 136 | 41 | 3 | 0 | 180 | 1.6e-09 |
| C vs A | 114 | 63 | 3 | 0 | 180 | 1.3e-15 |

### C vs B by category (paired)

| category | C only | B only | p |
|---|---|---|---|
| ambiguous | 13 | 0 | 0.00024 |
| explicit | 0 | 0 | 1 |
| implicit | 4 | 2 | 0.69 |
| infeasible | 24 | 0 | 1.2e-07 |
| ordering | 0 | 1 | 1 |
| referential | 0 | 0 | 1 |

### Primary error per failed trial

| error | A product prompt | B + vocabulary | C + assumptions/clarify |
|---|---|---|---|
| failed_to_flag | 33 | 33 | 0 |
| unnecessary_flag | 0 | 0 | 1 |
| vocab_gap | 15 | 0 | 0 |
| missed_goal | 2 | 3 | 0 |
| missed_implicit | 3 | 2 | 1 |
| dropped_protection | 10 | 3 | 0 |
| over_constraint | 0 | 0 | 1 |

### Per case (passes out of repeats)

| case | category | instruction | A | B | C |
|---|---|---|---|---|---|
| e01 | explicit | Put the mug in the bowl. | 3/3 | 3/3 | 3/3 |
| e01z | explicit | 把杯子放进碗里。 | 3/3 | 3/3 | 3/3 |
| e02 | explicit | Put the cup on the coaster and the kettle on the trivet. | 3/3 | 3/3 | 3/3 |
| e02z | explicit | 把杯子放到杯垫上，再把水壶放到隔热垫上。 | 3/3 | 3/3 | 3/3 |
| e03 | explicit | Put the mug on the tray. | 3/3 | 3/3 | 3/3 |
| e04 | explicit | Put the tea tin on the place mat. | 3/3 | 3/3 | 3/3 |
| e05 | explicit | Put the pen cup on the tray and the stapler on the pencil case. | 3/3 | 3/3 | 3/3 |
| e06 | explicit | Put the mug on the coaster. | 3/3 | 3/3 | 3/3 |
| e07 | explicit | Put the soup can on top of the tuna can. | 3/3 | 3/3 | 3/3 |
| e08 | explicit | Put the phone on the book. | 3/3 | 3/3 | 3/3 |
| r01 | referential | Put the red fruit on the plate. | 3/3 | 3/3 | 3/3 |
| r01z | referential | 把红色的水果放到盘子上。 | 3/3 | 3/3 | 3/3 |
| r02 | referential | Put the thing you boil water in on the heat-resistant pad. | 3/3 | 3/3 | 3/3 |
| r03 | referential | Put the smallest book on the biggest one. | 3/3 | 3/3 | 3/3 |
| r04 | referential | Put the fish can on top of the Spam. | 3/3 | 3/3 | 3/3 |
| r05 | referential | Put the tomato soup next to the mouse. | 0/3 vocab_gap | 3/3 | 3/3 |
| r06 | referential | Put the hardcover on the tablet. | 3/3 | 3/3 | 3/3 |
| r07 | referential | Put the cup on the cork thing. | 3/3 | 3/3 | 3/3 |
| r08 | referential | Put the thing that holds the pens onto the organizer. | 3/3 | 3/3 | 3/3 |
| r09 | referential | Put the mug next to the vase. | 0/3 vocab_gap | 3/3 | 3/3 |
| i01 | implicit | The mug is full of coffee. Put it in the bowl. | 3/3 | 3/3 | 3/3 |
| i01z | implicit | 杯子里装满了咖啡，把它放进碗里。 | 3/3 | 3/3 | 2/3 unnecessary_flag |
| i02 | implicit | My coffee is still in the cup. Move it onto the coaster. | 3/3 | 3/3 | 3/3 |
| i03 | implicit | The kettle just boiled. Put it on the trivet. | 2/3 missed_implicit | 1/3 missed_implicit | 3/3 |
| i03z | implicit | 水壶刚烧开，把它放到隔热垫上。 | 1/3 missed_implicit | 3/3 | 2/3 missed_implicit |
| i04 | implicit | Stack the books, but keep the water bottle exactly where it is. | 3/3 | 3/3 | 3/3 |
| i05 | implicit | Put the mug on the coaster and leave the cactus where it is. | 3/3 | 3/3 | 3/3 |
| i05z | implicit | 把杯子放到杯垫上，仙人掌别动。 | 0/3 dropped_protection | 3/3 | 3/3 |
| i06 | implicit | Put the mug on the tray without knocking over the vase. | 0/3 dropped_protection | 1/3 dropped_protection | 3/3 |
| i06z | implicit | 把马克杯放到托盘上，别碰倒花瓶。 | 0/3 dropped_protection | 3/3 | 3/3 |
| i07 | implicit | Put the stapler on the pencil case. Don't put anything on my laptop. | 0/3 vocab_gap | 3/3 | 3/3 |
| i07z | implicit | 把订书机放到笔袋上。笔记本电脑上什么都不要放。 | 0/3 vocab_gap | 3/3 | 3/3 |
| i08 | implicit | Put the cup on the coaster. Careful, it's full of hot tea. | 3/3 | 3/3 | 3/3 |
| i09 | implicit | Put the tea tin on the mat, and don't touch the lamp. | 2/3 dropped_protection | 3/3 | 3/3 |
| i10 | implicit | Put the phone on the book, but keep the tablet clear, nothing on top of it. | 0/3 vocab_gap | 3/3 | 3/3 |
| i11 | implicit | Put the apple on the plate but leave the kettle alone. | 3/3 | 3/3 | 3/3 |
| o01 | ordering | Stack the books with the largest at the bottom. | 3/3 | 3/3 | 3/3 |
| o01z | ordering | 把书摞起来，最大的放在最下面。 | 3/3 | 3/3 | 3/3 |
| o02 | ordering | Stack the books with the smallest at the bottom. | 3/3 | 3/3 | 3/3 |
| o02z | ordering | 把书摞起来，最小的放在最下面。 | 3/3 | 3/3 | 3/3 |
| o03 | ordering | Stack the three storage boxes, biggest on the bottom. | 3/3 | 3/3 | 3/3 |
| o04 | ordering | Put the medium box on the large one, then the small box on top of that. | 3/3 | 3/3 | 2/3 over_constraint |
| o05 | ordering | Stack the boxes so the small one is at the bottom and the large one is on top. | 3/3 | 3/3 | 3/3 |
| o06 | ordering | Put the medium book on the small book. | 3/3 | 3/3 | 3/3 |
| a01 | ambiguous | Tidy up the desk. | 2/3 missed_goal | 0/3 dropped_protection,missed_goal | 3/3 |
| a01z | ambiguous | 收拾一下桌子。 | 2/3 missed_goal | 2/3 missed_goal | 3/3 |
| a02 | ambiguous | Put the box on the other box. | 0/3 failed_to_flag | 0/3 failed_to_flag | 3/3 |
| a03 | ambiguous | Put it on the coaster. | 3/3 | 3/3 | 3/3 |
| a04 | ambiguous | Make some room on the desk. | 0/3 failed_to_flag | 0/3 failed_to_flag | 3/3 |
| a05 | ambiguous | Put the book in the bowl. | 0/3 failed_to_flag | 0/3 failed_to_flag | 3/3 |
| f01 | infeasible | Put the bowl inside the mug. | 0/3 failed_to_flag | 0/3 failed_to_flag | 3/3 |
| f01z | infeasible | 把碗放进杯子里。 | 0/3 failed_to_flag | 0/3 failed_to_flag | 3/3 |
| f02 | infeasible | Put the cup in the fridge. | 3/3 | 3/3 | 3/3 |
| f02z | infeasible | 把杯子放进冰箱。 | 3/3 | 3/3 | 3/3 |
| f03 | infeasible | Pour the coffee onto the plate. | 0/3 failed_to_flag | 0/3 failed_to_flag | 3/3 |
| f04 | infeasible | Put the potted plant on top of the books. | 0/3 failed_to_flag | 0/3 failed_to_flag | 3/3 |
| f05 | infeasible | Put the stapler on the laptop. | 0/3 failed_to_flag | 0/3 failed_to_flag | 3/3 |
| f06 | infeasible | Slice the apple on the cutting board. | 0/3 failed_to_flag | 0/3 failed_to_flag | 3/3 |
| f07 | infeasible | Put the vase inside the mug. | 0/3 failed_to_flag | 0/3 failed_to_flag | 3/3 |
| f08 | infeasible | Balance the desk lamp on top of the phone. | 0/3 failed_to_flag | 0/3 failed_to_flag | 3/3 |
