# Related work

How Rehearsal relates to prior work. Every entry below was checked on 2026-10-31 against arXiv,
the venue's proceedings page, or the project page. Venue years follow the conference, not the
proceedings print date (e.g. CoRL 2024 is printed as PMLR vol. 270, dated 2025).
"Unverified" marks anything we could not confirm.

**Our system in one line, for comparison.** One RGB phone photo plus an A4 sheet for metric scale
→ Muse Spark lists objects and guesses physical properties → SAM 3.1 segments them → a MuJoCo scene
built from **primitive shapes** (box, cylinder, bowl) with a Franka Panda that grasps by an
**idealised suction weld** → Muse Spark turns the instruction into predicates and writes a symbolic
plan (`transport(obj, dest, in|on, transit_height)`) → MuJoCo runs it, the predicates are checked
on the simulator's ground truth, and on failure the model gets the structured log back and
revises (at most 3 attempts) → the final plan is re-run under randomised perception error,
friction and mass, with common random numbers → re-observation is ranked by Monte-Carlo
value of information. **There is no real-robot execution.**

---

## 1. Real-to-sim scene reconstruction

| Work | What it does | How Rehearsal differs |
|---|---|---|
| **ACDC: Automated Creation of Digital Cousins for Robust Policy Learning.** Dai, Wong, Jiang, Wang, Gokmen, Zhang, Wu, Fei-Fei. CoRL 2024. [arXiv:2410.07408](https://arxiv.org/abs/2410.07408) | Turns one RGB image into many similar but not identical simulated scenes ("cousins") built from asset matching. Policies trained on cousins transfer zero-shot better than policies trained on a single twin (90% vs 25%). | ACDC uses the scene to **train policies**. We use it to **check one specific plan** before anything runs. Their result argues against trusting a single exact twin. Our stress test (many perturbed copies of the scene) follows the same idea for evaluation. *Weaker:* ACDC has articulated asset matching and real-robot transfer. We have primitive shapes and no robot. |
| **RialTo: Reconciling Reality through Simulation: A Real-to-Sim-to-Real Approach for Robust Manipulation.** Torne, Simeonov, Li, Chan, Chen, Gupta, Agrawal. RSS 2024. [arXiv:2403.03949](https://arxiv.org/abs/2403.03949) | Builds a digital twin from a scan, then fine-tunes imitation policies with RL inside it for robustness. | Their twin is built with a scanning step and a human GUI. Ours comes from one photo with no robotics expertise. They improve a *policy*. We verify and diagnose a *plan*. *Weaker:* their twins are much more faithful and validated on hardware. |
| **SIMPLER: Evaluating Real-World Robot Manipulation Policies in Simulation.** Li, Hsu, Gu, Pertsch, Mees, Walke, Fu, Lunawat, Sieh, Kirmani, Levine, Wu, Finn, Su, Vuong, Xiao. CoRL 2024. [arXiv:2405.05941](https://arxiv.org/abs/2405.05941) | Simulated environments whose policy rankings correlate strongly with real-world results, measured over ~1,500 paired sim/real episodes. | SIMPLER is the bar for "sim predicts real": it **measures** sim–real correlation. We have **not measured** this. Any "predicts real-world success" wording needs a SIMPLER-style paired study, which we cannot run. |
| **URDFormer: A Pipeline for Constructing Articulated Simulation Environments from Real-World Images.** Chen, Walsman, Memmel, Mo, Fang, Vemuri, Wu, Fox, Gupta. RSS 2024. [arXiv:2405.11656](https://arxiv.org/abs/2405.11656) | Image → articulated URDF scene (cabinets, drawers), trained on synthetic paired data. | We model only rigid tabletop objects as primitives. *Weaker:* no articulation. *Different:* we estimate per-object physical properties with ranges and use those ranges in the stress test. |
| **Real2Code: Reconstruct Articulated Objects via Code Generation.** Mandi, Weng, Bauer, Song. ICLR 2025. [arXiv:2406.08474](https://arxiv.org/abs/2406.08474) | A fine-tuned LLM writes code describing an articulated object's joints from part bounding boxes. | Same "LLM writes the structured spec" pattern, but applied to object geometry. Our LLM writes the *task spec* (predicates) and the plan. |
| **GRS: Generating Robotic Simulation Tasks from Real-World Images.** Zook et al. (NVIDIA, Stanford). CVPR 2025 Workshop (CV2). [arXiv:2410.15536](https://arxiv.org/abs/2410.15536) | One RGB-D image → SAM2 + VLM → digital twin built from matched assets, plus tasks and test suites that are refined in a loop until they can be solved. | Close in pipeline shape. Their loop refines the *task and sim code* so it is solvable. Ours refines the *robot plan* against fixed physics. They need depth. We use RGB plus a paper sheet for scale. |
| **RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins.** Mu, Chen, et al. CVPR 2025. [arXiv:2504.13059](https://arxiv.org/abs/2504.13059) (early version [arXiv:2409.02920](https://arxiv.org/abs/2409.02920)) | 2D images → generated 3D assets, and LLM-written expert programs for a dual-arm benchmark. | A benchmark and data generator, not a tool for one user's scene. Uses generative 3D assets, where we use primitives. |
| **SAM 3D: 3Dfy Anything in Images.** Meta (SAM 3D team). 2025. [arXiv:2511.16624](https://arxiv.org/abs/2511.16624) | Single-image object geometry, texture and layout. | This is the planned upgrade path from primitives to meshes. It is **not wired in** at the time of writing (see README). |
| **SAM 3: Segment Anything with Concepts.** Carion et al. (Meta). ICLR 2026. [arXiv:2511.16719](https://arxiv.org/abs/2511.16719) | Promptable concept segmentation: a noun phrase returns masks for every matching instance. | We use SAM 3.1 through the Meta Model API in exactly this way (Muse Spark lists the nouns, SAM segments them). We did not verify a separate paper for "3.1"; cite SAM 3. |

## 2. LLM task planning with feedback and self-correction

| Work | What it does | How Rehearsal differs |
|---|---|---|
| **Code as Policies: Language Model Programs for Embodied Control.** Liang, Huang, Xia, Xu, Hausman, Ichter, Florence, Zeng. ICRA 2023. [arXiv:2209.07753](https://arxiv.org/abs/2209.07753) | An LLM writes robot policy code from instructions, calling perception and control primitives. | Our action space is much narrower: one parameterised `transport`. We use the LLM for specification (predicates) and repair, not for writing open-ended code. |
| **Inner Monologue: Embodied Reasoning through Planning with Language Models.** Huang, Xia, Xiao, Chan, Liang, Florence, Zeng, Tompson, Mordatch, Chebotar, Sermanet, Brown, Jackson, Luu, Levine, Hausman, Ichter. CoRL 2022. [arXiv:2207.05608](https://arxiv.org/abs/2207.05608) | Closed-loop replanning from natural-language feedback (success detectors, scene descriptions, humans) on real robots. | The closest ancestor of our revise loop. Differences: our feedback comes from a **simulator before execution** (counterfactual and free to repeat) rather than from the real world during execution. It is **structured** (timestamped contacts between named bodies, IK errors, which predicate failed and by how many cm). *Weaker:* our feedback is only as true as the sim. Inner Monologue's feedback is real. |
| **REFLECT: Summarizing Robot Experiences for Failure Explanation and Correction.** Liu, Bahety, Song. CoRL 2023. [arXiv:2306.15724](https://arxiv.org/abs/2306.15724) | Turns multisensory robot logs into a hierarchical summary. An LLM explains the failure, and that explanation guides correction. | **The most direct prior for "the model reads the log and explains the failure."** Do not claim log-based failure explanation is new. What differs: our log comes from a simulator, so we have privileged ground truth (exact contacts, no perception noise in the log itself). And the explanation is *tested*: the revised plan is re-run and the predicates either pass or fail. |
| **DoReMi: Grounding Language Model by Detecting and Recovering from Plan-Execution Misalignment.** Guo, Wang, Zha, Jiang, Chen. arXiv 2307.00329 (2023; venue unverified). [arXiv:2307.00329](https://arxiv.org/abs/2307.00329) | The LLM generates constraints alongside the plan. A VLM watches execution for constraint violations and triggers recovery. | Their constraints resemble our compiled predicates, but are checked by a VLM on real video. Ours are checked geometrically on simulator state, which is exact but only valid inside the sim. |
| **LLM+P: Empowering Large Language Models with Optimal Planning Proficiency.** Liu, Jiang, Zhang, Liu, Zhang, Biswas, Stone. arXiv 2304.11477 (2023; venue unverified). [arXiv:2304.11477](https://arxiv.org/abs/2304.11477) | The LLM translates the problem into PDDL, and a classical planner solves it. | Same split as ours ("LLM translates, a sound tool decides"). Our sound tool is a physics engine checking continuous geometry, not a symbolic planner. We have no optimality or completeness guarantees. |
| **KnowNo: Robots That Ask For Help: Uncertainty Alignment for Large Language Model Planners.** Ren, Dixit, Bodrova, Singh, Tu, Brown, Xu, Takayama, Xia, Varley, Xu, Sadigh, Zeng, Majumdar. CoRL 2023. [arXiv:2307.01928](https://arxiv.org/abs/2307.01928) | Conformal prediction over LLM plan options decides when to ask a human, with a statistical coverage guarantee. | KnowNo addresses uncertainty in the *language* (which plan was meant). Our "look again" addresses *geometric* perception uncertainty. KnowNo has formal guarantees. Our stress-test numbers are Monte-Carlo estimates without calibration to reality. |

## 3. Simulation in the loop to verify or refine LLM/VLM plans

| Work | What it does | How Rehearsal differs |
|---|---|---|
| **SIMPACT: Simulation-Enabled Action Planning using Vision-Language Models.** Liu, Yao, Chen, Gao, Mao, Huang, Du. CVPR 2026. [arXiv:2512.05955](https://arxiv.org/abs/2512.05955) | **Closest prior work.** One RGB-D image → GroundedSAM2 + Hunyuan3D meshes + FoundationPose poses, with VLM-estimated mass, friction and CoM → MuJoCo (plus MPM for deformables). The VLM proposes primitive sequences, reads rendered and numeric rollout results, and refines them. A VLM judge decides success. Then it **executes on a real Franka** (7 tasks, 40–90% success). | Be explicit with judges that SIMPACT exists. Honest differences: (a) **success is decided by checkable predicates on sim state** (including an automatically added "nothing else moved" constraint), not by a VLM judge. (b) feedback is a structured contact/IK/predicate log, not renders. (c) **we quantify robustness** to perception and physics uncertainty with paired Monte-Carlo rollouts. SIMPACT reports low variance of VLM physics estimates but does not model perception uncertainty probabilistically. (d) **we rank which object to re-observe** (VoI); SIMPACT has no active perception. (e) RGB plus a paper sheet, no depth sensor. *Where we are weaker:* no real-robot validation at all, primitive geometry, an idealised suction grasp, no deformables, a much narrower action space. |
| **Mind's Eye: Grounded Language Model Reasoning through Simulation.** Liu, Wei, Gu, Wu, Vosoughi, Cui, Zhou, Dai. ICLR 2023. [arXiv:2210.05359](https://arxiv.org/abs/2210.05359) | Runs MuJoCo to simulate the outcome of a physics question and puts the result in the LLM's context. | The same principle ("let the physics engine answer what the LLM would guess"), applied to QA. We apply it to a user's real scene and a robot plan. |
| **SayPlan: Grounding Large Language Models using 3D Scene Graphs for Scalable Robot Task Planning.** Rana, Haviland, Garg, Abou-Chakra, Reid, Suenderhauf. CoRL 2023. [arXiv:2307.06135](https://arxiv.org/abs/2307.06135) | Iterative replanning against a *scene-graph* simulator that returns symbolic feedback (e.g. "door locked"). | Their verifier is symbolic and building-scale. Ours is physical (contacts, stability, IK) and tabletop-scale. |
| **Text2Motion: From Natural Language Instructions to Feasible Plans.** Lin, Agia, Migimatsu, Pavone, Bohg. Autonomous Robots 2023. [arXiv:2303.12153](https://arxiv.org/abs/2303.12153) | LLM task planning plus geometric feasibility checking with learned skill value functions, and a goal-satisfaction check. | They check geometric feasibility with learned value functions. We check with a full rigid-body rollout, which is slower but does not need trained skills. They have real-robot results. |
| **AutoTAMP: Autoregressive Task and Motion Planning with LLMs as Translators and Checkers.** Chen, Arkin, Dawson, Zhang, Roy, Fan. ICRA 2024. [arXiv:2306.06531](https://arxiv.org/abs/2306.06531) | The LLM translates NL into signal temporal logic, a TAMP solver plans, and syntax/semantic errors are re-prompted. | **Closest prior for "compile the instruction into checkable specs."** Their specs are STL with a formal solver. Ours are a small fixed predicate vocabulary checked after physical rollout. Theirs is more expressive and formal. Ours captures physical outcomes such as toppling and stability that a kinematic TAMP check misses. |
| **PRoC3S: Trust the PRoC3S: Solving Long-Horizon Robotics Problems with LLMs and Constraint Satisfaction.** Curtis, Kumar, Cao, Lozano-Pérez, Kaelbling. CoRL 2024. [arXiv:2406.05572](https://arxiv.org/abs/2406.05572) | The LLM writes a parameterised plan program. Parameters are sampled to satisfy kinematic, collision and goal constraints in sim, and failures go back to the LLM. | Very similar loop. They search over continuous parameters automatically, where we let the LLM pick them (e.g. transit height) from the log. Their approach is probably more sample-efficient at finding feasible parameters. Ours adds robustness under uncertainty, not just feasibility. |
| **RoboGen** (Wang, Xian, Chen, Wang, Wang, Fragkiadaki, Erickson, Held, Gan; ICML 2024; [arXiv:2311.01455](https://arxiv.org/abs/2311.01455)) and **GenSim** (Wang, Ling, Yuan, Shridhar, Bao, Qin, Wang, Xu, Wang; ICLR 2024; [arXiv:2310.01361](https://arxiv.org/abs/2310.01361)) | An LLM generates *tasks and environments* for data or skill learning. | Opposite direction: they invent scenes. We reconstruct one real scene and must stay faithful to it. |

## 4. Robustness, uncertainty, active perception, value of information

| Work | Relevance |
|---|---|
| **Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World.** Tobin, Fong, Ray, Schneider, Zaremba, Abbeel. IROS 2017. [arXiv:1703.06907](https://arxiv.org/abs/1703.06907) | Our "parallel universes" are domain randomization used for **evaluation** rather than training. Call it that. Judges will recognise the term. |
| **Active Domain Randomization.** Mehta, Diaz, Golemo, Pal, Paull. CoRL 2019. [arXiv:1904.04762](https://arxiv.org/abs/1904.04762) | Learns *which* randomizations are informative. Analogous to our question of which uncertainty matters, but for training. |
| **BayesSim: Adaptive Domain Randomization via Probabilistic Inference for Robotics Simulators.** Ramos, Possas, Fox. RSS 2019. [arXiv:1906.01728](https://arxiv.org/abs/1906.01728) | Infers a posterior over sim parameters from real trajectories. This is how our hand-set ranges (σ = 1.2 cm, friction ×0.6–1.4) *should* eventually be grounded. Ours are priors, not posteriors. |
| **Some Guidelines and Guarantees for Common Random Numbers.** Glasserman, Yao. Management Science 38(6):884–908, 1992. [doi:10.1287/mnsc.38.6.884](https://doi.org/10.1287/mnsc.38.6.884) | The standard reference for CRN as variance reduction when comparing two simulated systems. Our VoI comparison (same seeds, one σ changed) is a textbook CRN design. Cite this, not "parallel universes". |
| **Information Value Theory.** R. A. Howard. IEEE Trans. Systems Science and Cybernetics, 1966. [doi:10.1109/TSSC.1966.300074](https://doi.org/10.1109/TSSC.1966.300074) | The origin of decision-theoretic value of information: information is worth what it changes in expected outcome, not its entropy. Our score (Δ success probability after re-observing one object) is a Monte-Carlo, decision-relevant VoI in this sense, with a fixed symbolic plan (see claims.md). |
| **Active Perception.** R. Bajcsy. Proc. IEEE 76(8):966–1005, 1988. [doi:10.1109/5.5968](https://doi.org/10.1109/5.5968) | The classic framing: perception as control that chooses observations to reduce task loss. |
| **Integrated Task and Motion Planning in Belief Space.** Kaelbling, Lozano-Pérez. IJRR 32(9–10):1194–1227, 2013. [MIT DSpace](https://dspace.mit.edu/handle/1721.1/87038) | The principled version of what we approximate: planning *over beliefs*, with observation actions chosen inside the planner. A Meta researcher may ask "why not belief-space planning?" Answer: we do not plan in belief space. We **evaluate** a fixed plan under a belief, then rank one-step observations by how much they improve it. That is cheaper, myopic and greedy. |

## 5. Adam Harley's work (Harley is a Research Scientist at Meta, per [adamharley.com](https://adamharley.com))

| Work | Connection |
|---|---|
| **Move to See Better: Self-Improving Embodied Object Detection.** Fang, Jain, Sarch, Harley, Fragkiadaki. BMVC 2021. [arXiv:2012.00057](https://arxiv.org/abs/2012.00057) | An embodied agent moves to collect multi-view data and self-improves its detector. The spirit of "look again" is the same. The difference: they look again to **improve the perception model**, while we look again to **improve the robustness of one specific plan**, choosing which object by its downstream effect on task success. |
| **TIDEE: Tidying Up Novel Rooms using Visuo-Semantic Commonsense Priors.** Sarch, Fang, Harley, Schydlo, Tarr, Gupta, Fragkiadaki. ECCV 2022. [arXiv:2207.10761](https://arxiv.org/abs/2207.10761) | Commonsense rearrangement in AI2-THOR: detect out-of-place objects, propose receptacles, search for them. TIDEE decides *what tidy means* without instructions. We take an explicit instruction and decide *whether this robot can physically do it here*. These are complementary: TIDEE-style priors could propose our goals. |
| **AllTracker: Efficient Dense Point Tracking at High Resolution.** Harley et al. ICCV 2025. [arXiv:2506.07310](https://arxiv.org/abs/2506.07310) | Not used. Future work: a short phone video instead of a photo would let dense tracks recover depth and pose, and reduce the σ that drives our stress test. |
| **TAPIP3D: Tracking Any Point in Persistent 3D Geometry.** Zhang, Ke, Harley, Fragkiadaki. NeurIPS 2025. [arXiv:2504.14717](https://arxiv.org/abs/2504.14717) | Not used. 3D point tracks could close our sim-to-real loop: compare the real outcome's object tracks with the simulated trajectories we already record at 50 Hz. |

Do not claim these works as components. They are inspiration and future work.

## 6. Rearrangement benchmarks

| Work | Relevance |
|---|---|
| **Rearrangement: A Challenge for Embodied AI.** Batra et al. arXiv 2020. [arXiv:2011.01975](https://arxiv.org/abs/2011.01975) | Defines goal specification by predicates over object states and metrics for rearrangement. Our predicate goals (`inside`, `on`, `stacked`, `upright`) follow this "goal = checkable state predicate" framing. Cite it so the predicate design does not look ad hoc. |
| **Visual Room Rearrangement.** Weihs, Deitke, Kembhavi, Mottaghi. CVPR 2021. [arXiv:2103.16544](https://arxiv.org/abs/2103.16544) | The AI2-THOR rearrangement benchmark (RoomR). Navigation plus interaction. Not physics-faithful manipulation. |
| **Habitat 2.0: Training Home Assistants to Rearrange their Habitat.** Szot, Clegg, Undersander, et al. NeurIPS 2021. [arXiv:2106.14405](https://arxiv.org/abs/2106.14405) | Meta's physics-enabled rearrangement simulator and Home Assistant Benchmark. Relevant to Meta judges: it notes that sense-plan-act pipelines are brittle, which is exactly what our stress test measures for one scene. |
| **Housekeep: Tidying Virtual Households using Commonsense Reasoning.** Kant, Ramachandran, Yenamandra, Gilitschenski, Batra, Szot, Agrawal. ECCV 2022. [arXiv:2205.10712](https://arxiv.org/abs/2205.10712) | LLM-based commonsense tidying in Habitat (Meta co-authors). |

We do **not** evaluate on any of these benchmarks. Our scenes are our own. A judge may ask why;
the honest answer is scope (one-photo tabletop scenes vs. full apartments).

---

## What's new in Rehearsal (defensible version)

Each part (photo→sim, LLM plans checked in sim, log-based repair, domain randomization) has
prior work. SIMPACT (CVPR 2026) in particular already does single-image → MuJoCo → VLM plan
refinement. The defensible novelty is the **combination plus two uncertainty-aware additions**:

1. **Verification against compiled, checkable predicates, including side-effect constraints.**
   The model only *proposes* goals and plans. Success is decided by geometric predicates
   evaluated on simulator state, and a "nothing else moved by > 2 cm" constraint is added
   automatically for every object the instruction does not mention. Failures come back as a
   structured, timestamped log that the model must explain before it revises.
2. **Plan-level robustness estimates under perception and physics uncertainty, with paired
   (common-random-numbers) comparisons.** The output is not "the plan worked once in sim" but
   "k/N under σ-level position error and friction/mass variation". Plan variants are compared on
   identical noise draws, so small differences can be detected with N≈40.
3. **Task-driven active perception: ranking which object to re-observe by its Monte-Carlo value
   of information for *this* plan.** Information is valued by how much it raises predicted task
   success, not by uncertainty reduction in general. This is a cheap, myopic approximation of
   belief-space planning (Kaelbling & Lozano-Pérez 2013) that needs no POMDP solver.

To say alongside these: everything is validated in simulation only. Geometry is primitive, grasping is an
idealised suction weld, and the noise model (σ = 1.2 cm per axis, friction ×0.6–1.4, mass ×0.7–1.3)
is an assumption, not yet calibrated against measured perception error.
