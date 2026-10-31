// Small helpers shared by the app and demo mode.
export const $ = (id) => document.getElementById(id);
export const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export const PRED = { inside: "inside", on: "on top of", upright: "upright", stacked: "stacked", stable: "stable", undisturbed: "not disturbed" };
export const REL = { in: "into", on: "onto" };

// English gloss for known demo instructions (the agent receives the original text).
export const GLOSS = {
  "把杯子放进碗里，然后把书摞整齐": "Put the mug in the bowl, then stack the books neatly.",
};

// Contacts that matter: something the arm or the held object hit that is not the table or a destination.
export function unexpectedContacts(a) {
  const dests = new Set(a.plan.actions.map((x) => x.dest));
  return a.events.filter((e) => e.kind === "contact" && e.while_holding && !e.between.includes("floor")
    && !e.between.includes("table") && !e.between.some((x) => dests.has(x)));
}

export function hitList(a) {
  return unexpectedContacts(a).map((e) => {
    const obj = e.between.find((x) => x !== "robot_arm" && x !== e.while_holding) || e.between[0];
    return { t: e.t, obj, by: e.between.includes("robot_arm") ? "arm" : e.while_holding, text: `Collision · ${esc(obj)}` };
  });
}

export const col = (rgba) => `rgba(${rgba.slice(0, 3).map((c) => Math.round(c * 255)).join(",")},${rgba[3] ?? 1})`;

export function dims(o) {
  const s = o.shape;
  return s.type === "box" ? `${s.size.map((v) => Math.round(v * 100)).join("×")} cm`
    : `⌀${Math.round(s.radius * 200)} × ${Math.round(s.height * 100)} cm`;
}

export function goalText(g) { return `${g.pred}(${g.args.join(", ")})`; }

export function eventText(e) {
  if (e.kind === "contact") {
    const [p, q] = e.between;
    return `<code>${esc(p)}</code> ↔ <code>${esc(q)}</code>` + (e.while_holding ? ` · holding <code>${esc(e.while_holding)}</code> · tool z ${e.tcp_z} m` : "");
  }
  if (e.kind === "grasp") return `pick up <code>${esc(e.obj)}</code>`;
  if (e.kind === "release") return `release <code>${esc(e.obj)}</code>`;
  return `${esc(e.kind)} ${esc(e.msg || e.obj || "")}`;
}

export const LOGO = `<svg aria-hidden="true"><use href="#logo"/></svg>`;
export const MB = {
  spark: (what) => `<span class="mb"><i>✦</i><b>Muse Spark</b> ${what}</span>`,
  sam: (what) => `<span class="mb sam"><i>◎</i><b>SAM 3.1</b> ${what}</span>`,
  phys: (what) => `<span class="mb phys"><i>▣</i><b>MuJoCo</b> ${what}</span>`,
};
