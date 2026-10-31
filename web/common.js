// Small helpers shared by the app and demo mode.
export const $ = (id) => document.getElementById(id);
export const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export const PRED = { inside: "inside", on: "on top of", upright: "upright", stacked: "stacked", stable: "stable", undisturbed: "not disturbed" };
export const REL = { in: "into", on: "onto" };

// English gloss for known demo instructions (the agent receives the original text).
export const GLOSS = {
  "把杯子放进碗里，然后把书摞整齐": "Put the mug in the bowl, then stack the books neatly.",
  "把杯子放到杯垫上，再把水壶放到隔热垫上": "Put the cup on the coaster, then the kettle on the trivet.",
  "把马克杯放到托盘上": "Put the mug on the tray.",
};

// Contacts that matter: something the arm or the held object hit that is not the table or a destination.
// Protective stops always count.
export function unexpectedContacts(a) {
  const dests = new Set(a.plan.actions.map((x) => x.dest));
  return a.events.filter((e) => e.kind === "protective_stop" || (e.kind === "contact" && (e.while_holding || e.link) && !e.between.includes("floor")
    && !e.between.includes("table") && !e.between.some((x) => dests.has(x) || x === e.while_holding && !e.link)));
}

export function hitList(a) {
  return unexpectedContacts(a).map((e) => {
    if (e.kind === "protective_stop") {
      const f = e.force_n != null ? `${Math.round(e.force_n)} N` : "";
      return { t: e.t, obj: e.obj, kind: "stop", text: `Protective stop · ${f} on ${esc(e.obj)}`,
        headline: `Protective stop — ${f} on ${e.obj}`, logLine: `protective stop · ${e.link_role || "arm"} pressed ${e.obj} with ${f}` };
    }
    const obj = e.between.find((x) => x !== "robot_arm" && x !== e.while_holding) || e.between[0];
    const by = e.link_role ? `the ${e.link_role}` : e.while_holding || "the arm";
    return { t: e.t, obj, kind: "contact", text: `Collision · ${esc(obj)}`, headline: `Collision — ${by} hits ${obj}`,
      logLine: `${e.between.join(" ↔ ")}${e.while_holding ? ` · holding ${e.while_holding}` : ""}${e.tcp_z != null ? ` · tool z ${e.tcp_z} m` : ""}${e.phase ? ` · ${e.phase}` : ""}` };
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
    return `<code>${esc(p)}</code> ↔ <code>${esc(q)}</code>` + (e.link_role ? ` · ${esc(e.link_role)}` : "") +
      (e.while_holding ? ` · holding <code>${esc(e.while_holding)}</code>` : "") + (e.tcp_z != null && e.while_holding ? ` · tool z ${e.tcp_z} m` : "") + (e.phase ? ` <span style="color:var(--muted)">· ${esc(e.phase)}</span>` : "");
  }
  if (e.kind === "protective_stop") return `<b style="color:var(--bad-text)">protective stop</b> · ${esc(e.link_role || "arm")} pressed <code>${esc(e.obj)}</code>${e.force_n != null ? ` with ${Math.round(e.force_n)} N` : ""}`;
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
