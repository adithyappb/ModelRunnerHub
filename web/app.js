/**
 * Model Runner Hub — benchmark intelligence UI
 * Data: ./data/benchmarks.json (or ?data= URL)
 */

const DEFAULT_DATA_URL = "data/benchmarks.json";

const state = {
  raw: null,
  presetId: "balanced",
  categoryFilter: "all",
  sortKey: "composite",
  sortDir: "desc",
  search: "",
  openOnly: false,
  compareIds: new Set(),
  chart: null,
};

const COLORS = ["#22d3ee", "#a78bfa", "#fb7185", "#fbbf24", "#34d399"];

function $(sel, root = document) {
  return root.querySelector(sel);
}

function $all(sel, root = document) {
  return [...root.querySelectorAll(sel)];
}

function getDefMap(raw) {
  const m = new Map();
  for (const d of raw.benchmarkDefinitions || []) m.set(d.id, d);
  return m;
}

function normalizeScore(def, value) {
  if (value == null || Number.isNaN(value)) return null;
  if (!def) return null;
  if (def.higherIsBetter) {
    const max = def.scaleMax ?? 100;
    return Math.max(0, Math.min(1, value / max));
  }
  const cap = def.scaleMax ?? 20000;
  return Math.max(0, Math.min(1, 1 - Math.min(1, value / cap)));
}

function compositeScore(model, preset, defMap) {
  const w = preset.weights || {};
  const keys = Object.keys(w);
  if (!keys.length) return null;
  let sumW = 0;
  let acc = 0;
  for (const k of keys) {
    const def = defMap.get(k);
    const v = model.scores?.[k];
    const n = normalizeScore(def, v);
    if (n == null) continue;
    acc += n * w[k];
    sumW += w[k];
  }
  if (sumW === 0) return null;
  return acc / sumW;
}

function median(arr) {
  if (!arr.length) return null;
  const s = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

/** 0–100: how strong this value is vs cohort (higher = better for task). */
function cohortPercentile(values, value, higherIsBetter) {
  const valid = values.filter((v) => v != null && !Number.isNaN(v)).sort((a, b) => a - b);
  const n = valid.length;
  if (!n || value == null || Number.isNaN(value)) return null;
  if (n === 1) return 50;
  const below = valid.filter((x) => x < value).length;
  const equal = valid.filter((x) => x === value).length;
  const rank = higherIsBetter ? below + (equal - 1) / 2 : valid.filter((x) => x > value).length + (equal - 1) / 2;
  return (rank / (n - 1)) * 100;
}

function tierFromCompositePercentile(p) {
  if (p == null) return { label: "—", cls: "tier--c" };
  if (p >= 90) return { label: "S", cls: "tier--s" };
  if (p >= 72) return { label: "A", cls: "tier--a" };
  if (p >= 45) return { label: "B", cls: "tier--b" };
  return { label: "C", cls: "tier--c" };
}

function compositePercentiles(models, preset, defMap) {
  const comps = models.map((m) => compositeScore(m, preset, defMap)).filter((c) => c != null);
  return models.map((m) => {
    const c = compositeScore(m, preset, defMap);
    return cohortPercentile(comps, c, true);
  });
}

function explainTopModels(raw, preset, topN = 3) {
  const defMap = getDefMap(raw);
  const models = raw.models || [];
  const weights = preset.weights || {};
  const wKeys = Object.keys(weights);

  const scored = models
    .map((m) => ({
      model: m,
      composite: compositeScore(m, preset, defMap),
    }))
    .filter((x) => x.composite != null)
    .sort((a, b) => b.composite - a.composite);

  const medians = {};
  for (const k of wKeys) {
    const vals = models.map((m) => m.scores?.[k]).filter((v) => v != null);
    medians[k] = median(vals);
  }

  return scored.slice(0, topN).map(({ model, composite }, idx) => {
    const reasons = [];
    for (const k of wKeys) {
      const def = defMap.get(k);
      const v = model.scores?.[k];
      const med = medians[k];
      if (v == null || med == null || !def) continue;
      const unit = def.unit || "";
      const fmt = (x) => (typeof x === "number" ? x.toFixed(def.unit === "ms" ? 0 : 1) : x);
      if (def.higherIsBetter) {
        const delta = v - med;
        if (delta >= 1) {
          reasons.push({
            text: `${def.label} ${fmt(v)}${unit} vs median ${fmt(med)}${unit} (${delta >= 0 ? "+" : ""}${fmt(delta)}).`,
            strength: delta,
          });
        }
      } else {
        const delta = med - v;
        if (delta >= 200) {
          reasons.push({
            text: `${def.label} ${fmt(v)}${unit} — faster than median ${fmt(med)}${unit}.`,
            strength: delta,
          });
        }
      }
    }
    reasons.sort((a, b) => b.strength - a.strength);
    return {
      rank: idx + 1,
      model,
      composite,
      reasons: reasons.slice(0, 4).map((r) => r.text),
    };
  });
}

function visibleBenchmarkIds(raw) {
  const defs = raw.benchmarkDefinitions || [];
  if (state.categoryFilter === "all") return defs.map((d) => d.id);
  return defs.filter((d) => d.category === state.categoryFilter).map((d) => d.id);
}

function filteredModels(raw) {
  const q = state.search.trim().toLowerCase();
  return (raw.models || []).filter((m) => {
    if (state.openOnly && !m.openWeights) return false;
    if (!q) return true;
    const blob = `${m.name} ${m.provider} ${m.id}`.toLowerCase();
    return blob.includes(q);
  });
}

function renderPresets(raw) {
  const el = $("#preset-list");
  el.innerHTML = "";
  for (const p of raw.presets || []) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "preset-btn" + (state.presetId === p.id ? " active" : "");
    btn.dataset.id = p.id;
    btn.innerHTML = `<strong>${escapeHtml(p.label)}</strong><small>${escapeHtml(p.description || "")}</small>`;
    btn.addEventListener("click", () => {
      state.presetId = p.id;
      state.sortKey = "composite";
      state.sortDir = "desc";
      renderPresets(raw);
      renderAll();
    });
    el.appendChild(btn);
  }
}

function renderCategoryChips(raw) {
  const el = $("#category-chips");
  el.innerHTML = "";
  const all = document.createElement("button");
  all.type = "button";
  all.className = "chip" + (state.categoryFilter === "all" ? " on" : "");
  all.textContent = "All";
  all.addEventListener("click", () => {
    state.categoryFilter = "all";
    renderCategoryChips(raw);
    renderAll();
  });
  el.appendChild(all);
  for (const c of raw.meta?.categories || []) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "chip" + (state.categoryFilter === c.id ? " on" : "");
    b.textContent = c.label;
    b.addEventListener("click", () => {
      state.categoryFilter = c.id;
      renderCategoryChips(raw);
      renderAll();
    });
    el.appendChild(b);
  }
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderInsights(raw) {
  const preset = (raw.presets || []).find((p) => p.id === state.presetId);
  const box = $("#insights");
  if (!preset) {
    box.innerHTML = "";
    return;
  }
  const top = explainTopModels(raw, preset, 3);
  box.innerHTML = top
    .map(
      (t) => `
    <article class="insight-card" data-rank="${t.rank}">
      <div class="insight-card__head">
        <h3>${escapeHtml(t.model.name)}</h3>
        <span class="rank-badge">#${t.rank}</span>
      </div>
      <p class="score-pill"><span>Composite</span> ${(t.composite * 100).toFixed(1)}</p>
      <ul>
        ${t.reasons.map((r) => `<li>${escapeHtml(r)}</li>`).join("")}
        <li><strong>Preset</strong> — ${escapeHtml(preset.label)}: ${escapeHtml(preset.description || "")}</li>
      </ul>
    </article>`
    )
    .join("");
}

function renderHero(raw, modelsFiltered, preset) {
  const defMap = getDefMap(raw);
  $("#hero-preset-label").textContent = preset
    ? `${preset.label} · ${preset.description || ""}`
    : "—";
  $("#stat-models").textContent = String((raw.models || []).length);
  $("#stat-metrics").textContent = String((raw.benchmarkDefinitions || []).length);
  $("#stat-visible").textContent = String(modelsFiltered.length);
  const sorted = [...modelsFiltered]
    .map((m) => ({ m, c: compositeScore(m, preset, defMap) }))
    .filter((x) => x.c != null)
    .sort((a, b) => b.c - a.c);
  $("#stat-leader").textContent = sorted.length ? sorted[0].m.name : "—";
}

function sortModels(models, raw) {
  const defMap = getDefMap(raw);
  const preset = (raw.presets || []).find((p) => p.id === state.presetId);

  const arr = [...models];
  const dir = state.sortDir === "asc" ? 1 : -1;
  arr.sort((a, b) => {
    let va;
    let vb;
    if (state.sortKey === "composite") {
      va = compositeScore(a, preset, defMap);
      vb = compositeScore(b, preset, defMap);
    } else if (state.sortKey === "name") {
      return dir * a.name.localeCompare(b.name);
    } else {
      va = a.scores?.[state.sortKey];
      vb = b.scores?.[state.sortKey];
    }
    if (va == null && vb == null) return 0;
    if (va == null) return 1;
    if (vb == null) return -1;
    return dir * (va - vb);
  });
  return arr;
}

function toggleSort(key) {
  if (state.sortKey === key) {
    state.sortDir = state.sortDir === "asc" ? "desc" : "asc";
  } else {
    state.sortKey = key;
    state.sortDir = key === "name" ? "asc" : "desc";
  }
}

function getChartTextColor() {
  const cs = getComputedStyle(document.documentElement);
  return cs.getPropertyValue("--text").trim() || "#e9edf5";
}

function getChartMutedColor() {
  const cs = getComputedStyle(document.documentElement);
  return cs.getPropertyValue("--muted").trim() || "#8b95ad";
}

function renderTable(raw) {
  const defMap = getDefMap(raw);
  const preset = (raw.presets || []).find((p) => p.id === state.presetId);
  const benchIds = visibleBenchmarkIds(raw);
  const models = sortModels(filteredModels(raw), raw);

  const cohortByBench = {};
  for (const id of benchIds) {
    const d = defMap.get(id);
    const vals = models.map((m) => m.scores?.[id]);
    cohortByBench[id] = { def: d, vals };
  }

  const compPct = compositePercentiles(models, preset, defMap);

  const thead = $("#model-thead");
  const tbody = $("#model-tbody");

  const headCells = [
    `<th class="checkbox-cell" title="Pick up to 4 for radar">◇</th>`,
    `<th data-sort="name" class="${state.sortKey === "name" ? "sorted" : ""}">Model</th>`,
    `<th data-sort="composite" class="${state.sortKey === "composite" ? "sorted" : ""}">Composite</th>`,
    ...benchIds.map((id) => {
      const d = defMap.get(id);
      const cls = state.sortKey === id ? "sorted" : "";
      return `<th data-sort="${escapeHtml(id)}" class="${cls}" title="${escapeHtml(d?.short || "")}">${escapeHtml(d?.label || id)}</th>`;
    }),
  ];
  thead.innerHTML = `<tr>${headCells.join("")}</tr>`;

  $all("thead th[data-sort]", thead).forEach((th) => {
    th.addEventListener("click", () => {
      toggleSort(th.dataset.sort);
      renderAll();
    });
  });

  tbody.innerHTML = models
    .map((m, rowIdx) => {
      const comp = compositeScore(m, preset, defMap);
      const compStr = comp == null ? "—" : (comp * 100).toFixed(1);
      const tier = tierFromCompositePercentile(compPct[rowIdx]);
      const checked = state.compareIds.has(m.id) ? "checked" : "";
      const disabled = !state.compareIds.has(m.id) && state.compareIds.size >= 4 ? "disabled" : "";
      const rowClass = state.compareIds.has(m.id) ? "row--compare" : "";

      const cells = benchIds
        .map((id) => {
          const d = defMap.get(id);
          const v = m.scores?.[id];
          if (v == null) return `<td class="score-cell"><span class="score-cell__val">—</span></td>`;
          const suffix = d?.unit === "ms" ? " ms" : d?.unit === "%" || d?.unit === "acc" ? "%" : "";
          const shown = d?.unit === "ms" ? Math.round(v) : v.toFixed(1);
          const pct = cohortPercentile(cohortByBench[id].vals, v, d.higherIsBetter);
          const fill = pct == null ? 0 : Math.max(0, Math.min(100, pct));
          return `<td class="score-cell" title="Cohort percentile (visible rows)">
            <span class="score-cell__val">${shown}${suffix}</span>
            <span class="score-meter" aria-hidden="true"><span class="score-meter__fill" style="--fill:${fill.toFixed(1)}"></span></span>
          </td>`;
        })
        .join("");

      return `<tr data-id="${escapeHtml(m.id)}" class="${rowClass}">
        <td class="checkbox-cell"><input type="checkbox" class="cmp" data-id="${escapeHtml(m.id)}" ${checked} ${disabled} aria-label="Compare ${escapeHtml(m.name)}" /></td>
        <td class="name-cell">
          <strong>${escapeHtml(m.name)}</strong>
          <span>${escapeHtml(m.provider)}${m.sizeB != null ? ` · ${m.sizeB}B` : ""}</span>
          <div class="meta-inline">
            <span class="badge ${m.openWeights ? "open" : ""}">${m.openWeights ? "Open" : "Closed"}</span>
          </div>
        </td>
        <td class="score-composite">
          <div class="score-composite__row">
            <span class="score-composite__val">${compStr}</span>
            <span class="tier ${tier.cls}">${tier.label}</span>
          </div>
          <span class="score-meter" aria-hidden="true"><span class="score-meter__fill" style="--fill:${compPct[rowIdx] == null ? 0 : compPct[rowIdx].toFixed(1)}"></span></span>
        </td>
        ${cells}
      </tr>`;
    })
    .join("");

  $all("tbody input.cmp", tbody).forEach((cb) => {
    cb.addEventListener("change", (e) => {
      const id = e.target.dataset.id;
      if (e.target.checked) {
        if (state.compareIds.size >= 4) {
          e.target.checked = false;
          return;
        }
        state.compareIds.add(id);
      } else {
        state.compareIds.delete(id);
      }
      updateCompareChart(raw);
      renderTable(raw);
    });
  });
}

function updateCompareChart(raw) {
  const defMap = getDefMap(raw);
  const benchIds = visibleBenchmarkIds(raw).filter((id) => {
    const d = defMap.get(id);
    return d && d.higherIsBetter;
  });
  const take = benchIds.slice(0, 8);
  const models = (raw.models || []).filter((m) => state.compareIds.has(m.id));

  const canvas = $("#radar");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (state.chart) {
    state.chart.destroy();
    state.chart = null;
  }
  if (!models.length || !take.length || typeof Chart === "undefined") {
    return;
  }

  const labels = take.map((id) => defMap.get(id)?.short || id);
  const text = getChartTextColor();
  const muted = getChartMutedColor();

  const datasets = models.map((m, i) => ({
    label: m.name,
    data: take.map((id) => {
      const d = defMap.get(id);
      const v = m.scores?.[id];
      const n = normalizeScore(d, v);
      return n == null ? 0 : n * 100;
    }),
    borderColor: COLORS[i % COLORS.length],
    backgroundColor: `${COLORS[i % COLORS.length]}28`,
    borderWidth: 2,
    fill: true,
  }));

  state.chart = new Chart(ctx, {
    type: "radar",
    data: { labels, datasets },
    options: {
      responsive: true,
      scales: {
        r: {
          min: 0,
          max: 100,
          ticks: {
            stepSize: 20,
            backdropColor: "transparent",
            color: muted,
          },
          grid: { color: "rgba(128,140,160,0.15)" },
          angleLines: { color: "rgba(128,140,160,0.12)" },
          pointLabels: { color: muted, font: { size: 10, weight: "500" } },
        },
      },
      plugins: {
        legend: {
          labels: { color: text, font: { size: 11, weight: "500" } },
        },
      },
    },
  });
}

function renderMeta(raw) {
  $("#meta-title").textContent = raw.meta?.title || "Model Runner Hub";
  $("#meta-version").textContent = `v${raw.meta?.version || "?"} · ${raw.meta?.lastUpdated || "—"}`;
  $("#footer-note").textContent = raw.meta?.dataSourceNote || "";
}

function renderAll() {
  if (!state.raw) return;
  const raw = state.raw;
  const preset = (raw.presets || []).find((p) => p.id === state.presetId);
  const fm = filteredModels(raw);
  renderMeta(raw);
  renderHero(raw, fm, preset);
  renderInsights(raw);
  renderTable(raw);
  updateCompareChart(raw);
}

function exportCsv() {
  if (!state.raw) return;
  const raw = state.raw;
  const defMap = getDefMap(raw);
  const preset = (raw.presets || []).find((p) => p.id === state.presetId);
  const benchIds = visibleBenchmarkIds(raw);
  const models = sortModels(filteredModels(raw), raw);

  const headers = ["Model", "Provider", "Open weights", "Size (B)", "Composite", ...benchIds.map((id) => defMap.get(id)?.label || id)];

  const rows = models.map((m) => {
    const comp = compositeScore(m, preset, defMap);
    const compStr = comp == null ? "" : (comp * 100).toFixed(4);
    const base = [
      m.name,
      m.provider,
      m.openWeights ? "yes" : "no",
      m.sizeB != null ? String(m.sizeB) : "",
      compStr,
    ];
    const scores = benchIds.map((id) => {
      const v = m.scores?.[id];
      return v == null ? "" : String(v);
    });
    return [...base, ...scores];
  });

  const escape = (cell) => {
    const s = String(cell);
    if (/[",\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
    return s;
  };

  const csv = [headers.map(escape).join(","), ...rows.map((r) => r.map(escape).join(","))].join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `model-runner-hub-${state.presetId}.csv`;
  a.click();
  URL.revokeObjectURL(a.href);
}

async function loadData(url) {
  const status = $("#load-status");
  status.textContent = "Loading benchmarks…";
  status.classList.remove("error");
  try {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    state.raw = await res.json();
    const presets = state.raw.presets || [];
    state.presetId = presets.find((p) => p.id === "balanced")?.id || presets[0]?.id || "balanced";
    renderPresets(state.raw);
    renderCategoryChips(state.raw);
    renderAll();
    status.textContent = "Ready.";
    document.body.classList.add("is-loaded");
  } catch (e) {
    console.error(e);
    status.textContent = `Could not load data: ${e.message}`;
    status.classList.add("error");
  }
}

function initTheme() {
  const saved = localStorage.getItem("mrh-theme");
  const prefersLight = window.matchMedia?.("(prefers-color-scheme: light)").matches;
  const theme = saved || (prefersLight ? "light" : "dark");
  document.documentElement.setAttribute("data-theme", theme);
  $("#theme-toggle").addEventListener("click", () => {
    const next = document.documentElement.getAttribute("data-theme") === "light" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("mrh-theme", next);
    if (state.raw) updateCompareChart(state.raw);
  });
}

function setHelpOpen(open) {
  const modal = $("#help-modal");
  modal.hidden = !open;
}

function init() {
  initTheme();

  $("#reload-btn").addEventListener("click", () => {
    const custom = $("#data-url").value.trim();
    loadData(custom || DEFAULT_DATA_URL);
  });

  $("#export-btn").addEventListener("click", () => exportCsv());

  $("#help-btn").addEventListener("click", () => setHelpOpen(true));
  $all("[data-close]", $("#help-modal")).forEach((el) => {
    el.addEventListener("click", () => setHelpOpen(false));
  });

  $("#search").addEventListener("input", (e) => {
    state.search = e.target.value;
    renderAll();
  });

  $("#open-only").addEventListener("change", (e) => {
    state.openOnly = e.target.checked;
    renderAll();
  });

  $("#clear-compare").addEventListener("click", () => {
    state.compareIds.clear();
    renderAll();
  });

  document.addEventListener("keydown", (e) => {
    const tag = e.target && e.target.tagName;
    const typing = tag === "INPUT" || tag === "TEXTAREA";
    if (e.key === "?" && !e.ctrlKey && !e.metaKey && !typing) {
      e.preventDefault();
      const modal = $("#help-modal");
      setHelpOpen(modal.hidden);
    }
    if (e.key === "Escape") setHelpOpen(false);
    if (e.key === "/" && document.activeElement !== $("#search")) {
      const t = e.target;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA")) return;
      e.preventDefault();
      $("#search").focus();
    }
  });

  const params = new URLSearchParams(window.location.search);
  const dataParam = params.get("data");
  loadData(dataParam || DEFAULT_DATA_URL);

  if (dataParam) $("#data-url").value = dataParam;
}

document.addEventListener("DOMContentLoaded", init);
