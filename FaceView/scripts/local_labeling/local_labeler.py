#!/usr/bin/env python3
"""Local (PC-side) labeler for Nersemble + Ava-256 -- standard library only.

Same labeling as FaceView's frontend/backend, against local copies of the
overlay videos:
  * Nersemble: the five label statuses and the same one-step transitions as
    backend/app/services/label_tracker.py (Confirm Labeled, Unconfirm Labeled,
    Confirm, Unconfirm, Unconfirm a confirmed pair), with the Overlay Video and
    Overlay Landmarks videos.
  * Nersemble list: a "Re-wrapped" filter (All / Re-wrapped / Not re-wrapped)
    and badge, from the bundle's rewrapped.json (run_pipeline_batch_all.py
    --unconfirmed-only / --confirmed-only records; absent in older bundles).
  * Ava-256: the per-frame Valid / Invalid / Rerun range labeling, timeline,
    skin-error flags and two-step Done of Ava256Player.tsx /
    services/ava256_labels.py.

Everything is saved to ONE file, BUNDLE/label_tracker.json (atomic writes):
the real "Nersemble" / "Ava256" keys in their normal format, plus a
"_local_ava256" key holding the Ava-256 frame labels you changed
({capture: {"frames": [[frame_id, status], ...], "done": bool}}). Host-only
side effects (the .pt exports, renaming wrapped_mesh.obj, archiving CSVs)
happen when merge_local_labels.py replays your changes on the host.

Usage:
    python local_labeler.py --bundle ./local_bundle --videos ./VIDEOS [--port 8765]
then open http://localhost:8765
VIDEOS must contain line_renders/, point_overlay_renders/ and line_renders_ava256/.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

LOCAL_AVA_KEY = "_local_ava256"
FRAME_LABELS = ("Valid", "Invalid", "Rerun")
UNCONFIRMED = "unconfirmed"
OVERLAY_CAMERA = "222200046"
POINT_OVERLAY_CAMERA = "222200037"

# action -> (required current status, new status); mirrors label_tracker.py.
NERSEMBLE_ACTIONS = {
    "confirm_labeled": ("unlabeled", "unreviewed"),
    "unconfirm_labeled": ("unreviewed", "unlabeled"),
    "confirm": ("unconfirmed", "confirmed"),
    "unconfirm": ("unconfirmed", "unreviewed"),
    "unconfirm_confirmed": ("confirmed", "unconfirmed"),
}


class Store:
    """The local label_tracker.json plus the bundle's Ava-256 frame snapshot."""

    def __init__(self, bundle: Path, videos: Path) -> None:
        self.bundle, self.videos = bundle, videos
        self.tracker_path = bundle / "label_tracker.json"
        self.lock = threading.Lock()
        self.tracker = json.loads(self.tracker_path.read_text(encoding="utf-8"))
        frames_path = bundle / "ava256_frames.json"
        self.ava_base = json.loads(frames_path.read_text(encoding="utf-8")) if frames_path.exists() else {}
        # Pairs run_pipeline_batch_all.py --unconfirmed-only / --confirmed-only re-wrapped (bundles made
        # before export_local_bundle.py wrote rewrapped.json have none).
        rewrapped_path = bundle / "rewrapped.json"
        rewrapped = json.loads(rewrapped_path.read_text(encoding="utf-8")) if rewrapped_path.exists() else {}
        self.rewrapped = {(actor, seq) for record in rewrapped.values() for actor, seqs in record.items() for seq in seqs}

    def save(self) -> None:
        tmp = self.tracker_path.with_name(self.tracker_path.name + ".tmp")
        tmp.write_text(json.dumps(self.tracker, indent=2), encoding="utf-8")
        os.replace(tmp, self.tracker_path)

    # --- Nersemble -----------------------------------------------------------
    def nersemble_status(self, actor: str, seq: str) -> str:
        return self.tracker.get("Nersemble", {}).get(actor, {}).get(seq, "no_neutral_label")

    def nersemble_transition(self, actor: str, seq: str, action: str) -> str:
        if action not in NERSEMBLE_ACTIONS:
            raise ValueError(f"unknown action {action}")
        need, new = NERSEMBLE_ACTIONS[action]
        with self.lock:
            current = self.nersemble_status(actor, seq)
            if current != need:
                raise ValueError(f"{actor}/{seq} is {current}, not {need} -- {action} not allowed")
            self.tracker.setdefault("Nersemble", {}).setdefault(actor, {})[seq] = new
            self.save()
        return new

    def nersemble_video(self, actor: str, seq: str, kind: str) -> Path | None:
        if kind == "overlay":
            candidates = [self.videos / "line_renders" / f"{actor}_{seq}_{OVERLAY_CAMERA}.mp4"]
        else:
            candidates = [self.videos / "point_overlay_renders" / f"{actor}_{seq}_{c}.mp4"
                          for c in (POINT_OVERLAY_CAMERA, OVERLAY_CAMERA)]
        return next((p for p in candidates if p.exists()), None)

    # --- Ava-256 -------------------------------------------------------------
    def ava_frames(self, cap: str) -> list[list[str]]:
        local = self.tracker.get(LOCAL_AVA_KEY, {}).get(cap)
        if local is not None:
            return [list(r) for r in local["frames"]]
        return [list(r) for r in self.ava_base.get(cap, {}).get("frames", [])]

    def ava_done(self, cap: str) -> bool:
        local = self.tracker.get(LOCAL_AVA_KEY, {}).get(cap)
        return bool(local and local.get("done")) or bool(self.ava_base.get(cap, {}).get("is_done"))

    def ava_status(self, cap: str) -> str:
        return self.tracker.get("Ava256", {}).get(cap, "unlabeled")

    def ava_captures(self) -> list[dict]:
        ids = set(self.ava_base) | set(self.tracker.get(LOCAL_AVA_KEY, {}))
        out = []
        for cap in sorted(ids):
            counts = {label: 0 for label in (*FRAME_LABELS, UNCONFIRMED)}
            rows = self.ava_frames(cap)
            for _, status in rows:
                counts[status if status in counts else UNCONFIRMED] += 1
            out.append({"capture_id": cap, "label_status": self.ava_status(cap), "num_frames": len(rows),
                        "counts": counts, "is_done": self.ava_done(cap),
                        "has_video": (self.videos / "line_renders_ava256" / f"{cap}.mp4").exists()})
        return out

    def _set_local(self, cap: str, rows: list[list[str]], done: bool) -> None:
        self.tracker.setdefault(LOCAL_AVA_KEY, {})[cap] = {"frames": rows, "done": done}

    def ava_apply(self, cap: str, ranges_by_label: dict) -> tuple[list[list[str]], list[str]]:
        """services/ava256_labels.apply_ranges, on the local copy."""
        unknown = set(ranges_by_label) - set(FRAME_LABELS)
        if unknown:
            raise ValueError(f"Unknown label(s) {sorted(unknown)}; expected {list(FRAME_LABELS)}")
        with self.lock:
            if self.ava_done(cap):
                raise ValueError(f"{cap} is already done; its labels are read-only")
            rows = self.ava_frames(cap)
            if not rows:
                raise ValueError(f"No frame list for {cap}")
            n = len(rows)
            warnings: list[str] = []
            owner: dict[int, str] = {}
            for label, ranges in ranges_by_label.items():
                for pair in ranges:
                    if len(pair) != 2:
                        raise ValueError(f"{label}: each range must be [start, end], got {pair}")
                    start, end = sorted(int(v) for v in pair)
                    if end < 1 or start > n:
                        warnings.append(f"{label} {start}-{end} is outside 1-{n}; ignored")
                        continue
                    if start < 1 or end > n:
                        warnings.append(f"{label} {start}-{end} clamped to {max(start, 1)}-{min(end, n)}")
                    for pos in range(max(start, 1), min(end, n) + 1):
                        if pos in owner:
                            other = owner[pos]
                            raise ValueError(f"Frame #{pos} is in more than one range ({other} and {label})"
                                             if other != label else f"{label} ranges overlap at frame #{pos}")
                        owner[pos] = label
            rows = [[fid, owner.get(i + 1, status)] for i, (fid, status) in enumerate(rows)]
            self._set_local(cap, rows, False)
            self.save()
        return rows, warnings

    def ava_finish(self, cap: str) -> None:
        """finish_capture's checks; the export/archive run at merge time on the host."""
        with self.lock:
            if self.ava_done(cap):
                raise ValueError(f"{cap} is already done")
            rows = self.ava_frames(cap)
            pending = sum(1 for _, s in rows if s not in FRAME_LABELS)
            if pending:
                raise ValueError(f"{pending} frame(s) are still unconfirmed -- label every frame before Done")
            self._set_local(cap, rows, True)
            self.tracker.setdefault("Ava256", {})[cap] = "confirmed"
            self.save()


def make_handler(store: Store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet
            pass

        def _json(self, payload, status=HTTPStatus.OK):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status, detail):
            self._json({"detail": detail}, status)

        def _file(self, path: Path | None):
            if path is None or not path.is_file():
                return self._error(HTTPStatus.NOT_FOUND, "video not found")
            size = path.stat().st_size
            start, end = 0, size - 1
            match = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
            partial = bool(match)
            if match:
                if match.group(1):
                    start = int(match.group(1))
                    end = int(match.group(2)) if match.group(2) else size - 1
                else:  # suffix range: last N bytes
                    start = max(0, size - int(match.group(2) or 0))
                end = min(end, size - 1)
                if start > end:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
            self.send_response(HTTPStatus.PARTIAL_CONTENT if partial else HTTPStatus.OK)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as f:
                f.seek(start)
                left = end - start + 1
                try:
                    while left > 0:
                        chunk = f.read(min(1 << 20, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
                except ConnectionError:  # browser dropped the request (seek / switch video)
                    pass

        def do_GET(self):
            parts = [unquote(p) for p in urlparse(self.path).path.split("/") if p]
            if not parts:
                body = PAGE.encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif parts == ["api", "state"]:
                pairs = []
                for actor, seqs in sorted(store.tracker.get("Nersemble", {}).items()):
                    for seq, status in sorted(seqs.items()):
                        pairs.append({"actor_id": actor, "sequence_id": seq, "label_status": status,
                                      "rewrapped": (actor, seq) in store.rewrapped})
                self._json({"nersemble": pairs, "ava256": store.ava_captures()})
            elif len(parts) == 3 and parts[:2] == ["api", "ava"]:
                cap = parts[2]
                skin = store.ava_base.get(cap, {}).get("skin_errors", {})
                self._json({"capture_id": cap, "is_done": store.ava_done(cap), "label_status": store.ava_status(cap),
                            "frames": [{"frame_id": f, "status": s, "skin_error_px": skin.get(f)}
                                       for f, s in store.ava_frames(cap)]})
            elif len(parts) == 4 and parts[0] == "video" and parts[1] == "nersemble":
                actor, seq_kind = parts[2], parts[3]
                seq, _, kind = seq_kind.rpartition(".")
                self._file(store.nersemble_video(actor, seq, kind))
            elif len(parts) == 3 and parts[:2] == ["video", "ava"]:
                self._file(store.videos / "line_renders_ava256" / f"{parts[2]}.mp4")
            else:
                self._error(HTTPStatus.NOT_FOUND, "not found")

        def do_POST(self):
            parts = [unquote(p) for p in urlparse(self.path).path.split("/") if p]
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return self._error(HTTPStatus.BAD_REQUEST, "bad JSON")
            try:
                if parts == ["api", "nersemble", "transition"]:
                    status = store.nersemble_transition(body["actor_id"], body["sequence_id"], body["action"])
                    return self._json({"status": status})
                if len(parts) == 4 and parts[:2] == ["api", "ava"] and parts[3] == "frames":
                    rows, warnings = store.ava_apply(parts[2], body.get("ranges", {}))
                    return self._json({"ok": True, "warnings": warnings})
                if len(parts) == 4 and parts[:2] == ["api", "ava"] and parts[3] == "done":
                    store.ava_finish(parts[2])
                    return self._json({"ok": True})
            except ValueError as exc:
                return self._error(HTTPStatus.CONFLICT, str(exc))
            except KeyError as exc:
                return self._error(HTTPStatus.BAD_REQUEST, f"missing {exc}")
            self._error(HTTPStatus.NOT_FOUND, "not found")

    return Handler


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Local Labeler</title>
<style>
:root{--bg:#0f172a;--panel:#111c33;--line:#22304d;--text:#e2e8f0;--mute:#94a3b8;--accent:#38bdf8}
*{box-sizing:border-box}body{margin:0;font:14px system-ui,sans-serif;background:var(--bg);color:var(--text);display:flex;height:100vh}
#side{width:380px;min-width:300px;border-right:1px solid var(--line);display:flex;flex-direction:column}
#side .ctl{padding:10px;border-bottom:1px solid var(--line);display:flex;flex-wrap:wrap;gap:6px}
#list{overflow:auto;flex:1}#main{flex:1;overflow:auto;padding:16px}
button{background:#1e293b;color:var(--text);border:1px solid var(--line);border-radius:5px;padding:4px 8px;cursor:pointer;font-size:12px}
button:hover{background:#334155}button:disabled{opacity:.4;cursor:default}button.on{border-color:var(--accent);color:var(--accent)}
input{background:#1e293b;color:var(--text);border:1px solid var(--line);border-radius:5px;padding:4px 6px;font-size:12px}
.row{padding:7px 10px;border-bottom:1px solid #18233b;cursor:pointer}.row:hover{background:#16213a}.row.sel{background:#1c2b4a}
.row .t{display:flex;justify-content:space-between;gap:6px}.row .b{margin-top:5px;display:flex;gap:5px;flex-wrap:wrap}
.pill{font-size:11px;padding:1px 6px;border-radius:9px;border:1px solid var(--line);white-space:nowrap}
.s-no_neutral_label{color:#94a3b8}.s-unlabeled{color:#fbbf24}.s-unreviewed{color:#a78bfa}.s-unconfirmed{color:#38bdf8}.s-confirmed{color:#4ade80}
video{width:100%;max-height:68vh;background:#000}.card{border:1px solid var(--line);background:#0b1426;border-radius:8px;padding:12px;margin-bottom:12px}
.tabs button{margin-right:4px}.mute{color:var(--mute)}.err{color:#fb7185}.warn{color:#fbbf24}
#tl{display:flex;height:32px;border-radius:4px;overflow:hidden;user-select:none}#tl div{flex:1;min-width:0;position:relative;cursor:pointer}
.c-Valid{background:#16a34a}.c-Invalid{background:#e11d48}.c-Rerun{background:#d97706}.c-unconfirmed{background:#334155}
.t-Valid{color:#4ade80}.t-Invalid{color:#fb7185}.t-Rerun{color:#fbbf24}.t-unconfirmed{color:#94a3b8}
.conf{background:#dc2626!important;animation:p 1s infinite}@keyframes p{50%{opacity:.4}}
.lab{display:grid;grid-template-columns:70px 1fr;gap:6px;align-items:center;margin:6px 0}.lab input{width:100%}
.chip{display:inline-block;margin:2px;padding:1px 6px;border-radius:9px;background:#1e293b;font-size:11px}.chip b{cursor:pointer;margin-left:4px}
</style></head><body>
<div id="side">
 <div class="ctl"><button id="dsN" class="on">Nersemble</button><button id="dsA">Ava-256</button>
  <span class="mute" style="margin-left:auto" id="savedNote">saves to label_tracker.json</span></div>
 <div class="ctl" id="pills"></div>
 <div class="ctl"><input id="q" placeholder="search" style="flex:1">
  <input id="amin" placeholder="min actor" style="width:80px"><input id="amax" placeholder="max actor" style="width:80px"></div>
 <div class="ctl" id="rwctl"><span class="mute">Re-wrapped:</span><button data-rw="all" class="on">All</button>
  <button data-rw="yes">Re-wrapped</button><button data-rw="no">Not re-wrapped</button><span class="mute" id="rwcount" style="margin-left:auto"></span></div>
 <div id="list"></div>
</div>
<div id="main"><p class="mute">Select a pair / capture on the left.</p></div>
<script>
const NS=["no_neutral_label","unlabeled","unreviewed","unconfirmed","confirmed"];
const NAME={no_neutral_label:"Unlabeled Neutral",unlabeled:"Labeled Neutral",unreviewed:"Propagated & Unwrapped",unconfirmed:"Wrapped & Unconfirmed",confirmed:"Wrapped & Confirmed"};
const FAST_ACTOR_MIN=102,FAST_RATE=2.0;
const LABELS=["Valid","Invalid","Rerun"],SKIN_FLAG=4.0;
let S={ds:"N",state:null,on:new Set(NS),sel:null,tab:"overlay",ava:null,rw:"all"};
const $=id=>document.getElementById(id),esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
async function api(path,body){const r=await fetch(path,body?{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}:{});
 const j=await r.json().catch(()=>({}));if(!r.ok)throw new Error(j.detail||r.statusText);return j}
async function load(){S.state=await api("/api/state");drawPills();drawList()}
function drawPills(){const p=$("pills");p.innerHTML="";const statuses=S.ds==="N"?NS:["unlabeled","unconfirmed","confirmed"];
 for(const s of statuses){const b=document.createElement("button");b.className="pill s-"+s+(S.on.has(s)?" on":"");
  const n=S.ds==="N"?S.state.nersemble.filter(x=>x.label_status===s).length:S.state.ava256.filter(x=>x.label_status===s).length;
  b.textContent=(S.ds==="N"?NAME[s]:s)+" ("+n+")";b.onclick=()=>{S.on.has(s)?S.on.delete(s):S.on.add(s);drawPills();drawList()};p.appendChild(b)}}
function actorOk(a){const n=parseInt(a,10),lo=parseInt($("amin").value,10),hi=parseInt($("amax").value,10);
 if(!isNaN(lo)&&!(n>=lo))return false;if(!isNaN(hi)&&!(n<=hi))return false;return true}
function rwOk(x){return S.rw==="all"||(S.rw==="yes")===!!x.rewrapped}
function drawRw(){const c=$("rwctl");c.style.display=S.ds==="N"?"flex":"none";
 c.querySelectorAll("[data-rw]").forEach(b=>b.className=b.dataset.rw===S.rw?"on":"");
 const shown=S.state.nersemble.filter(x=>S.on.has(x.label_status));
 $("rwcount").textContent=`${shown.filter(x=>x.rewrapped).length} re-wrapped / ${shown.length} in shown statuses`}
function drawList(){const q=$("q").value.toLowerCase(),L=$("list");L.innerHTML="";drawRw();
 if(S.ds==="N"){for(const x of S.state.nersemble){if(!S.on.has(x.label_status))continue;const key=x.actor_id+"/"+x.sequence_id;
   if(q&&!key.toLowerCase().includes(q))continue;if(!actorOk(x.actor_id))continue;if(!rwOk(x))continue;
   const d=document.createElement("div");d.className="row"+(S.sel===key?" sel":"");
   d.innerHTML=`<div class="t"><span>${esc(key)}${x.rewrapped?' <span class="pill" style="color:#f0abfc">re-wrapped</span>':""}</span><span class="pill s-${x.label_status}">${NAME[x.label_status]}</span></div><div class="b"></div>`;
   const b=d.querySelector(".b"),btn=(t,a)=>{const e=document.createElement("button");e.textContent=t;e.onclick=ev=>{ev.stopPropagation();transition(x,a)};b.appendChild(e)};
   if(x.label_status==="unlabeled")btn("Confirm Labeled","confirm_labeled");
   if(x.label_status==="unreviewed")btn("Unconfirm Labeled","unconfirm_labeled");
   if(x.label_status==="unconfirmed"){btn("Unconfirm","unconfirm");btn("Confirm","confirm")}
   if(x.label_status==="confirmed")btn("Unconfirm","unconfirm_confirmed");
   d.onclick=()=>{S.sel=key;drawList();showPair(x)};L.appendChild(d)}}
 else{for(const c of S.state.ava256){if(!S.on.has(c.label_status))continue;if(q&&!c.capture_id.toLowerCase().includes(q))continue;
   const d=document.createElement("div");d.className="row"+(S.sel===c.capture_id?" sel":"");const k=c.counts;
   d.innerHTML=`<div class="t"><span>${esc(c.capture_id)}</span><span class="pill s-${c.label_status}">${c.is_done?"done":c.label_status}</span></div>
   <div class="mute" style="font-size:11px;margin-top:3px"><span class="t-Valid">V ${k.Valid}</span> · <span class="t-Invalid">I ${k.Invalid}</span> · <span class="t-Rerun">R ${k.Rerun}</span> · unlabeled ${k.unconfirmed} / ${c.num_frames}${c.has_video?"":" · <span class=err>no video</span>"}</div>`;
   d.onclick=()=>{S.sel=c.capture_id;drawList();showCapture(c.capture_id)};L.appendChild(d)}}}
async function transition(x,action){try{await api("/api/nersemble/transition",{actor_id:x.actor_id,sequence_id:x.sequence_id,action});
 await load();if(S.sel===x.actor_id+"/"+x.sequence_id)showPair(S.state.nersemble.find(p=>p.actor_id===x.actor_id&&p.sequence_id===x.sequence_id))}catch(e){alert(e.message)}}
function showPair(x){const M=$("main"),hasVid=x.label_status==="unconfirmed"||x.label_status==="confirmed";
 M.innerHTML=`<h2>${esc(x.actor_id)} / ${esc(x.sequence_id)} <span class="pill s-${x.label_status}">${NAME[x.label_status]}</span></h2>`+
 (hasVid?`<div class="tabs"><button id="tO" class="${S.tab==="overlay"?"on":""}">Overlay Video</button><button id="tP" class="${S.tab==="points"?"on":""}">Overlay Landmarks</button></div>
 <div class="card"><video id="nv" controls src="/video/nersemble/${encodeURIComponent(x.actor_id)}/${encodeURIComponent(x.sequence_id)}.${S.tab==="overlay"?"overlay":"points"}"></video></div>`
 :`<p class="mute">Overlay videos are shown once a pair is wrapped (Wrapped & Unconfirmed / Confirmed).</p>`);
 if(hasVid){$("tO").onclick=()=>{S.tab="overlay";showPair(x)};$("tP").onclick=()=>{S.tab="points";showPair(x)};
  // Nersemble overlay videos of actors FAST_ACTOR_MIN+ default to FAST_RATE (the player's speed menu still overrides).
  const v=$("nv"),rate=parseInt(x.actor_id,10)>=FAST_ACTOR_MIN?FAST_RATE:1;v.defaultPlaybackRate=rate;v.playbackRate=rate}}
// ---------------- Ava-256 (port of Ava256Player.tsx) ----------------
function parseRanges(text,n){const ranges=[],errors=[];for(const raw of text.split(",")){const t=raw.trim();if(!t)continue;
 const m=t.match(/^(\d+)\s*(?:-\s*(\d+))?$/);if(!m){errors.push(`"${t}" isn't a frame or range`);continue}
 let a=+m[1],b=m[2]?+m[2]:a;if(a>b)[a,b]=[b,a];if(a<1||b>n){errors.push(`${t} is outside 1-${n}`);continue}ranges.push([a,b])}return{ranges,errors}}
const fmt=r=>r.map(([a,b])=>a===b?`${a}`:`${a}-${b}`).join(", ");
function rangesFromStatuses(st){const o={Valid:[],Invalid:[],Rerun:[]};let i=0;while(i<st.length){let j=i;while(j+1<st.length&&st[j+1]===st[i])j++;
 if(st[i]!=="unconfirmed")o[st[i]].push([i+1,j+1]);i=j+1}return{Valid:fmt(o.Valid),Invalid:fmt(o.Invalid),Rerun:fmt(o.Rerun)}}
async function showCapture(cap){const d=await api("/api/ava/"+encodeURIComponent(cap));
 const A=S.ava={cap,frames:d.frames,done:d.is_done,status:d.label_status,saved:d.frames.map(f=>f.status),pos:1,dur:0,sel:null,drag:null,confirm:false,msg:"",msgCls:"mute"};
 A.text=rangesFromStatuses(A.saved);const has=S.state.ava256.find(c=>c.capture_id===cap)?.has_video;
 $("main").innerHTML=`<h2>${esc(cap)} <span class="mute" style="font-size:12px">${A.done?"Done -- labels read-only":esc(A.status)}</span></h2>
 ${has?`<div class="card"><video id="v" controls src="/video/ava/${encodeURIComponent(cap)}"></video>
 <div style="margin-top:8px;display:flex;gap:8px;align-items:center;flex-wrap:wrap"><button id="prev">◂ Prev</button><button id="next">Next ▸</button>
 <span id="readout"></span><label class="mute">Go to # <input id="goto" type="number" style="width:80px"></label></div></div>`:`<p class="mute">No line render video for this capture.</p>`}
 <div class="card"><div style="display:flex;justify-content:space-between" class="mute"><span>Timeline -- click a frame to seek${A.done?"":", drag to select a range"}</span><span id="counts"></span></div>
 <div id="tl" style="margin-top:6px"></div><div id="editor"></div></div>`;
 const v=$("v");if(v){v.onloadedmetadata=()=>{A.dur=v.duration;renderAva()};$("prev").onclick=()=>seek(A.pos-1);$("next").onclick=()=>seek(A.pos+1);
  $("goto").onkeydown=e=>{if(e.key==="Enter")seek(+e.target.value)};const tick=()=>{if(S.ava!==A)return;const n=A.frames.length;
  if(n&&A.dur){const p=Math.min(n,Math.floor(v.currentTime*fps()+1e-3)+1);if(p!==A.pos){A.pos=p;renderReadout();renderTimeline()}}requestAnimationFrame(tick)};requestAnimationFrame(tick)}
 window.onmouseup=()=>{A.drag=null};renderAva()}
const fps=()=>{const A=S.ava,n=A.frames.length;return A.dur>0&&n>0?n/A.dur:24};
function seek(p){const A=S.ava,v=$("v"),n=A.frames.length;if(!n)return;p=Math.max(1,Math.min(n,p));A.pos=p;if(v){v.pause();v.currentTime=(p-.5)/fps()}renderReadout();renderTimeline()}
function preview(){const A=S.ava,n=A.frames.length,owner=new Array(n).fill(null),conf=new Set(),parsed={};
 for(const l of LABELS){parsed[l]=parseRanges(A.text[l],n);for(const[a,b]of parsed[l].ranges)for(let p=a;p<=b;p++){if(owner[p-1]!==null)conf.add(p);owner[p-1]=l}}
 return{parsed,conf,prev:A.saved.map((s,i)=>owner[i]??s)}}
function renderReadout(){const A=S.ava,r=$("readout");if(!r)return;const f=A.frames[A.pos-1];
 r.innerHTML=`#${A.pos} / ${A.frames.length}`+(f?` · frame ${esc(f.frame_id)} · <span class="t-${f.status}">${f.status}</span>`+
 (f.skin_error_px!=null?` · <span class="${f.skin_error_px>SKIN_FLAG?"err":"mute"}">skin error ${f.skin_error_px.toFixed(2)} px</span>`:""):"")}
function renderTimeline(){const A=S.ava,{conf,prev}=preview(),T=$("tl");T.innerHTML="";
 prev.forEach((s,i)=>{const p=i+1,c=document.createElement("div"),f=A.frames[i],inSel=A.sel&&p>=A.sel[0]&&p<=A.sel[1];
  c.className=conf.has(p)?"conf":"c-"+s;if(s!==A.saved[i])c.style.opacity=.7;if(inSel)c.style.filter="brightness(1.5)";
  c.title=`#${p} · ${f.frame_id} · ${s}`+(f.skin_error_px!=null?` · skin error ${f.skin_error_px.toFixed(2)} px`:"");
  let h="";if(inSel)h+='<div style="position:absolute;left:0;right:0;top:0;height:3px;background:#fff"></div>';
  if((f.skin_error_px??0)>SKIN_FLAG)h+='<div style="position:absolute;left:0;right:0;bottom:0;height:3px;background:rgba(0,0,0,.6)"></div>';
  if(p===A.pos)h+='<div style="position:absolute;top:0;bottom:0;left:50%;width:2px;background:#fff"></div>';c.innerHTML=h;
  c.onmousedown=()=>{if(A.done){seek(p);return}A.drag=p;A.sel=[p,p];seek(p)};
  c.onmouseenter=()=>{if(A.drag!==null){A.sel=[Math.min(A.drag,p),Math.max(A.drag,p)];renderTimeline();renderEditor()}};T.appendChild(c)})}
function renderAva(){renderReadout();renderTimeline();renderEditor()}
function renderEditor(){const A=S.ava,{parsed,conf,prev}=preview(),E=$("editor"),cnt={Valid:0,Invalid:0,Rerun:0,unconfirmed:0};prev.forEach(s=>cnt[s]++);
 $("counts").innerHTML=LABELS.map(l=>`<span class="t-${l}" style="margin-left:10px">${l} ${cnt[l]}</span>`).join("")+`<span style="margin-left:10px">unlabeled ${cnt.unconfirmed}</span>`;
 if(A.done){E.innerHTML="";return}
 const errs=LABELS.flatMap(l=>parsed[l].errors.map(e=>`${l}: ${e}`)),dirty=prev.some((s,i)=>s!==A.saved[i]),savedUn=A.saved.filter(s=>s==="unconfirmed").length;
 const canApply=dirty&&!errs.length&&!conf.size;
 E.innerHTML=`<div style="margin-top:10px;display:flex;gap:6px;flex-wrap:wrap;align-items:center"><span class="mute">Selection:</span>
 <button id="ss">Start at playhead</button><button id="se">End at playhead</button><span>${A.sel?`#${A.sel[0]}–#${A.sel[1]}`:'<span class="mute">none</span>'}</span>
 ${LABELS.map(l=>`<button data-add="${l}" ${A.sel?"":"disabled"}>Add to ${l}</button>`).join("")}</div>
 <div style="margin-top:6px;display:flex;gap:6px;flex-wrap:wrap"><span class="mute">Unlabeled:</span><button data-un="Valid">Add Unlabeled to Valid</button><button data-un="Rerun">Add Unlabeled to Rerun</button></div>
 ${LABELS.map(l=>`<div class="lab"><span class="t-${l}">${l}</span><input data-l="${l}" value="${esc(A.text[l])}" placeholder="e.g. 1-100, 150-300"></div>
  <div style="margin-left:76px">${parsed[l].ranges.map(([a,b],i)=>`<span class="chip">${a===b?a:a+"-"+b}<b data-rm="${l}:${i}">×</b></span>`).join("")}</div>`).join("")}
 <div class="err">${errs.map(esc).join("<br>")}${conf.size?`<br>${conf.size} frame(s) are in more than one range`:""}</div>
 <div style="margin-top:8px;display:flex;gap:6px;align-items:center;flex-wrap:wrap"><button id="apply" ${canApply?"":"disabled"}>Apply</button>
 <button id="clear">Clear</button><button id="reset">Reset to saved</button><span style="margin-left:auto"></span>
 ${A.confirm?`<span class="warn">Mark this capture done?</span><button id="cdone">Confirm Done</button><button id="ndone">Cancel</button>`
  :`<button id="done" ${(savedUn||dirty)?"disabled":""}>Done</button>`}</div><div id="msg" class="${A.msgCls}" style="margin-top:6px">${esc(A.msg)}</div>`;
 E.querySelectorAll("input[data-l]").forEach(i=>i.oninput=()=>{A.text[i.dataset.l]=i.value;renderTimeline();renderEditorKeepFocus(i.dataset.l,i.selectionStart)});
 E.querySelectorAll("[data-add]").forEach(b=>b.onclick=()=>{const l=b.dataset.add;A.text[l]=fmt([...parsed[l].ranges,A.sel]);A.sel=null;renderAva()});
 E.querySelectorAll("[data-un]").forEach(b=>b.onclick=()=>{const l=b.dataset.un,runs=[];prev.forEach((s,i)=>{if(s!=="unconfirmed")return;const p=i+1,last=runs[runs.length-1];
  if(last&&last[1]===p-1)last[1]=p;else runs.push([p,p])});if(runs.length){A.text[l]=fmt([...parsed[l].ranges,...runs]);renderAva()}});
 E.querySelectorAll("[data-rm]").forEach(b=>b.onclick=()=>{const[l,i]=b.dataset.rm.split(":");A.text[l]=fmt(parsed[l].ranges.filter((_,k)=>k!==+i));renderAva()});
 $("ss").onclick=()=>{const end=A.sel?A.sel[1]:A.pos;A.sel=[Math.min(A.pos,end),Math.max(A.pos,end)];renderAva()};
 $("se").onclick=()=>{const st=A.sel?A.sel[0]:A.pos;A.sel=[Math.min(st,A.pos),Math.max(st,A.pos)];renderAva()};
 $("clear").onclick=()=>{A.text={Valid:"",Invalid:"",Rerun:""};A.sel=null;renderAva()};
 $("reset").onclick=()=>{A.text=rangesFromStatuses(A.saved);A.sel=null;renderAva()};
 $("apply").onclick=async()=>{const ranges={};for(const l of LABELS)ranges[l]=parsed[l].ranges;
  try{const r=await api(`/api/ava/${encodeURIComponent(A.cap)}/frames`,{ranges});await reloadCapture(r.warnings.length?"Saved. "+r.warnings.join("; "):"Saved.","mute")}catch(e){A.msg=e.message;A.msgCls="err";renderEditor()}};
 if($("done"))$("done").onclick=()=>{A.confirm=true;renderEditor()};
 if($("ndone"))$("ndone").onclick=()=>{A.confirm=false;renderEditor()};
 if($("cdone"))$("cdone").onclick=async()=>{try{await api(`/api/ava/${encodeURIComponent(A.cap)}/done`,{});await reloadCapture("Done (export/archive happen at merge on the host).","mute")}catch(e){A.msg=e.message;A.msgCls="err";A.confirm=false;renderEditor()}}}
function renderEditorKeepFocus(l,caret){renderEditor();const i=document.querySelector(`input[data-l="${l}"]`);if(i){i.focus();i.setSelectionRange(caret,caret)}}
async function reloadCapture(msg,cls){const A=S.ava,pos=A.pos;await load();const d=await api("/api/ava/"+encodeURIComponent(A.cap));
 A.frames=d.frames;A.saved=d.frames.map(f=>f.status);A.done=d.is_done;A.status=d.label_status;A.text=rangesFromStatuses(A.saved);A.sel=null;A.confirm=false;A.msg=msg;A.msgCls=cls;A.pos=pos;renderAva()}
$("dsN").onclick=()=>{S.ds="N";S.sel=null;S.on=new Set(NS);$("dsN").className="on";$("dsA").className="";$("main").innerHTML='<p class="mute">Select a pair on the left.</p>';drawPills();drawList()};
$("dsA").onclick=()=>{S.ds="A";S.sel=null;S.on=new Set(["unlabeled","unconfirmed","confirmed"]);$("dsA").className="on";$("dsN").className="";$("main").innerHTML='<p class="mute">Select a capture on the left.</p>';drawPills();drawList()};
for(const id of["q","amin","amax"])$(id).oninput=drawList;
document.querySelectorAll("#rwctl [data-rw]").forEach(b=>b.onclick=()=>{S.rw=b.dataset.rw;drawList()});
load();
</script></body></html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, required=True, help="Folder from export_local_bundle.py")
    parser.add_argument("--videos", type=Path, required=True,
                        help="Folder holding line_renders/, point_overlay_renders/, line_renders_ava256/")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    store = Store(args.bundle, args.videos)
    for sub in ("line_renders", "point_overlay_renders", "line_renders_ava256"):
        if not (args.videos / sub).is_dir():
            print(f"NOTE: {args.videos / sub} not found -- those videos won't play")
    class QuietServer(ThreadingHTTPServer):
        daemon_threads = True

        def handle_error(self, request, client_address):
            # A browser aborting a video request mid-transfer (seeking, switching
            # videos) is normal; only print real errors.
            import sys
            if isinstance(sys.exc_info()[1], ConnectionError):
                return
            super().handle_error(request, client_address)

    server = QuietServer(("127.0.0.1", args.port), make_handler(store))
    print(f"Local labeler on http://localhost:{args.port}  (saving to {store.tracker_path})")
    print("Leave this window open while labeling; press Ctrl+C to stop.", flush=True)
    started = time.monotonic()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        if time.monotonic() - started < 2:
            print("Stopped by an interrupt right after starting. In Git Bash on Windows this usually means the "
                  "terminal sent Ctrl+C -- run it as `winpty python local_labeler.py ...`, or from PowerShell / "
                  "Command Prompt instead.")
        else:
            print("Stopped (Ctrl+C).")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
