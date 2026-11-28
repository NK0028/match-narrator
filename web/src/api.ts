export type Event = {
  event: string;
  t: number;
  timestamp: string;
  team?: string | null;
  player?: string | null;
  confidence: number;
  source_label?: string | null;
  meta?: Record<string, unknown>;
};

export type Line = {
  t: number;
  end: number;
  text: string;
  event_index: number;
  event: string;
  source: "template" | "llm";
  excitement: number;
  audio?: string | null;
};

export type Track = {
  clip_id: string;
  duration: number;
  match: { home: string; away: string; home_colour?: string | null; away_colour?: string | null };
  events: Event[];
  lines: Line[];
  stats: Record<string, unknown> & { title?: string; ground_truth?: Event[] };
  audio_track?: string | null;
  video?: string | null;
};

export type ClipSummary = { id: string; title: string; duration: number; n_events: number; n_lines: number };

/** "api" when the FastAPI backend is reachable, otherwise a static build reading ./demo/. */
export type Mode = "api" | "static";

export async function detectMode(): Promise<{ mode: Mode; uploads: boolean; live: boolean }> {
  try {
    const r = await fetch("api/health");
    if (r.ok && (r.headers.get("content-type") || "").includes("json")) {
      const h = await r.json();
      return { mode: "api", uploads: !!h.uploads, live: !!h.live };
    }
  } catch {
    /* static hosting */
  }
  return { mode: "static", uploads: false, live: false };
}

export async function listClips(mode: Mode): Promise<ClipSummary[]> {
  const r = await fetch(mode === "api" ? "api/clips" : "demo/index.json");
  return r.ok ? r.json() : [];
}

export async function getTrack(mode: Mode, id: string): Promise<Track> {
  const r = await fetch(mode === "api" ? `api/clips/${id}` : `demo/${id}/track.json`);
  if (!r.ok) throw new Error(`clip ${id} not found`);
  return r.json();
}

export const mediaUrl = (mode: Mode, id: string, file: string) =>
  mode === "api" ? `media/${id}/${file}` : `demo/${id}/${file}`;

export async function upload(form: FormData): Promise<string> {
  const r = await fetch("api/process", { method: "POST", body: form });
  if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
  return (await r.json()).job_id;
}

export async function jobStatus(id: string): Promise<{ status: string; clip_id?: string; error?: string }> {
  return (await fetch(`api/jobs/${id}`)).json();
}

// ---- live mode ----------------------------------------------------------------------
export type LiveStats = {
  possession_pct?: Record<string, number>;
  passes?: Record<string, number>;
  turnovers?: number;
  roars?: number;
  longest_chain?: number;
};

export type LiveHandlers = {
  line: (l: Line) => void;
  progress: (p: { ready_t: number; duration: number; status: string; kits: string[] }) => void;
  stats: (s: LiveStats) => void;
  done: (d: { clip_id: string; perf: Record<string, unknown> }) => void;
  error: (msg: string) => void;
};

/** Upload with progress (fetch can't report upload progress), returns the live session id. */
export function startLive(file: File, language: string, onProgress: (f: number) => void): Promise<string> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("video", file);
    form.append("language", language);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "api/live");
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      let body: { session_id?: string; detail?: string } = {};
      try { body = JSON.parse(xhr.responseText); } catch { /* not JSON */ }
      if (xhr.status === 200 && body.session_id) resolve(body.session_id);
      else reject(new Error(body.detail || `upload failed (${xhr.status})`));
    };
    xhr.onerror = () => reject(new Error("upload failed: network error"));
    xhr.send(form);
  });
}

export type LiveSnapshot = { session_id: string; status: string; error: string | null; language: string; duration: number };

/** null when the server no longer has the session (restarted, or never existed). */
export async function getLiveSnapshot(sid: string): Promise<LiveSnapshot | null> {
  const r = await fetch(`api/live/${encodeURIComponent(sid)}?since=1000000000`);
  if (r.status === 404) return null;
  if (!r.ok) throw new Error(`live session lookup failed (${r.status})`);
  return r.json();
}

export function openLiveStream(sid: string, since: number, h: LiveHandlers): EventSource {
  const es = new EventSource(`api/live/${sid}/stream?since=${since}`);
  es.addEventListener("line", (e) => h.line(JSON.parse((e as MessageEvent).data)));
  es.addEventListener("progress", (e) => h.progress(JSON.parse((e as MessageEvent).data)));
  es.addEventListener("stats", (e) => h.stats(JSON.parse((e as MessageEvent).data)));
  es.addEventListener("done", (e) => h.done(JSON.parse((e as MessageEvent).data)));
  es.addEventListener("error", (e) => {
    const data = (e as MessageEvent).data;
    if (data) h.error(JSON.parse(data).error); // server-sent error event (not a connection drop)
  });
  return es;
}
