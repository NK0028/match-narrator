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

export async function detectMode(): Promise<{ mode: Mode; uploads: boolean }> {
  try {
    const r = await fetch("api/health");
    if (r.ok && (r.headers.get("content-type") || "").includes("json")) {
      const h = await r.json();
      return { mode: "api", uploads: !!h.uploads };
    }
  } catch {
    /* static hosting */
  }
  return { mode: "static", uploads: false };
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
