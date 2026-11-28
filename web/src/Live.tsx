import { useCallback, useEffect, useRef, useState } from "react";
import { getLiveSnapshot, Line, LiveStats, openLiveStream, startLive } from "./api";

const ICON: Record<string, string> = {
  pass: "↗️", possession: "🟢", turnover: "🔄", carry: "🏃", attack: "⚡", roar: "📣", close_up: "🎥",
  stats: "📊", goal: "⚽", shot: "🎯", corner: "⛳", card: "🟨", red_card: "🟥", penalty: "❗",
};
const fmt = (s: number) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

const START_BUFFER_S = 10; // broadcast delay before playback starts
const RESUME_BUFFER_S = 6; // after a stall, wait for this much commentary ahead
const STALL_MARGIN_S = 2; // pause when playback gets this close to ready commentary
const STALE_SPEECH_S = 1.5; // don't speak a line that's already this far behind the video
const posKey = (sid: string) => `live-pos-${sid}`;
const store = {
  get: (k: string) => { try { return localStorage.getItem(k); } catch { return null; } },
  set: (k: string, v: string) => { try { localStorage.setItem(k, v); } catch { /* storage blocked */ } },
};

type Status = "idle" | "uploading" | "running" | "finalizing" | "done" | "error";

export default function LivePlayer({ onFinished }: { onFinished?: (clipId: string) => void }) {
  const video = useRef<HTMLVideoElement>(null);
  const feed = useRef<HTMLOListElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [src, setSrc] = useState<string | null>(null);
  const [language, setLanguage] = useState("en");
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [uploadPct, setUploadPct] = useState(0);
  const [sid, setSid] = useState<string | null>(null);
  const [lines, setLines] = useState<Line[]>([]);
  const [readyT, setReadyT] = useState(0);
  const [duration, setDuration] = useState(0);
  const [stats, setStats] = useState<LiveStats | null>(null);
  const [kits, setKits] = useState<string[]>([]);
  const [perf, setPerf] = useState<Record<string, unknown> | null>(null);
  const [now, setNow] = useState(0);
  const [buffering, setBuffering] = useState(false);
  const [voice, setVoice] = useState(true);
  const [captions, setCaptions] = useState(true);
  const [resumeAt, setResumeAt] = useState<number | null>(null); // reloaded page: waits for a click

  // refs mirror state for the animation loop
  const r = useRef({ lines: [] as Line[], readyT: 0, status: "idle" as Status, autoPaused: false, started: false,
    spoken: new Set<number>(), voice: true, language: "en", needsClick: false, sid: null as string | null, savedAt: 0 });
  r.current.lines = lines; r.current.readyT = readyT; r.current.status = status;
  r.current.voice = voice; r.current.language = language; r.current.sid = sid;

  useEffect(() => () => { if (src?.startsWith("blob:")) URL.revokeObjectURL(src); }, [src]);

  // a reloaded page (#live/<session id>) reattaches to the session and plays the server's copy
  useEffect(() => {
    const m = location.hash.match(/^#live\/([\w-]+)$/);
    if (!m) return;
    const id = m[1];
    getLiveSnapshot(id).then((snap) => {
      if (!snap) { location.hash = "live"; return; }
      setLanguage(snap.language || "en");
      setDuration(snap.duration);
      setStatus(snap.status === "queued" ? "running" : (snap.status as Status));
      if (snap.error) setError(snap.error);
      r.current.needsClick = true;
      setResumeAt(Number(store.get(posKey(id)) ?? 0) || 0);
      setSrc(`api/live/${id}/video`);
      setSid(id);
    }).catch((e) => setError(String(e instanceof Error ? e.message : e)));
  }, []);

  const resume = () => {
    const v = video.current;
    window.speechSynthesis?.speak(new SpeechSynthesisUtterance("")); // unlock speech inside the click
    if (v && resumeAt) v.currentTime = Math.max(0, resumeAt - 2);
    r.current.needsClick = false;
    r.current.started = false; // the start gate plays it once enough commentary is ready
    setResumeAt(null);
  };

  const speak = useCallback((ln: Line) => {
    if (!("speechSynthesis" in window)) return;
    const synth = window.speechSynthesis;
    const u = new SpeechSynthesisUtterance(ln.text);
    u.lang = r.current.language === "ur" ? "ur-PK" : "en-GB";
    u.rate = 1.0 + 0.35 * ln.excitement;
    u.pitch = 1.0 + 0.25 * ln.excitement;
    const v = video.current;
    u.onstart = () => { if (v) v.volume = 0.3; }; // duck the match audio under the commentator
    u.onend = () => { if (v && !synth.speaking) v.volume = 1; };
    if (ln.excitement >= 0.7) synth.cancel(); // big moments interrupt
    synth.speak(u);
  }, []);

  // playback gate + speech scheduler
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const v = video.current, s = r.current;
      if (v) {
        const t = v.currentTime;
        setNow(t);
        const finished = s.status === "done" || s.status === "finalizing";
        if (s.sid && !v.paused && Math.abs(t - s.savedAt) > 1) { s.savedAt = t; store.set(posKey(s.sid), String(t)); }
        if (!s.needsClick && !s.started && (s.readyT >= START_BUFFER_S || (finished && s.readyT > 0)) && v.readyState >= 2) {
          s.started = true;
          v.play().catch(() => {});
        }
        if (!finished && !v.paused && t > s.readyT - STALL_MARGIN_S) {
          s.autoPaused = true; v.pause(); setBuffering(true);
        } else if (s.autoPaused && (finished || s.readyT >= t + RESUME_BUFFER_S)) {
          s.autoPaused = false; setBuffering(false); v.play().catch(() => {});
        }
        if (!v.paused) {
          s.lines.forEach((ln, i) => {
            if (!s.spoken.has(i) && t >= ln.t) {
              s.spoken.add(i);
              if (s.voice && t - ln.t <= STALE_SPEECH_S) speak(ln);
            }
          });
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [speak]);

  // on seek: lines before the new position count as already said
  const onSeeked = () => {
    const t = video.current?.currentTime ?? 0;
    window.speechSynthesis?.cancel();
    r.current.spoken = new Set(r.current.lines.map((ln, i) => (ln.t < t - 0.2 ? i : -1)).filter((i) => i >= 0));
  };

  // SSE with manual reconnect that resumes after the lines we already have
  useEffect(() => {
    if (!sid) return;
    let es: EventSource | null = null, closed = false, retry = 0;
    const connect = () => {
      es = openLiveStream(sid, r.current.lines.length, {
        line: (ln) => setLines((ls) => [...ls, ln]),
        progress: (p) => { setReadyT(p.ready_t); setDuration(p.duration); setKits(p.kits); if (p.status === "finalizing") setStatus("finalizing"); },
        stats: setStats,
        done: (d) => { setStatus("done"); setPerf(d.perf); closed = true; es?.close(); onFinished?.(d.clip_id); },
        error: (msg) => { setStatus("error"); setError(msg); closed = true; es?.close(); },
      });
      es.onerror = () => {
        es?.close();
        if (!closed) setTimeout(connect, Math.min(5000, 500 * 2 ** retry++));
      };
    };
    connect();
    return () => { closed = true; es?.close(); };
  }, [sid, onFinished]);

  // keep the newest spoken line in view
  const current = lines.findIndex((l) => now >= l.t && now < l.end + 0.6);
  useEffect(() => {
    if (current < 0 || !feed.current) return;
    (feed.current.children[current] as HTMLElement | undefined)?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [current]);

  const go = async () => {
    if (!file) return;
    setError(null); setLines([]); setReadyT(0); setStats(null); setPerf(null); setSid(null);
    Object.assign(r.current, { spoken: new Set<number>(), started: false, autoPaused: false, needsClick: false });
    setResumeAt(null);
    const url = URL.createObjectURL(file);
    setSrc(url);
    // unlock media + speech inside the click (browsers block them later without a gesture)
    window.speechSynthesis?.speak(new SpeechSynthesisUtterance(""));
    setTimeout(() => { const v = video.current; v?.play().then(() => v.pause()).catch(() => {}); }, 0);
    setStatus("uploading");
    try {
      const id = await startLive(file, language, setUploadPct);
      setSid(id);
      setStatus("running");
      location.hash = `live/${id}`; // a reload can find the session again
    } catch (e) {
      setStatus("error");
      setError(String(e instanceof Error ? e.message : e));
    }
  };

  const pct = (x: number) => `${Math.min(100, (100 * x) / Math.max(duration, 1))}%`;
  const possession = stats?.possession_pct ?? {};
  const shown = kits.length ? kits : Object.keys(possession);

  return (
    <div className="player live">
      <div className="livebar">
        <label className="file">
          <input type="file" accept="video/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            disabled={status === "uploading" || status === "running"} />
        </label>
        <select value={language} onChange={(e) => setLanguage(e.target.value)} disabled={status === "running"}>
          <option value="en">English</option><option value="ur">Urdu</option>
        </select>
        <button className="golive" onClick={go} disabled={!file || status === "uploading" || status === "running"}>
          {status === "running" ? "Live…" : "Go live"}
        </button>
        <span className={`livedot ${status}`}>{statusText(status, uploadPct)}</span>
      </div>
      {error && <div className="error">{error}</div>}

      <div className="stage">
        {src ? (
          <video ref={video} src={src} controls playsInline preload="auto" onSeeked={onSeeked} />
        ) : (
          <div className="novideo">Pick a match video (your own or permitted footage) and press <b>Go live</b>.
            Playback starts after a {START_BUFFER_S}-second commentary buffer.</div>
        )}
        {buffering && <div className="buffering">Buffering commentary…</div>}
        {resumeAt !== null && (
          <button className="resume" onClick={resume}>▶ Resume live commentary{resumeAt > 5 ? ` from ${fmt(resumeAt)}` : ""}</button>
        )}
        {src && resumeAt === null && !r.current.started && status !== "error" && <div className="buffering">Analysing the first {START_BUFFER_S} s…</div>}
        {captions && current >= 0 && (
          <div className={`caption ${lines[current].excitement >= 0.7 ? "big" : ""}`}>{lines[current].text}</div>
        )}
      </div>

      <div className="controls">
        <label><input type="checkbox" checked={voice} onChange={(e) => { setVoice(e.target.checked); if (!e.target.checked) window.speechSynthesis?.cancel(); }} /> Voice</label>
        <label><input type="checkbox" checked={captions} onChange={(e) => setCaptions(e.target.checked)} /> Captions</label>
        <span className="clock">{fmt(now)} / {fmt(duration)}</span>
      </div>
      <div className="analysed" title="commentary ready up to here">
        <div className="ready" style={{ width: pct(readyT) }} />
        <div className="playhead2" style={{ left: pct(now) }} />
      </div>

      <div className="panels">
        <section className="panel">
          <h3>Live commentary</h3>
          <ol className="feed" ref={feed}>
            {lines.map((l, i) => (
              <li key={i} className={i === current ? "now" : l.t < now ? "past" : "ahead"}
                onClick={() => { if (video.current) video.current.currentTime = Math.max(0, l.t - 1); }}>
                <span className="ts">{fmt(l.t)}</span>
                <span className="ic">{ICON[l.event] ?? "•"}</span>
                <span className="txt">{l.t > now + 0.5 ? "…" : l.text}</span>
                <span className={`src ${l.source}`}>{l.source === "llm" ? "Claude" : "template"}</span>
              </li>
            ))}
          </ol>
        </section>
        <section className="panel">
          <h3>Match stats <small>(from tracking)</small></h3>
          {shown.length >= 2 ? (
            <>
              <div className="possession">
                {shown.map((k) => (
                  <div key={k} className="pos" style={{ flexGrow: possession[k] ?? 1, background: kitColour(k) }}>
                    {k} {possession[k] != null ? `${possession[k]}%` : ""}
                  </div>
                ))}
              </div>
              <table className="stats">
                <thead><tr><th>{shown[0]}</th><th></th><th>{shown[1]}</th></tr></thead>
                <tbody>
                  <tr><td>{stats?.passes?.[shown[0]] ?? 0}</td><th>passes</th><td>{stats?.passes?.[shown[1]] ?? 0}</td></tr>
                </tbody>
              </table>
            </>
          ) : <p className="muted">Kit colours appear once enough players have been seen in wide shots.</p>}
          <table className="stats">
            <tbody>
              <tr><th>turnovers</th><td>{stats?.turnovers ?? 0}</td></tr>
              <tr><th>crowd roars</th><td>{stats?.roars ?? 0}</td></tr>
              <tr><th>longest pass chain</th><td>{stats?.longest_chain ?? 0}</td></tr>
            </tbody>
          </table>
          {perf && (
            <p className="muted">Analysed at {String(perf.x_realtime)}× real time
              {perf.llm_calls ? ` · ${perf.llm_calls} Claude lines, p95 ${perf.llm_latency_p95}s` : " · templates only"}.</p>
          )}
        </section>
      </div>
    </div>
  );
}

function statusText(s: Status, up: number) {
  switch (s) {
    case "idle": return "not live";
    case "uploading": return `uploading ${Math.round(up * 100)}%`;
    case "running": return "● LIVE";
    case "finalizing": return "finishing up";
    case "done": return "full time";
    case "error": return "error";
  }
}

const KIT: Record<string, string> = {
  white: "#e8e8e8", black: "#222", grey: "#888", red: "#c8202a", "dark red": "#7a1420", blue: "#2a48c8",
  "sky blue": "#6eaee6", navy: "#14204f", yellow: "#f0d228", orange: "#f08220", green: "#1e8c3c",
  purple: "#6e288c", pink: "#e678aa", claret: "#6e1e3c",
};
function kitColour(k: string) {
  const c = KIT[k.replace(/ \([AB]\)$/, "")] ?? "#3a4a42";
  return c;
}
