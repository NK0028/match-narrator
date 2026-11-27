import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ClipSummary, Event, Line, Mode, Track, detectMode, getTrack, jobStatus, listClips, mediaUrl, upload,
} from "./api";
import LivePlayer from "./Live";

const ICON: Record<string, string> = {
  goal: "⚽", shot: "🎯", corner: "⛳", card: "🟨", red_card: "🟥", kickoff: "▶️", substitution: "🔁",
  penalty: "❗", offside: "🚩", foul: "✋", free_kick: "🦶", clearance: "🧹", throw_in: "↩️", ball_out: "⤴️",
};
const LABEL: Record<string, string> = {
  goal: "Goal", shot: "Shot", corner: "Corner", card: "Yellow card", red_card: "Red card", kickoff: "Kick-off",
  substitution: "Substitution", penalty: "Penalty", offside: "Offside", foul: "Foul", free_kick: "Free kick",
  clearance: "Clearance", throw_in: "Throw-in", ball_out: "Ball out",
};
const fmt = (s: number) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

export default function App() {
  const [mode, setMode] = useState<Mode | null>(null);
  const [uploads, setUploads] = useState(false);
  const [live, setLive] = useState(false);
  const [view, setView] = useState<"clips" | "live">("clips");
  const [clips, setClips] = useState<ClipSummary[]>([]);
  const [clipId, setClipId] = useState<string | null>(null);
  const [track, setTrack] = useState<Track | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async (m: Mode) => {
    const cs = await listClips(m);
    setClips(cs);
    return cs;
  }, []);

  useEffect(() => {
    detectMode().then(async ({ mode, uploads, live }) => {
      setMode(mode);
      setUploads(uploads);
      setLive(live);
      if (live && location.hash === "#live") setView("live");
      const cs = await refresh(mode);
      const fromHash = location.hash.slice(1);
      setClipId(cs.find((c) => c.id === fromHash)?.id ?? cs[0]?.id ?? null);
    });
  }, [refresh]);

  useEffect(() => {
    const onHash = () => {
      const h = location.hash.slice(1);
      if (h === "live") setView("live");
      else if (h) { setView("clips"); setClipId(h); }
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    if (!mode || !clipId || view === "live") return;
    location.hash = clipId;
    setTrack(null);
    getTrack(mode, clipId).then(setTrack).catch((e) => setError(String(e)));
  }, [mode, clipId, view]);
  const onLiveFinished = useCallback(() => { if (mode) refresh(mode); }, [mode, refresh]);

  return (
    <div className="app">
      <header>
        <div className="brand">
          <span className="logo">⚽</span>
          <div>
            <h1>Match Narrator</h1>
            <p>Action spotting → structured events → grounded AI commentary → voice</p>
          </div>
        </div>
        {mode && <span className={`badge ${mode}`}>{mode === "api" ? "live backend" : "static demo"}</span>}
      </header>
      <div className="layout">
        <aside>
          {live && (
            <div className="tabs">
              <button className={view === "clips" ? "active" : ""} onClick={() => { setView("clips"); if (clipId) location.hash = clipId; }}>Clips</button>
              <button className={view === "live" ? "active" : ""} onClick={() => { setView("live"); location.hash = "live"; }}>● Live commentary</button>
            </div>
          )}
          <h2>Clips</h2>
          <ul className="clips">
            {clips.map((c) => (
              <li key={c.id}>
                <button className={c.id === clipId && view === "clips" ? "active" : ""} onClick={() => { setView("clips"); setClipId(c.id); }}>
                  <span className="title">{c.title}</span>
                  <span className="meta">{fmt(c.duration)} · {c.n_events} events · {c.n_lines} lines</span>
                </button>
              </li>
            ))}
            {mode && clips.length === 0 && <li className="muted">No clips yet.</li>}
          </ul>
          {uploads && mode === "api" && (
            <Uploader onDone={async (id) => { await refresh("api"); setClipId(id); }} />
          )}
          <Explainer />
        </aside>
        <main>
          {error && <div className="error">{error}</div>}
          {view === "live" && live ? <LivePlayer onFinished={onLiveFinished} />
            : track && mode ? <Player key={track.clip_id} mode={mode} track={track} /> : <div className="loading">Loading…</div>}
        </main>
      </div>
    </div>
  );
}

function Player({ mode, track }: { mode: Mode; track: Track }) {
  const video = useRef<HTMLVideoElement>(null);
  const audio = useRef<HTMLAudioElement>(null);
  const feed = useRef<HTMLOListElement>(null);
  const [now, setNow] = useState(0);
  const [voice, setVoice] = useState(true);
  const [captions, setCaptions] = useState(true);
  const [showGT, setShowGT] = useState(true);
  const hasVideo = !!track.video;
  const duration = track.duration;

  // Keep the commentary audio locked to the video clock.
  useEffect(() => {
    const v = video.current, a = audio.current;
    if (!v || !a) return;
    const sync = () => { if (Math.abs(a.currentTime - v.currentTime) > 0.25) a.currentTime = v.currentTime; };
    const play = () => { sync(); if (voice) a.play().catch(() => {}); };
    const pause = () => a.pause();
    v.addEventListener("play", play);
    v.addEventListener("pause", pause);
    v.addEventListener("seeked", sync);
    v.addEventListener("ratechange", () => (a.playbackRate = v.playbackRate));
    return () => { v.removeEventListener("play", play); v.removeEventListener("pause", pause); v.removeEventListener("seeked", sync); };
  }, [voice]);

  useEffect(() => {
    if (audio.current) audio.current.muted = !voice;
    if (voice && video.current && !video.current.paused) audio.current?.play().catch(() => {});
  }, [voice]);

  useEffect(() => {
    let raf = 0;
    const tick = () => { if (video.current) setNow(video.current.currentTime); raf = requestAnimationFrame(tick); };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  const current = track.lines.findIndex((l) => now >= l.t && now < l.end + 0.6);
  useEffect(() => {
    if (current < 0 || !feed.current) return;
    const el = feed.current.children[current] as HTMLElement | undefined;
    el?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [current]);

  const seek = (t: number) => { if (video.current) { video.current.currentTime = Math.max(0, t - 2); video.current.play().catch(() => {}); } };
  const stats = useMemo(() => liveStats(track.events, now), [track.events, now]);
  const gt = track.stats.ground_truth as Event[] | undefined;

  return (
    <div className="player">
      <div className="stage">
        {hasVideo ? (
          <video ref={video} src={mediaUrl(mode, track.clip_id, track.video!)} controls playsInline preload="metadata" muted={false} />
        ) : (
          <div className="novideo">No video for this clip — press play on the audio track below.</div>
        )}
        {stats.hasTeams && (
          <div className="scorebug">
            <span>{track.match.home}</span>
            <b>{stats.home.goals} – {stats.away.goals}</b>
            <span>{track.match.away}</span>
          </div>
        )}
        {captions && current >= 0 && (
          <div className={`caption ${track.lines[current].excitement >= 0.9 ? "big" : ""}`}>{track.lines[current].text}</div>
        )}
      </div>
      {track.audio_track && (
        <audio ref={audio} src={mediaUrl(mode, track.clip_id, track.audio_track)} preload="auto" controls={!hasVideo} />
      )}
      <div className="controls">
        <label><input type="checkbox" checked={voice} onChange={(e) => setVoice(e.target.checked)} /> Voice</label>
        <label><input type="checkbox" checked={captions} onChange={(e) => setCaptions(e.target.checked)} /> Captions</label>
        {gt && <label><input type="checkbox" checked={showGT} onChange={(e) => setShowGT(e.target.checked)} /> Ground truth</label>}
        <span className="clock">{fmt(now)} / {fmt(duration)}</span>
      </div>

      <Timeline events={track.events} gt={showGT ? gt : undefined} duration={duration} now={now} onSeek={seek} />

      <div className="panels">
        <section className="panel">
          <h3>Commentary</h3>
          <ol className="feed" ref={feed}>
            {track.lines.map((l: Line, i) => {
              const ev = track.events[l.event_index];
              return (
                <li key={i} className={i === current ? "now" : l.t < now ? "past" : ""} onClick={() => seek(l.t)}>
                  <span className="ts">{fmt(l.t)}</span>
                  <span className="ic">{ICON[l.event] ?? "•"}</span>
                  <span className="txt">{l.text}</span>
                  <span className={`src ${l.source}`} title={ev ? `confidence ${ev.confidence}` : ""}>
                    {l.source === "llm" ? "LLM" : "template"}{ev && ev.confidence < 1 ? ` · ${Math.round(ev.confidence * 100)}%` : ""}
                  </span>
                </li>
              );
            })}
          </ol>
        </section>
        <section className="panel">
          <h3>Live stats</h3>
          <table className="stats">
            <thead><tr>{stats.hasTeams ? <><th>{track.match.home}</th><th></th><th>{track.match.away}</th></> : <><th></th><th>spotted so far</th></>}</tr></thead>
            <tbody>
              {(["goals", "shots", "corners", "cards", "fouls"] as const).map((k) => (
                stats.hasTeams
                  ? <tr key={k}><td>{stats.home[k]}</td><th>{k}</th><td>{stats.away[k]}</td></tr>
                  : <tr key={k}><th>{k}</th><td>{stats.all[k]}</td></tr>
              ))}
            </tbody>
          </table>
          <h3>Event JSON <small>(all the LLM ever sees)</small></h3>
          <pre className="json">{JSON.stringify(
            track.events.filter((e) => e.t <= now).slice(-3).map(({ event, timestamp, team, player, confidence }) => ({ event, timestamp, team, player, confidence })),
            null, 1)}</pre>
        </section>
      </div>
    </div>
  );
}

function liveStats(events: Event[], now: number) {
  const z = () => ({ goals: 0, shots: 0, corners: 0, cards: 0, fouls: 0 });
  const s = { home: z(), away: z(), all: z(), hasTeams: events.some((e) => e.team === "home" || e.team === "away") };
  for (const e of events) {
    if (e.t > now) break;
    for (const side of [s.all, e.team === "home" ? s.home : e.team === "away" ? s.away : null]) {
    if (!side) continue;
    if (e.event === "goal") { side.goals++; side.shots++; }
    if (e.event === "shot") side.shots++;
    if (e.event === "corner") side.corners++;
    if (e.event === "card" || e.event === "red_card") side.cards++;
    if (e.event === "foul") side.fouls++;
    }
  }
  return s;
}

function Timeline({ events, gt, duration, now, onSeek }: {
  events: Event[]; gt?: Event[]; duration: number; now: number; onSeek: (t: number) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const pct = (t: number) => `${(100 * t) / Math.max(duration, 1)}%`;
  const big = new Set(["goal", "shot", "corner", "card", "red_card", "penalty", "kickoff", "substitution"]);
  return (
    <div className="timeline" ref={ref} onClick={(e) => {
      const r = ref.current!.getBoundingClientRect();
      onSeek(((e.clientX - r.left) / r.width) * duration);
    }}>
      <div className="row pred">
        <span className="rowlabel">spotted</span>
        {events.filter((e) => big.has(e.event)).map((e, i) => (
          <span key={i} className={`mk ${e.event}`} style={{ left: pct(e.t), opacity: 0.4 + 0.6 * e.confidence }}
            title={`${e.timestamp} ${LABEL[e.event]} (${Math.round(e.confidence * 100)}%)`}>{ICON[e.event]}</span>
        ))}
      </div>
      {gt && (
        <div className="row gt">
          <span className="rowlabel">truth</span>
          {gt.filter((e) => big.has(e.event)).map((e, i) => (
            <span key={i} className={`mk ${e.event}`} style={{ left: pct(e.t) }} title={`${e.timestamp} ${LABEL[e.event]}`}>{ICON[e.event]}</span>
          ))}
        </div>
      )}
      <div className="playhead" style={{ left: pct(now) }} />
    </div>
  );
}

function Uploader({ onDone }: { onDone: (clipId: string) => void }) {
  const [status, setStatus] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const submit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setBusy(true);
    setStatus("Uploading…");
    try {
      const id = await upload(new FormData(e.currentTarget));
      for (;;) {
        await new Promise((r) => setTimeout(r, 2000));
        const j = await jobStatus(id);
        setStatus(`Processing: ${j.status}`);
        if (j.status === "done" && j.clip_id) { setStatus("Done"); onDone(j.clip_id); break; }
        if (j.status === "error") { setStatus(`Error: ${j.error}`); break; }
      }
    } catch (err) {
      setStatus(`Error: ${String(err)}`);
    } finally {
      setBusy(false);
    }
  };
  return (
    <form className="upload" onSubmit={submit}>
      <h2>Narrate your clip</h2>
      <label>Video (mp4/mov, your own or permitted footage)<input name="video" type="file" accept="video/*" required /></label>
      <label>Events JSON (optional — else the spotter runs)<input name="events" type="file" accept=".json,application/json" /></label>
      <div className="teams">
        <input name="home" placeholder="Home team" />
        <input name="away" placeholder="Away team" />
      </div>
      <select name="language" defaultValue="en"><option value="en">English</option><option value="ur">Urdu</option></select>
      <button disabled={busy}>{busy ? "Working…" : "Generate commentary"}</button>
      {status && <p className="muted">{status}</p>}
    </form>
  );
}

function Explainer() {
  return (
    <details className="explainer">
      <summary>How it works</summary>
      <ol>
        <li><b>Spot</b> — a NetVLAD++ model scans SoccerNet ResNet features (2 fps) and outputs events + timestamps + team.</li>
        <li><b>Context</b> — on raw clips, YOLO + ByteTrack tag players and k-means on jersey colours assigns teams.</li>
        <li><b>Commentate</b> — templates narrate routine events; an LLM adds colour on big moments, and any line not grounded in the event JSON is rejected.</li>
        <li><b>Voice</b> — TTS voices each line, anchored to its event time, faster for big moments.</li>
      </ol>
      <p className="muted">Demo clips are a synthetic top-down view (match footage is copyrighted); commentary comes from the spotter's predictions on real SoccerNet match features.</p>
    </details>
  );
}
