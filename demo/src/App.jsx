import { useEffect, useRef, useState } from "react";
import { loadEngine, transliterate } from "./translit.js";

// word -> [top3 candidates]; survives re-renders, capped to keep memory sane
const cache = new Map();
const CACHE_MAX = 2000;

export default function App() {
  const [status, setStatus] = useState("starting…");
  const [input, setInput] = useState("");
  const [words, setWords] = useState([]); // [{raw, choices, picked}]
  const [busy, setBusy] = useState(false);
  const timer = useRef(null);

  useEffect(() => {
    loadEngine(setStatus).catch((e) => setStatus("failed to load: " + e.message));
  }, []);

  useEffect(() => {
    clearTimeout(timer.current);
    timer.current = setTimeout(async () => {
      const tokens = input.split(/\s+/).filter(Boolean);
      setBusy(true);
      const out = [];
      for (const raw of tokens) {
        if (!cache.has(raw)) {
          if (cache.size > CACHE_MAX) cache.clear();
          cache.set(raw, await transliterate(raw));
        }
        out.push({ raw, choices: cache.get(raw), picked: 0 });
      }
      setBusy(false);
      setWords(out);
    }, 350);
    return () => clearTimeout(timer.current);
  }, [input]);

  const pick = (i, j) =>
    setWords((ws) => ws.map((w, k) => (k === i ? { ...w, picked: j } : w)));

  const ready = status === null;

  return (
    <div style={styles.page}>
      <h1 style={styles.h1}>Manglish → മലയാളം</h1>
      <p style={styles.sub}>
        Char-level Transformer trained from scratch · runs entirely in your
        browser via ONNX Runtime Web — no server, no API.
      </p>

      {!ready && <div style={styles.loading}>{status}</div>}

      {ready && (
        <>
          <textarea
            style={styles.input}
            rows={3}
            placeholder="type manglish here… e.g. enthokke undu vishesham"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            autoFocus
          />
          <div style={styles.output}>
            {words.length === 0 && (
              <span style={styles.hint}>output appears here</span>
            )}
            {words.map((w, i) => (
              <span key={i} style={styles.wordWrap}>
                <WordChip word={w} onPick={(j) => pick(i, j)} />{" "}
              </span>
            ))}
            {busy && <span style={styles.hint}>…</span>}
          </div>
          <p style={styles.foot}>
            click a word to see the model's top-3 candidates
          </p>
        </>
      )}
    </div>
  );
}

function WordChip({ word, onPick }) {
  const [open, setOpen] = useState(false);
  return (
    <span style={{ position: "relative", display: "inline-block" }}>
      <span style={styles.chip} onClick={() => setOpen((o) => !o)}>
        {word.choices[word.picked]}
      </span>
      {open && word.choices.length > 1 && (
        <span style={styles.menu}>
          {word.choices.map((c, j) => (
            <span
              key={j}
              style={{
                ...styles.menuItem,
                fontWeight: j === word.picked ? 700 : 400,
              }}
              onClick={() => {
                onPick(j);
                setOpen(false);
              }}
            >
              {c}
            </span>
          ))}
        </span>
      )}
    </span>
  );
}

const styles = {
  page: {
    maxWidth: 640,
    margin: "8vh auto",
    padding: "0 20px",
    fontFamily: "system-ui, 'Noto Sans Malayalam', sans-serif",
    color: "#1a1a2e",
  },
  h1: { fontSize: 32, marginBottom: 4 },
  sub: { color: "#666", marginTop: 0, fontSize: 14 },
  loading: { padding: 24, color: "#888", fontStyle: "italic" },
  input: {
    width: "100%",
    fontSize: 18,
    padding: 12,
    borderRadius: 8,
    border: "1px solid #ccc",
    boxSizing: "border-box",
    resize: "vertical",
  },
  output: {
    minHeight: 64,
    marginTop: 16,
    padding: 16,
    background: "#f6f6fa",
    borderRadius: 8,
    fontSize: 24,
    lineHeight: 1.9,
  },
  hint: { color: "#aaa", fontSize: 14 },
  wordWrap: { whiteSpace: "nowrap" },
  chip: {
    cursor: "pointer",
    borderBottom: "1px dashed #99a",
  },
  menu: {
    position: "absolute",
    top: "100%",
    left: 0,
    zIndex: 10,
    background: "#fff",
    border: "1px solid #ddd",
    borderRadius: 6,
    boxShadow: "0 4px 12px rgba(0,0,0,.12)",
    display: "flex",
    flexDirection: "column",
    minWidth: 120,
  },
  menuItem: { padding: "6px 12px", cursor: "pointer", fontSize: 18 },
  foot: { color: "#999", fontSize: 12 },
};
