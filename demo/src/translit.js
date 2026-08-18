// Transliteration engine: loads the exported ONNX graphs and reimplements
// the decode loop in JS. Mirrors src/inference/decode.py on the Python side.
import * as ort from "onnxruntime-web";
// Vite bundles the ORT runtime as regular assets (self-hosted, no CDN).
// Files under /public can't be imported as modules, so ?url imports are
// the supported way to hand ORT its own loader.
import ortMjsUrl from "onnxruntime-web/ort-wasm-simd-threaded.jsep.mjs?url";
import ortWasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.jsep.wasm?url";

ort.env.wasm.wasmPaths = { mjs: ortMjsUrl, wasm: ortWasmUrl };

const SOS = 1, EOS = 2, UNK = 3;
const MAX_LEN = 40;

let encSession = null;
let decSession = null;
let srcStoi = null;   // char -> id
let tgtItos = null;   // id -> char
let specials = null;

export async function loadEngine(onProgress) {
  onProgress?.("loading vocabularies…");
  const [srcVocab, tgtVocab] = await Promise.all([
    fetch("/model/vocab_src.json").then((r) => r.json()),
    fetch("/model/vocab_tgt.json").then((r) => r.json()),
  ]);
  srcStoi = new Map(srcVocab.map((c, i) => [c, i]));
  tgtItos = tgtVocab;
  specials = new Set(tgtVocab.slice(0, 4));

  onProgress?.("loading encoder (10 MB)…");
  encSession = await ort.InferenceSession.create("/model/encoder.onnx");
  onProgress?.("loading decoder (14 MB)…");
  decSession = await ort.InferenceSession.create("/model/decoder_step.onnx");
  onProgress?.(null);
}

function encodeSrc(word) {
  return Array.from(word).map((c) => srcStoi.get(c) ?? UNK);
}

function decodeTgt(ids) {
  return ids
    .map((i) => tgtItos[i])
    .filter((c) => !specials.has(c))
    .join("");
}

function int64Tensor(arr, dims) {
  return new ort.Tensor("int64", BigInt64Array.from(arr.map(BigInt)), dims);
}

function causalMask(n) {
  const data = new Float32Array(n * n);
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) data[i * n + j] = -Infinity;
  return new ort.Tensor("float32", data, [n, n]);
}

async function runEncoder(srcIds) {
  const out = await encSession.run({ src: int64Tensor(srcIds, [1, srcIds.length]) });
  return out.memory;
}

async function nextLogits(memory, prefix) {
  const out = await decSession.run({
    memory,
    tgt_so_far: int64Tensor(prefix, [1, prefix.length]),
    causal_mask: causalMask(prefix.length),
  });
  return out.logits.data; // Float32Array (vocab,)
}

function logSoftmax(logits) {
  let max = -Infinity;
  for (const v of logits) if (v > max) max = v;
  let sum = 0;
  for (const v of logits) sum += Math.exp(v - max);
  const logZ = max + Math.log(sum);
  return Array.from(logits, (v) => v - logZ);
}

function topK(arr, k) {
  return arr
    .map((v, i) => [v, i])
    .sort((a, b) => b[0] - a[0])
    .slice(0, k);
}

// Beam search over the decoder step; mirrors beam_search_decode_transformer
// in Python. Returns up to `beamSize` candidate strings, best first.
export async function transliterate(word, beamSize = 4, topN = 3) {
  const cleaned = word.toLowerCase().replace(/[^a-z]/g, "");
  if (!cleaned) return [word];

  const memory = await runEncoder(encodeSrc(cleaned));

  let beams = [{ tokens: [SOS], score: 0, done: false }];
  const completed = [];

  for (let step = 0; step < MAX_LEN; step++) {
    const active = beams.filter((b) => !b.done);
    if (!active.length) break;

    const candidates = beams.filter((b) => b.done).slice();
    for (const beam of active) {
      const logits = await nextLogits(memory, beam.tokens);
      const logProbs = logSoftmax(logits);
      for (const [lp, idx] of topK(logProbs, beamSize)) {
        candidates.push({
          tokens: [...beam.tokens, idx],
          score: beam.score + lp,
          done: idx === EOS,
        });
      }
    }

    const norm = (b) => b.score / Math.pow(Math.max(b.tokens.length - 1, 1), 0.7);
    candidates.sort((a, b) => norm(b) - norm(a));

    beams = [];
    for (const cand of candidates) {
      if (cand.done) completed.push(cand);
      else beams.push(cand);
      if (beams.length === beamSize) break;
    }
  }

  completed.push(...beams);
  const norm = (b) => b.score / Math.pow(Math.max(b.tokens.length - 1, 1), 0.7);
  completed.sort((a, b) => norm(b) - norm(a));

  const seen = new Set();
  const results = [];
  for (const b of completed) {
    const eosPos = b.tokens.indexOf(EOS);
    const ids = b.tokens.slice(1, eosPos === -1 ? undefined : eosPos);
    const text = decodeTgt(ids);
    if (text && !seen.has(text)) {
      seen.add(text);
      results.push(text);
    }
    if (results.length === topN) break;
  }
  return results.length ? results : [word];
}
