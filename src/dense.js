import { env, pipeline } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";

const MODEL_ID = "Xenova/bge-small-zh-v1.5";
const BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章：";

env.allowLocalModels = true;
env.localModelPath = "/assets/models/";
env.allowRemoteModels = true;
env.useBrowserCache = true;

let extractorPromise = null;
let corpusCacheKey = "";
let corpusEmbeddings = [];

function toVector(output) {
  if (!output?.data) throw new Error("Transformer output did not include tensor data.");
  return Float32Array.from(output.data);
}

export async function loadModel(onProgress = () => {}) {
  if (!extractorPromise) {
    extractorPromise = pipeline("feature-extraction", MODEL_ID, {
      quantized: true,
      progress_callback: (event) => onProgress(event),
    });
  }
  return extractorPromise;
}

export async function embed(text) {
  const extractor = await loadModel();
  const output = await extractor(String(text || ""), {
    pooling: "mean",
    normalize: true,
  });
  return toVector(output);
}

export async function embedQuery(text) {
  return embed(`${BGE_QUERY_PREFIX}${String(text || "")}`);
}

function makeCorpusKey(docs) {
  return docs.map((doc) => `${doc.id}:${doc.text}`).join("|");
}

export async function embedCorpus(docs, onProgress = () => {}) {
  const key = makeCorpusKey(docs);
  if (key === corpusCacheKey && corpusEmbeddings.length) return corpusEmbeddings;

  await loadModel(onProgress);
  corpusEmbeddings = [];
  for (let i = 0; i < docs.length; i += 1) {
    const doc = docs[i];
    const vector = await embed(doc.text);
    corpusEmbeddings.push({ doc, vector });
    onProgress({
      status: "embedding",
      loaded: i + 1,
      total: docs.length,
      file: "corpus",
      progress: ((i + 1) / docs.length) * 100,
    });
  }
  corpusCacheKey = key;
  return corpusEmbeddings;
}

export function dot(a, b) {
  const size = Math.min(a.length, b.length);
  let sum = 0;
  for (let i = 0; i < size; i += 1) sum += a[i] * b[i];
  return sum;
}

export function cosine(a, b) {
  let normA = 0;
  let normB = 0;
  const size = Math.min(a.length, b.length);
  for (let i = 0; i < size; i += 1) {
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (!normA || !normB) return 0;
  return dot(a, b) / (Math.sqrt(normA) * Math.sqrt(normB));
}

export function l2(a, b) {
  const size = Math.min(a.length, b.length);
  let sum = 0;
  for (let i = 0; i < size; i += 1) {
    const delta = a[i] - b[i];
    sum += delta * delta;
  }
  return Math.sqrt(sum);
}

export async function searchDense(query, k = 5) {
  if (!corpusEmbeddings.length) {
    throw new Error("Dense corpus is not embedded yet. Call embedCorpus(docs) first.");
  }
  const queryVector = await embedQuery(query);
  return corpusEmbeddings
    .map(({ doc, vector }) => ({
      doc,
      vector,
      score: cosine(queryVector, vector),
      queryVector,
    }))
    .sort((a, z) => z.score - a.score)
    .slice(0, k);
}

export function getCorpusEmbeddings() {
  return corpusEmbeddings;
}
