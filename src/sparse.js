const HAN_RE = /\p{Script=Han}/u;
const WORD_RE = /[\p{L}\p{N}_]+/gu;

function normalizeText(text) {
  return String(text || "")
    .toLowerCase()
    .replace(/[，。！？、；：“”‘’（）【】《》—…]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export function tokenize(text) {
  const normalized = normalizeText(text);
  const hanRuns = [];
  let currentHanRun = [];
  const terms = [];
  const segments = [];
  let buffer = "";

  const flushWord = () => {
    if (!buffer) return;
    const words = buffer.match(WORD_RE) || [];
    for (const word of words) {
      if (!HAN_RE.test(word)) {
        terms.push(word);
        segments.push({ type: "word", value: word });
      }
    }
    buffer = "";
  };

  const flushHanRun = () => {
    if (!currentHanRun.length) return;
    hanRuns.push(currentHanRun);
    currentHanRun = [];
  };

  for (const char of normalized) {
    if (HAN_RE.test(char)) {
      flushWord();
      currentHanRun.push(char);
      terms.push(char);
      segments.push({ type: "han-char", value: char });
    } else {
      flushHanRun();
      buffer += char;
    }
  }
  flushHanRun();
  flushWord();

  for (const chars of hanRuns) {
    for (let i = 0; i < chars.length - 1; i += 1) {
      const bigram = `${chars[i]}${chars[i + 1]}`;
      terms.push(bigram);
      segments.push({ type: "han-bigram", value: bigram });
    }
  }

  return terms;
}

export function explainTokenization(text) {
  const terms = tokenize(text);
  const counts = countTerms(terms);
  return {
    text,
    terms,
    uniqueTerms: Array.from(counts.entries()).map(([term, count]) => ({ term, count })),
  };
}

function countTerms(terms) {
  const counts = new Map();
  for (const term of terms) counts.set(term, (counts.get(term) || 0) + 1);
  return counts;
}

function buildDocVector(termCounts, idf) {
  const total = Array.from(termCounts.values()).reduce((sum, count) => sum + count, 0) || 1;
  const vector = new Map();
  for (const [term, count] of termCounts.entries()) {
    const tf = count / total;
    vector.set(term, tf * (idf.get(term) || 0));
  }
  return vector;
}

export function cosineFromMaps(a, b) {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (const value of a.values()) normA += value * value;
  for (const value of b.values()) normB += value * value;
  const [small, large] = a.size <= b.size ? [a, b] : [b, a];
  for (const [term, value] of small.entries()) dot += value * (large.get(term) || 0);
  if (!normA || !normB) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

export function buildSparseIndex(corpus) {
  const docs = corpus.map((doc) => {
    const terms = tokenize(doc.text);
    const counts = countTerms(terms);
    return {
      ...doc,
      tokens: terms,
      termCounts: counts,
      length: terms.length,
    };
  });

  const df = new Map();
  for (const doc of docs) {
    for (const term of doc.termCounts.keys()) df.set(term, (df.get(term) || 0) + 1);
  }

  const n = docs.length;
  const idf = new Map();
  for (const [term, freq] of df.entries()) {
    idf.set(term, Math.log((n + 1) / (freq + 0.5)) + 1);
  }

  const avgDocLength = docs.reduce((sum, doc) => sum + doc.length, 0) / Math.max(1, docs.length);
  const indexedDocs = docs.map((doc) => ({
    ...doc,
    tfidf: buildDocVector(doc.termCounts, idf),
  }));

  return { docs: indexedDocs, df, idf, avgDocLength, documentCount: n };
}

export function bm25TermScore({ tf, idf, docLength, avgDocLength, k1 = 1.4, b = 0.72 }) {
  if (!tf) return 0;
  const denom = tf + k1 * (1 - b + b * (docLength / avgDocLength));
  return idf * ((tf * (k1 + 1)) / denom);
}

export function searchBM25(index, query, options = {}) {
  const { k = 5, k1 = 1.4, b = 0.72 } = options;
  const queryTerms = tokenize(query);
  const queryCounts = countTerms(queryTerms);

  const results = index.docs.map((doc) => {
    const contributions = [];
    let score = 0;
    for (const [term, qtf] of queryCounts.entries()) {
      const tf = doc.termCounts.get(term) || 0;
      const idf = index.idf.get(term) || 0;
      const termScore = bm25TermScore({
        tf,
        idf,
        docLength: doc.length,
        avgDocLength: index.avgDocLength,
        k1,
        b,
      });
      if (tf || idf) {
        contributions.push({
          term,
          query_tf: qtf,
          doc_tf: tf,
          idf,
          score: termScore,
        });
      }
      score += termScore;
    }
    return { doc, score, contributions: contributions.sort((a, z) => z.score - a.score) };
  });

  return {
    query,
    queryTerms,
    results: results.sort((a, z) => z.score - a.score).slice(0, k),
  };
}

export function searchTfIdfCosine(index, query, options = {}) {
  const { k = 5 } = options;
  const queryCounts = countTerms(tokenize(query));
  const queryVector = buildDocVector(queryCounts, index.idf);
  return index.docs
    .map((doc) => ({ doc, score: cosineFromMaps(queryVector, doc.tfidf) }))
    .sort((a, z) => z.score - a.score)
    .slice(0, k);
}

export function explainTfIdf(index, docId) {
  const doc = index.docs.find((item) => item.id === docId);
  if (!doc) return null;
  const total = doc.length || 1;
  return Array.from(doc.termCounts.entries())
    .map(([term, count]) => ({
      term,
      tf_raw: count,
      tf: count / total,
      df: index.df.get(term) || 0,
      idf: index.idf.get(term) || 0,
      weight: doc.tfidf.get(term) || 0,
    }))
    .sort((a, z) => z.weight - a.weight);
}
