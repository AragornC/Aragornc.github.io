// Cross-encoder 重排模块:用 bge-reranker-base(真·交叉编码器)
// 和 dense.js 的 bge(bi-encoder)是两种不同的模型:
//   bi-encoder  : query 和 doc 分别各编码成一个向量,再算 cosine —— 快,但两者从不"见面"
//   cross-encoder: query 和 doc 拼在一起喂进同一个模型,直接吐一个相关性分 —— 慢,但更准
import { env, AutoTokenizer, AutoModelForSequenceClassification }
  from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";

env.allowLocalModels = true;
env.localModelPath = "/assets/models/";
env.allowRemoteModels = true;
env.useBrowserCache = true;

const RERANKER_ID = "Xenova/bge-reranker-base";

let tokPromise = null;
let modelPromise = null;

export async function loadReranker(onProgress = () => {}) {
  if (!tokPromise) {
    tokPromise = AutoTokenizer.from_pretrained(RERANKER_ID, { progress_callback: onProgress });
  }
  if (!modelPromise) {
    modelPromise = AutoModelForSequenceClassification.from_pretrained(RERANKER_ID, {
      quantized: true,
      progress_callback: onProgress,
    });
  }
  await Promise.all([tokPromise, modelPromise]);
}

// 对一个 (query, doc) 对,返回 cross-encoder 的原始相关性 logit(越大越相关)
export async function rerankScore(query, doc) {
  const tokenizer = await tokPromise;
  const model = await modelPromise;
  const inputs = await tokenizer(query, { text_pair: doc, padding: true, truncation: true });
  const output = await model(inputs);
  return output.logits.data[0];
}

// 把 logit 压成 0~1 的"相关概率"(方便人看)
export function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

// 返回 (query, doc) 拼接后,真模型看到的 token 序列(含特殊标记)
// XLM-R 的拼法: <s> query </s></s> doc </s>
export async function tokenizePair(query, doc) {
  const tokenizer = await tokPromise;
  const enc = await tokenizer(query, { text_pair: doc });
  const ids = Array.from(enc.input_ids.data).map(Number);
  const SPECIAL = new Set([0, 1, 2, 3]); // <s> <pad> </s> <unk>
  const tokens = ids.map((id) => {
    let t = tokenizer.decode([id], { skip_special_tokens: false });
    return { id, text: t, special: SPECIAL.has(id) };
  });
  return tokens;
}

// 单段文本的 token(给 bi-encoder 那侧"分开看"用)
export async function tokenizeOne(text) {
  const tokenizer = await tokPromise;
  const enc = await tokenizer(text);
  const ids = Array.from(enc.input_ids.data).map(Number);
  const SPECIAL = new Set([0, 1, 2, 3]);
  return ids.map((id) => ({ id, text: tokenizer.decode([id], { skip_special_tokens: false }), special: SPECIAL.has(id) }));
}

// 对一批候选逐个打分,返回 [{...cand, logit, prob}],按 logit 降序
export async function rerank(query, candidates, getText = (c) => c.text, onStep = () => {}) {
  const scored = [];
  for (let i = 0; i < candidates.length; i += 1) {
    const c = candidates[i];
    const logit = await rerankScore(query, getText(c));
    scored.push({ ...c, logit, prob: sigmoid(logit) });
    onStep(i + 1, candidates.length);
  }
  return scored.sort((a, b) => b.logit - a.logit);
}
