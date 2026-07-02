// 生成模块:本地跑 Qwen2.5-0.5B-Instruct(真 LLM,小号,中文还行)
// Qwen2.5 较新,用 transformers.js v3(和 dense/rerank 的 v2 各自独立)
import { env, pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.3";

env.allowLocalModels = true;
env.localModelPath = "/assets/models/";
env.allowRemoteModels = true;
env.useBrowserCache = true;

const MODEL_ID = "onnx-community/Qwen2.5-0.5B-Instruct";

let genPromise = null;

export async function loadLLM(onProgress = () => {}) {
  if (!genPromise) {
    genPromise = pipeline("text-generation", MODEL_ID, {
      dtype: "q8",            // 对应 model_quantized.onnx
      progress_callback: onProgress,
    });
  }
  return genPromise;
}

// 给定 system + user 两段提示词,返回 LLM 生成的回答文本
export async function generate(systemPrompt, userPrompt, opts = {}) {
  const gen = await genPromise;
  const messages = [
    { role: "system", content: systemPrompt },
    { role: "user", content: userPrompt },
  ];
  const out = await gen(messages, {
    max_new_tokens: opts.max_new_tokens || 300,
    do_sample: false,           // 贪心解码,结果稳定可复现
    repetition_penalty: 1.1,
    ...opts,
  });
  // v3:out[0].generated_text 是完整对话数组,取最后一条 assistant
  const g = out?.[0]?.generated_text;
  if (Array.isArray(g)) {
    const last = g[g.length - 1];
    return (last && last.content ? last.content : String(last)).trim();
  }
  return String(g || "").trim();
}
