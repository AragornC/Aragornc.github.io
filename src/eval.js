function ids(items) {
  return items.map((item) => (typeof item === "string" ? item : item.id));
}

function relevantSet(relevant) {
  return new Set(ids(relevant));
}

function topK(returned, k) {
  return ids(returned).slice(0, k);
}

export function recallAtK(returned, relevant, k) {
  const rel = relevantSet(relevant);
  if (!rel.size) return 0;
  const hits = topK(returned, k).filter((id) => rel.has(id)).length;
  return hits / rel.size;
}

export function precisionAtK(returned, relevant, k) {
  if (!k) return 0;
  const rel = relevantSet(relevant);
  const hits = topK(returned, k).filter((id) => rel.has(id)).length;
  return hits / k;
}

export function mrr(returned, relevant) {
  const rel = relevantSet(relevant);
  const list = ids(returned);
  for (let i = 0; i < list.length; i += 1) {
    if (rel.has(list[i])) return 1 / (i + 1);
  }
  return 0;
}

export function ndcgAtK(returned, relevant, k) {
  const rel = relevantSet(relevant);
  const list = topK(returned, k);
  let dcg = 0;
  for (let i = 0; i < list.length; i += 1) {
    if (rel.has(list[i])) dcg += 1 / Math.log2(i + 2);
  }

  const idealHits = Math.min(rel.size, k);
  let idcg = 0;
  for (let i = 0; i < idealHits; i += 1) idcg += 1 / Math.log2(i + 2);
  return idcg ? dcg / idcg : 0;
}

export function hitCountAtK(returned, relevant, k) {
  const rel = relevantSet(relevant);
  return topK(returned, k).filter((id) => rel.has(id)).length;
}

export function evaluateMethod(perQueryReturned, queries, k = 5) {
  const perQuery = queries.map((query) => {
    const returned = perQueryReturned[query.id] || [];
    const relevant = query.relevant_ids || [];
    return {
      query,
      returned,
      hitCount: hitCountAtK(returned, relevant, k),
      relevantCount: relevant.length,
      recall: recallAtK(returned, relevant, k),
      precision: precisionAtK(returned, relevant, k),
      mrr: mrr(returned, relevant),
      ndcg: ndcgAtK(returned, relevant, k),
    };
  });

  const avg = (field) => perQuery.reduce((sum, item) => sum + item[field], 0) / Math.max(1, perQuery.length);
  return {
    macroRecall: avg("recall"),
    macroPrecision: avg("precision"),
    mrr: avg("mrr"),
    ndcg: avg("ndcg"),
    perQuery,
  };
}
