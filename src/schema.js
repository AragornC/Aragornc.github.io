const FIELD_WEIGHTS = {
  risk_type: 4,
  channel: 2,
  entities: 1,
};

function normalize(text) {
  return String(text || "").toLowerCase().trim();
}

function addToMapSet(map, key, value) {
  if (!map.has(key)) map.set(key, new Set());
  map.get(key).add(value);
}

function uniqueValues(doc) {
  const values = [];
  if (doc.schema?.risk_type) values.push({ field: "risk_type", value: doc.schema.risk_type });
  if (doc.schema?.channel) values.push({ field: "channel", value: doc.schema.channel });
  for (const entity of doc.schema?.entities || []) values.push({ field: "entities", value: entity });
  return values;
}

export function buildSchemaIndex(corpus) {
  const docsById = new Map(corpus.map((doc) => [doc.id, doc]));
  const schemaInverted = new Map();
  const adjacency = new Map();

  for (const doc of corpus) {
    if (!adjacency.has(doc.id)) adjacency.set(doc.id, []);
    for (const item of uniqueValues(doc)) {
      addToMapSet(schemaInverted, normalize(item.value), { field: item.field, docId: doc.id, value: item.value });
    }
    for (const edge of doc.relations || []) {
      if (!docsById.has(edge.target)) continue;
      adjacency.get(doc.id).push({ target: edge.target, rel: edge.rel, direction: "out" });
      if (!adjacency.has(edge.target)) adjacency.set(edge.target, []);
      adjacency.get(edge.target).push({ target: doc.id, rel: edge.rel, direction: "in" });
    }
  }

  const schemaTerms = Array.from(schemaInverted.keys()).sort((a, z) => z.length - a.length);
  return { docs: corpus, docsById, schemaInverted, schemaTerms, adjacency };
}

function alignedFieldsForDoc(doc, matchedValues) {
  const aligned = [];
  const schema = doc.schema || {};
  const entities = new Set(schema.entities || []);
  for (const match of matchedValues) {
    if (match.field === "risk_type" && schema.risk_type === match.value) aligned.push(match);
    if (match.field === "channel" && schema.channel === match.value) aligned.push(match);
    if (match.field === "entities" && entities.has(match.value)) aligned.push(match);
  }
  const deduped = new Map();
  for (const item of aligned) deduped.set(`${item.field}:${item.value}`, item);
  return Array.from(deduped.values());
}

function alignmentScore(alignedFields) {
  return alignedFields.reduce((sum, item) => sum + (FIELD_WEIGHTS[item.field] || 0), 0);
}

function parseQuerySchema(index, query) {
  const q = normalize(query);
  const matched = [];
  for (const term of index.schemaTerms) {
    if (!term || !q.includes(term)) continue;
    for (const entry of index.schemaInverted.get(term) || []) {
      matched.push({ field: entry.field, value: entry.value, term });
    }
  }
  const deduped = new Map();
  for (const item of matched) deduped.set(`${item.field}:${item.value}`, item);
  return Array.from(deduped.values());
}

function buildSeeds(index, matchedValues, limit) {
  return index.docs
    .map((doc) => {
      const alignedFields = alignedFieldsForDoc(doc, matchedValues);
      return {
        doc,
        alignedFields,
        alignScore: alignmentScore(alignedFields),
      };
    })
    .filter((item) => item.alignScore > 0)
    .sort((a, z) => z.alignScore - a.alignScore || a.doc.id.localeCompare(z.doc.id))
    .slice(0, limit);
}

function structuralScore(seedScore, dist, decay) {
  return seedScore * Math.pow(decay, dist) + (dist === 0 ? 0 : 0.35 / dist);
}

function expandGraph(index, seeds, maxHops, decay) {
  const reached = new Map();
  const queue = [];

  for (const seed of seeds) {
    reached.set(seed.doc.id, {
      dist: 0,
      via: "seed",
      seedId: seed.doc.id,
      seedScore: seed.alignScore,
      structuralScore: seed.alignScore,
      path: [seed.doc.id],
      relPath: [],
    });
    queue.push(seed.doc.id);
  }

  for (let cursor = 0; cursor < queue.length; cursor += 1) {
    const currentId = queue[cursor];
    const current = reached.get(currentId);
    if (!current || current.dist >= maxHops) continue;

    for (const edge of index.adjacency.get(currentId) || []) {
      const nextDist = current.dist + 1;
      const nextScore = structuralScore(current.seedScore, nextDist, decay);
      const existing = reached.get(edge.target);
      if (existing && existing.structuralScore >= nextScore) continue;
      reached.set(edge.target, {
        dist: nextDist,
        via: "hop",
        seedId: current.seedId,
        seedScore: current.seedScore,
        structuralScore: nextScore,
        path: [...current.path, edge.target],
        relPath: [...current.relPath, edge.rel],
      });
      queue.push(edge.target);
    }
  }

  return reached;
}

export function searchSchema(index, query, options = {}) {
  const { k = 5, hops = 2, seedLimit = 8, decay = 0.6 } = options;
  const matchedValues = parseQuerySchema(index, query);
  const seeds = buildSeeds(index, matchedValues, seedLimit);
  const reached = expandGraph(index, seeds, hops, decay);

  const results = Array.from(reached.entries()).map(([docId, reach]) => {
    const doc = index.docsById.get(docId);
    const alignedFields = alignedFieldsForDoc(doc, matchedValues);
    const ownAlign = alignmentScore(alignedFields);
    return {
      doc,
      score: Math.max(ownAlign, reach.structuralScore),
      via: reach.via,
      dist: reach.dist,
      alignedFields,
      seedId: reach.seedId,
      path: reach.path,
      relPath: reach.relPath,
      matchedValues,
    };
  });

  return results
    .sort((a, z) => z.score - a.score || a.dist - z.dist || a.doc.id.localeCompare(z.doc.id))
    .slice(0, k);
}

export function explainSchemaPath(index, docId, query, options = {}) {
  const results = searchSchema(index, query, { ...options, k: index.docs.length });
  return results.find((item) => item.doc.id === docId) || null;
}
