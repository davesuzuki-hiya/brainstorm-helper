const { useState, useMemo, useRef } = React;

// ---- Text utilities ----

const STOP_WORDS = new Set([
  "the","a","an","and","or","of","to","in","on","for","with",
  "is","are","this","that","it","be","by","from","as","at",
  "we","our","your","you","i","me","my","their","they","them",
  "was","were","have","has","had","do","did","does"
]);

function tokenize(text) {
  if (!text) return [];
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((word) => word && !STOP_WORDS.has(word));
}

function buildVocabulary(texts) {
  const vocab = new Map();
  texts.forEach((t) => {
    tokenize(t).forEach((word) => {
      if (!vocab.has(word)) {
        vocab.set(word, vocab.size);
      }
    });
  });
  return vocab;
}

function vectorize(text, vocab) {
  const vec = new Array(vocab.size).fill(0);
  const tokens = tokenize(text);
  tokens.forEach((word) => {
    const idx = vocab.get(word);
    if (idx !== undefined) {
      vec[idx] += 1;
    }
  });
  return vec;
}

function cosineSimilarity(a, b) {
  if (!a || !b || a.length !== b.length) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    const va = a[i];
    const vb = b[i];
    dot += va * vb;
    normA += va * va;
    normB += vb * vb;
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function capitalize(word) {
  if (!word) return "";
  return word.charAt(0).toUpperCase() + word.slice(1);
}

function extractTopKeywords(texts, maxWords) {
  const freq = new Map();
  texts.forEach((t) => {
    tokenize(t).forEach((word) => {
      freq.set(word, (freq.get(word) || 0) + 1);
    });
  });
  const sorted = Array.from(freq.entries()).sort((a, b) => b[1] - a[1]);
  return sorted.slice(0, maxWords).map(([w]) => capitalize(w));
}

// ---- Clustering & novelty ----

function clusterIdeas(ideas, ideaVectors) {
  const clusters = [];
  const centroids = [];
  const threshold = 0.35; // similarity threshold to join an existing cluster

  ideas.forEach((idea) => {
    const vec = ideaVectors[idea.id];
    if (!vec) return;

    if (clusters.length === 0) {
      clusters.push({
        id: "c1",
        ideaIds: [idea.id],
        primaryName: "",
        keywords: []
      });
      centroids.push(vec.slice());
      return;
    }

    let bestIdx = -1;
    let bestSim = -1;
    for (let i = 0; i < centroids.length; i++) {
      const sim = cosineSimilarity(vec, centroids[i]);
      if (sim > bestSim) {
        bestSim = sim;
        bestIdx = i;
      }
    }

    if (bestSim >= threshold) {
      const cluster = clusters[bestIdx];
      cluster.ideaIds.push(idea.id);

      // recompute centroid
      const newCentroid = new Array(vec.length).fill(0);
      cluster.ideaIds.forEach((id) => {
        const v = ideaVectors[id];
        if (!v) return;
        for (let d = 0; d < newCentroid.length; d++) {
          newCentroid[d] += v[d];
        }
      });
      for (let d = 0; d < newCentroid.length; d++) {
        newCentroid[d] /= cluster.ideaIds.length;
      }
      centroids[bestIdx] = newCentroid;
    } else {
      const clusterId = "c" + (clusters.length + 1);
      clusters.push({
        id: clusterId,
        ideaIds: [idea.id],
        primaryName: "",
        keywords: []
      });
      centroids.push(vec.slice());
    }
  });

  // assign names & keywords
  clusters.forEach((cluster, idx) => {
    const texts = cluster.ideaIds.map((id) => {
      const idea = ideas.find((i) => i.id === id);
      return idea ? idea.text : "";
    });
    const keywords = extractTopKeywords(texts, 3);
    cluster.keywords = keywords;
    cluster.primaryName = keywords.length
      ? keywords.join(" · ")
      : `Theme ${idx + 1}`;
  });

  return clusters;
}

function computeNoveltyScores(ideas, ideaVectors, problemVector, hasProblem) {
  const scores = {};
  const details = {};
  const combinedMap = {};

  if (ideas.length === 0) {
    return {
      scores,
      details,
      stats: { avgNovelty: null, maxNovelty: null }
    };
  }

  const idList = ideas.map((i) => i.id);
  let minCombined = null;
  let maxCombined = null;

  idList.forEach((id) => {
    const vec = ideaVectors[id];
    if (!vec) return;

    const relevance = hasProblem
      ? Math.max(0, Math.min(1, cosineSimilarity(vec, problemVector)))
      : 1;

    const sims = [];
    idList.forEach((otherId) => {
      if (otherId === id) return;
      const otherVec = ideaVectors[otherId];
      if (!otherVec) return;
      const s = cosineSimilarity(vec, otherVec);
      sims.push(s);
    });

    let meanNeighborSim;
    if (sims.length === 0) {
      meanNeighborSim = 0.5;
    } else {
      sims.sort((a, b) => b - a);
      const k = Math.min(3, sims.length);
      const top = sims.slice(0, k);
      const sum = top.reduce((acc, x) => acc + x, 0);
      meanNeighborSim = sum / k;
    }

    const rawNovelty = 1 - meanNeighborSim;
    const combined = rawNovelty * relevance;

    combinedMap[id] = combined;
    details[id] = {
      relevance,
      meanNeighborSim,
      rawNovelty,
      combined
    };

    if (minCombined === null || combined < minCombined) {
      minCombined = combined;
    }
    if (maxCombined === null || combined > maxCombined) {
      maxCombined = combined;
    }
  });

  let sumScores = 0;
  let count = 0;
  Object.keys(combinedMap).forEach((id) => {
    const combined = combinedMap[id];
    let score;
    if (maxCombined === minCombined) {
      score = 50;
    } else {
      score = 100 * (combined - minCombined) / (maxCombined - minCombined);
    }
    const rounded = Math.round(score);
    scores[id] = rounded;
    sumScores += score;
    count += 1;
  });

  const avgNovelty = count ? sumScores / count : null;
  const maxNovelty = count
    ? Math.max(...Object.values(scores))
    : null;

  return {
    scores,
    details,
    stats: { avgNovelty, maxNovelty }
  };
}

function computeAnalysis(problemStatement, ideas) {
  if (!ideas || ideas.length === 0) {
    return {
      vocab: null,
      problemVector: null,
      ideaVectors: {},
      clusters: [],
      noveltyById: {},
      stats: { avgNovelty: null, maxNovelty: null },
      topIdeas: []
    };
  }

  const hasProblemTokens = tokenize(problemStatement).length > 0;
  const texts = [];

  if (hasProblemTokens) {
    texts.push(problemStatement);
  }
  ideas.forEach((idea) => texts.push(idea.text));

  const vocab = buildVocabulary(texts);
  const problemVector = hasProblemTokens
    ? vectorize(problemStatement, vocab)
    : null;

  const ideaVectors = {};
  ideas.forEach((idea) => {
    ideaVectors[idea.id] = vectorize(idea.text, vocab);
  });

  const clusters = clusterIdeas(ideas, ideaVectors);
  const noveltyResult = computeNoveltyScores(
    ideas,
    ideaVectors,
    problemVector,
    hasProblemTokens
  );

  const noveltyById = noveltyResult.scores;
  const stats = noveltyResult.stats;

  const topIdeas = ideas
    .slice()
    .sort((a, b) => (noveltyById[b.id] || 0) - (noveltyById[a.id] || 0))
    .slice(0, 5);

  return {
    vocab,
    problemVector,
    ideaVectors,
    clusters,
    noveltyById,
    stats,
    topIdeas
  };
}

// ---- UI components ----

function NoveltyBadge({ score }) {
  let level = "low";
  let label = "Safe";

  if (score >= 80) {
    level = "high";
    label = "Bold";
  } else if (score >= 60) {
    level = "medium";
    label = "Fresh";
  }

  return (
    <span
      className={`novelty-badge novelty-${level}`}
      title={`Novelty score: ${score}`}
    >
      <span>{label}</span>
      <span>· {score}</span>
    </span>
  );
}

function App() {
  const [problemStatement, setProblemStatement] = useState(
    "How might we reduce churn and make our app more habit-forming?"
  );
  const [displayName, setDisplayName] = useState("");
  const [ideaText, setIdeaText] = useState("");
  const [ideas, setIdeas] = useState([]);
  const [viewMode, setViewMode] = useState("affinity");
  const [toast, setToast] = useState(null);

  const ideaIdRef = useRef(1);
  const toastTimeoutRef = useRef(null);

  const analysis = useMemo(
    () => computeAnalysis(problemStatement, ideas),
    [problemStatement, ideas]
  );

  function handleAddIdea() {
    const trimmed = ideaText.trim();
    if (!trimmed) return;

    const newIdea = {
      id: "i" + ideaIdRef.current++,
      text: trimmed,
      author: displayName.trim() || "Anonymous",
      createdAt: new Date().toISOString()
    };

    const nextIdeas = [...ideas, newIdea];

    // Compute novelty in advance for celebration
    const previewAnalysis = computeAnalysis(problemStatement, nextIdeas);
    const score = previewAnalysis.noveltyById[newIdea.id];

    if (typeof score === "number" && score >= 80) {
      if (toastTimeoutRef.current) {
        clearTimeout(toastTimeoutRef.current);
      }
      setToast("Standout idea! High novelty detected 🔥");
      toastTimeoutRef.current = setTimeout(() => {
        setToast(null);
      }, 3200);
    }

    setIdeas(nextIdeas);
    setIdeaText("");
  }

  function handleIdeaKeyDown(e) {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      handleAddIdea();
    }
  }

  const ideaCount = ideas.length;
  const clusterCount = analysis.clusters.length;
  const avgNovelty = analysis.stats.avgNovelty;
  const maxNovelty = analysis.stats.maxNovelty;

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <div className="app-title">
            AffinityBoard
            <span className="app-title-badge">Local prototype</span>
          </div>
          <div className="app-subtitle">
            Add ideas as digital stickies. The board groups them into themes and
            highlights novel directions automatically.
          </div>
        </div>
        <div className="app-header-right">
          <span>Tip: Press ⌘+Enter / Ctrl+Enter to add an idea quickly.</span>
        </div>
      </header>

      <main className="app-main">
        {/* Left column: configuration & input */}
        <section className="column column-left">
          <div className="panel-header">
            <div className="panel-title">Problem</div>
          </div>

          <div className="input-row">
            <div className="label">Problem statement</div>
            <textarea
              value={problemStatement}
              onChange={(e) => setProblemStatement(e.target.value)}
              placeholder="What are you brainstorming on this board?"
            />
          </div>

          <div className="panel-header" style={{ marginTop: "4px" }}>
            <div className="panel-title">New idea</div>
          </div>

          <div className="input-row">
            <div className="label">Your name (optional)</div>
            <input
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              placeholder="e.g. Alex"
            />
          </div>

          <div className="input-row">
            <div className="label">
              Idea text{" "}
              <span style={{ color: "#6b7280", fontSize: "0.75rem" }}>
                (one idea per sticky)
              </span>
            </div>
            <textarea
              value={ideaText}
              onChange={(e) => setIdeaText(e.target.value)}
              onKeyDown={handleIdeaKeyDown}
              placeholder="e.g. Daily streak rewards for completing a key action"
            />
          </div>

          <div className="button-row">
            <button
              className="button-primary"
              onClick={handleAddIdea}
              disabled={!ideaText.trim()}
            >
              <span>Drop sticky</span>
              <span>➜</span>
            </button>
            <button
              className="button-secondary"
              type="button"
              onClick={() => setIdeaText("")}
            >
              Clear
            </button>
          </div>

          <div
            style={{
              marginTop: "8px",
              fontSize: "0.75rem",
              color: "#9ca3af"
            }}
          >
            This prototype runs completely in your browser and uses a simple
            text-based vector model for clustering and novelty scoring. You can
            later swap in real embeddings/LLMs on a backend.
          </div>
        </section>

        {/* Center column: board */}
        <section className="column column-center">
          <div className="panel-header">
            <div className="panel-title">Board</div>
            <div className="chip-group">
              <button
                className={
                  "chip " + (viewMode === "affinity" ? "chip-active" : "")
                }
                type="button"
                onClick={() => setViewMode("affinity")}
              >
                Affinity view
              </button>
              <button
                className={
                  "chip " + (viewMode === "list" ? "chip-active" : "")
                }
                type="button"
                onClick={() => setViewMode("list")}
              >
                Flat list
              </button>
            </div>
          </div>

          {ideaCount === 0 ? (
            <div className="empty-state">
              Start by adding a few ideas. Once you have multiple stickies, the
              board will automatically group them into themes and score how
              novel each one is.
            </div>
          ) : viewMode === "affinity" ? (
            analysis.clusters.length === 0 ? (
              <div className="empty-state">
                We need a couple more ideas to form meaningful clusters.
              </div>
            ) : (
              <div className="clusters-grid">
                {analysis.clusters.map((cluster) => (
                  <div key={cluster.id} className="cluster-card">
                    <div className="cluster-header">
                      <div className="cluster-title">
                        {cluster.primaryName}
                      </div>
                      <div className="cluster-meta">
                        {cluster.ideaIds.length} idea
                        {cluster.ideaIds.length !== 1 ? "s" : ""}
                      </div>
                    </div>
                    {cluster.keywords && cluster.keywords.length > 0 && (
                      <div className="cluster-keywords">
                        {cluster.keywords.join(" · ")}
                      </div>
                    )}
                    <div className="cluster-ideas">
                      {cluster.ideaIds.map((id) => {
                        const idea = ideas.find((i) => i.id === id);
                        if (!idea) return null;
                        const score = analysis.noveltyById[idea.id] ?? null;
                        return (
                          <div key={idea.id} className="idea-card">
                            <div className="idea-text">{idea.text}</div>
                            <div className="idea-footer">
                              <div className="idea-author">
                                {idea.author || "Anonymous"}
                              </div>
                              <div className="idea-novelty">
                                {score !== null && score !== undefined && (
                                  <>
                                    <span className="idea-novelty-label">
                                      Novelty
                                    </span>
                                    <NoveltyBadge score={score} />
                                  </>
                                )}
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ))}
              </div>
            )
          ) : (
            // Flat list view
            <div className="clusters-grid">
              <div className="cluster-card" style={{ gridColumn: "1 / -1" }}>
                <div className="cluster-header">
                  <div className="cluster-title">All ideas</div>
                  <div className="cluster-meta">
                    Sorted by novelty score (highest first)
                  </div>
                </div>
                <div className="cluster-ideas">
                  {ideas
                    .slice()
                    .sort(
                      (a, b) =>
                        (analysis.noveltyById[b.id] || 0) -
                        (analysis.noveltyById[a.id] || 0)
                    )
                    .map((idea) => {
                      const score = analysis.noveltyById[idea.id] ?? null;
                      return (
                        <div key={idea.id} className="idea-card">
                          <div className="idea-text">{idea.text}</div>
                          <div className="idea-footer">
                            <div className="idea-author">
                              {idea.author || "Anonymous"}
                            </div>
                            <div className="idea-novelty">
                              {score !== null && score !== undefined && (
                                <>
                                  <span className="idea-novelty-label">
                                    Novelty
                                  </span>
                                  <NoveltyBadge score={score} />
                                </>
                              )}
                            </div>
                          </div>
                        </div>
                      );
                    })}
                </div>
              </div>
            </div>
          )}
        </section>

        {/* Right column: metrics & highlights */}
        <section className="column column-right">
          <div className="panel-header">
            <div className="panel-title">Insights</div>
          </div>

          <div className="stat-row">
            <div className="stat-label">Ideas</div>
            <div className="stat-value">{ideaCount}</div>
          </div>
          <div className="stat-row">
            <div className="stat-label">Themes</div>
            <div className="stat-value">
              {clusterCount}
              {clusterCount === 0 && ideaCount > 0 && (
                <span style={{ marginLeft: 4, color: "#9ca3af" }}>
                  (forming…)
                </span>
              )}
            </div>
          </div>
          <div className="stat-row">
            <div className="stat-label">Average novelty</div>
            <div className="stat-value">
              {avgNovelty != null ? `${avgNovelty.toFixed(1)}` : "–"}
            </div>
          </div>
          <div className="stat-row" style={{ marginBottom: 10 }}>
            <div className="stat-label">Highest novelty</div>
            <div className="stat-value">
              {maxNovelty != null ? `${maxNovelty}` : "–"}
            </div>
          </div>

          <div className="stat-row" style={{ marginBottom: 8 }}>
            <div className="stat-label">Top ideas</div>
            <div className="stat-chip">
              Highlighting the boldest directions
            </div>
          </div>

          {analysis.topIdeas.length === 0 ? (
            <div
              style={{
                fontSize: "0.78rem",
                color: "#9ca3af",
                marginTop: "4px"
              }}
            >
              Once you have a few ideas, the most novel ones will show up here.
            </div>
          ) : (
            <div className="top-ideas-list">
              {analysis.topIdeas.map((idea, idx) => {
                const score = analysis.noveltyById[idea.id] ?? null;
                return (
                  <div key={idea.id} className="top-idea">
                    <div className="top-idea-title">
                      {idx + 1}. {idea.text}
                    </div>
                    <div className="top-idea-meta">
                      <span>{idea.author || "Anonymous"}</span>
                      <span>
                        {score != null ? `Novelty ${score}` : "Novelty –"}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </section>
      </main>

      <footer className="app-footer">
        <span>
          Local-only prototype · No data leaves the browser · Replace scoring
          logic with real embeddings when you add a backend.
        </span>
        <span>Built for quick GitHub Pages hosting.</span>
      </footer>

      {toast && (
        <div className="toast">
          <span>{toast}</span>
        </div>
      )}
    </div>
  );
}

const rootElement = document.getElementById("root");
const root = ReactDOM.createRoot(rootElement);
root.render(<App />);

