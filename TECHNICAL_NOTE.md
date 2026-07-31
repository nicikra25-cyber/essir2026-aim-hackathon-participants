# TECHNICAL NOTE — Bayes Retriever

**Nicole Kraemer | ESSIR 2026 AIMultimediaLab Hackathon**

## 1. System

The application answers questions about one academic PDF through a grounded RAG pipeline. `POST /query` resolves Level-2 follow-ups, retrieves evidence from Qdrant, and sends only grounded context to `google/gemma-4-e4b` through LM Studio. Narrative pages are extracted with PyMuPDF, split into page-bounded three-sentence chunks with one-sentence overlap (362 chunks), embedded with `intfloat/multilingual-e5-large`, and reranked with `cross-encoder/ms-marco-MiniLM-L6-v2`. Level 3 decomposes one question into separate evidence obligations; table tasks use a physically separate pdfplumber/Qdrant index.

```text
POST /query
→ memory-aware rewrite / Level-3 decomposition
→ Qdrant candidate retrieval
→ CrossEncoder + dense + rank-fusion reranking
→ labelled evidence
→ Gemma answer
→ complete supporting quote + PDF page + diagnostics
```

| Stage | Final implementation | Change from scaffold |
|---|---|---|
| Extraction | PyMuPDF for prose; pdfplumber only for reconstructed Level-3 tables | Yes |
| Chunking | 3 sentences, 1-sentence overlap, never across pages | Yes |
| Index | E5 query/passage prefixes; cosine Qdrant; separate narrative/table collections | Extended |
| Retrieval | 100–120 candidates per internal query; CrossEncoder, dense and reciprocal-rank fusion; bibliography/evidence-type controls | Yes |
| Answer/citation | Gemma uses labelled evidence; citation selector returns a complete sentence and stored page | Yes |

PyMuPDF was chosen after comparison with pypdf and pdfplumber: it produced the cleanest narrative reading order and the fewest broken-word indicators. pdfplumber was retained for tables because ordinary prose extraction did not preserve Table 5. Sentence-sized chunks replaced full-page chunks because full pages produced broad, truncated citations. Broad retrieval plus reranking reduced bibliography hits, while sentence-aware citation selection removed artificial ellipses.

## 2. Level 2 — Conversational memory

Level 2 is anchored in both **conversational dependency** and **question type**. The system first decides whether a request is standalone or a follow-up. Earlier user questions recover an omitted subject, while the original question type is preserved: **Why** searches for a causal mechanism; **How** searches for a response and its means.

```text
load history by conversation_id
→ detect vague follow-up
→ preserve Why/How intent
→ rewrite to standalone retrieval query
→ retrieve and rerank
→ answer using original question + resolved interpretation + evidence + history
→ append turn to memory
```

**Required real example**

| Item | System behaviour |
|---|---|
| Earlier q4 | “What problems can intense competitive forces create for an organisation?” |
| q5 as asked | “Why does that happen?” |
| Standalone query | “Why can intense competitive forces create problems for an organisation?” |

The raw word “that” has no searchable subject. The rewrite recovers the topic from q4, keeps the causal intent, and allows retrieval to rank the page-2 value-division mechanism first. q6 similarly resolves “And how is it solved?” into a management/mitigation query and prioritises response-plus-means evidence. History is stored in a process-local dictionary keyed by `conversation_id`; rewriting uses up to four messages (two previous turns). The original question, rewrite, history count and rewrite method are logged. The limitation is that memory is lost on restart.

| Question | Before rewriting | Final system |
|---|---|---|
| q4 | Passed; already standalone | Passed; no regression |
| q5 | Failed: “that” unresolved; irrelevant pages | Passed: page-2 cause ranked first |
| q6 | Failed: “it” unresolved; bibliography retrieved | Passed with minor wording omission; page-6 management-accounting mechanism ranked first |

History alone was insufficient because vague follow-ups still produced weak embeddings. Separating conversational interpretation from retrieval fixed q5 and q6 while keeping the original wording natural in the final answer.

## 3. Level 3 — Whole-document reasoning

Level 3 begins by identifying the required reasoning structure: **cross-section combination, comparison, or synthesis**. This controls decomposition; each resulting task is then assigned an evidence type such as narrative, table, result, comparison item, or synthesis component.

```text
identify reasoning type
→ create EvidenceTask objects
→ retrieve each task independently
→ use narrative or table index
→ keep labelled EvidenceGroups
→ deduplicate without losing coverage
→ synthesise one answer
→ return one evidence-type-specific quote per group
```

This is controlled multi-hop retrieval, not a larger `top_k`. It is not an open-ended agent and does not use a graph: deterministic routing was easier to audit and measure.

- **q7 — combination:** three tasks (`literature_review`, `table_5`, `results_interpretation`) retrieved pages 6, 17 and 18. The answer correctly reported β = 0.154, p = 0.028 and partial support for H3.
- **q8 — comparison:** separate labelled tasks kept the overall sample, low-cost/high-force group and product-differentiation/high-force group distinct. The correct outcomes, coefficients and p-values remained attached to their groups.
- **q9 — synthesis:** the main conclusion was retrieved correctly, but the recommendation component accepted a page-1 empirical finding as advice and missed the condition-specific directives on page 22. Result: partial pass.

## 4. Measurement

Nine fixed questions were evaluated against manually prepared expected answers, designated quotations and PDF pages.

| Q | L | Measured result | Pages | Latency | Verdict |
|---|---:|---|---|---:|---|
| q1 | 1 | 5/5 forces; exact evidence | 2 | 20.5 s | Pass |
| q2 | 1 | 2/2 outcomes; direct evidence | 2, 20 | 21.0 s | Pass |
| q3 | 1 | 505; exact evidence | 9 | 32.2 s | Pass |
| q4 | 2 | 3/3 benchmark effects | 5 | 28.5 s | Pass; minor extra |
| q5 | 2 | Reference resolved; cause ranked first | 2 | 34.2 s | Pass |
| q6 | 2 | Response + information/decision roles | 6 | 41.2 s | Pass; minor omission |
| q7 | 3 | 3/3 evidence groups; β, p, H3 correct | 6, 17, 18 | 66.0 s | Pass; noisy table quote |
| q8 | 3 | 3/3 groups and statistics correct | 18, 20 | 46.0 s | Pass |
| q9 | 3 | Conclusion correct; recommendations incomplete | 22; wrong p.1 source | 46.1 s | Partial |

| Level | Full passes | Mean latency |
|---|---:|---:|
| Level 1 | 3/3 | 24.5 s |
| Level 2 | 3/3* | 34.6 s |
| Level 3 | 2/3 | 52.7 s |
| **Overall** | **8/9 + 1 partial** | **37.3 s/query** |

\*q6 omitted the explicit statement that the problem is not completely eliminated.

**Ablations.** Sentence-aware citation selection changed Level-1 exact supporting quotes from **0/3 to 3/3**. Before Level-2 rewriting, only q4 passed (**1/3**); the final history-based, question-type-aware stack made q4–q6 answerable (**3/3**). Level-3 decomposition produced full evidence coverage for q7 and q8 (**2/3** Level-3 questions).

All models and Qdrant ran locally, so there was no external per-query API charge. Diagnostics returned `tokens = null`, so token consumption cannot be reported honestly. Total measured latency was 335.7 seconds; Level 3 was slowest because it may run several searches, reranking passes, table retrieval and synthesis.

## 5. What broke

The first q7 run produced the right statistic in the answer but the wrong supporting evidence. A page-3 methodology paragraph was selected as “Table 5” evidence, while a page-18 contemporary-practices/H2 sentence contradicted the requested traditional-practices/H3 result. Comparing the answer with its returned sources showed that the failure occurred before generation: ordinary extraction had flattened or fragmented the table, so the correct row was not searchable.

I kept PyMuPDF for prose and added a separate pdfplumber table path. The extractor tests 90°, 270° and 180° rotations and normal/reversed reading directions, extracts words with coordinates, reconstructs rows and cells, merges wrapped labels, assigns stable row/column/cell identifiers, and stores coefficients with same-row p-values. Semantic column labels such as “Overall sample” are added before indexing in a separate Qdrant collection. Table candidates must now contain the requested interaction, outcome and statistical pattern; methodology passages without numbers are demoted.

After this change, q7 retrieved the correct page-17 row and reported β = 0.154, p = 0.028 and H3 correctly. The remaining negative result is evidence cleanliness: the returned table quote still includes neighbouring values.

## 6. Limitations and next steps

The clearest limitation is q9. The system handles clearly named evidence groups well, but recommendation synthesis requires exhaustive condition coverage. It retrieved the main conclusion yet treated an empirical finding as managerial advice and missed four condition-specific recommendations.

With another day I would:

1. require directive language such as “it is recommended”, “managers should”, “should employ”, “should not”, or “it is inappropriate”;
2. split recommendations into condition–action tasks and block synthesis until every required condition has evidence;
3. store structured recommendation units: condition, action, practice, outcome, exact quote and page;
4. add token counting, persistent memory, and evaluation on unseen questions from multiple PDFs.

The target is not a longer answer, but complete evidence coverage: every recommendation must be attached to its own condition, exact quotation and page before synthesis.
