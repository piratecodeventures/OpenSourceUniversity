# RAG Advanced Implementation Guide: Deep Technical Reference

> Comprehensive advanced implementations, mathematical proofs, performance benchmarks, and production-ready code patterns for RAG systems.

---

## Table of Contents

1. [Mathematical Foundations](#mathematical-foundations)
2. [Advanced Retrieval Algorithms](#advanced-retrieval-algorithms)
3. [Production Code Patterns](#production-code-patterns)
4. [Performance Optimization](#performance-optimization)
5. [Enterprise Patterns](#enterprise-patterns)
6. [Troubleshooting Guide](#troubleshooting-guide)

---

## Mathematical Foundations

### Vector Space Mathematics

**Cosine Similarity** (Standard):
$$\text{cos\_sim}(a, b) = \frac{a \cdot b}{||a|| \cdot ||b||} = \frac{\sum_{i=1}^{n} a_i b_i}{\sqrt{\sum_{i=1}^{n} a_i^2} \cdot \sqrt{\sum_{i=1}^{n} b_i^2}}$$

Range: [-1, 1] where 1 = identical, -1 = opposite, 0 = orthogonal

**Euclidean Distance**:
$$d(a, b) = \sqrt{\sum_{i=1}^{n} (a_i - b_i)^2}$$

Normalized embeddings: $d(a,b) = \sqrt{2 - 2 \cdot \cos\_sim(a,b)}$

**For normalized vectors: Cosine similarity = Dot product**

### Information Theoretic Metrics

**Mutual Information Between Query and Document**:
$$I(Q; D) = \sum_{q,d} P(q,d) \log \frac{P(q,d)}{P(q)P(d)}$$

Higher MI = better retrieval signal

**Jensen-Shannon Divergence** (Symmetric KL divergence):
$$JS(P||Q) = \frac{1}{2}KL(P||M) + \frac{1}{2}KL(Q||M)$$ where $M = \frac{P+Q}{2}$

Better than KL divergence for comparing distributions

### TF-IDF Mathematical Foundation

**Term Frequency (TF)**:
$$\text{TF}(t, d) = \frac{\text{count}(t, d)}{|d|}$$

or log-normalized: $1 + \log(\text{count}(t, d))$

**Inverse Document Frequency (IDF)**:
$$\text{IDF}(t) = \log\left(\frac{N}{n_t}\right)$$

where N = total documents, $n_t$ = documents containing term t

**Gives rare terms higher weight**

---

## Advanced Retrieval Algorithms

### Multi-Stage Retrieval Pipeline

```python
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum

class RetrievalStage(Enum):
    CANDIDATE_GENERATION = 1
    FIRST_STAGE_RANKING = 2
    SECOND_STAGE_RANKING = 3
    FINAL_SELECTION = 4

@dataclass
class RetrievalResult:
    document_id: str
    content: str
    score: float
    stage_passed: RetrievalStage
    retrieval_method: str  # "lexical", "dense", "semantic"

class MultiStageRetriever:
    """
    Production-grade multi-stage retrieval:
    1. Candidate generation: Fast, broad search
    2. First-stage ranking: Quick filtering
    3. Second-stage ranking: Slow, accurate
    4. Final selection: User requirements
    """
    
    def __init__(self, lexical_index, dense_index, cross_encoder_model):
        self.lexical = lexical_index
        self.dense = dense_index
        self.cross_encoder = cross_encoder_model
    
    def retrieve(self, query: str, final_top_k: int = 10) -> List[RetrievalResult]:
        """Execute full retrieval pipeline"""
        
        # STAGE 1: Candidate Generation (100-500 candidates)
        # Use fast, approximate methods
        lexical_results = self.lexical.search(query, top_k=100)
        dense_results = self.dense.search(query, top_k=100)
        
        # Merge, deduplicate
        candidates = self._merge_results(lexical_results, dense_results)
        print(f"Stage 1 candidates: {len(candidates)}")
        
        # STAGE 2: First-Stage Ranking (narrow to 50)
        # Lightweight ranking
        first_stage = self._first_stage_rank(query, candidates)
        first_stage = first_stage[:50]
        print(f"Stage 2 candidates: {len(first_stage)}")
        
        # STAGE 3: Second-Stage Ranking (narrow to 10)
        # Expensive cross-encoder ranking
        second_stage = self._second_stage_rank(query, first_stage)
        second_stage = second_stage[:final_top_k]
        print(f"Stage 3 candidates: {len(second_stage)}")
        
        # STAGE 4: Final Selection
        # Apply domain-specific filters
        final_results = self._apply_final_filters(query, second_stage)
        
        return final_results
    
    def _merge_results(self, lexical, dense):
        """Merge and deduplicate results from multiple methods"""
        merged = {}
        
        for rank, (doc_id, score) in enumerate(lexical):
            if doc_id not in merged:
                merged[doc_id] = {"lexical_score": score, "lexical_rank": rank}
            else:
                merged[doc_id]["lexical_score"] = score
                merged[doc_id]["lexical_rank"] = rank
        
        for rank, (doc_id, score) in enumerate(dense):
            if doc_id not in merged:
                merged[doc_id] = {"dense_score": score, "dense_rank": rank}
            else:
                merged[doc_id]["dense_score"] = score
                merged[doc_id]["dense_rank"] = rank
        
        # Normalize and combine scores using RRF
        k = 60  # Reciprocal rank fusion parameter
        results = []
        
        for doc_id, scores in merged.items():
            # RRF: 1/(k + rank)
            rrf_score = 0
            
            if "lexical_rank" in scores:
                rrf_score += 1 / (k + scores["lexical_rank"])
            
            if "dense_rank" in scores:
                rrf_score += 1 / (k + scores["dense_rank"])
            
            results.append((doc_id, rrf_score))
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def _first_stage_rank(self, query, candidates):
        """Quick ranking (e.g., BM25 + simple features)"""
        # Just return with updated scores from BM25
        scored = []
        for doc_id, initial_score in candidates:
            # Rerank with additional signals
            length_score = np.log(self.get_doc_length(doc_id))
            freshness_score = self.get_freshness_score(doc_id)
            
            combined_score = (0.6 * initial_score + 
                            0.2 * length_score + 
                            0.2 * freshness_score)
            
            scored.append((doc_id, combined_score))
        
        return sorted(scored, key=lambda x: x[1], reverse=True)
    
    def _second_stage_rank(self, query, first_stage_docs):
        """Expensive but accurate cross-encoder ranking"""
        doc_texts = [self.get_document_text(doc_id) for doc_id, _ in first_stage_docs]
        
        # Cross-encoder: pair-wise ranking
        pairs = [(query, doc_text) for doc_text in doc_texts]
        scores = self.cross_encoder.predict(pairs)
        
        # Zip back with doc IDs
        results = list(zip(
            [doc_id for doc_id, _ in first_stage_docs],
            scores
        ))
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def _apply_final_filters(self, query, second_stage_docs):
        """Apply domain-specific filters and post-processing"""
        results = []
        
        for doc_id, score in second_stage_docs:
            doc = self.get_document(doc_id)
            
            # Check filters
            if self._passes_quality_filter(doc):
                if self._passes_recency_filter(doc):
                    if self._passes_domain_filter(query, doc):
                        results.append(RetrievalResult(
                            document_id=doc_id,
                            content=doc["text"],
                            score=score,
                            stage_passed=RetrievalStage.FINAL_SELECTION,
                            retrieval_method="multi-stage"
                        ))
        
        return results
    
    def _passes_quality_filter(self, doc):
        """Check document quality"""
        return len(doc["text"]) > 50 and doc.get("quality_score", 0) > 0.5
    
    def _passes_recency_filter(self, doc):
        """Check document recency"""
        from datetime import datetime, timedelta
        doc_age = datetime.now() - doc.get("timestamp", datetime.now())
        return doc_age < timedelta(days=365)
    
    def _passes_domain_filter(self, query, doc):
        """Domain-specific filtering"""
        domain_keywords = self._extract_domain_keywords(query)
        return any(kw in doc["text"] for kw in domain_keywords)
```

### Approximate Nearest Neighbor (ANN) Search Details

```python
import heapq
from typing import Tuple

class HNSWIndex:
    """
    Hierarchical Navigable Small World Graph
    - O(log n) search complexity
    - Excellent recall in practice
    - Memory efficient
    """
    
    def __init__(self, dim: int, max_m: int = 16, max_layer: int = None):
        self.dim = dim
        self.max_m = max_m
        self.max_layer = max_layer or int(np.log(1_000_000))
        
        self.graph = {}  # level -> {node_id: [neighbors]}
        self.data = {}   # node_id -> vector
        self.entry_point = None
    
    def add(self, node_id: str, vector: np.ndarray):
        """Add vector to HNSW"""
        if len(self.data) == 0:
            # First node
            self.entry_point = node_id
            self.data[node_id] = vector
            self.graph[0] = {node_id: []}
            return
        
        # Assign layer to new node
        layer = self._assign_layer()
        
        # Find nearest neighbors at all layers
        candidates = self.entry_point
        
        for lc in range(max(self.graph.keys()), layer - 1, -1):
            nearest = self._search_layer(vector, [candidates], lc, 1)
            candidates = nearest[0]
        
        # Insert into all layers
        for lc in range(layer, -1, -1):
            candidates = self._search_layer(vector, [candidates], lc, self.max_m)
            
            if lc not in self.graph:
                self.graph[lc] = {}
            
            self.graph[lc][node_id] = candidates
            
            # Update reverse links
            for neighbor in candidates:
                if neighbor not in self.graph[lc]:
                    self.graph[lc][neighbor] = []
                
                self.graph[lc][neighbor].append(node_id)
                
                # Prune if needed
                if len(self.graph[lc][neighbor]) > self.max_m:
                    self.graph[lc][neighbor] = self._prune_neighbors(
                        self.graph[lc][neighbor],
                        vector,
                        self.max_m
                    )
        
        self.data[node_id] = vector
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors"""
        # Layer search from top to entry point
        candidates = [self.entry_point]
        
        for lc in range(max(self.graph.keys()), 0, -1):
            nearest = self._search_layer(query_vector, candidates, lc, 1)
            candidates = nearest
        
        # Layer 0 search
        nearest = self._search_layer(query_vector, candidates, 0, k)
        
        return nearest
    
    def _search_layer(self, query_vector, entry_points, layer, k):
        """Search single layer"""
        visited = set()
        candidates = []
        w = []
        
        # Initialize with entry points
        for ep in entry_points:
            dist = self._distance(query_vector, self.data[ep])
            heapq.heappush(candidates, (-dist, ep))
            heapq.heappush(w, (dist, ep))
            visited.add(ep)
        
        # Greedy search
        while candidates:
            lowerbound = -candidates[0][0]
            
            if lowerbound > w[0][0]:
                break
            
            current = heapq.heappop(candidates)[1]
            
            # Check neighbors in layer
            if layer in self.graph and current in self.graph[layer]:
                neighbors = self.graph[layer][current]
            else:
                neighbors = []
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    dist = self._distance(query_vector, self.data[neighbor])
                    
                    if dist < -w[0][0] or len(w) < k:
                        heapq.heappush(candidates, (-dist, neighbor))
                        heapq.heappush(w, (dist, neighbor))
                        
                        if len(w) > k:
                            heapq.heappop(w)
        
        return sorted([(node_id, dist) for dist, node_id in w], key=lambda x: x[1])
    
    def _distance(self, a, b):
        """Cosine distance"""
        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _assign_layer(self):
        """Randomly assign layer with exponential decay"""
        return int(-np.log(np.random.uniform()) * (1.0 / np.log(2.0)))
    
    def _prune_neighbors(self, neighbors, vector, m):
        """Keep m most relevant neighbors"""
        scored = [(self._distance(vector, self.data[n]), n) for n in neighbors]
        scored.sort()
        return [n for _, n in scored[:m]]
```

---

## Production Code Patterns

### Error Handling & Resilience

```python
from typing import Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
from functools import wraps

@dataclass
class RAGResponse:
    answer: str
    confidence: float
    sources: List[str]
    retrieved_docs: int
    latency_ms: float
    error: Optional[str] = None
    fallback_used: bool = False

class RAGErrorHandler:
    """Production error handling for RAG"""
    
    def __init__(self, max_retries=3, timeout_seconds=30):
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.circuit_breaker = CircuitBreaker(failure_threshold=5)
    
    async def execute_rag_with_fallback(self, query: str) -> RAGResponse:
        """Execute RAG with multiple fallback strategies"""
        
        start_time = datetime.now()
        
        # Strategy 1: Primary RAG
        try:
            result = await self._execute_rag_primary(query)
            return result
        except retrieval_error:
            pass
        except timeout_error:
            pass
        
        # Strategy 2: Fallback to cache
        try:
            cached_result = self.query_cache.get(query)
            if cached_result:
                return RAGResponse(
                    answer=cached_result,
                    confidence=0.7,
                    sources=["cache"],
                    retrieved_docs=0,
                    latency_ms=(datetime.now() - start_time).total_seconds() * 1000,
                    fallback_used=True
                )
        except:
            pass
        
        # Strategy 3: Fallback to LLM alone (no retrieval)
        try:
            llm_only_result = await self.llm.generate(query)
            return RAGResponse(
                answer=llm_only_result,
                confidence=0.5,
                sources=["llm_only"],
                retrieved_docs=0,
                latency_ms=(datetime.now() - start_time).total_seconds() * 1000,
                fallback_used=True
            )
        except:
            pass
        
        # Strategy 4: Return error response
        return RAGResponse(
            answer="I encountered an error processing your query. Please try again.",
            confidence=0.0,
            sources=[],
            retrieved_docs=0,
            latency_ms=(datetime.now() - start_time).total_seconds() * 1000,
            error="All RAG strategies failed",
            fallback_used=False
        )

class CircuitBreaker:
    """Prevent cascading failures"""
    
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker"""
        
        if self.state == "OPEN":
            if datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout):
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker OPEN")
        
        try:
            result = await func(*args, **kwargs)
            
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            
            return result
        
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise
```

### Batch Processing & Optimization

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

class BatchRAGProcessor:
    """Process multiple queries efficiently"""
    
    def __init__(self, batch_size=32, max_workers=4):
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.query_cache = {}
    
    async def process_batch(self, queries: List[str]) -> List[RAGResponse]:
        """Process batch of queries with optimization"""
        
        results = []
        
        # Deduplicate identical queries
        unique_queries = {}
        for idx, query in enumerate(queries):
            if query not in unique_queries:
                unique_queries[query] = []
            unique_queries[query].append(idx)
        
        print(f"Deduped {len(queries)} queries to {len(unique_queries)}")
        
        # Process unique queries
        unique_results = {}
        for query_batch in self._chunk(list(unique_queries.keys()), self.batch_size):
            batch_results = await asyncio.gather(*[
                self.process_single(q) for q in query_batch
            ])
            
            for query, result in zip(query_batch, batch_results):
                unique_results[query] = result
        
        # Map back to original order
        for original_idx, query in enumerate(queries):
            results.append(unique_results[query])
        
        return results
    
    async def process_single(self, query: str) -> RAGResponse:
        """Process single query"""
        
        # Check cache first
        if query in self.query_cache:
            return self.query_cache[query]
        
        # Process
        result = await self.rag_system.query(query)
        
        # Cache with TTL
        self.query_cache[query] = result
        
        return result
    
    def _chunk(self, items, chunk_size):
        """Chunk items into batches"""
        for i in range(0, len(items), chunk_size):
            yield items[i:i + chunk_size]
```

---

## Performance Optimization

### Embedding Cache Strategy

```python
import hashlib
from functools import lru_cache
import pickle

class SmartEmbeddingCache:
    """Multi-level embedding cache with smart invalidation"""
    
    def __init__(self, redis_client, local_cache_size=10000):
        self.redis = redis_client
        self.local_cache = {}
        self.local_cache_size = local_cache_size
        self.stats = {"hits": 0, "misses": 0}
    
    def get_or_compute(self, text: str, embedding_model) -> np.ndarray:
        """Get embedding from cache or compute"""
        
        # Hash text for key
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Level 1: Local cache (fastest)
        if text_hash in self.local_cache:
            self.stats["hits"] += 1
            return self.local_cache[text_hash]
        
        # Level 2: Redis cache (medium speed)
        redis_key = f"embedding:{text_hash}"
        cached = self.redis.get(redis_key)
        if cached:
            embedding = pickle.loads(cached)
            self.local_cache[text_hash] = embedding
            self._evict_if_needed()
            self.stats["hits"] += 1
            return embedding
        
        # Level 3: Compute (slow)
        embedding = embedding_model.encode(text)
        
        # Store in both caches
        self.local_cache[text_hash] = embedding
        self._evict_if_needed()
        self.redis.setex(
            redis_key,
            86400,  # 24 hours TTL
            pickle.dumps(embedding)
        )
        
        self.stats["misses"] += 1
        return embedding
    
    def _evict_if_needed(self):
        """LRU eviction when cache full"""
        if len(self.local_cache) > self.local_cache_size:
            # Remove oldest 10%
            to_remove = int(self.local_cache_size * 0.1)
            keys_to_remove = list(self.local_cache.keys())[:to_remove]
            for key in keys_to_remove:
                del self.local_cache[key]
    
    def get_stats(self):
        """Cache hit rate"""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0
        return {"hit_rate": hit_rate, **self.stats}
```

### Query Latency Optimization

```python
class LatencyOptimizer:
    """Reduce end-to-end latency"""
    
    def __init__(self):
        self.profiler = LatencyProfiler()
    
    async def optimize_query(self, query: str) -> RAGResponse:
        """Optimized query execution"""
        
        with self.profiler.measure("total"):
            # Parallel retrieval + embedding
            with self.profiler.measure("retrieval"):
                retrieval_task = asyncio.create_task(
                    self.retrieve_documents(query)
                )
            
            # Pre-compute query embedding in parallel
            with self.profiler.measure("embedding"):
                embedding_task = asyncio.create_task(
                    self.embed_query(query)
                )
            
            # Get results
            docs, embedding = await asyncio.gather(
                retrieval_task,
                embedding_task
            )
            
            # Generate answer
            with self.profiler.measure("generation"):
                answer = await self.generate_answer(query, docs)
            
            return RAGResponse(
                answer=answer,
                confidence=0.9,
                sources=[d["source"] for d in docs],
                retrieved_docs=len(docs),
                latency_ms=self.profiler.get_total_time()
            )

class LatencyProfiler:
    """Measure and analyze latency breakdown"""
    
    def __init__(self):
        self.measurements = {}
    
    @contextmanager
    def measure(self, operation_name: str):
        """Context manager for timing"""
        start = time.time()
        try:
            yield
        finally:
            elapsed = (time.time() - start) * 1000
            self.measurements[operation_name] = elapsed
            print(f"{operation_name}: {elapsed:.1f}ms")
    
    def get_breakdown(self) -> Dict[str, float]:
        """Get latency breakdown"""
        total = sum(self.measurements.values())
        return {
            op: (time_ms / total * 100, time_ms)
            for op, time_ms in self.measurements.items()
        }
```

---

## Enterprise Patterns

### Multi-Tenant RAG

```python
from typing import Dict

class MultiTenantRAG:
    """Serve multiple organizations with isolated knowledge bases"""
    
    def __init__(self):
        self.tenant_stores = {}  # tenant_id -> vector_store
        self.tenant_configs = {}  # tenant_id -> config
    
    async def onboard_tenant(self, tenant_id: str, config: Dict):
        """Onboard new tenant"""
        
        # Create isolated vector store
        store = self._create_isolated_store(tenant_id)
        self.tenant_stores[tenant_id] = store
        self.tenant_configs[tenant_id] = config
        
        # Load tenant-specific knowledge
        await self.load_tenant_knowledge(tenant_id, config)
    
    async def query(self, tenant_id: str, query: str) -> RAGResponse:
        """Query with tenant isolation"""
        
        if tenant_id not in self.tenant_stores:
            raise ValueError(f"Unknown tenant: {tenant_id}")
        
        store = self.tenant_stores[tenant_id]
        config = self.tenant_configs[tenant_id]
        
        # Retrieve from tenant's store only
        docs = store.search(query, top_k=config.get("top_k", 5))
        
        # Generate with tenant-specific LLM
        llm = self._get_tenant_llm(tenant_id)
        answer = await llm.generate(f"Q: {query}\nContext: {docs}")
        
        return RAGResponse(
            answer=answer,
            sources=[d["source"] for d in docs],
            retrieved_docs=len(docs),
            confidence=0.85,
            latency_ms=0
        )
    
    def _create_isolated_store(self, tenant_id: str):
        """Create isolated vector store"""
        # Could use separate Pinecone index, Qdrant namespace, etc.
        return VectorStore(namespace=f"tenant_{tenant_id}")
    
    def _get_tenant_llm(self, tenant_id: str):
        """Get tenant-specific LLM config"""
        config = self.tenant_configs[tenant_id]
        return ChatOpenAI(
            model=config.get("model", "gpt-4"),
            temperature=config.get("temperature", 0),
            api_key=config.get("api_key")
        )
```

### Knowledge Update Strategy

```python
class KnowledgeUpdateManager:
    """Manage document lifecycle and updates"""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.document_versions = {}  # doc_id -> [versions]
        self.update_queue = asyncio.Queue()
    
    async def ingest_document(self, doc_id: str, content: str, metadata: Dict):
        """Ingest new document"""
        
        # Version the document
        version_id = f"{doc_id}_v{len(self.document_versions.get(doc_id, []))}"
        
        # Process and chunk
        chunks = self.chunk_document(content)
        
        # Add to vector store with version info
        for chunk_idx, chunk in enumerate(chunks):
            self.vector_store.add({
                "id": f"{version_id}_chunk_{chunk_idx}",
                "text": chunk,
                "doc_id": doc_id,
                "version_id": version_id,
                "chunk_idx": chunk_idx,
                "metadata": metadata,
                "timestamp": datetime.now()
            })
        
        # Record version
        if doc_id not in self.document_versions:
            self.document_versions[doc_id] = []
        self.document_versions[doc_id].append(version_id)
    
    async def update_document(self, doc_id: str, new_content: str):
        """Update existing document"""
        
        # Retrieve old document
        old_chunks = self.vector_store.search(f"doc_id:{doc_id}", top_k=1000)
        
        # Mark old as deprecated
        for chunk in old_chunks:
            chunk["deprecated"] = True
            self.vector_store.update(chunk)
        
        # Ingest new version
        await self.ingest_document(doc_id, new_content, {})
    
    async def delete_document(self, doc_id: str):
        """Soft delete document"""
        
        old_chunks = self.vector_store.search(f"doc_id:{doc_id}", top_k=1000)
        
        for chunk in old_chunks:
            chunk["deleted"] = True
            self.vector_store.update(chunk)
```

---

## Troubleshooting Guide

### Common Issues & Solutions

**Issue 1: Poor Retrieval Quality**

```python
class RetrievalDiagnoser:
    """Diagnose and fix poor retrieval"""
    
    def diagnose(self, query, retrieved_docs, expected_relevant):
        """Identify retrieval problems"""
        
        issues = []
        
        # Check if relevant docs even in index
        for relevant_doc in expected_relevant:
            if not self.is_in_index(relevant_doc):
                issues.append(f"Missing document: {relevant_doc}")
        
        # Check query-document similarity
        query_embedding = self.embed_query(query)
        for doc in retrieved_docs:
            doc_embedding = self.embed_doc(doc)
            similarity = cosine_similarity(query_embedding, doc_embedding)
            
            if similarity < 0.3:
                issues.append(f"Low similarity ({similarity:.2f}) for doc {doc['id']}")
        
        # Check for semantic gaps
        query_keywords = extract_keywords(query)
        for doc in retrieved_docs:
            doc_keywords = extract_keywords(doc['text'])
            overlap = len(set(query_keywords) & set(doc_keywords))
            
            if overlap == 0:
                issues.append(f"No keyword overlap with doc {doc['id']}")
        
        return issues
    
    def recommend_fixes(self, issues):
        """Recommend improvements"""
        
        fixes = []
        
        if "Missing document" in str(issues):
            fixes.append("Add missing documents to index")
            fixes.append("Increase index refresh frequency")
        
        if "Low similarity" in str(issues):
            fixes.append("Try different embedding model")
            fixes.append("Increase top_k for retrieval")
            fixes.append("Implement query expansion")
        
        if "No keyword overlap" in str(issues):
            fixes.append("Use hybrid search (BM25 + dense)")
            fixes.append("Expand query with synonyms")
        
        return fixes
```

---

This comprehensive guide provides production-ready patterns and deep technical knowledge for implementing enterprise-grade RAG systems.

**Latest Update**: May 2026
