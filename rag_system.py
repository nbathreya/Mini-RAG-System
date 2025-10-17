"""
Minimal RAG System with Vector Database
Demonstrates: embeddings, retrieval, evaluation metrics
No external LLM required - uses sentence-transformers
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

@dataclass
class Document:
    """Document with metadata"""
    id: str
    text: str
    metadata: Dict
    embedding: np.ndarray = None

class VectorStore:
    """In-memory vector database with efficient retrieval"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.documents: List[Document] = []
        self.embeddings: np.ndarray = None
        
    def add_documents(self, texts: List[str], metadatas: List[Dict] = None):
        """Batch embed and store documents"""
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Batch embedding for efficiency
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        for i, (text, metadata, embedding) in enumerate(zip(texts, metadatas, embeddings)):
            doc = Document(
                id=f"doc_{len(self.documents) + i}",
                text=text,
                metadata=metadata,
                embedding=embedding
            )
            self.documents.append(doc)
        
        # Update embedding matrix
        self._rebuild_index()
    
    def _rebuild_index(self):
        """Rebuild embedding matrix for fast search"""
        if self.documents:
            self.embeddings = np.vstack([doc.embedding for doc in self.documents])
    
    def search(self, query: str, k: int = 5, threshold: float = 0.0) -> List[Tuple[Document, float]]:
        """
        Semantic search with cosine similarity
        Returns: List of (document, score) tuples
        """
        if not self.documents:
            return []
        
        # Embed query
        query_embedding = self.model.encode([query])[0]
        
        # Compute similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get top-k above threshold
        top_indices = np.argsort(similarities)[::-1][:k]
        results = [
            (self.documents[idx], float(similarities[idx]))
            for idx in top_indices
            if similarities[idx] >= threshold
        ]
        
        return results
    
    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.7) -> List[Tuple[Document, float]]:
        """
        Hybrid search: semantic + keyword matching
        alpha: weight for semantic similarity (1-alpha for keyword)
        """
        # Semantic scores
        semantic_results = self.search(query, k=len(self.documents))
        semantic_scores = {doc.id: score for doc, score in semantic_results}
        
        # Keyword scores (simple TF-IDF-like)
        query_words = set(query.lower().split())
        keyword_scores = {}
        for doc in self.documents:
            doc_words = set(doc.text.lower().split())
            overlap = len(query_words & doc_words)
            keyword_scores[doc.id] = overlap / max(len(query_words), 1)
        
        # Normalize and combine
        max_semantic = max(semantic_scores.values()) if semantic_scores else 1
        max_keyword = max(keyword_scores.values()) if keyword_scores else 1
        
        combined_scores = {}
        for doc in self.documents:
            sem_score = semantic_scores.get(doc.id, 0) / max_semantic
            kw_score = keyword_scores.get(doc.id, 0) / max_keyword
            combined_scores[doc.id] = alpha * sem_score + (1 - alpha) * kw_score
        
        # Return top-k
        sorted_docs = sorted(
            [(doc, combined_scores[doc.id]) for doc in self.documents],
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_docs[:k]


class RAGEvaluator:
    """Evaluate RAG system performance"""
    
    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """Precision@K metric"""
        retrieved_k = retrieved[:k]
        relevant_retrieved = len(set(retrieved_k) & set(relevant))
        return relevant_retrieved / k if k > 0 else 0.0
    
    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """Recall@K metric"""
        retrieved_k = retrieved[:k]
        relevant_retrieved = len(set(retrieved_k) & set(relevant))
        return relevant_retrieved / len(relevant) if relevant else 0.0
    
    @staticmethod
    def mean_reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
        """MRR metric"""
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                return 1.0 / (i + 1)
        return 0.0
    
    @staticmethod
    def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """Normalized Discounted Cumulative Gain"""
        retrieved_k = retrieved[:k]
        
        # DCG
        dcg = sum([
            1.0 / np.log2(i + 2) if doc_id in relevant else 0.0
            for i, doc_id in enumerate(retrieved_k)
        ])
        
        # IDCG (ideal)
        idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(relevant), k))])
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate_retrieval(self, 
                          vector_store: VectorStore,
                          test_queries: List[Dict]) -> Dict:
        """
        Evaluate on test set
        test_queries: [{"query": str, "relevant_ids": List[str]}]
        """
        metrics = {
            "precision@5": [],
            "recall@5": [],
            "mrr": [],
            "ndcg@5": []
        }
        
        for test in test_queries:
            query = test["query"]
            relevant = test["relevant_ids"]
            
            # Retrieve
            results = vector_store.search(query, k=5)
            retrieved = [doc.id for doc, _ in results]
            
            # Calculate metrics
            metrics["precision@5"].append(
                self.precision_at_k(retrieved, relevant, 5)
            )
            metrics["recall@5"].append(
                self.recall_at_k(retrieved, relevant, 5)
            )
            metrics["mrr"].append(
                self.mean_reciprocal_rank(retrieved, relevant)
            )
            metrics["ndcg@5"].append(
                self.ndcg_at_k(retrieved, relevant, 5)
            )
        
        # Average metrics
        return {
            metric: np.mean(values)
            for metric, values in metrics.items()
        }


def benchmark_rag_system():
    """Benchmark and demo"""
    print("=== RAG System Benchmark ===\n")
    
    # Sample documents (scientific abstracts)
    documents = [
        "CUDA enables GPU-accelerated computing for parallel processing tasks.",
        "Signal processing algorithms detect events in time-series data.",
        "Vector databases store embeddings for semantic search.",
        "Machine learning models require large datasets for training.",
        "Natural language processing uses transformers for text understanding.",
        "Deep learning frameworks like PyTorch enable neural network development.",
        "Real-time systems require low-latency processing pipelines.",
        "Distributed computing handles large-scale data processing.",
    ]
    
    metadatas = [{"source": f"paper_{i}"} for i in range(len(documents))]
    
    # Initialize vector store
    print("Initializing vector store...")
    vector_store = VectorStore()
    
    # Add documents
    start = time.perf_counter()
    vector_store.add_documents(documents, metadatas)
    index_time = time.perf_counter() - start
    
    # Search
    query = "How to accelerate computing with GPUs?"
    start = time.perf_counter()
    results = vector_store.search(query, k=3)
    search_time = time.perf_counter() - start
    
    print(f"\nIndexing time: {index_time*1000:.2f}ms")
    print(f"Search time: {search_time*1000:.2f}ms")
    print(f"\nQuery: {query}\n")
    print("Top results:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. [Score: {score:.3f}] {doc.text[:80]}...")
    
    # Evaluation
    test_queries = [
        {"query": "GPU computing", "relevant_ids": ["doc_0", "doc_6"]},
        {"query": "signal analysis", "relevant_ids": ["doc_1"]},
    ]
    
    evaluator = RAGEvaluator()
    metrics = evaluator.evaluate_retrieval(vector_store, test_queries)
    
    print("\n=== Retrieval Metrics ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")
    
    # Compare search methods
    print("\n=== Semantic vs Hybrid Search ===")
    query = "parallel processing"
    
    semantic = vector_store.search(query, k=3)
    hybrid = vector_store.hybrid_search(query, k=3, alpha=0.7)
    
    print(f"Query: {query}\n")
    print("Semantic only:")
    for doc, score in semantic:
        print(f"  {score:.3f}: {doc.text[:60]}...")
    
    print("\nHybrid (semantic + keyword):")
    for doc, score in hybrid:
        print(f"  {score:.3f}: {doc.text[:60]}...")


if __name__ == "__main__":
    benchmark_rag_system()
