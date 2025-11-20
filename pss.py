import os
import re
import json
import traceback
import torch
import zipfile
import sys
from typing import List, Dict, Any, Optional, TypedDict, Tuple
from collections import defaultdict, Counter
import numpy as np
from dataclasses import dataclass
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from threading import Lock
import time
import gc

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.cross_encoder import CrossEncoder 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp

@dataclass
class Config:
    EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
    RERANKER_MODEL = "BAAI/bge-reranker-base"  
    TOP_K_RETRIEVAL = 50
    TOP_K_FINAL = 5
    MIN_CHUNK_SIZE = 50
    MAX_CHUNK_SIZE = 1000
    CHUNK_OVERLAP_RATIO = 0.3
    FAISS_BATCH_SIZE = 1000
    ZIP_SUBMISSION_NAME = "submission.zip"
    CACHE_DIR = "./cache"

    MAX_WORKERS = 16
    QUERY_BATCH_SIZE = 256

    USE_FP16 = True
    PIN_MEMORY = True

class GPUAccelerator:
    """Ensures all operations run on GPU when available."""

    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"[GPU] Using CUDA: {torch.cuda.get_device_name(0)}")
            print(f"[GPU] Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            return device
        else:
            device = 'cpu'
            print("[WARNING] CUDA not available. Using CPU.")
            return device

    @staticmethod
    def clear_cache():
        """Clear GPU cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

class AdaptiveTextProcessor:
    """Dynamic text processing that adapts to data characteristics."""

    def __init__(self):
        self.stopwords = self._load_stopwords()

    def _load_stopwords(self) -> set:
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'what', 'which', 'who', 'when', 'where', 'why', 'how'
        }

    def analyze_document_structure(self, text: str) -> Dict[str, Any]:
        lines = text.split('\n')
        sentences = re.split(r'[.!?]+', text)
        words = text.split()

        analysis = {
            'total_length': len(text),
            'num_lines': len(lines),
            'num_sentences': len([s for s in sentences if s.strip()]),
            'num_words': len(words),
            'avg_line_length': np.mean([len(line) for line in lines]) if lines else 0,
            'avg_sentence_length': np.mean([len(s) for s in sentences if s.strip()]) if sentences else 0,
            'has_structure': len([l for l in lines if l.strip()]) / max(len(lines), 1) < 0.8,
        }

        return analysis

    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\'\"]', ' ', text)
        text = re.sub(r'([\.!?]){2,}', r'\1', text)
        return text.strip()

    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        filtered = [w for w in words if w not in self.stopwords]

        if not filtered:
            return []

        word_freq = Counter(filtered)
        position_scores = {}
        for idx, word in enumerate(filtered[:100]):
            if word not in position_scores:
                position_scores[word] = 1.0 - (idx / 100)

        combined_scores = {}
        for word, freq in word_freq.items():
            pos_score = position_scores.get(word, 0)
            combined_scores[word] = freq * (1 + pos_score)

        top_keywords = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [word for word, score in top_keywords[:top_n]]

    def adaptive_chunk(self, text: str, doc_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        text = self.clean_text(text)

        if doc_analysis['has_structure']:
            chunk_size = min(Config.MAX_CHUNK_SIZE, int(doc_analysis['avg_line_length'] * 10))
        else:
            chunk_size = Config.MAX_CHUNK_SIZE

        chunk_size = max(Config.MIN_CHUNK_SIZE, chunk_size)
        overlap = int(chunk_size * Config.CHUNK_OVERLAP_RATIO)

        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)

        current_chunk = ""
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sent_len = len(sentence)

            if current_length + sent_len > chunk_size and current_chunk:
                keywords = self.extract_keywords(current_chunk, 8)
                chunks.append({
                    'text': current_chunk,
                    'keywords': keywords,
                    'length': current_length,
                    'sentences': current_chunk.count('.') + current_chunk.count('!') + current_chunk.count('?')
                })

                if len(current_chunk) > overlap:
                    overlap_text = current_chunk[-overlap:]
                    current_chunk = overlap_text + " " + sentence
                    current_length = len(current_chunk)
                else:
                    current_chunk = sentence
                    current_length = sent_len
            else:
                current_chunk += (" " + sentence) if current_chunk else sentence
                current_length += sent_len

        if current_chunk and len(current_chunk) >= Config.MIN_CHUNK_SIZE:
            keywords = self.extract_keywords(current_chunk, 8)
            chunks.append({
                'text': current_chunk,
                'keywords': keywords,
                'length': current_length,
                'sentences': current_chunk.count('.') + current_chunk.count('!') + current_chunk.count('?')
            })

        return chunks

class DynamicQueryProcessor:
    """Lightweight query processing - OPTIMIZED."""

    def __init__(self):
        self.processor = AdaptiveTextProcessor()

    def analyze_query(self, query: str) -> Dict[str, Any]:
        words = query.lower().split()
        return {
            'length': len(query),
            'num_words': len(words),
            'keywords': self.processor.extract_keywords(query, 3)
        }

    def expand_query(self, query: str, analysis: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Minimal expansion for speed - only 2 queries."""
        queries = [(query, 1.0)]

        if analysis['keywords'] and len(analysis['keywords']) >= 2:
            keyword_query = " ".join(analysis['keywords'])
            if keyword_query.lower() != query.lower():
                queries.append((keyword_query, 0.8))

        return queries

class GPUEmbeddingEngine:
    """GPU-accelerated embedding with FP16 support."""

    def __init__(self, model_name: str, device: str):
        self.device = device
        print(f"[Embedding] Loading {model_name} on {device}...")

        model_kwargs = {'device': device}
        if "bge" in model_name:
            encode_kwargs = {'normalize_embeddings': True}
            self.model = SentenceTransformer(model_name, **model_kwargs)
            self.model.encode_kwargs = encode_kwargs
        else:
            self.model = SentenceTransformer(model_name, **model_kwargs)
            self.model.encode_kwargs = {'normalize_embeddings': True}

        if sys.version_info >= (3, 8) and device == 'cuda':
            try:
                self.model._first_module().auto_model = torch.compile(self.model._first_module().auto_model)
                print("[Embedding] ✓ torch.compile() enabled (PyTorch 2.0+ speedup)")
            except Exception as e:
                print(f"[Embedding] Warning: torch.compile() failed: {e}")

        if Config.USE_FP16 and device == 'cuda':
            self.model.half()
            print("[Embedding] ✓ FP16 (half precision) enabled - 2x faster")

        print(f"[Embedding] Model loaded. Max sequence length: {self.model.max_seq_length}")

    def encode_batch(self, texts: List[str], batch_size: int = 128, show_progress: bool = True) -> torch.Tensor:
        """Encode texts in batches on GPU with FP16."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_tensor=True,
            device=self.device,
        )
        return embeddings

    def encode_single(self, text: str) -> torch.Tensor:
        embedding = self.model.encode(
            text,
            convert_to_tensor=True,
            device=self.device,
        )
        return embedding

class GPUTFIDFRetriever:
    """GPU-accelerated TF-IDF retrieval using PyTorch SPARSE tensors."""

    def __init__(self, tfidf_matrix: sp.csr_matrix, device: str):
        self.device = device

        print("      → Converting TF-IDF matrix to GPU (SPARSE)...")

        tfidf_coo = tfidf_matrix.tocoo()

        indices = torch.from_numpy(np.vstack((tfidf_coo.row, tfidf_coo.col))).long()
        values = torch.from_numpy(tfidf_coo.data)
        shape = torch.Size(tfidf_coo.shape)

        dtype = torch.float32 
        values = values.to(dtype)

        self.tfidf_tensor = torch.sparse_coo_tensor(
            indices, values, shape, device=self.device
        )

        print(f"      → SPARSE TF-IDF matrix on GPU: {self.tfidf_tensor.shape} (dtype: {dtype})")

    def retrieve(self, query_vec: sp.csr_matrix, k: int) -> List[Tuple[int, float]]:
        """Fast GPU-based sparse-dense matrix multiplication."""

        query_dense = query_vec.toarray()[0]
        dtype = torch.float32 
        query_tensor = torch.tensor(query_dense, dtype=dtype, device=self.device).view(-1, 1)

        similarities = torch.sparse.mm(self.tfidf_tensor, query_tensor).squeeze()

        top_values, top_indices = torch.topk(similarities, k=min(k, len(similarities)))

        results = [
            (int(idx.cpu()), float(val.cpu()))
            for idx, val in zip(top_indices, top_values)
        ]

        return results


class MultiMethodRetriever:
    """Ultra-optimized retrieval with GPU acceleration (STAGE 1)."""

    def __init__(self, documents: List[Document], store_path: str, device: str):
        self.documents = documents
        self.store_path = store_path
        self.device = device

        print(f"[Retriever] Initializing on {device}...")
        print(f"[Retriever] Total documents: {len(documents)}")

        self._build_indices()

        print("[Retriever] All indices built successfully!")

    def _build_indices(self):
        self._build_document_mapping()
        self._build_dense_index()
        self._build_tfidf_index()

    def _build_document_mapping(self):
        print("  [1/3] Building document mapping...")
        self.chunk_to_source = {}
        self.source_to_chunks = defaultdict(list)
        self.source_metadata = defaultdict(lambda: {
            'chunk_count': 0,
            'total_length': 0,
            'keywords': set()
        })

        for idx, doc in enumerate(self.documents):
            source = doc.metadata.get("source_document", "Unknown")
            keywords = doc.metadata.get("keywords", [])

            self.chunk_to_source[idx] = source
            self.source_to_chunks[source].append(idx)

            self.source_metadata[source]['chunk_count'] += 1
            self.source_metadata[source]['total_length'] += len(doc.page_content)
            self.source_metadata[source]['keywords'].update(keywords)

        print(f"      → Mapped {len(self.documents)} chunks to {len(self.source_to_chunks)} documents")

    def _build_dense_index(self):
        print("  [2/3] Building dense embeddings (GPU + FP16)...")

        model_name_slug = re.sub(r'[^a-zA-Z0-9_-]', '_', Config.EMBEDDING_MODEL)
        cache_file_name = f"embeddings_cache_{model_name_slug}.pt"
        cache_file = Path(self.store_path) / cache_file_name

        if cache_file.exists():
            print(f"      → Loading cached embeddings: {cache_file_name}")
            self.dense_embeddings = torch.load(cache_file, map_location=self.device)

            if Config.USE_FP16 and self.device == 'cuda' and self.dense_embeddings.dtype != torch.float16:
                self.dense_embeddings = self.dense_embeddings.half()

            print(f"      → Loaded {self.dense_embeddings.shape[0]} embeddings on GPU")
        else:
            print(f"      → No cache found. Generating new embeddings with {Config.EMBEDDING_MODEL}")
            texts = [doc.page_content for doc in self.documents]

            embedding_engine = GPUEmbeddingEngine(Config.EMBEDDING_MODEL, self.device)
            self.dense_embeddings = embedding_engine.encode_batch(texts, batch_size=128)

            print(f"      → Generated {self.dense_embeddings.shape[0]} embeddings")
            print(f"      → Caching embeddings to {cache_file_name}...")
            torch.save(self.dense_embeddings, cache_file)

        self.dense_embeddings = self.dense_embeddings.to(self.device)
        print(f"      → Embeddings permanently on GPU (dtype: {self.dense_embeddings.dtype})")

        self.embedding_engine = GPUEmbeddingEngine(Config.EMBEDDING_MODEL, self.device)

    def _build_tfidf_index(self):
        print("  [3/3] Building TF-IDF index (GPU-accelerated)...")

        cache_file_vectorizer = Path(self.store_path) / "tfidf_vectorizer.pkl"
        cache_file_matrix = Path(self.store_path) / "tfidf_matrix.pkl"

        if cache_file_vectorizer.exists() and cache_file_matrix.exists():
            print("      → Loading cached TF-IDF index...")
            with open(cache_file_vectorizer, 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            with open(cache_file_matrix, 'rb') as f:
                tfidf_matrix = pickle.load(f)
            print(f"      → TF-IDF index loaded: {tfidf_matrix.shape}")
        else:
            corpus = [doc.page_content for doc in self.documents]

            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95,
                sublinear_tf=True
            )

            tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)

            print("      → Caching TF-IDF index...")
            with open(cache_file_vectorizer, 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
            with open(cache_file_matrix, 'wb') as f:
                pickle.dump(tfidf_matrix, f)
            print(f"      → Built matrix: {tfidf_matrix.shape}")

        self.gpu_tfidf = GPUTFIDFRetriever(tfidf_matrix, self.device)

    def _retrieve_dense_batch(self, query_embeddings: torch.Tensor, k: int) -> List[List[Tuple[int, float]]]:
        """Ultra-fast batch dense retrieval on GPU."""
        similarities = torch.matmul(query_embeddings, self.dense_embeddings.T)

        batch_results = []
        for sim_scores in similarities:
            top_results = torch.topk(sim_scores, k=min(k, len(sim_scores)))
            results = [
                (int(idx.cpu()), float(score.cpu()))
                for idx, score in zip(top_results.indices, top_results.values)
            ]
            batch_results.append(results)

        return batch_results

    def _retrieve_tfidf(self, query: str, k: int) -> List[Tuple[int, float]]:
        """GPU-accelerated TF-IDF retrieval."""
        query_vec = self.tfidf_vectorizer.transform([query])
        return self.gpu_tfidf.retrieve(query_vec, k)

    def _aggregate_to_documents(self, chunk_results: List[Tuple[int, float]]) -> List[Tuple[str, float]]:
        """Public method so RAGEngine can call it after reranking."""
        doc_scores = defaultdict(list)

        for chunk_idx, score in chunk_results:
            source = self.chunk_to_source.get(chunk_idx, "Unknown")
            doc_scores[source].append(score)

        final_scores = {}
        for doc, scores in doc_scores.items():
            if not scores:
                continue

            max_score = max(scores)
            avg_score = np.mean(scores)

            chunk_count = self.source_metadata[doc]['chunk_count']

            if chunk_count > 0:
                count_score = len(scores) / chunk_count
            else:
                count_score = 0.0

            final_scores[doc] = (0.5 * max_score + 0.3 * avg_score + 0.2 * count_score)

        sorted_docs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs

    def _retrieve_single_sparse_set(self, query: str, k: int) -> Dict[int, float]:
        """Helper to get TF-IDF scores for one query."""
        sparse_results = defaultdict(float)

        tfidf_results = self._retrieve_tfidf(query, k)
        for doc_idx, score in tfidf_results:
            sparse_results[doc_idx] += 0.3 * score 

        return sparse_results

    def retrieve_chunk_batch(self, queries: List[str], expanded_queries_list: List[List[Tuple[str, float]]]) -> List[List[Tuple[int, float]]]:
        """Retrieves top_k chunks for a batch of queries."""

        primary_queries = [expanded[0][0] for expanded in expanded_queries_list]

        query_embeddings = self.embedding_engine.encode_batch(
            primary_queries,
            batch_size=len(primary_queries),
            show_progress=False
        )
        dense_results_batch = self._retrieve_dense_batch(query_embeddings, Config.TOP_K_RETRIEVAL)

        sparse_results_list = [None] * len(primary_queries)
        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            futures = {
                executor.submit(self._retrieve_single_sparse_set, query, Config.TOP_K_RETRIEVAL): i
                for i, query in enumerate(primary_queries)
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    sparse_results_list[idx] = future.result()
                except Exception as e:
                    print(f"\n[ERROR] Sparse retrieval failed for query {idx}: {e}")
                    sparse_results_list[idx] = defaultdict(float)

        final_batch_chunks = []
        for i in range(len(primary_queries)):
            all_results = defaultdict(float)

            # Add dense results
            for doc_idx, score in dense_results_batch[i]:
                all_results[doc_idx] += 0.7 * score  # 70% weight

            # Add parallel sparse results
            for doc_idx, score in sparse_results_list[i].items():
                all_results[doc_idx] += score # 30% weight (already applied in helper)

            # Aggregate and get top chunks
            top_chunks = sorted(all_results.items(), key=lambda x: x[1], reverse=True)[:Config.TOP_K_RETRIEVAL]
            final_batch_chunks.append(top_chunks)

        return final_batch_chunks

class RAGEngine:
    """Ultra-fast RAG engine with 2-stage reranking."""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(RAGEngine, cls).__new__(cls)
        return cls._instance

    def __init__(self, documents: List[Document] = None, store_path: str = None):
        if not hasattr(self, 'initialized') and documents and store_path:
            print("\n" + "="*70)
            print("INITIALIZING ULTRA-FAST RAG ENGINE")
            print("="*70)

            self.device = GPUAccelerator.get_device()

            self.retriever = MultiMethodRetriever(documents, store_path, self.device)
            self.query_processor = DynamicQueryProcessor()

            print(f"[Reranker] Loading {Config.RERANKER_MODEL} on {self.device}...")
            self.reranker = CrossEncoder(Config.RERANKER_MODEL, device=self.device, max_length=512)
            if Config.USE_FP16 and self.device == 'cuda':
                self.reranker.model.half()
                print("[Reranker] ✓ FP16 (half precision) enabled")
            print("[Reranker] Reranker model loaded.")

            self.initialized = True

            print("\n" + "="*70)
            print(f"RAG ENGINE READY (Model: {Config.EMBEDDING_MODEL})")
            print("="*70)

    def _rerank_batch(self, queries: List[str], top_chunks_batch: List[List[Tuple[int, float]]]) -> List[List[str]]:
        """Reranks retrieved chunks for a batch of queries."""
        final_batch_results = []

        for i, query in enumerate(queries):
            chunk_list = top_chunks_batch[i] 

            if not chunk_list:
                final_batch_results.append([])
                continue

            # Create (query, chunk_text) pairs for the reranker
            chunk_texts = [self.retriever.documents[idx].page_content for idx, score in chunk_list]
            sentence_pairs = [(query, text) for text in chunk_texts]

            # Run reranker in a small batch (just 50 pairs)
            # This is a separate GPU call for *each* query
            rerank_scores = self.reranker.predict(sentence_pairs, show_progress_bar=False)

            # Combine original chunk_idx with new reranker_score
            reranked_chunks = [
                (chunk_list[j][0], rerank_scores[j])
                for j in range(len(rerank_scores))
            ]

            # Sort by the new, more accurate reranker score
            reranked_chunks.sort(key=lambda x: x[1], reverse=True)

            # Aggregate the reranked chunks into documents
            doc_results = self.retriever._aggregate_to_documents(reranked_chunks)

            # Append the top N final documents
            final_batch_results.append([doc for doc, score in doc_results[:Config.TOP_K_FINAL]])

        return final_batch_results

    def run_query_batch(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Execute batch of queries efficiently."""
        if not hasattr(self, 'retriever'):
            return [{"query": q, "response": ["Error: Engine not initialized"]} for q in queries]

        try:
            expanded_queries_list = []
            for query in queries:
                query_analysis = self.query_processor.analyze_query(query)
                expanded_queries = self.query_processor.expand_query(query, query_analysis)
                expanded_queries_list.append(expanded_queries)

            # Stage 1: Fast Retrieval
            top_chunks_batch = self.retriever.retrieve_chunk_batch(queries, expanded_queries_list)

            # Stage 2: Accurate Reranking
            final_doc_results = self._rerank_batch(queries, top_chunks_batch)

            return [
                {"query": query, "response": results}
                for query, results in zip(queries, final_doc_results)
            ]

        except Exception as e:
            traceback.print_exc()
            return [{"query": q, "response": [f"Error: {str(e)}"]} for q in queries]

class ProgressTracker:
    """Thread-safe progress tracking."""

    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.lock = Lock()
        self.start_time = time.time()
        self.last_update = time.time()

    def update(self, n: int = 1):
        """Update progress."""
        with self.lock:
            self.completed += n
            current_time = time.time()

            if current_time - self.last_update < 2:
                return

            self.last_update = current_time
            elapsed = current_time - self.start_time
            rate = self.completed / elapsed if elapsed > 0 else 0
            remaining = (self.total - self.completed) / rate if rate > 0 else 0

            print(f"\rProgress: {self.completed}/{self.total} "
                  f"({self.completed*100/self.total:.1f}%) | "
                  f"Rate: {rate:.1f} q/s | "
                  f"ETA: {remaining/60:.1f} min", end="", flush=True)

    def finish(self):
        """Mark as finished."""
        elapsed = time.time() - self.start_time
        print(f"\n✓ Completed {self.total} queries in {elapsed/60:.1f} minutes "
              f"(avg: {self.total/elapsed:.1f} q/s)")

def normalize_query_data(query_dict: Dict) -> Dict:
    """Normalize query data to handle different key names."""
    query_text = query_dict.get('query') or query_dict.get('Query')

    query_num = (query_dict.get('query_num') or
                 query_dict.get('querynum') or
                 query_dict.get('query_number') or
                 query_dict.get('Query_num'))

    return {
        'query': query_text,
        'query_num': query_num
    }

def process_query_batch(batch_queries: List[Dict], engine: RAGEngine, output_folder: str) -> int:
    """Process a batch of queries."""
    normalized_batch = [normalize_query_data(q) for q in batch_queries]

    queries = [q['query'] for q in normalized_batch]

    # Batch processing
    results = engine.run_query_batch(queries)

    # <<< --- BOTTLENECK: This loop writes 256 files per batch --- >>>
    for query_info, result in zip(normalized_batch, results):
        query_num = query_info['query_num']
        output_path = os.path.join(output_folder, f"query_{query_num}.json")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

    return len(batch_queries)

def read_file_content(file_path: str) -> str:
    """Read file content dynamically."""
    try:
        if file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        elif file_path.endswith(".json"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return json.dumps(data, ensure_ascii=False, indent=2)
                elif isinstance(data, list):
                    return json.dumps(data, ensure_ascii=False, indent=2)
                else:
                    return str(data)
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception as e:
        print(f"[Warning] Could not read {file_path}: {e}")
        return ""

def _process_file_chunk(file_path: str, processor: AdaptiveTextProcessor) -> Optional[Dict[str, Any]]:
    """Helper for parallel file processing."""
    try:
        content = read_file_content(file_path)
        if not content.strip():
            return None

        analysis = processor.analyze_document_structure(content)
        chunks = processor.adaptive_chunk(content, analysis)

        if chunks:
            return {
                "document_name": os.path.basename(file_path),
                "chunks": chunks,
                "analysis": analysis
            }
        return None
    except Exception as e:
        print(f"[Warning] Failed to process {file_path}: {e}")
        return None

def structure_data_from_folder(input_folder: str) -> List[Dict[str, Any]]:
    """Dynamic data structuring from folder - NOW IN PARALLEL."""
    print(f"\n[DataLoader] Processing folder: {input_folder}")

    processor = AdaptiveTextProcessor()
    structured_data = []

    file_paths = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if os.path.isfile(os.path.join(input_folder, f))
    ]
    print(f"[DataLoader] Found {len(file_paths)} files. Processing in parallel...")

    with ProcessPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
        futures = (
            executor.submit(_process_file_chunk, fp, processor)
            for fp in file_paths
        )

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result:
                structured_data.append(result)

            if (i+1) % 200 == 0:
                print(f"  ...processed {i+1}/{len(file_paths)} files")

    print(f"\n[DataLoader] Processed {len(structured_data)} valid documents")
    return structured_data

def parse_structured_data(data: List[Dict[str, Any]]) -> List[Document]:
    """Parse structured data into Document objects."""
    print(f"\n[Parser] Creating document chunks...")

    all_documents = []

    for doc_record in data:
        document_name = doc_record.get("document_name", "Unknown")
        chunks = doc_record.get("chunks", [])

        for chunk_info in chunks:
            if isinstance(chunk_info, dict):
                text = chunk_info.get('text', '')
                keywords = chunk_info.get('keywords', [])
                length = chunk_info.get('length', 0)
            else:
                text = str(chunk_info)
                keywords = []
                length = len(text)

            if text and len(text) >= Config.MIN_CHUNK_SIZE:
                metadata = {
                    "source_document": document_name,
                    "keywords": keywords,
                    "chunk_length": length
                }
                all_documents.append(Document(page_content=text, metadata=metadata))

    print(f"[Parser] Created {len(all_documents)} document chunks")
    return all_documents

def check_existing_index(store_path: str) -> bool:
    """Check if ALL required index files exist."""

    model_name_slug = re.sub(r'[^a-zA-Z0-9_-]', '_', Config.EMBEDDING_MODEL)
    cache_file_name = f"embeddings_cache_{model_name_slug}.pt"

    required_files = [
        cache_file_name,
        "documents_cache.pkl",
        "tfidf_vectorizer.pkl",
        "tfidf_matrix.pkl"
    ]

    store_path_obj = Path(store_path)

    for file_name in required_files:
        if not (store_path_obj / file_name).exists():
            print(f"[Index Check] Missing file: {file_name}")
            return False

    return True

class AgentState(TypedDict):
    user_inputs: Dict[str, str]
    next_action: str
    single_query_input: str
    multi_query_file_path: str
    skip_index_build: bool

def knowledge_base_agent(state: AgentState) -> Dict:
    """Build knowledge base or load existing."""

    store_path = state['user_inputs']['vector_store_path']

    if state.get('skip_index_build', False):
        print("\n" + "="*70)
        print(f"LOADING EXISTING INDEX (Model: {Config.EMBEDDING_MODEL})")
        print("="*70)

        cache_file = Path(store_path) / "documents_cache.pkl"

        if not cache_file.exists():
            print("[ERROR] documents_cache.pkl not found!")
            sys.exit(1)

        print(f"[Loading] Loading cached documents...")
        with open(cache_file, 'rb') as f:
            documents = pickle.load(f)
        print(f"[Loading] ✓ Loaded {len(documents)} document chunks")

        print("[Loading] Initializing GPU-optimized RAG engine...")
        RAGEngine(documents, store_path)

        GPUAccelerator.clear_cache()

        print("\n" + "="*70)
        print("✓ READY FOR ULTRA-FAST QUERY PROCESSING")
        print("="*70)
        return {}

    print("\n" + "="*70)
    print(f"BUILDING INDEX FROM SCRATCH (Model: {Config.EMBEDDING_MODEL})")
    print("="*70)

    input_folder = state['user_inputs']['input_folder']

    if not os.path.isdir(input_folder):
        print(f"[ERROR] Folder not found: {input_folder}")
        sys.exit(1)

    structured_data = structure_data_from_folder(input_folder)

    if not structured_data:
        print("[ERROR] No valid data found")
        sys.exit(1)

    documents = parse_structured_data(structured_data)

    if not documents:
        print("[ERROR] No document chunks created")
        sys.exit(1)

    cache_file = Path(store_path) / "documents_cache.pkl"
    print(f"\n[Caching] Saving documents...")
    with open(cache_file, 'wb') as f:
        pickle.dump(documents, f)

    RAGEngine(documents, store_path)

    GPUAccelerator.clear_cache()

    print("\n" + "="*70)
    print("✓ INDEX BUILT SUCCESSFULLY")
    print("="*70)

    return {}

def user_interface_agent(state: AgentState) -> Dict[str, str]:
    print("\n" + "="*50)
    print("MAIN MENU")
    print("="*50)
    print("1. Single query")
    print("2. Batch queries (ULTRA-FAST SEQUENTIAL)")
    print("3. Exit")
    print("="*50)

    while True:
        try:
            choice = input("Select option (1-3): ").strip()
            if choice == '1':
                query = input("Enter query: ").strip()
                return {"next_action": "single_query", "single_query_input": query}
            elif choice == '2':
                path = input("Path to queries.json: ").strip()
                return {"next_action": "multi_query", "multi_query_file_path": path}
            elif choice == '3':
                return {"next_action": "quit"}
            else:
                print("Invalid option.")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            return {"next_action": "quit"}

def single_query_agent(state: AgentState) -> Dict[str, str]:
    print("\n[Single Query Mode]")

    query = state.get('single_query_input')
    if not query:
        return {"single_query_input": ""}

    engine = RAGEngine()
    results = engine.run_query_batch([query])
    result = results[0]

    sanitized = re.sub(r'[\W_]+', '_', query.lower())[:50].strip('_')
    output_path = os.path.join(
        state['user_inputs']['output_folder'],
        f"query_single_{sanitized}.json"
    )

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"✓ Saved: {output_path}")
    print(f"✓ Results: {result['response']}")

    return {"single_query_input": ""}

def multi_query_agent(state: AgentState) -> Dict[str, str]:
    """ULTRA-FAST sequential batch query processing."""
    print("\n" + "="*70)
    print("ULTRA-FAST SEQUENTIAL BATCH MODE (GPU-OPTIMIZED)")
    print("="*70)

    file_path = state.get('multi_query_file_path')

    if not file_path or not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return {"multi_query_file_path": ""}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            queries = json.load(f)

        if not isinstance(queries, list):
            raise ValueError("Expected JSON array")

    except Exception as e:
        print(f"[ERROR] Could not parse JSON: {e}")
        return {"multi_query_file_path": ""}

    total_queries = len(queries)
    print(f"\n[INFO] Total queries: {total_queries}")
    print(f"[INFO] Batch size: {Config.QUERY_BATCH_SIZE} (GPU-optimized)")
    print(f"[INFO] Workers: {Config.MAX_WORKERS} (for data loading only)")
    print(f"[INFO] GPU Memory: 15GB | FP16: {Config.USE_FP16}")

    engine = RAGEngine()
    output_folder = state['user_inputs']['output_folder']

    batches = []
    for i in range(0, total_queries, Config.QUERY_BATCH_SIZE):
        batch = queries[i:i + Config.QUERY_BATCH_SIZE]
        batches.append(batch)

    print(f"[INFO] Created {len(batches)} batches\n")

    tracker = ProgressTracker(total_queries)

    print(f"\n[INFO] Starting sequential batch processing...")
    for batch in batches:
        try:
            processed_count = process_query_batch(batch, engine, output_folder)
            tracker.update(processed_count)
        except Exception as e:
            print(f"\n[ERROR] Batch failed: {e}")
            traceback.print_exc()

    tracker.finish()

    print(f"\n✓ All queries processed")
    print(f"✓ Results in: {output_folder}")

    return {"multi_query_file_path": ""}

def supervisor_router(state: AgentState) -> str:
    action = state.get('next_action')
    if action == "single_query":
        return "single_query_agent"
    elif action == "multi_query":
        return "multi_query_agent"
    elif action == "quit":
        return END
    return END

def build_main_workflow() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("knowledge_base_agent", knowledge_base_agent)
    workflow.add_node("user_interface_agent", user_interface_agent)
    workflow.add_node("single_query_agent", single_query_agent)
    workflow.add_node("multi_query_agent", multi_query_agent)

    workflow.set_entry_point("knowledge_base_agent")
    workflow.add_edge("knowledge_base_agent", "user_interface_agent")

    workflow.add_conditional_edges(
        "user_interface_agent",
        supervisor_router,
        {
            "single_query_agent": "single_query_agent",
            "multi_query_agent": "multi_query_agent",
            END: END
        }
    )

    workflow.add_edge("single_query_agent", "user_interface_agent")
    workflow.add_edge("multi_query_agent", "user_interface_agent")

    return workflow.compile()

def get_initial_user_inputs() -> Dict[str, str]:
    print("="*70)
    print("ULTRA-FAST GPU-OPTIMIZED RAG SYSTEM")
    print(f"Model: {Config.EMBEDDING_MODEL}")
    print("15GB GPU | FP16 | Batch Size 256 | 16 Workers")
    print("="*70)
    print()

    inputs = {}

    inputs['input_folder'] = input("1. Input data folder: ").strip()
    inputs['vector_store_path'] = input("2. Vector store path: ").strip()
    inputs['output_folder'] = input("3. Output folder: ").strip()

    os.makedirs(inputs['vector_store_path'], exist_ok=True)
    os.makedirs(inputs['output_folder'], exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    skip_build = False
    if check_existing_index(inputs['vector_store_path']):
        print("\n" + "="*70)
        print("✓ COMPLETE INDEX FOUND (for current model)")
        print("="*70)
        choice = input("Use existing index? (y/n): ").strip().lower()
        if choice == 'y':
            skip_build = True
            print("\n✓ Will use existing index")
        else:
            print("\n✓ Will rebuild")
    else:
        print("\n[INFO] No matching index found. Will build from scratch.")

    inputs['skip_index_build'] = skip_build

    return inputs

def create_submission_zip(output_folder: str):
    """Create submission zip file."""
    zip_path = os.path.join(output_folder, Config.ZIP_SUBMISSION_NAME)
    json_files = [f for f in os.listdir(output_folder) if f.endswith('.json')]

    if not json_files:
        print("\n[Warning] No JSON files found")
        return

    print(f"\nCreating submission: {zip_path}")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in json_files:
            file_path = os.path.join(output_folder, file)
            zipf.write(file_path, arcname=file)

    print(f"✓ Created {Config.ZIP_SUBMISSION_NAME} with {len(json_files)} files")

def main():
    user_inputs = None
    try:
        user_inputs = get_initial_user_inputs()

        app = build_main_workflow()
        initial_state = {
            "user_inputs": user_inputs,
            "next_action": "",
            "single_query_input": "",
            "multi_query_file_path": "",
            "skip_index_build": user_inputs.get('skip_index_build', False)
        }

        app.invoke(initial_state)

        print("\n" + "="*70)
        print("PIPELINE COMPLETED")
        print("="*70)

    except SystemExit as e:
        if e.code != 0:
            print(f"\n[ERROR] Exit code: {e.code}")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        traceback.print_exc()
    finally:
        if user_inputs and 'output_folder' in user_inputs:
            create_submission_zip(user_inputs['output_folder'])

        GPUAccelerator.clear_cache()

if __name__ == "__main__":
    main()
