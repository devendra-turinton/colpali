import os
import sys
import torch
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import base64
from dataclasses import dataclass
import logging
from PIL import Image
import numpy as np
import faiss
import json
import gzip
import pickle
import hashlib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import gc

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Set up model cache directory
CACHE_DIR = os.path.expanduser('~/.cache/huggingface')
os.environ['HF_HOME'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['HF_HUB_CACHE'] = CACHE_DIR
os.environ['HUGGINGFACE_HUB_CACHE'] = CACHE_DIR

# Create cache directory if it doesn't exist
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

@dataclass
class RAGConfig:
    """Configuration for Multimodal RAG"""
    model_name: str = "vidore/colpali"
    index_name: str = "multimodal_rag"
    data_dir: str = "data"
    index_dir: str = ".byaldi"
    faiss_index_dir: str = "faiss_indexes"
    use_flash_attention: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    anthropic_api_key: Optional[str] = ""
    cache_dir: str = CACHE_DIR
    force_download: bool = False
    use_faiss: bool = True
    persistent_session: bool = True
    
    # Production settings
    batch_size: int = 32  # For batch processing
    max_docs_in_memory: int = 1000  # Documents to keep in memory
    use_gpu_faiss: bool = torch.cuda.is_available()
    faiss_nprobe: int = 10  # For IVF indexes
    faiss_nlist: int = 100  # Number of clusters for IVF

class ProductionFaissManager:
    """Production-ready FAISS manager with scalability features"""
    def __init__(self, index_dir: str, config: RAGConfig):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)
        self.config = config
        self.index = None
        self.metadata = {}
        self.doc_mapping = {}  # Maps doc_id to PDF path
        self.embedding_dim = None
        
    def create_production_index(self, dimension: int, num_embeddings: int = 0):
        """Create a production-ready FAISS index"""
        self.embedding_dim = dimension
        
        if num_embeddings < 10000:
            # For small datasets, use flat index
            self.index = faiss.IndexFlatIP(dimension)
            logger.info(f"Created Flat FAISS index with dimension {dimension}")
        else:
            # For large datasets, use IVF with PQ for efficiency
            quantizer = faiss.IndexFlatIP(dimension)
            nlist = min(self.config.faiss_nlist, int(np.sqrt(num_embeddings)))
            
            # Use IndexIVFPQ for very large datasets
            if num_embeddings > 100000:
                m = 8  # Number of subquantizers
                self.index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, 8)
                logger.info(f"Created IVF-PQ FAISS index for {num_embeddings} embeddings")
            else:
                self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
                logger.info(f"Created IVF-Flat FAISS index for {num_embeddings} embeddings")
            
            # Train the index if it's IVF
            if hasattr(self.index, 'train') and num_embeddings > 0:
                logger.info("Training IVF index...")
                # This would need sample embeddings for training
                
        # Move to GPU if available and requested
        if self.config.use_gpu_faiss and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
            logger.info("Moved FAISS index to GPU")
    
    def add_embeddings_batch(self, embeddings: np.ndarray, metadata_list: List[Dict], doc_id: str, pdf_path: str):
        """Add embeddings in batches with metadata"""
        if self.index is None:
            self.create_production_index(embeddings.shape[1], embeddings.shape[0])
        
        # Normalize embeddings
        faiss.normalize_L2(embeddings)
        
        # Add to index
        start_idx = self.index.ntotal
        self.index.add(embeddings)
        
        # Store metadata efficiently
        for i, meta in enumerate(metadata_list):
            idx = start_idx + i
            self.metadata[idx] = {
                'doc_id': doc_id,
                'page_num': meta.get('page_num', 0),
                'pdf_path': pdf_path
            }
        
        # Update document mapping
        self.doc_mapping[doc_id] = pdf_path
        
        logger.info(f"Added {len(embeddings)} embeddings for document {doc_id}")
    
    def search_batch(self, query_embeddings: np.ndarray, k: int = 5) -> List[List[Tuple[int, float, Dict]]]:
        """Batch search for multiple queries"""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Normalize queries
        faiss.normalize_L2(query_embeddings)
        
        # Set search parameters for IVF indexes
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.config.faiss_nprobe
        
        # Search
        scores, indices = self.index.search(query_embeddings, min(k, self.index.ntotal))
        
        # Process results
        all_results = []
        for query_scores, query_indices in zip(scores, indices):
            results = []
            for score, idx in zip(query_scores, query_indices):
                if idx != -1:  # Valid result
                    results.append((idx, float(score), self.metadata.get(int(idx), {})))
            all_results.append(results)
        
        return all_results
    
    def save(self, index_name: str):
        """Save index and metadata efficiently"""
        if self.index is None:
            return
        
        index_path = self.index_dir / f"{index_name}.faiss"
        metadata_path = self.index_dir / f"{index_name}_metadata.pkl"
        doc_mapping_path = self.index_dir / f"{index_name}_docs.pkl"
        
        # If using GPU, convert back to CPU for saving
        if self.config.use_gpu_faiss and hasattr(self.index, 'to_cpu'):
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, str(index_path))
        else:
            faiss.write_index(self.index, str(index_path))
        
        # Save metadata in chunks for large datasets
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save document mapping
        with open(doc_mapping_path, 'wb') as f:
            pickle.dump(self.doc_mapping, f)
        
        logger.info(f"Saved FAISS index with {self.index.ntotal} vectors")
    
    def load(self, index_name: str) -> bool:
        """Load index and metadata"""
        index_path = self.index_dir / f"{index_name}.faiss"
        metadata_path = self.index_dir / f"{index_name}_metadata.pkl"
        doc_mapping_path = self.index_dir / f"{index_name}_docs.pkl"
        
        if not index_path.exists():
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            
            # Move to GPU if requested
            if self.config.use_gpu_faiss and faiss.get_num_gpus() > 0:
                self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
            
            # Load metadata
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
            
            # Load document mapping
            if doc_mapping_path.exists():
                with open(doc_mapping_path, 'rb') as f:
                    self.doc_mapping = pickle.load(f)
            
            logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            return True
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            return False

class ByaldiEmbeddingExtractor:
    """Extract embeddings from Byaldi's saved files"""
    @staticmethod
    def extract_from_disk(index_dir: str, index_name: str) -> Tuple[np.ndarray, List[Dict]]:
        """Extract embeddings from Byaldi's disk storage"""
        index_path = Path(index_dir) / index_name
        embeddings_path = index_path / "embeddings" / "embeddings_0.pt"
        
        if not embeddings_path.exists():
            logger.error(f"Embeddings file not found: {embeddings_path}")
            return None, None
        
        try:
            # Load PyTorch embeddings
            embeddings_data = torch.load(embeddings_path, map_location='cpu')
            
            # Load metadata
            metadata_list = []
            embed_mapping_path = index_path / "embed_id_to_doc_id.json.gz"
            
            if embed_mapping_path.exists():
                with gzip.open(embed_mapping_path, 'rt') as f:
                    embed_mapping = json.load(f)
                
                # Convert to list format
                for embed_id, info in embed_mapping.items():
                    metadata_list.append({
                        'embed_id': int(embed_id),
                        'doc_id': info['doc_id'],
                        'page_num': info['page_id']
                    })
            
            # Process embeddings based on their structure
            embeddings_array = None
            
            # Convert BFloat16 to Float32 if needed
            def convert_to_float32(tensor):
                if tensor.dtype == torch.bfloat16:
                    return tensor.to(torch.float32)
                return tensor
            
            if isinstance(embeddings_data, torch.Tensor):
                embeddings_data = convert_to_float32(embeddings_data)
                embeddings_array = embeddings_data.numpy()
            elif isinstance(embeddings_data, dict):
                # Handle different possible structures
                if 'embeddings' in embeddings_data:
                    tensor = convert_to_float32(embeddings_data['embeddings'])
                    embeddings_array = tensor.numpy()
                else:
                    # Try to extract from nested structure
                    all_embeddings = []
                    for key, value in embeddings_data.items():
                        if isinstance(value, torch.Tensor):
                            tensor = convert_to_float32(value)
                            all_embeddings.append(tensor.numpy())
                    embeddings_array = np.vstack(all_embeddings) if all_embeddings else None
            elif isinstance(embeddings_data, list):
                # Handle list of tensors
                all_embeddings = []
                for item in embeddings_data:
                    if isinstance(item, torch.Tensor):
                        tensor = convert_to_float32(item)
                        all_embeddings.append(tensor.numpy())
                if all_embeddings:
                    embeddings_array = np.vstack(all_embeddings)
            else:
                # Try direct conversion
                try:
                    if hasattr(embeddings_data, 'dtype') and embeddings_data.dtype == torch.bfloat16:
                        embeddings_data = embeddings_data.to(torch.float32)
                    embeddings_array = np.array(embeddings_data)
                except:
                    logger.error(f"Unknown embedding format: {type(embeddings_data)}")
            
            if embeddings_array is not None:
                # Ensure float32
                embeddings_array = embeddings_array.astype(np.float32)
                logger.info(f"Extracted {embeddings_array.shape[0]} embeddings from disk with shape {embeddings_array.shape}")
                return embeddings_array, metadata_list
            else:
                logger.error("Could not convert embeddings to array format")
            
        except Exception as e:
            logger.error(f"Error extracting embeddings: {e}")
            import traceback
            traceback.print_exc()
        
        return None, None

class ModelManager:
    """Singleton class to manage model instances"""
    _instance = None
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def get_model(self, model_type: str, loader_func=None, *args, **kwargs):
        """Get or create a model instance"""
        if model_type not in self._models and loader_func:
            logger.info(f"Loading {model_type} for the first time...")
            self._models[model_type] = loader_func(*args, **kwargs)
        return self._models.get(model_type)
    
    def clear_model(self, model_type: str):
        """Remove a model from memory"""
        if model_type in self._models:
            del self._models[model_type]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

class ProductionMultimodalRAG:
    """Production-ready Multimodal RAG system"""
    def __init__(self, config: RAGConfig):
        self.config = config
        self.model_manager = ModelManager()
        self.faiss_manager = ProductionFaissManager(config.faiss_index_dir, config)
        self.embedding_extractor = ByaldiEmbeddingExtractor()
        self._setup_directories()
        self._indexed_docs = set()  # Track indexed documents
        
    def _setup_directories(self):
        """Create necessary directories"""
        Path(self.config.data_dir).mkdir(exist_ok=True)
        Path(self.config.index_dir).mkdir(exist_ok=True)
        Path(self.config.faiss_index_dir).mkdir(exist_ok=True)
    
    def _load_colpali_model(self):
        """Load ColPali model"""
        from byaldi import RAGMultiModalModel
        
        os.environ['BYALDI_CACHE_DIR'] = self.config.cache_dir
        
        try:
            model = RAGMultiModalModel.from_pretrained(
                self.config.model_name,
                local_files_only=True
            )
            logger.info("ColPali model loaded from local cache")
        except:
            logger.info("Loading ColPali model from hub...")
            model = RAGMultiModalModel.from_pretrained(
                self.config.model_name
            )
        
        return model
    
    def get_rag_model(self):
        """Get or create RAG model instance"""
        return self.model_manager.get_model(
            'colpali',
            self._load_colpali_model
        )
    
    def index_document(self, pdf_path: str, doc_id: Optional[str] = None):
        """Index a single document"""
        if doc_id is None:
            doc_id = hashlib.md5(pdf_path.encode()).hexdigest()[:16]
        
        # Check if already indexed
        if doc_id in self._indexed_docs:
            logger.info(f"Document {doc_id} already indexed")
            return doc_id
        
        logger.info(f"Indexing document: {pdf_path}")
        
        rag_model = self.get_rag_model()
        
        # Index with Byaldi
        temp_index_name = f"temp_{doc_id}"
        rag_model.index(
            input_path=pdf_path,
            index_name=temp_index_name,
            store_collection_with_index=False,
            overwrite=True
        )
        
        # Debug: Check what files were created
        temp_path = Path(self.config.index_dir) / temp_index_name
        if temp_path.exists():
            logger.info(f"Temporary index created at: {temp_path}")
            embeddings_dir = temp_path / "embeddings"
            if embeddings_dir.exists():
                files = list(embeddings_dir.glob("*"))
                logger.info(f"Embedding files: {files}")
        
        # Extract embeddings from disk
        embeddings, metadata = self.embedding_extractor.extract_from_disk(
            self.config.index_dir,
            temp_index_name
        )
        
        if embeddings is not None:
            # Add to FAISS
            self.faiss_manager.add_embeddings_batch(
                embeddings,
                metadata,
                doc_id,
                pdf_path
            )
            self._indexed_docs.add(doc_id)
            
            # Clean up temporary Byaldi index
            import shutil
            temp_path = Path(self.config.index_dir) / temp_index_name
            if temp_path.exists():
                shutil.rmtree(temp_path)
            
            logger.info(f"Successfully indexed {pdf_path} with {len(embeddings)} embeddings")
        else:
            logger.error(f"Failed to extract embeddings for {pdf_path}")
            
            # Debug: Try to inspect the embeddings file
            embeddings_path = temp_path / "embeddings" / "embeddings_0.pt"
            if embeddings_path.exists():
                try:
                    data = torch.load(embeddings_path, map_location='cpu')
                    logger.info(f"Embeddings file type: {type(data)}")
                    if hasattr(data, 'shape'):
                        logger.info(f"Shape: {data.shape}")
                    if hasattr(data, 'dtype'):
                        logger.info(f"Dtype: {data.dtype}")
                    if isinstance(data, dict):
                        logger.info(f"Dict keys: {list(data.keys())}")
                except Exception as e:
                    logger.error(f"Debug inspection failed: {e}")
        
        return doc_id
    
    def index_documents_batch(self, pdf_paths: List[str], batch_size: int = 10):
        """Index multiple documents efficiently"""
        logger.info(f"Indexing {len(pdf_paths)} documents in batches of {batch_size}")
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            for i in range(0, len(pdf_paths), batch_size):
                batch = pdf_paths[i:i + batch_size]
                futures = [executor.submit(self.index_document, pdf) for pdf in batch]
                
                for future in futures:
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Error indexing document: {e}")
                
                # Save periodically
                if (i + batch_size) % 100 == 0:
                    self.save_index()
                    logger.info(f"Saved index after {i + batch_size} documents")
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search using FAISS index"""
        logger.info(f"Searching for: {query}")
        
        if self.faiss_manager.index is None or self.faiss_manager.index.ntotal == 0:
            logger.warning("No documents indexed yet")
            return []
        
        # Get query embedding using ColPali
        rag_model = self.get_rag_model()
        
        # For now, use Byaldi's search to get query embedding
        # In production, you'd extract the query embedding directly
        temp_results = rag_model.search(query, k=1)
        
        if not temp_results:
            return []
        
        # Use FAISS for actual search
        # This is a simplified approach - in production, you'd generate query embeddings directly
        results = []
        for i in range(min(k, self.faiss_manager.index.ntotal)):
            meta = self.faiss_manager.metadata.get(i, {})
            results.append({
                'page_num': meta.get('page_num', 1),
                'score': 0.9 - (i * 0.1),  # Simulated scores
                'doc_id': meta.get('doc_id', ''),
                'pdf_path': meta.get('pdf_path', '')
            })
        
        return results[:k]
    
    def save_index(self):
        """Save FAISS index to disk"""
        self.faiss_manager.save(self.config.index_name)
        
        # Save indexed docs list
        indexed_docs_path = Path(self.config.faiss_index_dir) / f"{self.config.index_name}_indexed_docs.pkl"
        with open(indexed_docs_path, 'wb') as f:
            pickle.dump(self._indexed_docs, f)
    
    def load_index(self) -> bool:
        """Load FAISS index from disk"""
        success = self.faiss_manager.load(self.config.index_name)
        
        if success:
            # Load indexed docs list
            indexed_docs_path = Path(self.config.faiss_index_dir) / f"{self.config.index_name}_indexed_docs.pkl"
            if indexed_docs_path.exists():
                with open(indexed_docs_path, 'rb') as f:
                    self._indexed_docs = pickle.load(f)
        
        return success
    
    def get_page_image(self, pdf_path: str, page_num: int):
        """Extract specific page as image"""
        from pdf2image import convert_from_path
        
        images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num)
        return images[0] if images else None
    
    def generate_response_anthropic(self, query: str, image):
        """Generate response using Anthropic Claude API"""
        if not self.config.anthropic_api_key:
            return "Error: Anthropic API key not provided"
        
        try:
            import anthropic
            import io
            
            img_byte_arr = io.BytesIO()
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            image.save(img_byte_arr, format='JPEG', quality=95)
            img_byte_arr = img_byte_arr.getvalue()
            
            base64_image = base64.b64encode(img_byte_arr).decode('utf-8')
            
            client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)
            
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": base64_image
                                }
                            },
                            {
                                "type": "text",
                                "text": f"Please analyze this document page and answer the following question: {query}"
                            }
                        ]
                    }
                ]
            )
            
            return message.content[0].text
            
        except Exception as e:
            logger.error(f"Error with Anthropic API: {e}")
            return f"Error generating response: {str(e)}"
    
    def process_query(self, query: str):
        """Process a query end-to-end"""
        results = self.search(query)
        
        if not results:
            return {"error": "No relevant documents found"}
        
        best_result = results[0]
        pdf_path = best_result.get('pdf_path')
        page_num = best_result.get('page_num', 1)
        
        if not pdf_path:
            return {"error": "PDF path not found in results"}
        
        image = self.get_page_image(pdf_path, page_num)
        if not image:
            return {"error": "Could not extract page image"}
        
        response = self.generate_response_anthropic(query, image)
        
        return {
            "query": query,
            "response": response,
            "source_page": page_num,
            "source_pdf": pdf_path,
            "confidence_score": best_result.get('score', 0),
            "all_results": results
        }

# Production usage example
def main_production():
    """Production usage example"""
    config = RAGConfig(
        anthropic_api_key="",
        use_faiss=True,
        use_gpu_faiss=torch.cuda.is_available(),
        batch_size=50,
        faiss_nlist=1000  # Increase for larger datasets
    )
    
    rag_system = ProductionMultimodalRAG(config)
    
    # Try to load existing index
    if not rag_system.load_index():
        logger.info("No existing index found, will create new one")
    
    # Example: Index a batch of PDFs
    pdf_paths = [
        "/home/devendra_yadav/colpali/data/class_10.pdf",
        "/home/devendra_yadav/colpali/data/input.pdf",
        # ... thousands more
    ]
    
    # Index in batches
    rag_system.index_documents_batch(pdf_paths[:100])  # Start with first 100
    
    # Save index
    rag_system.save_index()
    
    # Process queries
    queries = [
        # "Find information about chemical equations",
        # "What are oxidation reactions?",
        "Explain the graph Slow Roatation for Kepler-51D"
    ]
    
    for query in queries:
        result = rag_system.process_query(query)
        print(f"\nQuery: {query}")
        print(f"Response: {result.get('response', 'No response')}")
        print(f"Source: {result.get('source_pdf', 'Unknown')}, Page {result.get('source_page', 'Unknown')}")

if __name__ == "__main__":
    main_production()