from concurrent.futures import ThreadPoolExecutor
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
import gc

# Fix tqdm multiprocessing issue
import multiprocessing
multiprocessing.set_start_method('spawn', force=True) if sys.platform != 'win32' else None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")
# Suppress tqdm warnings
import tqdm
tqdm.tqdm.monitor_interval = 0

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
    anthropic_api_key: Optional[str] = "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"
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

class ColPaliQueryEncoder:
    """Dedicated query encoder for ColPali"""
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        
    def load_model(self):
        """Load ColPali model components for query encoding"""
        if self.model is not None:
            return

        try:
            # Try using the model through Byaldi's internals
            from byaldi import RAGMultiModalModel
            
            # Create a temporary Byaldi model to access its components
            temp_model = RAGMultiModalModel.from_pretrained(self.model_name)
            
            if hasattr(temp_model, 'model') and hasattr(temp_model, 'processor'):
                self.model = temp_model.model
                self.processor = temp_model.processor
                logger.info("Successfully loaded model through Byaldi")
            else:
                raise AttributeError("Cannot access model components from Byaldi")
                
        except Exception as e2:
            logger.warning(f"Byaldi method failed: {e2}")
            
            # Final fallback: Use the correct model class for PaliGemma
            try:
                from transformers import AutoModelForCausalLM, AutoProcessor
                
                logger.info("Loading model using AutoModelForCausalLM...")
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
                    device_map="auto" if self.device == "cuda" else None
                )
                
                if self.device == "cpu":
                    self.model = self.model.to(self.device)
                
                self.model.eval()
                
                self.processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                
                logger.info("Model loaded successfully using AutoModelForCausalLM")
                
            except Exception as e3:
                logger.error(f"All loading methods failed: {e3}")
                raise RuntimeError("Failed to load ColPali model through any available method")        
    
    def encode_queries(self, queries: List[str], batch_size: int = 1) -> np.ndarray:
        """Encode text queries into embeddings"""
        self.load_model()
        
        if self.model is None or self.processor is None:
            raise RuntimeError("Model or processor not loaded properly")
        
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(queries), batch_size):
                batch_queries = queries[i:i + batch_size]
                
                try:
                    # Method 1: Try standard text encoding
                    batch_inputs = self.processor(
                        text=batch_queries,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=256  # Increased from 50
                    )
                    
                    # Move inputs to device
                    batch_inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                  for k, v in batch_inputs.items()}
                    
                    # Get embeddings based on model type
                    if hasattr(self.model, 'get_text_features'):
                        # CLIP-like interface
                        embeddings = self.model.get_text_features(**batch_inputs)
                    elif hasattr(self.model, 'encode_text'):
                        # Alternative encoding method
                        embeddings = self.model.encode_text(**batch_inputs)
                    else:
                        # Standard forward pass
                        outputs = self.model(**batch_inputs, output_hidden_states=True)
                        
                        # Extract embeddings from outputs
                        if hasattr(outputs, 'hidden_states'):
                            # Use last hidden state
                            embeddings = outputs.hidden_states[-1]
                        elif hasattr(outputs, 'last_hidden_state'):
                            embeddings = outputs.last_hidden_state
                        else:
                            # Try to get any tensor output
                            embeddings = outputs[0] if isinstance(outputs, tuple) else outputs
                    
                    # Handle different embedding shapes
                    if len(embeddings.shape) == 3:  # [batch, seq, dim]
                        # Use mean pooling over sequence
                        embeddings = embeddings.mean(dim=1)
                    elif len(embeddings.shape) == 1:  # Single embedding
                        embeddings = embeddings.unsqueeze(0)
                    
                except Exception as e:
                    logger.warning(f"Standard encoding failed: {e}, trying image-based encoding")
                    
                    # Method 2: Create dummy images with text for vision-language models
                    from PIL import Image, ImageDraw, ImageFont
                    import io
                    
                    dummy_images = []
                    for query in batch_queries:
                        # Create image with text
                        img = Image.new('RGB', (224, 224), color='white')
                        draw = ImageDraw.Draw(img)
                        
                        try:
                            font = ImageFont.load_default()
                        except:
                            font = None
                        
                        # Simple text wrapping
                        words = query.split()
                        lines = []
                        current_line = []
                        
                        for word in words:
                            current_line.append(word)
                            if len(' '.join(current_line)) > 30:
                                if len(current_line) > 1:
                                    lines.append(' '.join(current_line[:-1]))
                                    current_line = [word]
                                else:
                                    lines.append(' '.join(current_line))
                                    current_line = []
                        
                        if current_line:
                            lines.append(' '.join(current_line))
                        
                        # Draw text
                        y = 10
                        for line in lines[:6]:  # Max 6 lines
                            draw.text((10, y), line, fill='black', font=font)
                            y += 20
                        
                        dummy_images.append(img)
                    
                    # Process with images
                    batch_inputs = self.processor(
                        images=dummy_images,
                        text=batch_queries,
                        return_tensors="pt",
                        padding=True
                    )
                    
                    # Move to device
                    batch_inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                  for k, v in batch_inputs.items()}
                    
                    outputs = self.model(**batch_inputs, output_hidden_states=True)
                    
                    if hasattr(outputs, 'hidden_states'):
                        embeddings = outputs.hidden_states[-1].mean(dim=1)
                    else:
                        embeddings = outputs[0] if isinstance(outputs, tuple) else outputs
                        if len(embeddings.shape) == 3:
                            embeddings = embeddings.mean(dim=1)
                
                # Move to CPU and convert to numpy
                embeddings = embeddings.cpu().numpy()
                all_embeddings.append(embeddings)
        
        # Concatenate all batches
        query_embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
        
        # Normalize embeddings
        if query_embeddings.size > 0:
            norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
            query_embeddings = query_embeddings / (norms + 1e-8)
        
        return query_embeddings.astype(np.float32)

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
        if self.embedding_dim is None:
            self.embedding_dim = embeddings.shape[1]
        
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
            
            # Set embedding dimension from loaded index
            if hasattr(self.index, 'd'):
                self.embedding_dim = self.index.d
            elif hasattr(self.index, 'quantizer') and hasattr(self.index.quantizer, 'd'):
                self.embedding_dim = self.index.quantizer.d
            else:
                # Try to infer from index
                logger.warning("Could not determine embedding dimension from index")
            
            logger.info(f"Loaded FAISS index with dimension {self.embedding_dim}")
            
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
    
    def recreate_model(self, model_type: str, loader_func, *args, **kwargs):
        """Clear and recreate a model"""
        self.clear_model(model_type)
        return self.get_model(model_type, loader_func, *args, **kwargs)

class ProductionMultimodalRAG:
    """Production-ready Multimodal RAG system"""
    def __init__(self, config: RAGConfig):
        self.config = config
        self.model_manager = ModelManager()
        self.faiss_manager = ProductionFaissManager(config.faiss_index_dir, config)
        self.embedding_extractor = ByaldiEmbeddingExtractor()
        self._setup_directories()
        self._indexed_docs = set()  # Track indexed documents
        
        # Initialize query encoder with proper device handling
        device = "cpu"  # Default to CPU for query encoding
        if config.device == "cuda" and torch.cuda.is_available():
            # Check GPU memory for query encoder
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory >= 6:  # Need at least 6GB for query encoder
                device = "cuda"
            else:
                logger.warning(f"GPU has only {gpu_memory:.1f}GB. Using CPU for query encoder.")
        
        self.query_encoder = ColPaliQueryEncoder(config.model_name, device)
        self._query_embedding_cache = {}  # Cache for query embeddings
        
    def _setup_directories(self):
        """Create necessary directories"""
        Path(self.config.data_dir).mkdir(exist_ok=True)
        Path(self.config.index_dir).mkdir(exist_ok=True)
        Path(self.config.faiss_index_dir).mkdir(exist_ok=True)
    
    def _load_colpali_model(self):
        """Load ColPali model with CPU fallback for large models"""
        from byaldi import RAGMultiModalModel
        
        os.environ['BYALDI_CACHE_DIR'] = self.config.cache_dir
        
        # Force CPU if GPU memory is limited
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            if gpu_memory < 12:  # If GPU has less than 12GB
                logger.warning(f"GPU has only {gpu_memory:.1f}GB. Using CPU mode for stability.")
                os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA
        
        try:
            model = RAGMultiModalModel.from_pretrained(
                self.config.model_name,
                local_files_only=True
            )
            logger.info("ColPali model loaded from local cache")
        except:
            logger.info("Loading ColPali model from hub...")
            try:
                model = RAGMultiModalModel.from_pretrained(
                    self.config.model_name
                )
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise
        
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
        
        try:
            # Clear any existing Byaldi model to avoid state conflicts
            self.model_manager.clear_model('colpali')
            
            # Get a fresh RAG model instance
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
                
                # Clear the Byaldi model after successful indexing to free memory
                self.model_manager.clear_model('colpali')
                
            else:
                logger.error(f"Failed to extract embeddings for {pdf_path}")
                
        except Exception as e:
            logger.error(f"Error in index_document: {e}")
            # Clear the model on error as well
            self.model_manager.clear_model('colpali')
            raise
        finally:
            # Clean up any tqdm instances
            if hasattr(tqdm, '_instances'):
                tqdm.tqdm._instances.clear()
        
        return doc_id

    def index_documents_batch(self, pdf_paths: List[str], batch_size: int = 10):
        """Index multiple documents efficiently"""
        logger.info(f"Indexing {len(pdf_paths)} documents in batches of {batch_size}")
        
        with ThreadPoolExecutor(max_workers=1) as executor:
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
    
    def _generate_query_embeddings_v2(self, query: str) -> Optional[np.ndarray]:
        """Generate query embeddings using dedicated query encoder"""
        try:
            # Check cache first
            query_hash = hashlib.md5(query.encode()).hexdigest()
            if query_hash in self._query_embedding_cache:
                logger.info("Using cached query embeddings")
                return self._query_embedding_cache[query_hash]
            
            logger.info("Generating query embeddings with ColPali query encoder")
            
            # For ColPali, we need to generate multiple query variations
            # This mimics the multi-vector approach used in the paper
            query_variations = [
                query,  # Original query
                f"Find information about {query}",  # Augmented version
                f"Documents containing {query}",  # Another variation
            ]
            
            # Encode all variations
            query_embeddings = self.query_encoder.encode_queries(query_variations)
            
            # Ensure embedding dimension matches FAISS index
            if self.faiss_manager.embedding_dim and query_embeddings.shape[1] != self.faiss_manager.embedding_dim:
                logger.warning(f"Query embedding dim {query_embeddings.shape[1]} doesn't match index dim {self.faiss_manager.embedding_dim}")
                
                # If dimensions don't match, try to project or pad
                if query_embeddings.shape[1] > self.faiss_manager.embedding_dim:
                    # Truncate
                    query_embeddings = query_embeddings[:, :self.faiss_manager.embedding_dim]
                else:
                    # Pad with zeros
                    padding = np.zeros((query_embeddings.shape[0], 
                                       self.faiss_manager.embedding_dim - query_embeddings.shape[1]))
                    query_embeddings = np.hstack([query_embeddings, padding])
            
            # Cache the result
            self._query_embedding_cache[query_hash] = query_embeddings
            
            logger.info(f"Generated {query_embeddings.shape[0]} query embeddings with dimension {query_embeddings.shape[1]}")
            return query_embeddings
            
        except Exception as e:
            logger.error(f"Error generating query embeddings: {e}")
            import traceback
            traceback.print_exc()
            
            # Last resort fallback - return properly sized random embeddings
            if self.faiss_manager.embedding_dim:
                num_embeddings = 3  # Minimal number of query embeddings
                fallback = np.random.randn(num_embeddings, self.faiss_manager.embedding_dim).astype(np.float32)
                # Normalize
                for i in range(num_embeddings):
                    fallback[i] = fallback[i] / np.linalg.norm(fallback[i])
                logger.warning("Using random fallback embeddings")
                return fallback
            
            return None
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search using FAISS index with proper query embeddings"""
        logger.info(f"Searching for: {query}")
        
        if self.faiss_manager.index is None or self.faiss_manager.index.ntotal == 0:
            logger.warning("No documents indexed yet")
            return []
        
        # Ensure embedding dimension is set
        if self.faiss_manager.embedding_dim is None:
            logger.error("Embedding dimension not set in FAISS manager")
            return []
        
        # Generate query embeddings using the new method
        query_embeddings = self._generate_query_embeddings_v2(query)
        
        if query_embeddings is None or len(query_embeddings) == 0:
            logger.error("Failed to generate query embeddings")
            return []
        
        # Search FAISS
        results = []
        
        # Search for each query embedding
        all_scores = []
        all_indices = []
        
        for query_emb in query_embeddings:
            query_emb = query_emb.reshape(1, -1).astype(np.float32)
            
            # Ensure normalization
            norm = np.linalg.norm(query_emb)
            if norm > 0:
                query_emb = query_emb / norm
            
            # Search
            scores, indices = self.faiss_manager.index.search(query_emb, min(k * 3, self.faiss_manager.index.ntotal))
            all_scores.extend(scores[0])
            all_indices.extend(indices[0])
        
        # Aggregate results by page
        page_scores = {}
        for score, idx in zip(all_scores, all_indices):
            if idx != -1 and score > 0:  # Valid result with positive score
                meta = self.faiss_manager.metadata.get(int(idx), {})
                page_key = (meta.get('pdf_path', ''), meta.get('page_num', 0))
                
                if page_key not in page_scores:
                    page_scores[page_key] = {
                        'scores': [],
                        'meta': meta
                    }
                page_scores[page_key]['scores'].append(float(score))
        
        # Create result list
        for (pdf_path, page_num), data in page_scores.items():
            if pdf_path:  # Only include results with valid paths
                avg_score = np.mean(data['scores'])
                max_score = np.max(data['scores'])
                results.append({
                    'page_num': page_num,
                    'score': float(max_score),
                    'avg_score': float(avg_score),
                    'doc_id': data['meta'].get('doc_id', ''),
                    'pdf_path': pdf_path,
                    'num_matches': len(data['scores'])
                })
        
        # Sort by score (descending)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"Found {len(results)} unique pages, returning top {k}")
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
    
    def search_with_fallback(self, query: str, k: int = 5) -> List[Dict]:
        """Search with fallback to Byaldi if FAISS fails"""
        # Try FAISS search first
        results = self.search(query, k)
        
        if not results:
            logger.info("FAISS search failed, falling back to Byaldi search")
            try:
                # Get the current model
                rag_model = self.get_rag_model()
                
                # Store all results from all documents
                all_results = []
                
                # Try to search across all indexed documents
                if self.faiss_manager.doc_mapping:
                    logger.info(f"Searching across {len(self.faiss_manager.doc_mapping)} documents")
                    
                    # Use the current model's search
                    current_results = rag_model.search(query, k=k)
                    
                    for res in current_results:
                        # Check if this result might belong to this document
                        page_num = getattr(res, 'page_num', 1)
                        doc_idx = getattr(res, 'doc_id', 0)
                        
                        # Try to match the document
                        doc_keys = list(self.faiss_manager.doc_mapping.keys())
                        if doc_idx < len(doc_keys):
                            doc_id = doc_keys[doc_idx]
                            pdf_path = self.faiss_manager.doc_mapping[doc_id]
                        else:
                            # Default to first document if mapping fails
                            doc_id = doc_keys[0] if doc_keys else ""
                            pdf_path = list(self.faiss_manager.doc_mapping.values())[0] if self.faiss_manager.doc_mapping else ""
                        
                        results.append({
                            'page_num': page_num,
                            'score': float(getattr(res, 'score', 0.0)),
                            'doc_id': doc_id,
                            'pdf_path': pdf_path,
                            'num_matches': 1,
                            'search_type': 'byaldi_fallback'
                        })
                        
                        logger.info(f"Result: doc={doc_id}, page={page_num}, score={getattr(res, 'score', 0.0):.3f}")
                
            except Exception as e:
                logger.error(f"Error in fallback search: {e}")
                import traceback
                traceback.print_exc()
        
        return results

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
    
    def get_page_image(self, pdf_path: str, page_num: int):
        """Extract specific page as image"""
        from pdf2image import convert_from_path
        
        images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num)
        return images[0] if images else None
    
    def process_query(self, query: str):
        """Process a query end-to-end using FAISS"""
        logger.info(f"Processing query: {query}")
        
        # Use FAISS search directly
        results = self.search(query)
        
        if not results:
            logger.warning("FAISS search returned no results, trying fallback")
            # Only use fallback if FAISS truly fails
            results = self.search_with_fallback(query)
            
            if not results:
                return {"error": "No relevant documents found"}
        
        best_result = results[0]
        pdf_path = best_result.get('pdf_path')
        page_num = best_result.get('page_num', 1)
        
        if not pdf_path or not os.path.exists(pdf_path):
            return {"error": f"PDF path not found or doesn't exist: {pdf_path}"}
        
        logger.info(f"Best match: {pdf_path}, Page {page_num}, Score: {best_result.get('score', 0):.4f}")
        
        try:
            image = self.get_page_image(pdf_path, page_num)
            if not image:
                return {"error": f"Could not extract page {page_num} from {pdf_path}"}
            
            response = self.generate_response_anthropic(query, image)
            
            return {
                "query": query,
                "response": response,
                "source_page": page_num,
                "source_pdf": pdf_path,
                "confidence_score": best_result.get('score', 0),
                "search_method": "faiss",
                "all_results": results[:5]  # Return top 5 results
            }
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Error processing query: {str(e)}"}

# Production usage example
def main_production():
    """Production usage example"""
    config = RAGConfig(
        anthropic_api_key="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",  # Replace with your actual API key
        use_faiss=True,
        use_gpu_faiss=torch.cuda.is_available(),
        batch_size=50,
        faiss_nlist=1000  # Increase for larger datasets
    )
    
    logger.info("Starting production RAG system with configuration:")
    logger.info(f"Config: {config}")
    rag_system = ProductionMultimodalRAG(config)
    
    # Try to load existing index
    if not rag_system.load_index():
        logger.info("No existing index found, will create new one")
    
    # Example: Index PDFs
    pdf_paths = [
        "/home/devendra_yadav/colpali/data/class_10.pdf",
        "/home/devendra_yadav/colpali/data/input.pdf",
    ]
    
    # Index documents
    for pdf_path in pdf_paths:
        if os.path.exists(pdf_path):
            rag_system.index_document(pdf_path)
    
    logger.info(f"Indexed {len(rag_system._indexed_docs)} documents")
    
    # Save index
    rag_system.save_index()
    logger.info("Index saved successfully")
    logger.info("RAG system is ready for queries")
    
    # Process queries
    queries = [
        "What is chemical reactions?",
        # "What do you understand by balancing chemical equations?",
        # "Explain in details about the Kepler-51d's JWSTlight curve?"
    ]
    
    for query in queries:
        result = rag_system.process_query(query)
        print(f"\nQuery: {query}")
        print(f"Response: {result.get('response', 'No response')}")
        print(f"Source: {result.get('source_pdf', 'Unknown')}, Page {result.get('source_page', 'Unknown')}")
        print(f"Score: {result.get('confidence_score', 0):.3f}")
        print("-" * 80)

if __name__ == "__main__":
    main_production()