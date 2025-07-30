import os
import torch
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import base64
from dataclasses import dataclass
import logging
from PIL import Image, ImageDraw, ImageEnhance
import io
import numpy as np
from datetime import datetime
import time
import json
import re

# Computer Vision imports
import cv2
from pdf2image import convert_from_path

# Elasticsearch
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Model cache directory
CACHE_DIR = os.path.expanduser('~/.cache/huggingface')
os.environ['HF_HOME'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)


@dataclass
class ChunkInfo:
    """Information about a document chunk"""
    chunk_id: str
    document_id: str
    page_num: int
    chunk_num: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    chunk_type: str  # 'text', 'table', 'figure', 'header'
    image: Image.Image
    embedding: Optional[np.ndarray] = None
    text_preview: Optional[str] = None
    confidence: float = 0.0

class DocumentSegmenter:
    """Enhanced document segmentation using computer vision"""
    
    def __init__(self):
        self.morph_kernel_size = 5
        self.min_contour_area = 2000  # Increased for better chunks
        self.max_chunks_per_page = 15  # Reduced for quality
        
    def segment_page(self, page_image: Image.Image, page_num: int, doc_id: str) -> List[ChunkInfo]:
        """Segment a page into chunks using enhanced computer vision"""
        try:
            # Convert PIL to OpenCV format
            img_array = np.array(page_image)
            
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
                
            # Enhanced preprocessing
            gray = self._enhance_image_for_segmentation(gray)
            
            # Apply adaptive thresholding for better text detection
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.morph_kernel_size, self.morph_kernel_size))
            morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Dilate to connect nearby text with adaptive kernel
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 15))
            dilated = cv2.dilate(morph, dilate_kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter and process contours
            valid_contours = self._filter_contours(contours, page_image.width, page_image.height)
            
            # Sort by reading order (top to bottom, left to right)
            valid_contours.sort(key=lambda c: (c[1] // 50, c[0]))  # Group by approximate rows
            
            # Create chunks
            chunks = []
            for idx, (x, y, w, h, area) in enumerate(valid_contours[:self.max_chunks_per_page]):
                chunk_info = self._create_chunk(page_image, x, y, w, h, area, idx, page_num, doc_id)
                if chunk_info:
                    chunks.append(chunk_info)
                    
            logger.info(f"Segmented page {page_num} into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error segmenting page {page_num}: {e}")
            return []
    
    def _enhance_image_for_segmentation(self, gray: np.ndarray) -> np.ndarray:
        """Enhance image for better segmentation"""
        # Noise reduction
        denoised = cv2.medianBlur(gray, 3)
        
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        return enhanced
    
    def _filter_contours(self, contours: List, width: int, height: int) -> List[Tuple[int, int, int, int, float]]:
        """Filter and validate contours"""
        valid_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < self.min_contour_area:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter out contours that are too small or too large
            if w < 50 or h < 30:  # Too small
                continue
            if w > width * 0.95 or h > height * 0.95:  # Too large (likely full page)
                continue
                
            # Aspect ratio filtering
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < 0.1 or aspect_ratio > 20:  # Too thin or too wide
                continue
                
            valid_contours.append((x, y, w, h, area))
            
        return valid_contours
    
    def _create_chunk(self, page_image: Image.Image, x: int, y: int, w: int, h: int, 
                     area: float, idx: int, page_num: int, doc_id: str) -> Optional[ChunkInfo]:
        """Create a chunk with enhanced metadata"""
        try:
            # Add intelligent padding
            padding = self._calculate_padding(w, h)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(page_image.width, x + w + padding)
            y2 = min(page_image.height, y + h + padding)
            
            # Extract chunk image
            chunk_image = page_image.crop((x1, y1, x2, y2))
            
            # Determine chunk type with better heuristics
            chunk_type = self._determine_chunk_type(w, h, y, page_image.height, area)
            
            # Calculate confidence based on multiple factors
            confidence = self._calculate_confidence(w, h, area, page_image.width, page_image.height)
            
            # Generate text preview for better searchability
            text_preview = self._generate_text_preview(chunk_image)
            
            chunk_info = ChunkInfo(
                chunk_id=f"{doc_id}_p{page_num}_c{idx}",
                document_id=doc_id,
                page_num=page_num,
                chunk_num=idx,
                bbox=(x1, y1, x2, y2),
                chunk_type=chunk_type,
                image=chunk_image,
                confidence=confidence,
                text_preview=text_preview
            )
            
            return chunk_info
            
        except Exception as e:
            logger.error(f"Error creating chunk {idx}: {e}")
            return None
    
    def _calculate_padding(self, w: int, h: int) -> int:
        """Calculate adaptive padding based on chunk size"""
        base_padding = 15
        size_factor = min(w, h) / 200  # Normalize by typical text size
        return max(10, min(30, int(base_padding * size_factor)))
    
    def _determine_chunk_type(self, width: int, height: int, y_pos: int, page_height: int, area: float) -> str:
        """Enhanced chunk type determination"""
        aspect_ratio = width / height if height > 0 else 0
        position_ratio = y_pos / page_height if page_height > 0 else 0
        
        # Header detection
        if position_ratio < 0.15:  # Top 15% of page
            return "header"
        
        # Footer detection
        if position_ratio > 0.85:  # Bottom 15% of page
            return "footer"
        
        # Table detection (wide, rectangular)
        if aspect_ratio > 2.5 and area > 10000:
            return "table"
        
        # Figure detection (square-ish, medium to large)
        if 0.5 < aspect_ratio < 2.0 and area > 15000:
            return "figure"
        
        # List detection (narrow, tall)
        if aspect_ratio < 0.5 and height > 200:
            return "list"
        
        # Default to text
        return "text"
    
    def _calculate_confidence(self, w: int, h: int, area: float, page_w: int, page_h: int) -> float:
        """Calculate confidence score for chunk quality"""
        # Size factor (prefer medium-sized chunks)
        size_ratio = area / (page_w * page_h)
        size_score = 1.0 - abs(size_ratio - 0.1) * 5  # Optimal around 10% of page
        
        # Aspect ratio factor (prefer reasonable ratios)
        aspect_ratio = w / h if h > 0 else 0
        aspect_score = 1.0 if 0.2 < aspect_ratio < 5.0 else 0.5
        
        # Area factor
        area_score = min(area / 20000, 1.0)  # Normalize by reasonable area
        
        confidence = (size_score * 0.4 + aspect_score * 0.3 + area_score * 0.3)
        return max(0.1, min(1.0, confidence))
    
    def _generate_text_preview(self, chunk_image: Image.Image) -> str:
        """Generate text preview using OCR for better searchability"""
        try:
            import pytesseract
            
            # Enhance image for OCR
            enhanced = self._enhance_for_ocr(chunk_image)
            
            # Extract text
            text = pytesseract.image_to_string(enhanced, config='--psm 6')
            
            # Clean and truncate
            cleaned_text = ' '.join(text.split())
            logger.info(f"Extracted text preview: {cleaned_text[:200]}...")
            return cleaned_text[:200] if cleaned_text else ""
            
        except Exception as e:
            logger.debug(f"OCR preview generation failed: {e}")
            return ""
    
    def _enhance_for_ocr(self, image: Image.Image) -> Image.Image:
        """Enhance image for better OCR results"""
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)
        
        return image

class ElasticsearchManager:
    """Enhanced Elasticsearch manager with optimizations"""
    
    def __init__(self, host: str = "http://localhost:9200", 
                 username: str = "elastic", 
                 password: str = "315KJhzg_XxHzpPzLFKk"):
        
        self.client = Elasticsearch(
            host,
            basic_auth=(username, password),
            verify_certs=False,
            request_timeout=60,
            max_retries=3,
            retry_on_timeout=True,
            retry_on_status={502, 503, 504}
        )
        self.index_name = "multimodal_document_chunks"
        
        # Test connection
        try:
            info = self.client.info()
            logger.info(f"Connected to Elasticsearch: {info['cluster_name']}")
        except Exception as e:
            logger.error(f"Elasticsearch connection failed: {e}")
            raise
    
    def create_index(self, embedding_dim: int = 128):
        """Create optimized Elasticsearch index"""
        if self.client.indices.exists(index=self.index_name):
            logger.info(f"Index {self.index_name} already exists")
            return
            
        mappings = {
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "document_id": {"type": "keyword"},
                    "page_num": {"type": "integer"},
                    "chunk_num": {"type": "integer"},
                    "chunk_type": {"type": "keyword"},
                    "bbox": {"type": "integer"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": embedding_dim,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "text_preview": {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "confidence": {"type": "float"},
                    "timestamp": {"type": "date"},
                    "image_base64": {"type": "text", "index": False}  # Store for retrieval
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "index.knn": True,
                "index.refresh_interval": "1s"
            }
        }
        
        self.client.indices.create(index=self.index_name, body=mappings)
        logger.info(f"Created index: {self.index_name}")
    
    def index_chunks(self, chunks: List[ChunkInfo]):
        """Index chunks with image data"""
        actions = []
        
        for chunk in chunks:
            # Convert image to base64 for storage
            buffered = io.BytesIO()
            chunk.image.save(buffered, format='PNG')
            image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            doc = {
                "_index": self.index_name,
                "_id": chunk.chunk_id,
                "_source": {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "page_num": chunk.page_num,
                    "chunk_num": chunk.chunk_num,
                    "chunk_type": chunk.chunk_type,
                    "bbox": list(chunk.bbox),
                    "embedding": chunk.embedding.tolist() if chunk.embedding is not None else None,
                    "text_preview": chunk.text_preview or "",
                    "confidence": chunk.confidence,
                    "timestamp": datetime.now(),
                    "image_base64": image_base64
                }
            }
            actions.append(doc)
        
        if actions:
            try:
                success, failed = bulk(self.client, actions, chunk_size=100)
                logger.info(f"Indexed {success} chunks, {len(failed)} failed")
                
                if failed:
                    for failure in failed:
                        logger.error(f"Indexing failed: {failure}")
                        
            except Exception as e:
                logger.error(f"Bulk indexing failed: {e}")
    
    def search_chunks(self, query_embedding: np.ndarray, 
                     query_text: str = "", 
                     top_k: int = 10, 
                     min_score: float = 0.6,
                     chunk_types: List[str] = None) -> List[Dict]:
        """Enhanced hybrid search with filters"""
        
        # Build query
        query_parts = []
        
        # 1. Vector similarity search (primary)
        knn_query = {
            "field": "embedding",
            "query_vector": query_embedding.tolist(),
            "k": top_k * 3,
            "num_candidates": top_k * 5
        }
        
        # 2. Text search (secondary boost)
        text_query = None
        if query_text.strip():
            text_query = {
                "multi_match": {
                    "query": query_text,
                    "fields": ["text_preview^2", "chunk_type"],
                    "type": "best_fields",
                    "boost": 0.3
                }
            }
        
        # 3. Filters
        filters = []
        if chunk_types:
            filters.append({"terms": {"chunk_type": chunk_types}})
        
        # Combine queries
        search_query = {
            "knn": knn_query,
            "_source": {
                "excludes": ["image_base64"]  # Don't return image data in search
            },
            "size": top_k
        }
        
        # Add text query if available
        if text_query:
            search_query["query"] = {
                "bool": {
                    "should": [text_query],
                    "filter": filters if filters else []
                }
            }
        elif filters:
            search_query["query"] = {
                "bool": {
                    "filter": filters
                }
            }
        
        try:
            response = self.client.search(index=self.index_name, body=search_query)
            
            results = []
            for hit in response['hits']['hits']:
                if hit['_score'] >= min_score:
                    result = hit['_source']
                    result['score'] = hit['_score']
                    results.append(result)
                    
            logger.info(f"Found {len(results)} relevant chunks (min_score: {min_score})")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_chunk_image(self, chunk_id: str) -> Optional[Image.Image]:
        """Retrieve chunk image by ID"""
        try:
            response = self.client.get(index=self.index_name, id=chunk_id)
            
            if response['found']:
                image_base64 = response['_source']['image_base64']
                image_data = base64.b64decode(image_base64)
                return Image.open(io.BytesIO(image_data))
                
        except Exception as e:
            logger.error(f"Failed to retrieve chunk image {chunk_id}: {e}")
            
        return None

class ColPaliEmbedder:
    """ColPali embeddings implementation using direct model access"""
    
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.embedding_dim = 128
        
    def initialize(self):
        """Initialize ColPali model"""
        try:
            from colpali_engine.models import ColPali, ColPaliProcessor
            
            logger.info("Loading ColPali model...")
            
            # Load model
            self.model = ColPali.from_pretrained(
                "vidore/colpali-v1.2",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device
            ).eval()
            
            # Load processor
            self.processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")
            
            # ColPali v1.2 outputs 128-dimensional embeddings per patch
            self.embedding_dim = 128
            
            logger.info(f"ColPali model loaded successfully (embedding_dim: {self.embedding_dim})")
            
        except Exception as e:
            logger.error(f"ColPali initialization failed: {e}")
            raise
          
    def generate_embedding(self, image: Image.Image) -> np.ndarray:
        """Generate ColPali embedding for image"""
        if self.model is None:
            raise ValueError("ColPali model not initialized")
            
        try:
            with torch.no_grad():
                # Ensure RGB
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Process image
                batch_images = self.processor.process_images([image]).to(self.device)
                
                # Get embeddings
                embeddings = self.model(**batch_images)
                
                # ColPali returns embeddings for each patch
                # Average pool across patches to get single embedding
                image_embedding = embeddings.mean(dim=1).squeeze(0)
                
                return image_embedding.cpu().numpy().astype(np.float32)
                
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)

    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for query text"""
        if self.model is None:
            raise ValueError("ColPali model not initialized")
            
        try:
            with torch.no_grad():
                # Process query
                batch_queries = self.processor.process_queries([query]).to(self.device)
                
                # Get query embeddings
                embeddings = self.model(**batch_queries)
                
                # Average pool to get single embedding
                query_embedding = embeddings.mean(dim=1).squeeze(0)
                
                return query_embedding.cpu().numpy().astype(np.float32)
                
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)

class VisionLLM:
    """Enhanced Vision Language Model with better error handling"""
    
    def __init__(self, base_url: str = "https://6bgg1rhitot0zg-11434.proxy.runpod.net"):
        self.base_url = base_url
        self.model = "llama3.2-vision:11b"
        self.llm = None
        
    def initialize(self):
        """Initialize Vision Language Model"""
        try:
            from langchain_ollama import ChatOllama
            
            self.llm = ChatOllama(
                base_url=self.base_url,
                model=self.model,
                temperature=0.1,
                max_tokens=3000,
                timeout=300,  # 5 minutes
                request_timeout=300,
                num_predict=3000
            )
            logger.info("Vision Language Model initialized")
            
        except ImportError:
            logger.error("langchain_ollama not installed. Install with: pip install langchain-ollama")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Vision LLM: {e}")
            raise
            
    def generate_response(self, query: str, image: Image.Image, 
                         max_retries: int = 3, 
                         context: str = "") -> str:
        """Generate response with enhanced error handling"""
        
        if self.llm is None:
            return "Vision model not available"
            
        for attempt in range(max_retries):
            try:
                # Optimize image
                optimized_image = self._optimize_image(image)
                
                # Convert to base64
                buffered = io.BytesIO()
                optimized_image.save(buffered, format='JPEG', quality=85)
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                logger.info(f"Attempt {attempt + 1}: Image size: {len(img_base64)} bytes")
                
                # Create enhanced prompt
                prompt = self._create_prompt(query, context)
                
                # Create message
                from langchain_core.messages import HumanMessage
                
                message = HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url", 
                            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                        }
                    ]
                )
                
                logger.info(f"Sending request to vision model (attempt {attempt + 1})...")
                start_time = time.time()
                
                # Generate response
                response = self.llm.invoke([message])
                
                elapsed_time = time.time() - start_time
                logger.info(f"Received response in {elapsed_time:.2f} seconds")
                
                if response and hasattr(response, 'content') and response.content:
                    content = response.content.strip()
                    if len(content) > 10:  # Ensure meaningful response
                        return content
                    else:
                        raise ValueError("Response too short")
                else:
                    raise ValueError("Empty response")
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < max_retries - 1:
                    sleep_time = 2 ** attempt
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error("All attempts failed")
                    return f"Vision processing failed after {max_retries} attempts. Error: {str(e)}"
        
        return "Vision processing failed"
    
    def _optimize_image(self, image: Image.Image) -> Image.Image:
        """Optimize image for vision model processing"""
        
        # Resize if too large
        max_size = 1024
        if image.width > max_size or image.height > max_size:
            ratio = min(max_size/image.width, max_size/image.height)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.debug(f"Resized image to {new_size}")
        
        # Ensure RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Enhance for better recognition
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)
        
        return image
    
    def _create_prompt(self, query: str, context: str = "") -> str:
        """Create enhanced prompt for vision model"""
        
        base_prompt = f"""Analyze this document image and answer the following question: {query}

Instructions:
1. Look carefully at all text, tables, figures, and visual elements in the image
2. Provide a detailed and accurate response based on what you can see
3. If you can't find specific information, mention what you can see instead
4. Be specific and cite exact details from the image when possible

"""
        
        if context:
            base_prompt += f"Additional context: {context}\n\n"
        
        base_prompt += "Question: " + query
        
        return base_prompt

class EnhancedMultimodalRAG:
    """Main RAG system with enhanced capabilities"""
    
    def __init__(self, 
                 elasticsearch_host: str = "http://localhost:9200",
                 elasticsearch_user: str = "elastic",
                 elasticsearch_password: str = "315KJhzg_XxHzpPzLFKk",
                 vision_model_url: str = "https://6bgg1rhitot0zg-11434.proxy.runpod.net"):
        
        self.segmenter = DocumentSegmenter()
        logger.info("Document segmenter initialized")
        self.es_manager = ElasticsearchManager(elasticsearch_host, elasticsearch_user, elasticsearch_password)
        logger.info("Elasticsearch manager initialized")
        self.embedder = ColPaliEmbedder()
        logger.info("ColPali embedder initialized")
        self.vision_llm = VisionLLM(vision_model_url)
        
    def initialize(self):
        """Initialize all components"""
        logger.info("Initializing Enhanced Multimodal RAG System")
        
        # Initialize embeddings model
        self.embedder.initialize()
        
        # Create Elasticsearch index with correct dimensions
        self.es_manager.create_index(self.embedder.embedding_dim)
        
        # Initialize vision model
        self.vision_llm.initialize()
        
        logger.info("System initialized successfully")
        
    def process_document(self, pdf_path: str, doc_id: Optional[str] = None) -> List[ChunkInfo]:
        """Process document with enhanced pipeline"""
        
        if doc_id is None:
            doc_id = Path(pdf_path).stem
            
        logger.info(f"Processing document: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            # Convert PDF to images
            pages = convert_from_path(pdf_path, dpi=200)
            logger.info(f"Converted PDF to {len(pages)} pages")
            
            all_chunks = []
            
            for page_num, page_image in enumerate(pages, 1):
                logger.info(f"Processing page {page_num}/{len(pages)}")
                
                # Segment page
                chunks = self.segmenter.segment_page(page_image, page_num, doc_id)
                # Generate embeddings
                for chunk in chunks:
                    try:
                        chunk.embedding = self.embedder.generate_embedding(chunk.image)
                    except Exception as e:
                        logger.error(f"Failed to generate embedding for chunk {chunk.chunk_id}: {e}")
                        chunk.embedding = np.zeros(self.embedder.embedding_dim, dtype=np.float32)
                
                all_chunks.extend(chunks)
                
            # Index in Elasticsearch
            if all_chunks:
                self.es_manager.index_chunks(all_chunks)
                logger.info(f"Successfully processed {len(all_chunks)} chunks from {len(pages)} pages")
            else:
                logger.warning("No chunks were generated from the document")
                
            return all_chunks
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise
    
    def search_chunks(self, query: str, 
                     top_k: int = 10, 
                     min_score: float = 0.6,
                     chunk_types: List[str] = None) -> List[Dict]:
        """Search for relevant chunks"""
        
        logger.info(f"Searching for: '{query}'")
        
        try:
            # Generate query embedding
            query_embedding = self.embedder.generate_query_embedding(query)
            
            # Search in Elasticsearch
            results = self.es_manager.search_chunks(
                query_embedding=query_embedding,
                query_text=query,
                top_k=top_k,
                min_score=min_score,
                chunk_types=chunk_types
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def process_query(self, query: str, 
                     top_k: int = 10,
                     use_vision: bool = True,
                     chunk_types: List[str] = None) -> Dict[str, Any]:
        """Process complete query pipeline"""
        
        # Search for relevant chunks
        chunks = self.search_chunks(query, top_k=top_k, chunk_types=chunk_types)
        
        if not chunks:
            return {
                "query": query,
                "response": "No relevant chunks found for your query.",
                "chunks": [],
                "total_chunks": 0,
                "processing_time": 0
            }
        
        start_time = time.time()
        
        if use_vision:
            # Process with vision model
            response = self._process_with_vision(query, chunks)
        else:
            # Process with text only
            response = self._process_with_text(query, chunks)
        
        processing_time = time.time() - start_time
        
        return {
            "query": query,
            "response": response,
            "chunks": chunks,
            "total_chunks": len(chunks),
            "processing_time": processing_time
        }
    
    def _process_with_vision(self, query: str, chunks: List[Dict]) -> str:
        """Process query using vision model"""
        
        # Select best chunks (limit to avoid timeout)
        best_chunks = sorted(chunks, key=lambda x: x.get('score', 0), reverse=True)[:3]
        
        responses = []
        
        for i, chunk_data in enumerate(best_chunks):
            logger.info(f"Processing chunk {i+1}/{len(best_chunks)}")
            
            # Get chunk image
            chunk_image = self.es_manager.get_chunk_image(chunk_data['chunk_id'])
            
            if chunk_image:
                # Create context from text preview
                context = chunk_data.get('text_preview', '')
                
                # Generate response
                response = self.vision_llm.generate_response(query, chunk_image, context=context)
                
                if response and "failed" not in response.lower():
                    responses.append({
                        'chunk_id': chunk_data['chunk_id'],
                        'page': chunk_data['page_num'],
                        'type': chunk_data['chunk_type'],
                        'response': response,
                        'score': chunk_data.get('score', 0)
                    })
        
        if responses:
            return self._combine_responses(responses, query)
        else:
            return "Could not process the document chunks with the vision model."
    
    def _process_with_text(self, query: str, chunks: List[Dict]) -> str:
        """Process query using text previews only"""
        
        # Combine text previews
        context_parts = []
        for chunk in chunks[:5]:  # Limit context length
            text_preview = chunk.get('text_preview', '').strip()
            if text_preview:
                context_parts.append(f"Page {chunk['page_num']}: {text_preview}")
        
        if context_parts:
            context = "\n\n".join(context_parts)
            response = f"Based on the document content:\n\n{context}\n\nAnswer to '{query}':\n"
            response += "The relevant information can be found in the document sections above."
            return response
        else:
            return "No text content available for processing."
    
    def _combine_responses(self, responses: List[Dict], query: str) -> str:
        """Combine multiple chunk responses intelligently"""
        
        if len(responses) == 1:
            return responses[0]['response']
        
        # Sort by score
        responses.sort(key=lambda x: x['score'], reverse=True)
        
        combined = f"Based on analysis of {len(responses)} document sections:\n\n"
        
        for i, resp in enumerate(responses, 1):
            combined += f"**Section {i}** (Page {resp['page']}, {resp['type']}):\n"
            combined += f"{resp['response']}\n\n"
        
        return combined
    
    def get_document_stats(self, doc_id: str) -> Dict[str, Any]:
        """Get statistics about processed document"""
        
        try:
            # Search for all chunks of this document
            query = {
                "query": {"term": {"document_id": doc_id}},
                "size": 1000,
                "aggs": {
                    "pages": {"terms": {"field": "page_num"}},
                    "types": {"terms": {"field": "chunk_type"}},
                    "avg_confidence": {"avg": {"field": "confidence"}}
                }
            }
            
            response = self.es_manager.client.search(
                index=self.es_manager.index_name, 
                body=query
            )
            
            stats = {
                "document_id": doc_id,
                "total_chunks": response['hits']['total']['value'],
                "pages": len(response['aggregations']['pages']['buckets']),
                "chunk_types": {
                    bucket['key']: bucket['doc_count'] 
                    for bucket in response['aggregations']['types']['buckets']
                },
                "average_confidence": response['aggregations']['avg_confidence']['value']
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get document stats: {e}")
            return {"error": str(e)}

def main():
    """Main execution function with comprehensive examples"""
    
    # Configuration
    config = {
        "pdf_path": "/home/devendra_yadav/colpali/data/class_10.pdf",  # Update this path
        "elasticsearch_host": "http://localhost:9200",
        "elasticsearch_user": "elastic",
        "elasticsearch_password": "315KJhzg_XxHzpPzLFKk",
        "vision_model_url": "https://6bgg1rhitot0zg-11434.proxy.runpod.net"
    }
    
    # Initialize RAG system
    logger.info("Starting Enhanced Multimodal RAG System...")
    rag_system = EnhancedMultimodalRAG(
        elasticsearch_host=config["elasticsearch_host"],
        elasticsearch_user=config["elasticsearch_user"],
        elasticsearch_password=config["elasticsearch_password"],
        vision_model_url=config["vision_model_url"]
    )
    
    try:
        # Initialize system
        logger.info("Initializing Enhanced Multimodal RAG System...")
        rag_system.initialize()
        
        # Check if PDF exists
        pdf_path = config["pdf_path"]
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            logger.info("Please update the pdf_path in the config section")
            return
        
        # Process document
        doc_id = "sample_document"
        logger.info(f"Processing document: {pdf_path}")
        chunks = rag_system.process_document(pdf_path, doc_id)
        
        if not chunks:
            logger.error("No chunks were generated from the document")
            return
        
        # Get document statistics
        stats = rag_system.get_document_stats(doc_id)
        logger.info(f"Document stats: {json.dumps(stats, indent=2)}")
        
        # Example queries
        queries = [
            "What is Chemical Reaction?",
            # "What is the main topic of this document?",
            # "Can you summarize the key points?",
            # "Are there any tables or figures in the document?",
            # "What specific information is mentioned on the first page?",
            # "List any important dates or numbers mentioned"
        ]
        
        # Process queries
        for query in queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print(f"{'='*60}")
            
            # Process with vision
            result = rag_system.process_query(query, use_vision=True)
            
            print(f"\nResponse: {result['response']}")
            print(f"\nMetadata:")
            print(f"  - Total chunks found: {result['total_chunks']}")
            print(f"  - Processing time: {result['processing_time']:.2f} seconds")
            
            if result['chunks']:
                print(f"  - Top relevant chunks:")
                for i, chunk in enumerate(result['chunks'][:3], 1):
                    print(f"    {i}. Page {chunk['page_num']}, {chunk['chunk_type']} "
                          f"(score: {chunk.get('score', 0):.3f})")
            
            print(f"\n{'-'*60}")
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()