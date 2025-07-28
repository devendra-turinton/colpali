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
import pickle
import hashlib
from datetime import datetime

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
    index_dir: str = ".byaldi"  # Directory for storing indexes
    use_flash_attention: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    anthropic_api_key: Optional[str] = ""
    cache_dir: str = CACHE_DIR
    force_download: bool = False
    use_local_vlm: bool = False  # Set to True to use Qwen2-VL instead of API
    persistent_session: bool = True  # Keep models in memory

class ModelManager:
    """Singleton class to manage model instances"""
    _instance = None
    _models = {}
    _initialized = False
    
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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def clear_all(self):
        """Clear all models from memory"""
        self._models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class IndexManager:
    """Manage document indexes to avoid rebuilding"""
    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)
        self.metadata_file = self.index_dir / "index_metadata.pkl"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load index metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return {}
        return {}
    
    def _save_metadata(self):
        """Save index metadata"""
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def get_document_hash(self, pdf_path: str) -> str:
        """Generate hash for document"""
        stat = os.stat(pdf_path)
        content = f"{pdf_path}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def is_indexed(self, pdf_path: str, index_name: str) -> bool:
        """Check if document is already indexed"""
        doc_hash = self.get_document_hash(pdf_path)
        return (index_name in self.metadata and 
                self.metadata[index_name].get('doc_hash') == doc_hash and
                (self.index_dir / index_name).exists())
    
    def mark_indexed(self, pdf_path: str, index_name: str):
        """Mark document as indexed"""
        doc_hash = self.get_document_hash(pdf_path)
        self.metadata[index_name] = {
            'doc_hash': doc_hash,
            'pdf_path': pdf_path,
            'indexed_at': datetime.now().isoformat()
        }
        self._save_metadata()

class MultimodalRAGSystem:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.model_manager = ModelManager()
        self.index_manager = IndexManager(config.index_dir)
        self._setup_directories()
        self._current_pdf_path = None
        
    def _setup_directories(self):
        """Create necessary directories"""
        Path(self.config.data_dir).mkdir(exist_ok=True)
        Path(self.config.index_dir).mkdir(exist_ok=True)
    
    def _load_colpali_model(self):
        """Load ColPali model"""
        from byaldi import RAGMultiModalModel
        
        # Configure Byaldi to use our cache directory
        os.environ['BYALDI_CACHE_DIR'] = self.config.cache_dir
        
        # Try to load with local_files_only first
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
    
    def _load_vlm_model(self):
        """Load Qwen2-VL model"""
        if not self.config.use_local_vlm:
            logger.info("Skipping local VLM loading, will use API")
            return None, None
            
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        except ImportError:
            logger.warning("Qwen2VL not available in this transformers version")
            return None, None
        
        model_id = "Qwen/Qwen2-VL-7B-Instruct"
        
        # Model configuration
        model_kwargs = {
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            "device_map": "auto" if torch.cuda.is_available() else "cpu",
            "cache_dir": self.config.cache_dir,
            "local_files_only": True  # Try local first
        }
        
        # Add flash attention if available and requested
        if self.config.use_flash_attention:
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            except:
                logger.warning("Flash attention not available")
        
        try:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                **model_kwargs
            )
            processor = AutoProcessor.from_pretrained(
                model_id,
                cache_dir=self.config.cache_dir,
                local_files_only=True
            )
            return model, processor
        except:
            logger.info("Local VLM not found, downloading...")
            model_kwargs.pop("local_files_only", None)
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                **model_kwargs
            )
            processor = AutoProcessor.from_pretrained(
                model_id,
                cache_dir=self.config.cache_dir
            )
            return model, processor
    
    def get_rag_model(self):
        """Get or create RAG model instance"""
        return self.model_manager.get_model(
            'colpali',
            self._load_colpali_model
        )
    
    def get_vlm_model(self) -> Tuple[Optional[any], Optional[any]]:
        """Get or create VLM model instance"""
        if not self.config.use_local_vlm:
            return None, None
            
        vlm = self.model_manager.get_model('vlm')
        processor = self.model_manager.get_model('processor')
        
        if vlm is None and self.config.use_local_vlm:
            vlm, processor = self._load_vlm_model()
            if vlm is not None:
                self.model_manager._models['vlm'] = vlm
                self.model_manager._models['processor'] = processor
        
        return vlm, processor
    
    def initialize_models(self, force_reload: bool = False):
        """Initialize all required models"""
        logger.info("Initializing models...")
        
        if force_reload:
            self.model_manager.clear_all()
        
        # Load ColPali model
        try:
            rag_model = self.get_rag_model()
            logger.info("ColPali model ready")
        except Exception as e:
            logger.error(f"Error loading ColPali model: {e}")
            raise
        
        # Load VLM if requested
        if self.config.use_local_vlm:
            try:
                vlm, processor = self.get_vlm_model()
                if vlm:
                    logger.info("Vision Language Model ready")
                else:
                    logger.info("Using API mode for VLM")
            except Exception as e:
                logger.error(f"Error loading VLM: {e}")
                logger.info("Will use API mode")
    
    def index_document(self, pdf_path: str, force_reindex: bool = False):
        """Index a PDF document with caching"""
        logger.info(f"Checking index for document: {pdf_path}")
        
        # Check if already indexed
        if not force_reindex and self.index_manager.is_indexed(pdf_path, self.config.index_name):
            logger.info("Document already indexed, skipping indexing")
            self._current_pdf_path = pdf_path
            return
        
        logger.info("Indexing document...")
        
        rag_model = self.get_rag_model()
        if not rag_model:
            raise ValueError("RAG model not initialized")
        
        # Check if poppler is installed
        try:
            from pdf2image import convert_from_path
            test_images = convert_from_path(pdf_path, last_page=1)
            logger.info("Poppler is installed and working")
        except Exception as e:
            logger.error(f"Poppler not found or not working: {e}")
            raise RuntimeError("Poppler is required for PDF processing")
        
        try:
            rag_model.index(
                input_path=pdf_path,
                index_name=self.config.index_name,
                store_collection_with_index=False,
                overwrite=True
            )
            logger.info("Document indexed successfully")
            
            # Mark as indexed
            self.index_manager.mark_indexed(pdf_path, self.config.index_name)
            self._current_pdf_path = pdf_path
            
        except Exception as e:
            logger.error(f"Error indexing document: {e}")
            raise
    
    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for relevant pages"""
        logger.info(f"Searching for: {query}")
        
        rag_model = self.get_rag_model()
        if not rag_model:
            raise ValueError("RAG model not initialized")
        
        results = rag_model.search(query, k=k)
        logger.info(f"Found {len(results)} relevant pages")
        return results
    
    def get_page_image(self, pdf_path: str, page_num: int):
        """Extract specific page as image"""
        from pdf2image import convert_from_path
        
        images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num)
        return images[0] if images else None
    
    def generate_response_qwen(self, query: str, image, max_tokens: int = 256):
        """Generate response using Qwen2-VL"""
        vlm_model, processor = self.get_vlm_model()
        
        if not vlm_model or not processor:
            logger.info("VLM not available, using Anthropic API")
            return self.generate_response_anthropic(query, image)
        
        try:
            # Similar to your original implementation
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": query}
                ]
            }]
            
            text = processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = processor(
                text=text,
                images=image,
                return_tensors="pt"
            )
            
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")
            
            with torch.no_grad():
                generate_ids = vlm_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens
                )
            
            output_text = processor.decode(
                generate_ids[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            return output_text
            
        except Exception as e:
            logger.error(f"Error with Qwen2-VL: {e}")
            return self.generate_response_anthropic(query, image)
    
    def generate_response_anthropic(self, query: str, image):
        """Generate response using Anthropic Claude API"""
        if not self.config.anthropic_api_key:
            return "Error: Anthropic API key not provided and local VLM not available"
        
        try:
            import anthropic
            import io
            
            # Convert PIL image to bytes
            img_byte_arr = io.BytesIO()
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            image.save(img_byte_arr, format='JPEG', quality=95)
            img_byte_arr = img_byte_arr.getvalue()
            
            # Encode image to base64
            base64_image = base64.b64encode(img_byte_arr).decode('utf-8')
            
            # Initialize Anthropic client
            client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)
            
            # Create message with image
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
    
    def process_query(self, query: str, pdf_path: str = None, use_anthropic: bool = True):
        """Complete pipeline to process a query"""
        # Use stored PDF path if not provided
        if pdf_path is None:
            pdf_path = self._current_pdf_path
        
        if pdf_path is None:
            return "Error: No PDF document loaded"
        
        # Search for relevant pages
        results = self.search(query)
        
        if not results:
            return "No relevant pages found for your query."
        
        # Get the most relevant page
        best_result = results[0]
        page_num = best_result['page_num']
        score = best_result['score']
        
        logger.info(f"Best match: Page {page_num} with score {score}")
        
        # Extract page image
        image = self.get_page_image(pdf_path, page_num)
        
        if not image:
            return "Error extracting page image"
        
        # Generate response
        if use_anthropic or not self.config.use_local_vlm:
            response = self.generate_response_anthropic(query, image)
        else:
            response = self.generate_response_qwen(query, image)
        
        return {
            "query": query,
            "response": response,
            "source_page": page_num,
            "confidence_score": score,
            "all_results": results
        }
    
    def cleanup(self, keep_colpali: bool = True):
        """Clean up models from memory"""
        if not keep_colpali:
            self.model_manager.clear_all()
        else:
            # Keep ColPali but clear VLM to save memory
            self.model_manager.clear_model('vlm')
            self.model_manager.clear_model('processor')

# Convenience functions for easy usage
_global_rag_system = None

def get_rag_system(config: RAGConfig = None) -> MultimodalRAGSystem:
    """Get or create global RAG system instance"""
    global _global_rag_system
    
    if _global_rag_system is None:
        if config is None:
            config = RAGConfig()
        _global_rag_system = MultimodalRAGSystem(config)
        _global_rag_system.initialize_models()
    
    return _global_rag_system

def process_pdf_query(query: str, pdf_path: str = None, api_key: str = None):
    """Simple interface to process queries"""
    config = RAGConfig(
        anthropic_api_key=api_key,
        use_local_vlm=False  # Use API by default for faster responses
    )
    
    rag = get_rag_system(config)
    
    # Index document if needed
    if pdf_path and pdf_path != rag._current_pdf_path:
        rag.index_document(pdf_path)
    
    # Process query
    return rag.process_query(query, pdf_path)

# Example usage
def main():
    """Main execution function"""
    # Configuration
    config = RAGConfig(
        anthropic_api_key="",  # Replace with your key
        use_flash_attention=False,
        use_local_vlm=False,  # Set to True to use local Qwen2-VL
        persistent_session=True
    )
    
    # Get persistent RAG system
    rag_system = get_rag_system(config)
    
    try:
        # Your PDF path
        pdf_path = "/home/devendra_yadav/colpali/data/class_10.pdf"
        
        # Index the document (only happens once)
        rag_system.index_document(pdf_path)
        
        # Example queries - models stay in memory between queries
        queries = [
            "Explain in details about the balancing a chemical equations?",
            "What are the different types of chemical reactions?",
            "Give examples of oxidation reactions"
        ]
        
        # Process queries efficiently
        for query in queries:
            print(f"\n{'='*50}")
            print(f"Query: {query}")
            print(f"{'='*50}")
            
            result = rag_system.process_query(query, pdf_path, use_anthropic=True)
            
            if isinstance(result, dict):
                print(f"Response: {result['response']}")
                print(f"Source: Page {result['source_page']} (Score: {result['confidence_score']:.2f})")
            else:
                print(f"Response: {result}")
        
        # Optional: Clean up VLM memory but keep ColPali loaded
        rag_system.cleanup(keep_colpali=True)
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
   