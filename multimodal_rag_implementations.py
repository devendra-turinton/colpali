import os
import sys
import torch
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Union
import base64
from dataclasses import dataclass
import logging
from PIL import Image

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
os.environ['anthropic_api_key'] = ""  # Add your Anthropic API key here
# Create cache directory if it doesn't exist
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

@dataclass
class RAGConfig:
    """Configuration for Multimodal RAG"""
    model_name: str = "vidore/colpali"
    index_name: str = "multimodal_rag"
    data_dir: str = "Data"
    use_flash_attention: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    anthropic_api_key: Optional[str] = ""  # Add your Anthropic API key here
    cache_dir: str = CACHE_DIR
    force_download: bool = False  # Set to True to force re-download

class MultimodalRAGSystem:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.rag_model = None
        self.vlm_model = None
        self.processor = None
        self.tokenizer = None
        self._setup_directories()
        
    def _setup_directories(self):
        """Create necessary directories"""
        Path(self.config.data_dir).mkdir(exist_ok=True)
        
    def check_model_cache(self, model_name: str) -> bool:
        """Check if model is already downloaded"""
        from huggingface_hub import snapshot_download, model_info
        
        try:
            # Check if model exists in cache
            model_path = Path(self.config.cache_dir) / "hub" / f"models--{model_name.replace('/', '--')}"
            
            if model_path.exists() and any(model_path.iterdir()):
                logger.info(f"Model found in cache: {model_path}")
                
                # List cached files
                cached_files = []
                for root, dirs, files in os.walk(model_path):
                    for file in files:
                        if file.endswith(('.safetensors', '.bin', '.json')):
                            cached_files.append(file)
                
                if cached_files:
                    logger.info(f"Cached files: {', '.join(cached_files[:5])}...")
                    return True
                    
        except Exception as e:
            logger.warning(f"Error checking cache: {e}")
            
        return False
    
    def initialize_models(self):
        """Initialize all required models"""
        logger.info("Initializing models...")
        
        # Check cache first
        if self.check_model_cache(self.config.model_name):
            logger.info("Using cached ColPali model")
        else:
            logger.info("Model not found in cache. Will download on first use.")
            logger.info("This may take 15-30 minutes depending on your connection.")
            logger.info(f"Models will be saved to: {self.config.cache_dir}")
        
        # Initialize ColPali RAG model
        try:
            from byaldi import RAGMultiModalModel
            
            # Configure Byaldi to use our cache directory
            os.environ['BYALDI_CACHE_DIR'] = self.config.cache_dir
            
            logger.info("Loading ColPali model...")
            # Byaldi doesn't support cache_dir parameter, it uses environment variables
            self.rag_model = RAGMultiModalModel.from_pretrained(
                self.config.model_name
            )
            logger.info("ColPali model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading ColPali model: {e}")
            raise
            
        # Initialize Vision Language Model (Qwen2-VL)
        try:
            self._initialize_vlm()
            logger.info("Vision Language Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading VLM: {e}")
            logger.info("Falling back to API mode")
            self._initialize_vlm_fallback()
    
    def _initialize_vlm(self):
        """Initialize Qwen2-VL model"""
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        except ImportError:
            # Fallback for older transformers versions
            logger.warning("Qwen2VL not available in this transformers version")
            self._initialize_vlm_fallback()
            return
        
        model_id = "Qwen/Qwen2-VL-7B-Instruct"
        
        # Check if VLM is cached
        if self.check_model_cache(model_id):
            logger.info("Using cached Qwen2-VL model")
        else:
            logger.info("Qwen2-VL not in cache. This is a 15GB+ download.")
            response = input("Download Qwen2-VL now? (y/n): ")
            if response.lower() != 'y':
                logger.info("Skipping VLM download. Will use Anthropic API for responses.")
                self._initialize_vlm_fallback()
                return
        
        # Model configuration
        model_kwargs = {
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            "device_map": "auto" if torch.cuda.is_available() else "cpu",
            "cache_dir": self.config.cache_dir,
        }
        
        # Add flash attention if available and requested
        if self.config.use_flash_attention:
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            except:
                logger.warning("Flash attention not available, using default attention")
        
        self.vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            **model_kwargs
        )
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=self.config.cache_dir
        )
        
    def _initialize_vlm_fallback(self):
        """Fallback to API model if main model fails"""
        logger.info("Using fallback VLM configuration (Anthropic API)")
        self.vlm_model = None
        self.processor = None
        
    def download_sample_pdf(self, url: str = "https://arxiv.org/pdf/2409.06697"):
        """Download sample PDF for testing"""
        import urllib.request
        
        output_path = Path(self.config.data_dir) / "input.pdf"
        if not output_path.exists():
            logger.info(f"Downloading PDF from {url}")
            urllib.request.urlretrieve(url, output_path)
            logger.info(f"PDF saved to {output_path}")
        return str(output_path)
        
    def index_document(self, pdf_path: str):
        """Index a PDF document"""
        logger.info(f"Indexing document: {pdf_path}")
        
        if not self.rag_model:
            raise ValueError("RAG model not initialized")
        
        # Check if poppler is installed
        try:
            from pdf2image import convert_from_path
            # Test poppler installation
            test_images = convert_from_path(pdf_path, last_page=1)
            logger.info("Poppler is installed and working")
        except Exception as e:
            logger.error(f"Poppler not found or not working: {e}")
            logger.error("Please install poppler-utils: sudo apt-get install poppler-utils")
            raise RuntimeError("Poppler is required for PDF processing. Please install it.")
        
        try:
            self.rag_model.index(
                input_path=pdf_path,
                index_name=self.config.index_name,
                store_collection_with_index=False,
                overwrite=True
            )
            logger.info("Document indexed successfully")
        except Exception as e:
            logger.error(f"Error indexing document: {e}")
            raise
        
    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for relevant pages"""
        logger.info(f"Searching for: {query}")
        
        if not self.rag_model:
            raise ValueError("RAG model not initialized")
            
        results = self.rag_model.search(query, k=k)
        logger.info(f"Found {len(results)} relevant pages")
        return results
        
    def get_page_image(self, pdf_path: str, page_num: int):
        """Extract specific page as image"""
        from pdf2image import convert_from_path
        
        images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num)
        return images[0] if images else None
        
    def generate_response_qwen(self, query: str, image, max_tokens: int = 256):
        """Generate response using Qwen2-VL"""
        if not self.vlm_model or not self.processor:
            logger.error("VLM not available, using Anthropic API instead")
            return self.generate_response_anthropic(query, image)
            
        try:
            # Try to import qwen_vl_utils, fallback if not available
            try:
                from qwen_vl_utils import process_vision_info
            except ImportError:
                logger.warning("qwen_vl_utils not available, using basic processing")
                # Basic processing without qwen_vl_utils
                import io
                from transformers import AutoTokenizer
                
                # Convert PIL image to format suitable for processor
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": query}
                    ]
                }]
                
                # Process without qwen_vl_utils
                text = self.processor.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                # Simple image processing
                inputs = self.processor(
                    text=text,
                    images=image,
                    return_tensors="pt"
                )
                
                if torch.cuda.is_available():
                    inputs = inputs.to("cuda")
                
                # Generate response
                with torch.no_grad():
                    generate_ids = self.vlm_model.generate(
                        **inputs,
                        max_new_tokens=max_tokens
                    )
                
                # Decode response
                output_text = self.processor.decode(
                    generate_ids[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
                
                return output_text
            
            # If qwen_vl_utils is available
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": query}
                ]
            }]
            
            # Process inputs
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")
            
            # Generate response
            with torch.no_grad():
                generate_ids = self.vlm_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens
                )
            
            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generate_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            return output_text[0]
            
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
            # Convert to RGB if necessary (Claude doesn't support RGBA)
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
                model="claude-3-5-sonnet-20241022",  # or "claude-3-opus-20240229" for more powerful model
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
    
    def process_query(self, query: str, pdf_path: str, use_anthropic: bool = True):
        """Complete pipeline to process a query"""
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
        if use_anthropic or not self.vlm_model:
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

def main():
    """Main execution function"""
    # Configuration
    config = RAGConfig(
        anthropic_api_key= "",  # Set your API key as environment variable
        use_flash_attention=False  # Set to True if you have flash attention installed
    )
    
    # Initialize system
    rag_system = MultimodalRAGSystem(config)
    
    try:
        # Initialize models
        rag_system.initialize_models()
        
        # Download sample PDF
        #pdf_path = rag_system.download_sample_pdf()
        pdf_path = "/home/devendra_yadav/colpali/Data/pnid.pdf"
        
        # Index the document
        rag_system.index_document(pdf_path)
        
        # Example queries
        queries = [
            "Give me the list of all instruments on red line?",
            
        ]
        
        # Process queries
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
                
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()