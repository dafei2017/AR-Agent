#!/usr/bin/env python3
"""
Medical Image Analyzer using LLaVA-NeXT-Med
Core module for medical image analysis and interpretation
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoProcessor,
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration
)
from PIL import Image
import base64
import io
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import json
import re
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalImageAnalyzer:
    """
    Medical Image Analyzer using LLaVA-NeXT-Med for medical image understanding
    """
    
    def __init__(self, 
                 model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf",
                 device: str = "auto",
                 load_in_4bit: bool = True,
                 max_new_tokens: int = 512):
        """
        Initialize the medical image analyzer
        
        Args:
            model_name: HuggingFace model name for LLaVA-NeXT
            device: Device to run the model on ('auto', 'cuda', 'cpu')
            load_in_4bit: Whether to use 4-bit quantization
            max_new_tokens: Maximum number of tokens to generate
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = self._setup_device(device)
        self.load_in_4bit = load_in_4bit
        
        # Model components
        self.model = None
        self.processor = None
        self.tokenizer = None
        
        # Medical prompts and templates
        self.medical_prompts = self._load_medical_prompts()
        
        # Analysis history
        self.analysis_history = []
        
        logger.info(f"Initializing MedicalImageAnalyzer with model: {model_name}")
        logger.info(f"Device: {self.device}, 4-bit loading: {load_in_4bit}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup the appropriate device for model inference"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"CUDA available, using GPU: {torch.cuda.get_device_name()}")
            else:
                device = "cpu"
                logger.info("CUDA not available, using CPU")
        
        return torch.device(device)
    
    def _load_medical_prompts(self) -> Dict[str, str]:
        """Load medical-specific prompts for different analysis types"""
        return {
            "general": (
                "You are a medical AI assistant specialized in analyzing medical images. "
                "Provide a detailed, professional analysis of this medical image. "
                "Include: 1) Description of visible structures, 2) Notable findings, "
                "3) Potential abnormalities, 4) Clinical significance. "
                "Be precise and use appropriate medical terminology."
            ),
            "radiology": (
                "As a radiologist AI, analyze this medical imaging study. "
                "Describe the imaging modality, anatomical structures visible, "
                "any pathological findings, and provide a structured report "
                "with impression and recommendations."
            ),
            "pathology": (
                "Analyze this pathology image as a pathologist would. "
                "Describe the tissue type, cellular morphology, "
                "any abnormal findings, and suggest potential diagnoses "
                "based on the histological features observed."
            ),
            "dermatology": (
                "Examine this dermatological image. Describe the skin lesion "
                "characteristics including size, color, texture, borders, "
                "and any concerning features. Suggest differential diagnoses "
                "and recommend next steps."
            ),
            "ophthalmology": (
                "Analyze this ophthalmological image. Describe the retinal "
                "or anterior segment findings, identify any pathological "
                "changes, and provide clinical interpretation relevant "
                "to eye health and vision."
            )
        }
    
    def load_model(self) -> bool:
        """Load the LLaVA-NeXT model and processor"""
        try:
            logger.info("Loading LLaVA-NeXT model...")
            
            # Configure model loading parameters
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device.type == "cuda" else torch.float32,
                "device_map": "auto" if self.device.type == "cuda" else None,
            }
            
            if self.load_in_4bit and self.device.type == "cuda":
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model_kwargs["quantization_config"] = quantization_config
            
            # Load processor and model
            self.processor = LlavaNextProcessor.from_pretrained(self.model_name)
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            if not self.load_in_4bit and self.device.type != "cuda":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def preprocess_image(self, image_input) -> Image.Image:
        """Preprocess image input (base64, PIL, or file path)"""
        if isinstance(image_input, str):
            if image_input.startswith('data:image') or len(image_input) > 100:
                # Base64 encoded image
                if 'base64,' in image_input:
                    image_data = image_input.split('base64,')[1]
                else:
                    image_data = image_input
                
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            else:
                # File path
                image = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise ValueError("Unsupported image input type")
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    
    def analyze_image(self, 
                     image_input,
                     prompt: Optional[str] = None,
                     analysis_type: str = "general",
                     custom_instructions: Optional[str] = None) -> Dict[str, Any]:
        """Analyze a medical image and return structured results"""
        
        if self.model is None:
            if not self.load_model():
                return {"error": "Failed to load model"}
        
        try:
            # Preprocess image
            image = self.preprocess_image(image_input)
            
            # Prepare prompt
            if prompt is None:
                prompt = self.medical_prompts.get(analysis_type, self.medical_prompts["general"])
            
            if custom_instructions:
                prompt += f" Additional instructions: {custom_instructions}"
            
            # Format conversation
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image"},
                    ],
                },
            ]
            
            # Apply chat template
            prompt_text = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
            
            # Process inputs
            inputs = self.processor(
                prompt_text, 
                image, 
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode response
            generated_text = self.processor.decode(
                output[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            # Parse and structure the response
            analysis_result = self._parse_analysis_response(
                generated_text, analysis_type
            )
            
            # Add metadata
            analysis_result.update({
                "timestamp": datetime.now().isoformat(),
                "model_name": self.model_name,
                "analysis_type": analysis_type,
                "image_size": image.size
            })
            
            # Store in history
            self.analysis_history.append(analysis_result)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _parse_analysis_response(self, response_text: str, analysis_type: str) -> Dict[str, Any]:
        """Parse the model response into structured format"""
        result = {
            "description": response_text.strip(),
            "findings": [],
            "recommendations": None,
            "confidence": None,
            "structured_report": {}
        }
        
        # Extract structured information using regex patterns
        patterns = {
            "findings": r"(?:findings?|observations?):\s*([^\n]+(?:\n[^\n]+)*)",
            "impression": r"(?:impression|conclusion):\s*([^\n]+(?:\n[^\n]+)*)",
            "recommendations": r"(?:recommendations?|suggestions?):\s*([^\n]+(?:\n[^\n]+)*)",
            "diagnosis": r"(?:diagnosis|differential):\s*([^\n]+(?:\n[^\n]+)*)"
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE)
            if match:
                content = match.group(1).strip()
                if key == "findings":
                    # Split findings into list
                    result["findings"] = [
                        f.strip() for f in re.split(r'[\n•-]', content) 
                        if f.strip()
                    ]
                else:
                    result["structured_report"][key] = content
        
        # Extract recommendations if found
        if "recommendations" in result["structured_report"]:
            result["recommendations"] = result["structured_report"]["recommendations"]
        
        # Estimate confidence based on language certainty
        confidence_indicators = {
            "high": ["clearly", "definitely", "obvious", "evident", "certain"],
            "medium": ["likely", "probable", "suggests", "indicates", "appears"],
            "low": ["possible", "may", "might", "could", "uncertain", "unclear"]
        }
        
        text_lower = response_text.lower()
        confidence_score = 0.5  # Default medium confidence
        
        for level, indicators in confidence_indicators.items():
            count = sum(1 for indicator in indicators if indicator in text_lower)
            if count > 0:
                if level == "high":
                    confidence_score = min(0.9, confidence_score + count * 0.1)
                elif level == "low":
                    confidence_score = max(0.1, confidence_score - count * 0.1)
        
        result["confidence"] = confidence_score
        
        return result
    
    def batch_analyze(self, 
                     image_list: List,
                     analysis_type: str = "general",
                     custom_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
        """Analyze multiple images in batch"""
        results = []
        
        for i, image_input in enumerate(image_list):
            logger.info(f"Analyzing image {i+1}/{len(image_list)}")
            result = self.analyze_image(
                image_input, 
                prompt=custom_prompt,
                analysis_type=analysis_type
            )
            results.append(result)
        
        return results
    
    def get_analysis_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get analysis history"""
        if limit:
            return self.analysis_history[-limit:]
        return self.analysis_history
    
    def clear_history(self):
        """Clear analysis history"""
        self.analysis_history.clear()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "load_in_4bit": self.load_in_4bit,
            "max_new_tokens": self.max_new_tokens,
            "model_loaded": self.model is not None,
            "available_analysis_types": list(self.medical_prompts.keys())
        }
    
    def export_analysis(self, 
                       analysis_result: Dict[str, Any], 
                       format: str = "json") -> str:
        """Export analysis result in specified format"""
        if format == "json":
            return json.dumps(analysis_result, indent=2, ensure_ascii=False)
        elif format == "text":
            return self._format_text_report(analysis_result)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _format_text_report(self, analysis_result: Dict[str, Any]) -> str:
        """Format analysis result as text report"""
        report = []
        report.append("MEDICAL IMAGE ANALYSIS REPORT")
        report.append("=" * 40)
        report.append(f"Timestamp: {analysis_result.get('timestamp', 'N/A')}")
        report.append(f"Model: {analysis_result.get('model_name', 'N/A')}")
        report.append(f"Analysis Type: {analysis_result.get('analysis_type', 'N/A')}")
        report.append("")
        
        report.append("DESCRIPTION:")
        report.append(analysis_result.get('description', 'No description available'))
        report.append("")
        
        if analysis_result.get('findings'):
            report.append("KEY FINDINGS:")
            for finding in analysis_result['findings']:
                report.append(f"• {finding}")
            report.append("")
        
        if analysis_result.get('recommendations'):
            report.append("RECOMMENDATIONS:")
            report.append(analysis_result['recommendations'])
            report.append("")
        
        if analysis_result.get('confidence'):
            confidence_pct = int(analysis_result['confidence'] * 100)
            report.append(f"CONFIDENCE: {confidence_pct}%")
        
        return "\n".join(report)

# Utility functions
def create_analyzer(model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf",
                   **kwargs) -> MedicalImageAnalyzer:
    """Factory function to create a medical image analyzer"""
    return MedicalImageAnalyzer(model_name=model_name, **kwargs)

def analyze_medical_image(image_input, 
                         model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf",
                         **kwargs) -> Dict[str, Any]:
    """Quick function to analyze a single medical image"""
    analyzer = create_analyzer(model_name)
    return analyzer.analyze_image(image_input, **kwargs)

if __name__ == "__main__":
    # Example usage
    analyzer = MedicalImageAnalyzer()
    
    # Test with a sample image (you would provide actual medical image)
    # result = analyzer.analyze_image("path/to/medical/image.jpg")
    # print(json.dumps(result, indent=2))
    
    print("Medical Image Analyzer initialized successfully")
    print(f"Model info: {analyzer.get_model_info()}")