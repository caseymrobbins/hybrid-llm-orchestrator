from pytector import PromptInjectionDetector
from scrubadub import Scrubber
import re
import logging
from typing import Tuple
import hashlib

logger = logging.getLogger(__name__)

class SecurityManager:
    """Enhanced security manager with working implementations."""
    
    def __init__(self):
        # Initialize with working patterns
        self.injection_patterns = [
            # Common injection patterns
            r'ignore\s+previous\s+instructions',
            r'forget\s+everything\s+above',
            r'you\s+are\s+now\s+a\s+different',
            r'system\s*:\s*ignore',
            r'new\s+instructions',
            r'disregard\s+all\s+previous',
            r'override\s+previous\s+commands',
            # Jailbreak attempts
            r'roleplay\s+as\s+a\s+malicious',
            r'pretend\s+to\s+be\s+evil',
            r'act\s+as\s+if\s+you\s+are\s+not\s+ai',
        ]
        
        # PII patterns
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}-\d{3}-\d{4}\b|\b\(\d{3}\)\s*\d{3}-\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
        }
        
        # Initialize the prompt injection detector with a local model [2]
        self.injection_detector = PromptInjectionDetector(model_name_or_url="deberta")
        
        # Initialize the PII scrubber [7]
        self.pii_scrubber = Scrubber()

    def detect_injection(self, prompt: str) -> Tuple[bool, float]:
        """
        Detects potential prompt injection attacks.
        Returns (is_injection, confidence_score)
        """
        try:
            prompt_lower = prompt.lower()
            matches = 0
            total_patterns = len(self.injection_patterns)
            
            for pattern in self.injection_patterns:
                if re.search(pattern, prompt_lower, re.IGNORECASE):
                    matches += 1
                    logger.warning(f"Injection pattern detected: {pattern}")
            
            confidence = matches / total_patterns if total_patterns > 0 else 0.0
            is_injection = confidence > 0.1  # Lower threshold for detection
            
            return is_injection, confidence, self.injection_detector.detect_injection(prompt)
            
        except Exception as e:
            logger.error(f"Injection detection failed: {e}")
            return False, 0.0

    def scrub_pii(self, text: str) -> str:
        """
        Remove PII from text with proper replacements.
        """
        try:
            scrubbed_text = text
            
            for pii_type, pattern in self.pii_patterns.items():
                replacement = f"[{pii_type.upper()}_REDACTED]"
                scrubbed_text = re.sub(pattern, replacement, scrubbed_text)
            
            return  self.pii_scrubber.clean(scrubbed_text)
            
        except Exception as e:
            logger.error(f"PII scrubbing failed: {e}")
            return text