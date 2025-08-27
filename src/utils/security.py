from pytector import PromptInjectionDetector
from scrubadub import Scrubber

class SecurityManager:
    """Handles prompt injection detection and PII scrubbing."""

    def __init__(self):
        # Initialize the prompt injection detector with a local model [2]
        self.injection_detector = PromptInjectionDetector(model_name_or_url="deberta")
        
        # Initialize the PII scrubber [7]
        self.pii_scrubber = Scrubber()

    def detect_injection(self, prompt: str) -> (bool, float):
        """
        Detects if a prompt contains a potential injection attack.
        Returns a tuple of (is_injection, probability).
        """
        return self.injection_detector.detect_injection(prompt)

    def scrub_pii(self, text: str) -> str:
        """
        Removes Personally Identifiable Information from a string.
        """
        return self.pii_scrubber.clean(text)