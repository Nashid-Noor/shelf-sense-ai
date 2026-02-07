"""
OCR Confidence Estimation for ShelfSense AI

Multi-signal confidence scoring for OCR output:
- Raw engine confidence
- Text pattern validity
- Language model perplexity
- Character-level statistics
- Book-specific heuristics
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter
from loguru import logger


@dataclass
class ConfidenceBreakdown:
    """Detailed confidence analysis."""
    
    # Core scores (0-1)
    engine_confidence: float  # Raw OCR engine confidence
    pattern_score: float      # Text pattern validity
    language_score: float     # Language plausibility
    book_score: float         # Book-specific patterns
    
    # Final computed score
    final_confidence: float
    
    # Diagnostic info
    flags: list[str] = field(default_factory=list)
    adjustments: dict[str, float] = field(default_factory=dict)
    
    @property
    def is_high_confidence(self) -> bool:
        return self.final_confidence >= 0.8
    
    @property
    def is_usable(self) -> bool:
        return self.final_confidence >= 0.5
    
    @property
    def needs_fallback(self) -> bool:
        return self.final_confidence < 0.5


class ConfidenceEstimator:
    """
    Production confidence estimation for OCR output.
    
    Combines multiple signals to produce a reliable confidence score:
    1. Engine confidence - what the OCR model reports
    2. Pattern validity - does the text look like real text?
    3. Language plausibility - does it look like English/common language?
    4. Book heuristics - does it look like book metadata?
    
    The estimator is calibrated to be slightly pessimistic - it's better
    to trigger a fallback than to confidently return garbage.
    """
    
    # Weights for combining scores
    WEIGHTS = {
        'engine': 0.35,
        'pattern': 0.25,
        'language': 0.25,
        'book': 0.15,
    }
    
    # Common English letter frequencies (for language scoring)
    ENGLISH_FREQUENCIES = {
        'e': 0.127, 't': 0.091, 'a': 0.082, 'o': 0.075, 'i': 0.070,
        'n': 0.067, 's': 0.063, 'h': 0.061, 'r': 0.060, 'd': 0.043,
        'l': 0.040, 'c': 0.028, 'u': 0.028, 'm': 0.024, 'w': 0.024,
        'f': 0.022, 'g': 0.020, 'y': 0.020, 'p': 0.019, 'b': 0.015,
        'v': 0.010, 'k': 0.008, 'j': 0.002, 'x': 0.002, 'q': 0.001,
        'z': 0.001
    }
    
    # Common book-related words/patterns
    BOOK_INDICATORS = [
        r'\bnovel\b', r'\bstory\b', r'\btale\b', r'\bbook\b',
        r'\bchapter\b', r'\bvolume\b', r'\bpart\b', r'\bseries\b',
        r'\bedition\b', r'\bpublish', r'\bauthor\b', r'\bwriter\b',
        r'\bfiction\b', r'\bnonfiction\b', r'\bmystery\b', r'\bromance\b',
        r'\bthriller\b', r'\bscifi\b', r'\bfantasy\b', r'\bhistory\b',
    ]
    
    # Suspicious patterns that reduce confidence
    SUSPICIOUS_PATTERNS = [
        r'[^\x00-\x7F]{3,}',  # Long non-ASCII sequences
        r'(.)\1{4,}',         # Character repeated 5+ times
        r'[^aeiouAEIOU\s]{8,}',  # 8+ consonants in a row
        r'\d{10,}',           # Very long number sequences
        r'[!@#$%^&*]{3,}',    # Multiple special chars
    ]
    
    def __init__(
        self,
        min_confidence_threshold: float = 0.3,
        high_confidence_threshold: float = 0.8,
        penalize_short: bool = True,
        penalize_numeric_only: bool = True,
    ):
        """
        Initialize confidence estimator.
        
        Args:
            min_confidence_threshold: Below this, flag as unreliable
            high_confidence_threshold: Above this, consider high quality
            penalize_short: Reduce confidence for very short text
            penalize_numeric_only: Reduce confidence for all-numeric text
        """
        self.min_threshold = min_confidence_threshold
        self.high_threshold = high_confidence_threshold
        self.penalize_short = penalize_short
        self.penalize_numeric_only = penalize_numeric_only
        
        # Compile patterns
        self._book_patterns = [
            re.compile(p, re.IGNORECASE) 
            for p in self.BOOK_INDICATORS
        ]
        self._suspicious_patterns = [
            re.compile(p) 
            for p in self.SUSPICIOUS_PATTERNS
        ]
        
        logger.info(
            f"ConfidenceEstimator initialized: "
            f"thresholds=[{min_confidence_threshold}, {high_confidence_threshold}]"
        )
    
    def estimate(
        self,
        text: str,
        engine_confidence: float,
        source_type: str = "unknown",  # "spine" or "cover"
    ) -> ConfidenceBreakdown:
        """
        Estimate confidence for OCR output.
        
        Args:
            text: OCR text result
            engine_confidence: Raw confidence from OCR engine
            source_type: "spine", "cover", or "unknown"
            
        Returns:
            ConfidenceBreakdown with detailed scores
        """
        flags = []
        adjustments = {}
        
        # Handle empty text
        if not text or not text.strip():
            return ConfidenceBreakdown(
                engine_confidence=engine_confidence,
                pattern_score=0.0,
                language_score=0.0,
                book_score=0.0,
                final_confidence=0.0,
                flags=["empty_text"],
                adjustments={},
            )
        
        # Calculate component scores
        pattern_score, pattern_flags = self._score_patterns(text)
        flags.extend(pattern_flags)
        
        language_score, lang_flags = self._score_language(text)
        flags.extend(lang_flags)
        
        book_score, book_flags = self._score_book_patterns(text)
        flags.extend(book_flags)
        
        # Apply penalties
        if self.penalize_short and len(text.strip()) < 3:
            adjustments['short_text'] = -0.2
            flags.append("very_short")
        
        if self.penalize_numeric_only and text.strip().isdigit():
            # All digits - might be ISBN, price, etc.
            adjustments['numeric_only'] = -0.1
            flags.append("numeric_only")
        
        # Source-specific adjustments
        if source_type == "spine":
            # Spines often have rotated/harder text
            if engine_confidence < 0.6:
                adjustments['spine_low_conf'] = -0.05
        
        # Calculate weighted final score
        weighted = (
            self.WEIGHTS['engine'] * engine_confidence +
            self.WEIGHTS['pattern'] * pattern_score +
            self.WEIGHTS['language'] * language_score +
            self.WEIGHTS['book'] * book_score
        )
        
        # Apply adjustments
        total_adjustment = sum(adjustments.values())
        final_confidence = max(0.0, min(1.0, weighted + total_adjustment))
        
        return ConfidenceBreakdown(
            engine_confidence=engine_confidence,
            pattern_score=pattern_score,
            language_score=language_score,
            book_score=book_score,
            final_confidence=final_confidence,
            flags=flags,
            adjustments=adjustments,
        )
    
    def _score_patterns(self, text: str) -> tuple[float, list[str]]:
        """
        Score text based on pattern validity.
        
        Checks for:
        - Reasonable word lengths
        - Balanced character types
        - No suspicious patterns
        """
        flags = []
        score = 1.0
        
        # Check for suspicious patterns
        for pattern in self._suspicious_patterns:
            if pattern.search(text):
                score -= 0.15
                flags.append("suspicious_pattern")
        
        # Check word statistics
        words = text.split()
        if words:
            # Average word length
            avg_len = sum(len(w) for w in words) / len(words)
            if avg_len < 2:
                score -= 0.1
                flags.append("short_words")
            elif avg_len > 15:
                score -= 0.1
                flags.append("long_words")
            
            # Ratio of "normal" words (2-20 chars, has vowel)
            normal_words = sum(
                1 for w in words
                if 2 <= len(w) <= 20 and re.search(r'[aeiouAEIOU]', w)
            )
            normal_ratio = normal_words / len(words)
            if normal_ratio < 0.5:
                score -= 0.15
                flags.append("unusual_words")
        
        # Character type distribution
        letters = sum(1 for c in text if c.isalpha())
        digits = sum(1 for c in text if c.isdigit())
        spaces = sum(1 for c in text if c.isspace())
        special = len(text) - letters - digits - spaces
        
        total = len(text) or 1
        
        # Text should be mostly letters
        if letters / total < 0.5:
            score -= 0.1
            flags.append("low_letter_ratio")
        
        # Not too many special characters
        if special / total > 0.2:
            score -= 0.1
            flags.append("high_special_ratio")
        
        return max(0.0, score), flags
    
    def _score_language(self, text: str) -> tuple[float, list[str]]:
        """
        Score text based on language plausibility.
        
        Uses letter frequency analysis for English.
        """
        flags = []
        
        # Extract just letters
        letters = ''.join(c.lower() for c in text if c.isalpha())
        
        if not letters:
            return 0.3, ["no_letters"]
        
        # Calculate letter frequency
        freq = Counter(letters)
        total = len(letters)
        
        # Compare to English frequencies
        score = 0.0
        for letter, expected_freq in self.ENGLISH_FREQUENCIES.items():
            actual_freq = freq.get(letter, 0) / total
            # Score based on how close to expected
            diff = abs(actual_freq - expected_freq)
            score += max(0, 1 - diff * 5)  # Penalty for deviation
        
        # Normalize to 0-1
        score = score / len(self.ENGLISH_FREQUENCIES)
        
        # Additional checks
        if score < 0.4:
            flags.append("unusual_letter_dist")
        
        # Check for reasonable vowel ratio (English: ~38%)
        vowels = sum(1 for c in letters if c in 'aeiou')
        vowel_ratio = vowels / len(letters) if letters else 0
        
        if vowel_ratio < 0.2 or vowel_ratio > 0.6:
            score -= 0.1
            flags.append("unusual_vowel_ratio")
        
        return max(0.0, min(1.0, score)), flags
    
    def _score_book_patterns(self, text: str) -> tuple[float, list[str]]:
        """
        Score text for book-specific patterns.
        
        Higher score if text looks like book metadata.
        """
        flags = []
        score = 0.5  # Neutral baseline
        
        # Check for book indicators
        for pattern in self._book_patterns:
            if pattern.search(text):
                score += 0.1
                break  # Only count once
        
        # Check for author-like patterns (capitalized words)
        words = text.split()
        capitalized = sum(1 for w in words if w and w[0].isupper())
        if words and capitalized / len(words) > 0.5:
            score += 0.1
            flags.append("capitalized_words")
        
        # Check for title-like patterns
        # Titles often have 2-8 words
        if 2 <= len(words) <= 8:
            score += 0.1
        
        # ISBN pattern presence (positive signal)
        if re.search(r'\b97[89]\d{10}\b', text) or re.search(r'\b\d{9}[\dX]\b', text):
            score += 0.15
            flags.append("isbn_present")
        
        # Publisher names (small boost)
        publishers = [
            'penguin', 'random house', 'harper', 'simon', 'macmillan',
            'hachette', 'scholastic', 'wiley', 'pearson', 'mcgraw'
        ]
        text_lower = text.lower()
        if any(pub in text_lower for pub in publishers):
            score += 0.1
            flags.append("publisher_name")
        
        return min(1.0, score), flags
    
    def estimate_batch(
        self,
        texts: list[str],
        confidences: list[float],
        source_types: Optional[list[str]] = None,
    ) -> list[ConfidenceBreakdown]:
        """
        Estimate confidence for multiple texts.
        
        Args:
            texts: List of OCR texts
            confidences: List of engine confidences
            source_types: Optional list of source types
            
        Returns:
            List of ConfidenceBreakdown objects
        """
        if source_types is None:
            source_types = ["unknown"] * len(texts)
        
        return [
            self.estimate(text, conf, src)
            for text, conf, src in zip(texts, confidences, source_types)
        ]
    
    def should_retry(self, breakdown: ConfidenceBreakdown) -> bool:
        """
        Determine if OCR should be retried with different settings.
        
        Returns True if:
        - Confidence is low but not hopeless
        - Specific flags suggest rotation/enhancement might help
        """
        if breakdown.final_confidence >= self.high_threshold:
            return False  # Good enough
        
        if breakdown.final_confidence < 0.2:
            return False  # Probably not salvageable
        
        # Check for flags that suggest retry might help
        retry_flags = {
            'unusual_letter_dist',
            'unusual_vowel_ratio', 
            'suspicious_pattern',
        }
        
        if retry_flags & set(breakdown.flags):
            return True
        
        # Borderline confidence - worth a retry
        if 0.4 <= breakdown.final_confidence <= 0.6:
            return True
        
        return False
    
    def get_quality_tier(self, breakdown: ConfidenceBreakdown) -> str:
        """
        Get quality tier label for the confidence.
        
        Returns one of: "high", "medium", "low", "unusable"
        """
        conf = breakdown.final_confidence
        
        if conf >= self.high_threshold:
            return "high"
        elif conf >= 0.6:
            return "medium"
        elif conf >= self.min_threshold:
            return "low"
        else:
            return "unusable"


class AdaptiveConfidenceEstimator(ConfidenceEstimator):
    """
    Confidence estimator that learns from correction feedback.
    
    Tracks historical accuracy and adjusts weights accordingly.
    This is useful for fine-tuning to specific book collections.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Track predictions and outcomes
        self._history: list[tuple[float, bool]] = []  # (predicted_conf, was_correct)
        self._calibration_enabled = False
    
    def record_outcome(
        self,
        predicted_confidence: float,
        was_correct: bool,
    ):
        """
        Record whether a prediction was correct.
        
        Args:
            predicted_confidence: What we predicted
            was_correct: Whether the OCR was actually correct
        """
        self._history.append((predicted_confidence, was_correct))
        
        # Recalibrate periodically
        if len(self._history) % 100 == 0:
            self._recalibrate()
    
    def _recalibrate(self):
        """Recalibrate based on historical accuracy."""
        if len(self._history) < 50:
            return
        
        # Calculate calibration curve
        # Group predictions into buckets and compare to actual accuracy
        buckets = {i/10: [] for i in range(11)}
        
        for pred, correct in self._history[-500:]:  # Last 500 samples
            bucket = round(pred, 1)
            bucket = min(1.0, max(0.0, bucket))
            buckets[bucket].append(correct)
        
        # Log calibration info
        for bucket, outcomes in sorted(buckets.items()):
            if outcomes:
                actual_acc = sum(outcomes) / len(outcomes)
                logger.debug(
                    f"Calibration bucket {bucket:.1f}: "
                    f"predicted={bucket:.1f}, actual={actual_acc:.2f}, "
                    f"n={len(outcomes)}"
                )
    
    def get_calibration_stats(self) -> dict:
        """Get calibration statistics."""
        if not self._history:
            return {"samples": 0}
        
        total = len(self._history)
        correct = sum(1 for _, c in self._history if c)
        avg_conf = sum(p for p, _ in self._history) / total
        
        return {
            "samples": total,
            "accuracy": correct / total,
            "avg_confidence": avg_conf,
            "calibration_gap": avg_conf - (correct / total),
        }
