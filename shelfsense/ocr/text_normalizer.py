"""
Text Normalizer for ShelfSense AI

Post-processing pipeline for OCR output:
- Unicode normalization
- OCR error correction
- Case normalization
- Punctuation cleanup
- Common book-related text patterns
"""

from typing import List, Optional, Dict, Tuple
import re
import unicodedata
from dataclasses import dataclass
from loguru import logger


@dataclass
class NormalizationResult:
    """Result of text normalization."""
    original: str
    normalized: str
    corrections: List[Tuple[str, str]]  # (original, corrected) pairs
    confidence_adjustment: float  # Adjustment to OCR confidence
    
    @property
    def was_modified(self) -> bool:
        return self.original != self.normalized


class TextNormalizer:
    """
    Text normalization and cleaning for book metadata.
    
    Handles common OCR errors specific to book spines and covers:
    - Character substitutions (0/O, 1/I/l, etc.)
    - Unicode normalization (ligatures, combining characters)
    - Publisher-specific patterns
    - Author name formatting
    
    Usage:
        normalizer = TextNormalizer()
        result = normalizer.normalize("J0HN  GRISH4M")
        print(result.normalized)  # "JOHN GRISHAM"
    """
    
    # Common OCR character substitutions
    CHAR_SUBSTITUTIONS = {
        '0': 'O',  # Zero to O (context-dependent)
        '1': 'I',  # One to I (context-dependent)
        '|': 'I',  # Pipe to I
        '!': 'I',  # Exclamation to I (in some contexts)
        '$': 'S',
        '@': 'A',
        '€': 'E',
        '&': 'AND',  # Context-dependent
        '+': 'T',  # Plus to T (in some fonts)
        '¡': 'I',
        '¿': '?',
    }
    
    # Number to letter mappings for names/titles (not ISBNs)
    NUM_TO_LETTER = {
        '0': 'O',
        '1': 'I',
        '3': 'E',
        '4': 'A',
        '5': 'S',
        '6': 'G',
        '7': 'T',
        '8': 'B',
    }
    
    # Common OCR error patterns (regex pattern, replacement)
    ERROR_PATTERNS = [
        (r'\brn\b', 'm'),  # rn -> m in context
        (r'\bvv\b', 'w'),  # vv -> w
        (r'\bIl\b', 'H'),  # Il -> H (font-dependent)
        (r'(?<=[A-Z])0(?=[A-Z])', 'O'),  # 0 between letters
        (r'(?<=[a-z])0(?=[a-z])', 'o'),  # 0 between lowercase
        (r'(?<=[A-Z])1(?=[A-Z])', 'I'),  # 1 between uppercase
        (r'(?<=[a-z])1(?=[a-z])', 'l'),  # 1 between lowercase
        (r'["""]', '"'),  # Smart quotes to standard
        (r"[''']", "'"),  # Smart apostrophes
    ]
    
    # Book-specific patterns
    BOOK_PATTERNS = [
        # Volume markers
        (r'V01\.?\s*(\d+)', r'Vol. \1'),
        (r'V0LUME\s*(\d+)', r'Volume \1'),
        # Edition markers
        (r'(\d+)[stndrdth]+\s*[Ee]d\.?', r'\1st Ed.'),
        # Publisher abbreviations
        (r'\bPENGU1N\b', 'PENGUIN'),
        (r'\bRAND0M\b', 'RANDOM'),
        (r'\bH0USE\b', 'HOUSE'),
    ]
    
    def __init__(
        self,
        fix_unicode: bool = True,
        fix_ocr_errors: bool = True,
        normalize_case: bool = False,
        remove_noise: bool = True,
        min_word_length: int = 1
    ):
        """
        Initialize the text normalizer.
        
        Args:
            fix_unicode: Normalize Unicode characters
            fix_ocr_errors: Apply OCR error corrections
            normalize_case: Convert to title case
            remove_noise: Remove non-text characters
            min_word_length: Minimum word length to keep
        """
        self.fix_unicode = fix_unicode
        self.fix_ocr_errors = fix_ocr_errors
        self.normalize_case = normalize_case
        self.remove_noise = remove_noise
        self.min_word_length = min_word_length
        
        # Compile patterns for efficiency
        self._compiled_patterns = [
            (re.compile(p, re.IGNORECASE), r)
            for p, r in self.ERROR_PATTERNS + self.BOOK_PATTERNS
        ]
    
    def normalize(self, text: str) -> NormalizationResult:
        """
        Normalize OCR text output.
        
        Args:
            text: Raw OCR text
            
        Returns:
            NormalizationResult with normalized text and metadata
        """
        if not text:
            return NormalizationResult(
                original="",
                normalized="",
                corrections=[],
                confidence_adjustment=0.0
            )
        
        original = text
        corrections = []
        confidence_adj = 0.0
        
        # Step 1: Unicode normalization
        if self.fix_unicode:
            text = self._normalize_unicode(text)
        
        # Step 2: Remove noise characters
        if self.remove_noise:
            text = self._remove_noise(text)
        
        # Step 3: Fix OCR errors
        if self.fix_ocr_errors:
            text, ocr_corrections = self._fix_ocr_errors(text)
            corrections.extend(ocr_corrections)
            # Each correction slightly reduces confidence
            confidence_adj -= len(ocr_corrections) * 0.02
        
        # Step 4: Normalize whitespace
        text = self._normalize_whitespace(text)
        
        # Step 5: Apply case normalization
        if self.normalize_case:
            text = self._normalize_case(text)
        
        # Step 6: Filter short words
        if self.min_word_length > 1:
            text = self._filter_short_words(text)
        
        return NormalizationResult(
            original=original,
            normalized=text.strip(),
            corrections=corrections,
            confidence_adjustment=max(-0.3, confidence_adj)  # Cap adjustment
        )
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters."""
        # NFKC normalization: compatibility decomposition + canonical composition
        # This handles ligatures (ﬁ -> fi), superscripts, etc.
        text = unicodedata.normalize('NFKC', text)
        
        # Remove combining characters (accents on separate codepoints)
        text = ''.join(
            c for c in text
            if unicodedata.category(c) != 'Mn'  # Non-spacing marks
        )
        
        return text
    
    def _remove_noise(self, text: str) -> str:
        """Remove non-text noise characters."""
        # Keep letters, numbers, basic punctuation, and whitespace
        # Remove control characters and unusual symbols
        allowed = set(
            'abcdefghijklmnopqrstuvwxyz'
            'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            '0123456789'
            " .,;:!?'-\"()[]&"
        )
        
        cleaned = ''.join(c if c in allowed else ' ' for c in text)
        return cleaned
    
    def _fix_ocr_errors(self, text: str) -> Tuple[str, List[Tuple[str, str]]]:
        """Apply OCR error corrections."""
        corrections = []
        
        # Apply compiled regex patterns
        for pattern, replacement in self._compiled_patterns:
            matches = pattern.findall(text)
            if matches:
                new_text = pattern.sub(replacement, text)
                if new_text != text:
                    corrections.append((text, new_text))
                    text = new_text
        
        return text, corrections
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace."""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove spaces around hyphens in compound words
        text = re.sub(r'\s*-\s*', '-', text)
        return text.strip()
    
    def _normalize_case(self, text: str) -> str:
        """
        Normalize case for book titles.
        
        Uses smart title casing that handles:
        - Articles (a, an, the)
        - Prepositions
        - Conjunctions
        """
        # Words to keep lowercase (unless first/last)
        lowercase_words = {
            'a', 'an', 'the', 'and', 'but', 'or', 'nor', 'for', 'yet', 'so',
            'at', 'by', 'for', 'in', 'of', 'on', 'to', 'up', 'as', 'is'
        }
        
        words = text.split()
        if not words:
            return text
        
        result = []
        for i, word in enumerate(words):
            # First and last words always capitalized
            if i == 0 or i == len(words) - 1:
                result.append(word.capitalize())
            elif word.lower() in lowercase_words:
                result.append(word.lower())
            else:
                result.append(word.capitalize())
        
        return ' '.join(result)
    
    def _filter_short_words(self, text: str) -> str:
        """Filter out words shorter than minimum length."""
        words = text.split()
        filtered = [w for w in words if len(w) >= self.min_word_length]
        return ' '.join(filtered)
    
    def normalize_author_name(self, name: str) -> str:
        """
        Normalize author name specifically.
        
        Handles:
        - "LASTNAME, FIRSTNAME" -> "Firstname Lastname"
        - All caps conversion
        - Initial handling
        """
        name = self.normalize(name).normalized
        
        # Handle "LASTNAME, FIRSTNAME" format
        if ',' in name:
            parts = [p.strip() for p in name.split(',', 1)]
            if len(parts) == 2:
                name = f"{parts[1]} {parts[0]}"
        
        # Title case
        name = name.title()
        
        # Fix common author name OCR errors
        author_fixes = [
            (r'\bJr\b', 'Jr.'),
            (r'\bSr\b', 'Sr.'),
            (r'\bDr\b', 'Dr.'),
            (r'\bMc ([A-Z])', r'Mc\1'),  # McDonald, McKenzie
            (r"\bO ([A-Z])", r"O'\1"),   # O'Brien, O'Connor
        ]
        
        for pattern, replacement in author_fixes:
            name = re.sub(pattern, replacement, name)
        
        return name.strip()
    
    def normalize_title(self, title: str) -> str:
        """
        Normalize book title specifically.
        
        Handles:
        - Subtitle separation
        - Series markers
        - Edition markers
        """
        title = self.normalize(title).normalized
        
        # Normalize subtitle separators
        title = re.sub(r'\s*[:|]\s*', ': ', title)
        
        # Handle series markers
        title = re.sub(
            r'(?:Book|Volume|Vol\.?|Part|#)\s*(\d+)',
            r'(Book \1)',
            title,
            flags=re.IGNORECASE
        )
        
        return title.strip()
    
    def extract_potential_isbn(self, text: str) -> Optional[str]:
        """
        Extract potential ISBN from text.
        
        Handles both ISBN-10 and ISBN-13 formats.
        """
        # Remove common OCR artifacts
        clean = re.sub(r'[^0-9X-]', '', text.upper())
        
        # ISBN-13 pattern
        isbn13_match = re.search(r'97[89]\d{10}', clean)
        if isbn13_match:
            isbn = isbn13_match.group()
            if self._validate_isbn13(isbn):
                return isbn
        
        # ISBN-10 pattern
        isbn10_match = re.search(r'\d{9}[\dX]', clean)
        if isbn10_match:
            isbn = isbn10_match.group()
            if self._validate_isbn10(isbn):
                return isbn
        
        return None
    
    def _validate_isbn10(self, isbn: str) -> bool:
        """Validate ISBN-10 checksum."""
        if len(isbn) != 10:
            return False
        
        try:
            total = sum(
                (10 if c == 'X' else int(c)) * (10 - i)
                for i, c in enumerate(isbn)
            )
            return total % 11 == 0
        except ValueError:
            return False
    
    def _validate_isbn13(self, isbn: str) -> bool:
        """Validate ISBN-13 checksum."""
        if len(isbn) != 13:
            return False
        
        try:
            total = sum(
                int(c) * (1 if i % 2 == 0 else 3)
                for i, c in enumerate(isbn)
            )
            return total % 10 == 0
        except ValueError:
            return False


class BatchTextNormalizer:
    """Batch normalization for multiple text strings."""
    
    def __init__(self, normalizer: Optional[TextNormalizer] = None):
        self.normalizer = normalizer or TextNormalizer()
    
    def normalize_batch(
        self,
        texts: List[str],
        as_titles: bool = False,
        as_authors: bool = False
    ) -> List[NormalizationResult]:
        """
        Normalize multiple texts.
        
        Args:
            texts: List of text strings
            as_titles: Use title-specific normalization
            as_authors: Use author-specific normalization
            
        Returns:
            List of NormalizationResult objects
        """
        results = []
        for text in texts:
            if as_titles:
                normalized = self.normalizer.normalize_title(text)
                result = NormalizationResult(
                    original=text,
                    normalized=normalized,
                    corrections=[],
                    confidence_adjustment=0.0
                )
            elif as_authors:
                normalized = self.normalizer.normalize_author_name(text)
                result = NormalizationResult(
                    original=text,
                    normalized=normalized,
                    corrections=[],
                    confidence_adjustment=0.0
                )
            else:
                result = self.normalizer.normalize(text)
            
            results.append(result)
        
        return results
