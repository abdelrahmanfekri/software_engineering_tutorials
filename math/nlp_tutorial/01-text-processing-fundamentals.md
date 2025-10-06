# NLP Tutorial Module 1: Text Processing Fundamentals

## Learning Objectives
By the end of this module, you will be able to:
- Understand the nature of human language and text data
- Implement text preprocessing pipelines
- Apply various tokenization techniques
- Handle different text encodings and formats
- Clean and normalize text data
- Extract linguistic features from text

## Introduction to Text Processing

### What is Natural Language?
Natural language is the complex, ambiguous, and context-dependent communication system used by humans. Unlike programming languages, natural language exhibits:

- **Ambiguity**: Words and phrases can have multiple meanings
- **Context Dependency**: Meaning depends on surrounding text
- **Variation**: Multiple ways to express the same concept
- **Evolution**: Language changes over time and across communities

### Challenges in Text Processing

1. **Morphological Variation**: "run", "runs", "running", "ran"
2. **Syntactic Ambiguity**: "I saw the man with binoculars"
3. **Semantic Ambiguity**: "The bank is closed" (financial institution vs. river bank)
4. **Pragmatic Context**: "It's cold in here" (statement vs. request to close window)

## Text Data Representation

### Character Encoding

Text data is stored as sequences of bytes, requiring encoding schemes:

#### ASCII (American Standard Code for Information Interchange)
- 7-bit encoding (128 characters)
- Covers English letters, digits, and basic symbols
- Limited for international text

#### Unicode
- Universal character encoding standard
- Supports over 1 million characters
- UTF-8: Variable-length encoding (1-4 bytes per character)
- UTF-16: 2-4 bytes per character
- UTF-32: 4 bytes per character

```python
# Character encoding examples
text = "Hello, 世界! 🌍"

# UTF-8 encoding
utf8_bytes = text.encode('utf-8')
print(f"UTF-8: {utf8_bytes}")

# UTF-16 encoding
utf16_bytes = text.encode('utf-16')
print(f"UTF-16: {utf16_bytes}")

# Character code points
for char in text:
    print(f"'{char}' -> U+{ord(char):04X}")
```

## Text Preprocessing Pipeline

### 1. Text Loading and Encoding

```python
import chardet
import pandas as pd

def detect_encoding(file_path):
    """Detect file encoding automatically"""
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result['encoding']

def load_text_file(file_path):
    """Load text file with proper encoding"""
    encoding = detect_encoding(file_path)
    with open(file_path, 'r', encoding=encoding) as f:
        return f.read()

# Example usage
text = load_text_file('sample.txt')
```

### 2. Text Cleaning

```python
import re
import string

class TextCleaner:
    def __init__(self):
        self.punctuation_table = str.maketrans('', '', string.punctuation)
        
    def remove_html_tags(self, text):
        """Remove HTML tags from text"""
        html_pattern = re.compile(r'<[^>]+>')
        return html_pattern.sub('', text)
    
    def remove_urls(self, text):
        """Remove URLs from text"""
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        return url_pattern.sub('', text)
    
    def remove_emails(self, text):
        """Remove email addresses from text"""
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        return email_pattern.sub('', text)
    
    def remove_punctuation(self, text):
        """Remove punctuation marks"""
        return text.translate(self.punctuation_table)
    
    def normalize_whitespace(self, text):
        """Normalize whitespace characters"""
        return re.sub(r'\s+', ' ', text).strip()
    
    def clean_text(self, text):
        """Apply all cleaning steps"""
        text = self.remove_html_tags(text)
        text = self.remove_urls(text)
        text = self.remove_emails(text)
        text = self.remove_punctuation(text)
        text = self.normalize_whitespace(text)
        return text

# Example usage
cleaner = TextCleaner()
dirty_text = "Hello! This is a test. Visit https://example.com or email me@test.com"
clean_text = cleaner.clean_text(dirty_text)
print(clean_text)  # "Hello This is a test Visit or email"
```

### 3. Text Normalization

```python
import unicodedata

class TextNormalizer:
    def __init__(self):
        self.contractions = {
            "don't": "do not",
            "can't": "cannot",
            "won't": "will not",
            "I'm": "I am",
            "you're": "you are",
            "it's": "it is",
            "we're": "we are",
            "they're": "they are"
        }
    
    def expand_contractions(self, text):
        """Expand common contractions"""
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)
        return text
    
    def normalize_unicode(self, text):
        """Normalize Unicode characters"""
        # NFD: Normalization Form Decomposed
        text = unicodedata.normalize('NFD', text)
        return text
    
    def lowercase_text(self, text):
        """Convert text to lowercase"""
        return text.lower()
    
    def normalize_text(self, text):
        """Apply all normalization steps"""
        text = self.expand_contractions(text)
        text = self.normalize_unicode(text)
        text = self.lowercase_text(text)
        return text

# Example usage
normalizer = TextNormalizer()
text = "I'm happy! Don't worry."
normalized_text = normalizer.normalize_text(text)
print(normalized_text)  # "i am happy do not worry"
```

## Tokenization

Tokenization is the process of splitting text into smaller units (tokens) such as words, subwords, or characters.

### 1. Word Tokenization

```python
import nltk
import spacy
from nltk.tokenize import word_tokenize, WordPunctTokenizer
import re

# NLTK word tokenization
nltk.download('punkt')
text = "Hello, world! How are you today?"
tokens = word_tokenize(text)
print(tokens)  # ['Hello', ',', 'world', '!', 'How', 'are', 'you', 'today', '?']

# SpaCy word tokenization
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
tokens = [token.text for token in doc]
print(tokens)  # ['Hello', ',', 'world', '!', 'How', 'are', 'you', 'today', '?']

# Custom word tokenization
def simple_word_tokenize(text):
    """Simple word tokenization using regex"""
    pattern = r'\b\w+\b'
    return re.findall(pattern, text)

tokens = simple_word_tokenize(text)
print(tokens)  # ['Hello', 'world', 'How', 'are', 'you', 'today']
```

### 2. Sentence Tokenization

```python
from nltk.tokenize import sent_tokenize

text = "Hello world! How are you? I'm doing great."
sentences = sent_tokenize(text)
print(sentences)  # ['Hello world!', 'How are you?', "I'm doing great."]

# SpaCy sentence tokenization
doc = nlp(text)
sentences = [sent.text for sent in doc.sents]
print(sentences)  # ['Hello world!', 'How are you?', "I'm doing great."]
```

### 3. Subword Tokenization

```python
from transformers import AutoTokenizer

# BERT tokenizer (WordPiece)
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
text = "Hello, world! How are you today?"
bert_tokens = bert_tokenizer.tokenize(text)
print("BERT tokens:", bert_tokens)

# GPT-2 tokenizer (BPE)
gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')
gpt2_tokens = gpt2_tokenizer.tokenize(text)
print("GPT-2 tokens:", gpt2_tokens)

# T5 tokenizer (SentencePiece)
t5_tokenizer = AutoTokenizer.from_pretrained('t5-small')
t5_tokens = t5_tokenizer.tokenize(text)
print("T5 tokens:", t5_tokens)
```

### 4. Character Tokenization

```python
def character_tokenize(text):
    """Character-level tokenization"""
    return list(text)

def character_tokenize_with_spaces(text):
    """Character-level tokenization preserving word boundaries"""
    tokens = []
    for char in text:
        if char.isspace():
            tokens.append('<SPACE>')
        else:
            tokens.append(char)
    return tokens

text = "Hello world"
char_tokens = character_tokenize(text)
print("Character tokens:", char_tokens)

char_tokens_with_spaces = character_tokenize_with_spaces(text)
print("Character tokens with spaces:", char_tokens_with_spaces)
```

## Stop Words Removal

Stop words are common words that typically don't carry much semantic meaning.

```python
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

# Get stop words for different languages
english_stopwords = set(stopwords.words('english'))
spanish_stopwords = set(stopwords.words('spanish'))

def remove_stopwords(tokens, language='english'):
    """Remove stop words from token list"""
    if language == 'english':
        stop_words = english_stopwords
    elif language == 'spanish':
        stop_words = spanish_stopwords
    else:
        return tokens
    
    return [token for token in tokens if token.lower() not in stop_words]

# Example usage
tokens = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
filtered_tokens = remove_stopwords(tokens)
print("Original tokens:", tokens)
print("Filtered tokens:", filtered_tokens)
```

## Stemming and Lemmatization

### Stemming

Stemming reduces words to their root form by removing suffixes.

```python
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer

# Different stemming algorithms
porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer('english')

words = ['running', 'runs', 'ran', 'easily', 'fairly', 'fairness']

print("Original words:", words)
print("Porter stems:", [porter.stem(word) for word in words])
print("Lancaster stems:", [lancaster.stem(word) for word in words])
print("Snowball stems:", [snowball.stem(word) for word in words])
```

### Lemmatization

Lemmatization reduces words to their dictionary form (lemma) using vocabulary and morphological analysis.

```python
import spacy
from nltk.stem import WordNetLemmatizer

# NLTK lemmatization
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()

words = ['running', 'runs', 'ran', 'better', 'best', 'mice', 'geese']

print("Original words:", words)
print("NLTK lemmas:", [lemmatizer.lemmatize(word) for word in words])

# SpaCy lemmatization
nlp = spacy.load("en_core_web_sm")
text = " ".join(words)
doc = nlp(text)
spacy_lemmas = [token.lemma_ for token in doc]
print("SpaCy lemmas:", spacy_lemmas)
```

## Part-of-Speech Tagging

POS tagging assigns grammatical categories to words.

```python
import nltk
import spacy

# Download required NLTK data
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')

# NLTK POS tagging
text = "The quick brown fox jumps over the lazy dog"
tokens = word_tokenize(text)
pos_tags = nltk.pos_tag(tokens)
print("NLTK POS tags:", pos_tags)

# SpaCy POS tagging
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
spacy_pos = [(token.text, token.pos_, token.tag_) for token in doc]
print("SpaCy POS tags:", spacy_pos)

# POS tag meanings
print("\nPOS tag meanings:")
for token, tag in pos_tags:
    print(f"{token}: {nltk.help.upenn_tagset(tag)}")
```

## Named Entity Recognition (NER)

NER identifies and classifies named entities in text.

```python
import spacy
from nltk import ne_chunk
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('maxent_ne_chunker')
nltk.download('words')

# NLTK NER
text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
tokens = word_tokenize(text)
pos_tags = nltk.pos_tag(tokens)
ner_tags = ne_chunk(pos_tags)
print("NLTK NER:", ner_tags)

# SpaCy NER
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
spacy_ner = [(ent.text, ent.label_) for ent in doc.ents]
print("SpaCy NER:", spacy_ner)

# Detailed NER with SpaCy
for ent in doc.ents:
    print(f"Entity: {ent.text}, Label: {ent.label_}, Start: {ent.start_char}, End: {ent.end_char}")
```

## Complete Text Processing Pipeline

```python
class TextProcessor:
    def __init__(self):
        self.cleaner = TextCleaner()
        self.normalizer = TextNormalizer()
        self.nlp = spacy.load("en_core_web_sm")
        
    def process_text(self, text):
        """Complete text processing pipeline"""
        # Step 1: Clean text
        cleaned_text = self.cleaner.clean_text(text)
        
        # Step 2: Normalize text
        normalized_text = self.normalizer.normalize_text(cleaned_text)
        
        # Step 3: Create SpaCy document
        doc = self.nlp(normalized_text)
        
        # Step 4: Extract features
        features = {
            'tokens': [token.text for token in doc],
            'lemmas': [token.lemma_ for token in doc],
            'pos_tags': [(token.text, token.pos_) for token in doc],
            'entities': [(ent.text, ent.label_) for ent in doc.ents],
            'sentences': [sent.text for sent in doc.sents]
        }
        
        return features

# Example usage
processor = TextProcessor()
text = """
    Hello! I'm John from Apple Inc. 
    Visit our website at https://apple.com for more information.
    You can also email me at john@apple.com.
"""

features = processor.process_text(text)
print("Processed features:")
for key, value in features.items():
    print(f"{key}: {value}")
```

## Practical Exercises

### Exercise 1: Text Cleaning Pipeline
Create a comprehensive text cleaning pipeline that handles:
- HTML tags removal
- URL removal
- Email removal
- Special characters normalization
- Multiple whitespace handling

### Exercise 2: Custom Tokenizer
Implement a custom tokenizer that:
- Handles contractions properly
- Preserves important punctuation
- Splits on whitespace and punctuation
- Handles edge cases (numbers, abbreviations)

### Exercise 3: Text Preprocessing for Different Tasks
Create specialized preprocessing pipelines for:
- Sentiment analysis
- Machine translation
- Question answering
- Text summarization

## Assessment Questions

1. **Why is text preprocessing important in NLP?**
   - Removes noise and irrelevant information
   - Standardizes text format
   - Improves model performance
   - Reduces computational complexity

2. **What are the differences between stemming and lemmatization?**
   - Stemming: Heuristic-based, faster, less accurate
   - Lemmatization: Dictionary-based, slower, more accurate
   - Stemming may produce non-words
   - Lemmatization produces valid dictionary words

3. **How do you handle out-of-vocabulary words in tokenization?**
   - Character-level tokenization
   - Subword tokenization (BPE, WordPiece)
   - Unknown token placeholder
   - Morphological analysis

## Key Takeaways

- Text preprocessing is crucial for NLP model performance
- Different tasks may require different preprocessing approaches
- Tokenization choice affects model behavior significantly
- Text normalization helps with data consistency
- Linguistic features (POS, NER) provide valuable information
- Preprocessing should be tailored to the specific use case

## Next Steps

In the next module, we'll explore the mathematical foundations of NLP, including vector spaces, distance metrics, and statistical methods for text analysis. This will prepare you for understanding how text is represented mathematically in modern NLP systems.
