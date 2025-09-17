import ssl
import random
import warnings

import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer, util

warnings.filterwarnings("ignore", category=FutureWarning)

NLP_GLOBAL = spacy.load("en_core_web_sm")

def download_nltk_resources():
    """
    Download required NLTK resources if not already installed.
    """
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    resources = ['punkt', 'averaged_perceptron_tagger', 'punkt_tab','wordnet','averaged_perceptron_tagger_eng']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Error downloading {resource}: {str(e)}")


# This class  contains methods to humanize academic text, such as improving readability or
# simplifying complex language.
class AcademicTextHumanizer:
    """
    Transforms text into a more formal (academic) style:
      - Expands contractions
      - Adds academic transitions
      - Optionally converts some sentences to passive voice
      - Optionally replaces words with synonyms for more formality
    """

    def __init__(
        self,
        model_name='paraphrase-MiniLM-L6-v2',
        p_passive=0.2,
        p_synonym_replacement=0.3,
        p_academic_transition=0.3,
        seed=None
    ):
        if seed is not None:
            random.seed(seed)

        self.nlp = spacy.load("en_core_web_sm")
        self.model = SentenceTransformer(model_name)

        # Transformation probabilities
        self.p_passive = p_passive
        self.p_synonym_replacement = p_synonym_replacement
        self.p_academic_transition = p_academic_transition

        # Common academic transitions
        self.academic_transitions = [
            "Moreover,", "Additionally,", "Furthermore,", "Hence,", 
            "Therefore,", "Consequently,", "Nonetheless,", "Nevertheless,"
        ]

    def humanize_text(self, text, use_passive=False, use_synonyms=False):
        # Split text into paragraphs to preserve structure
        paragraphs = self._split_into_paragraphs(text)
        transformed_paragraphs = []
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
                
            doc = self.nlp(paragraph)
            transformed_sentences = []
            
            for i, sent in enumerate(doc.sents):
                sentence_str = sent.text.strip()
                
                # 1. Expand contractions
                sentence_str = self.expand_contractions(sentence_str)
                
                # 2. Possibly add academic transitions (but not to first sentence of paragraph)
                if i > 0 and random.random() < self.p_academic_transition:
                    sentence_str = self.add_academic_transitions(sentence_str)
                
                # 3. Optionally convert to passive
                if use_passive and random.random() < self.p_passive:
                    sentence_str = self.convert_to_passive(sentence_str)
                
                # 4. Optionally replace words with synonyms
                if use_synonyms and random.random() < self.p_synonym_replacement:
                    sentence_str = self.replace_with_synonyms(sentence_str)
                
                transformed_sentences.append(sentence_str)
            
            # Join sentences in paragraph with proper spacing
            if transformed_sentences:
                paragraph_text = ' '.join(transformed_sentences)
                # Clean up spacing issues
                paragraph_text = self._clean_spacing(paragraph_text)
                transformed_paragraphs.append(paragraph_text)
        
        # Join paragraphs with double newlines to preserve structure
        return '\n\n'.join(transformed_paragraphs)

    def expand_contractions(self, sentence):
        """Expand contractions while preserving original spacing and punctuation."""
        import re
        
        contraction_map = {
            r"\bn't\b": " not",
            r"\b're\b": " are", 
            r"\b's\b": " is",
            r"\b'll\b": " will",
            r"\b've\b": " have",
            r"\b'd\b": " would",
            r"\b'm\b": " am"
        }
        
        result = sentence
        for contraction_pattern, expansion in contraction_map.items():
            # Use regex substitution to preserve spacing and capitalization
            result = re.sub(contraction_pattern, expansion, result, flags=re.IGNORECASE)
        
        return result

    def add_academic_transitions(self, sentence):
        transition = random.choice(self.academic_transitions)
        # Make the first word of the original sentence lowercase unless it's a proper noun
        words = sentence.split()
        if words and not self._is_proper_noun(words[0], sentence):
            words[0] = words[0].lower()
            sentence = ' '.join(words)
        return f"{transition} {sentence}"
    
    def _is_proper_noun(self, word, context_sentence):
        """Check if a word is a proper noun using spaCy NER."""
        # Use spaCy to analyze the sentence and check if the first word is a named entity
        doc = self.nlp(context_sentence)
        if doc and len(doc) > 0:
            first_token = doc[0]
            # Check if it's a named entity or has proper noun POS tag
            if first_token.ent_type_ or first_token.pos_ == 'PROPN':
                return True
        
        # Additional check for common proper nouns that might be missed
        proper_indicators = {'Mr', 'Mrs', 'Ms', 'Dr', 'Prof', 'President', 'January', 'February', 'March', 
                           'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 
                           'December', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 
                           'Sunday', 'America', 'Europe', 'Asia', 'Africa', 'Australia', 'Antarctica'}
        
        return word in proper_indicators

    def convert_to_passive(self, sentence):
        doc = self.nlp(sentence)
        subj_tokens = [t for t in doc if t.dep_ == 'nsubj' and t.head.dep_ == 'ROOT']
        dobj_tokens = [t for t in doc if t.dep_ == 'dobj']

        if subj_tokens and dobj_tokens:
            subject = subj_tokens[0]
            dobj = dobj_tokens[0]
            verb = subject.head
            if subject.i < verb.i < dobj.i:
                passive_str = f"{dobj.text} {verb.lemma_} by {subject.text}"
                original_str = ' '.join(token.text for token in doc)
                chunk = f"{subject.text} {verb.text} {dobj.text}"
                if chunk in original_str:
                    sentence = original_str.replace(chunk, passive_str)
        return sentence

    def replace_with_synonyms(self, sentence):
        tokens = word_tokenize(sentence)
        pos_tags = nltk.pos_tag(tokens)

        new_tokens = []
        for (word, pos) in pos_tags:
            if pos.startswith(('J', 'N', 'V', 'R')) and wordnet.synsets(word):
                if random.random() < 0.5:
                    synonyms = self._get_synonyms(word, pos)
                    if synonyms:
                        best_synonym = self._select_closest_synonym(word, synonyms)
                        new_tokens.append(best_synonym if best_synonym else word)
                    else:
                        new_tokens.append(word)
                else:
                    new_tokens.append(word)
            else:
                new_tokens.append(word)

        return ' '.join(new_tokens)

    def _split_into_paragraphs(self, text):
        """Split text into paragraphs, preserving paragraph structure."""
        import re
        # Split on double newlines or more, preserving the paragraph breaks
        paragraphs = re.split(r'\n\s*\n', text.strip())
        return [p.strip() for p in paragraphs if p.strip()]

    def _clean_spacing(self, text):
        """Clean up unnecessary spaces while preserving proper formatting."""
        import re
        
        # First, normalize all whitespace to single spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Fix spacing around punctuation marks
        text = re.sub(r'\s+([.!?,:;])', r'\1', text)
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        # Comprehensive quotation mark handling
        # Pattern 1: Fix spaces inside quotes - remove space after opening quote
        text = re.sub(r'(["\'])\s+', r'\1', text)
        # Pattern 2: Fix spaces inside quotes - remove space before closing quote
        text = re.sub(r'\s+(["\'])', r'\1', text)
        # Pattern 3: Ensure space before opening quote (when preceded by word)
        text = re.sub(r'([a-zA-Z])(["\'])', r'\1 \2', text)
        # Pattern 4: Ensure space after closing quote (when followed by word)
        text = re.sub(r'(["\'])([a-zA-Z])', r'\1 \2', text)
        # Pattern 5: Handle punctuation after quotes
        text = re.sub(r'(["\'])\s*([.!?,:;])', r'\1\2', text)
        
        # Fix spacing around parentheses and brackets
        text = re.sub(r'\s+([)\]])', r'\1', text)  # Remove space before closing
        text = re.sub(r'([\(\[])\s+', r'\1', text)  # Remove space after opening
        
        # Clean up any remaining multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text

    def _get_synonyms(self, word, pos):
        wn_pos = None
        if pos.startswith('J'):
            wn_pos = wordnet.ADJ
        elif pos.startswith('N'):
            wn_pos = wordnet.NOUN
        elif pos.startswith('R'):
            wn_pos = wordnet.ADV
        elif pos.startswith('V'):
            wn_pos = wordnet.VERB

        synonyms = set()
        for syn in wordnet.synsets(word, pos=wn_pos):
            for lemma in syn.lemmas():
                lemma_name = lemma.name().replace('_', ' ')
                if lemma_name.lower() != word.lower():
                    synonyms.add(lemma_name)
        return list(synonyms)

    def _select_closest_synonym(self, original_word, synonyms):
        if not synonyms:
            return None
        original_emb = self.model.encode(original_word, convert_to_tensor=True)
        synonym_embs = self.model.encode(synonyms, convert_to_tensor=True)
        cos_scores = util.cos_sim(original_emb, synonym_embs)[0]
        max_score_index = cos_scores.argmax().item()
        max_score = cos_scores[max_score_index].item()
        if max_score >= 0.5:
            return synonyms[max_score_index]
        return None
