import re

try:
    import spacy
    # try to load common local models
    nlp = None
    for mdl in ("en_core_web_sm", "en_core_web_md", "en"):
        try:
            nlp = spacy.load(mdl)
            break
        except Exception:
            nlp = None
except Exception:
    nlp = None

common_genres = ['pop','rock','jazz','classical','hip hop','rap','country','electronic','edm','metal','blues','r&b','soul','romance','love','thriller','mystery','comedy','drama']

content_nouns = set(['movie','movies','film','films','song','songs','album','albums','book','books'])


def parse_query_constraints(text):
    """Parse natural language query and extract simple constraints:
    - number: requested number of items (int) or None
    - genre: requested genre/topic string or None
    - sort_by: e.g. 'trending' or None
    Uses spaCy NER when available; falls back to heuristics.
    """
    result = {'number': None, 'genre': None, 'sort_by': None}
    if not text:
        return result
    s = text.lower()

    # Use spaCy if available
    if nlp is not None:
        doc = nlp(text)
        chosen_number = None
        for ent in doc.ents:
            if ent.label_ in ("CARDINAL", "QUANTITY", "ORDINAL"):
                # if numeric entity is part of a larger work title, skip
                parent_ent = None
                for e2 in doc.ents:
                    if e2.start <= ent.start and e2.end >= ent.end and e2 != ent and e2.label_ in ("WORK_OF_ART", "PRODUCT"):
                        parent_ent = e2
                        break
                if parent_ent:
                    continue
                # if numeric adjacent to content noun -> likely a title
                after = doc[ent.end: ent.end+2]
                before = doc[max(0, ent.start-2): ent.start]
                if any(t.text.lower() in content_nouns for t in after) or any(t.text.lower() in content_nouns for t in before):
                    continue
                # check surrounding tokens for count indicators
                left = doc[max(0, ent.start-4): ent.end+4]
                left_text = left.text.lower()
                if any(w in left_text for w in ['top', 'give', 'show', 'list', 'recommend', 'suggest', 'popular', 'trending', 'best']):
                    try:
                        chosen_number = int(''.join(ch for ch in ent.text if ch.isdigit()))
                        break
                    except Exception:
                        continue
        if chosen_number is None:
            for token in doc:
                if token.like_num:
                    window = [t.text.lower() for t in doc[max(0, token.i-3): token.i+4]]
                    if any(w in window for w in content_nouns) and not any(w in ' '.join(window) for w in ['top','give','show','list','recommend','suggest']):
                        continue
                    try:
                        chosen_number = int(token.text)
                        break
                    except Exception:
                        continue
        result['number'] = chosen_number

        # detect genre
        found_genre = None
        for token in doc:
            t = token.text.lower()
            if t in common_genres:
                found_genre = t
                break
        if not found_genre:
            for chunk in doc.noun_chunks:
                m = re.search(r"genre\s+(\w[\w\- ]*)", chunk.text.lower())
                if m:
                    found_genre = m.group(1).strip().split()[0]
                    break
        result['genre'] = found_genre

        if any(tok in s for tok in ['trending', 'trends', 'trend', 'popular', 'top', 'best']):
            result['sort_by'] = 'trending'

        return result

    # fallback heuristics
    # find numeric tokens
    chosen_number = None
    for m in re.finditer(r"\b(\d{1,4})\b", s):
        num_text = m.group(1)
        start = m.start()
        prefix = s[:start]
        left_tokens = re.findall(r"\w+", prefix)
        idx = len(left_tokens)
        tokens = re.findall(r"\w+|[^\s\w]+", s)
        window = tokens[max(0, idx-3): idx+4]
        # adjacency
        adjacent_to_content = any(t in content_nouns for t in window)
        has_count_indicator = any(t in window for t in ['give','show','suggest','recommend','top','best','list','find'])
        if adjacent_to_content and not has_count_indicator:
            continue
        try:
            n = int(num_text)
            if 1 <= n <= 500:
                chosen_number = n
                break
        except:
            pass
    result['number'] = chosen_number

    gm = re.search(r"genre[:\s]+([a-zA-Z\-\s]+)", s)
    if gm:
        result['genre'] = gm.group(1).strip().split()[0]
    else:
        for g in common_genres:
            if g in s:
                result['genre'] = g
                break

    if any(tok in s for tok in ['trending', 'trends', 'trend', 'popular', 'top', 'best']):
        result['sort_by'] = 'trending'

    return result


def pos_tag_extract_keywords(text, k=5):
    """Extract up to k keywords from text using POS tagging (prefer nouns/adjectives).
    Uses spaCy when available; falls back to a simple heuristic extractor.
    Returns a list of keywords (strings).
    """
    if not text or k is None or k <= 0:
        return []
    text = text.strip()
    # Use spaCy if available
    if nlp is not None:
        doc = nlp(text)
        # prefer NOUN, PROPN, ADJ
        candidates = []
        for token in doc:
            if token.is_stop or token.is_punct or token.like_num:
                continue
            if token.pos_ in ("NOUN", "PROPN", "ADJ"):
                candidates.append(token.lemma_.lower())
        # preserve order and uniqueness
        seen = set()
        keywords = []
        for w in candidates:
            if w not in seen and len(w) > 1:
                seen.add(w)
                keywords.append(w)
            if len(keywords) >= k:
                break
        return keywords

    # fallback naive extractor: return first k non-stop, non-numeric words
    stopwords = set(["the","a","an","and","or","of","in","on","for","with","to","from","by","about","that","this","it","is","are","be","as","at","you","your"])
    tokens = re.findall(r"\w+", text.lower())
    keywords = []
    seen = set()
    for t in tokens:
        if t in stopwords or t.isdigit() or len(t) <= 2:
            continue
        if t in seen:
            continue
        seen.add(t)
        keywords.append(t)
        if len(keywords) >= k:
            break
    return keywords
