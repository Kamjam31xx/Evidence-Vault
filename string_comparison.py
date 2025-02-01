from difflib import SequenceMatcher
import numpy as np

def string_similarity(str1, str2, method='levenshtein'):

    # Handle empty strings
    if not str1 or not str2:
        return 0.0
    
    # Normalize strings
    str1 = str1.lower().strip()
    str2 = str2.lower().strip()
    
    if method == 'levenshtein':
        return _levenshtein_similarity(str1, str2)
    elif method == 'jaccard':
        return _jaccard_similarity(str1, str2)
    elif method == 'sequence':
        return _sequence_matcher(str1, str2)
    elif method == 'combined':
        scores = [
            _levenshtein_similarity(str1, str2),
            _jaccard_similarity(str1, str2),
            _sequence_matcher(str1, str2)
        ]
        return np.mean(scores)
    else:
        raise ValueError("Invalid method. Choose from: 'levenshtein', 'jaccard', 'sequence', 'combined'")

def _levenshtein_similarity(s, t):

    # Compute Levenshtein distance
    rows = len(s) + 1
    cols = len(t) + 1
    distance = np.zeros((rows, cols), dtype=int)

    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if s[i-1] == t[j-1] else 2  # Substitution cost
            distance[i][j] = min(
                distance[i-1][j] + 1,     # Deletion
                distance[i][j-1] + 1,     # Insertion
                distance[i-1][j-1] + cost # Substitution
            )
    
    # Calculate similarity ratio
    max_len = max(len(s), len(t))
    if max_len == 0:
        return 1.0
    return 1 - (distance[-1][-1] / max_len)

def _jaccard_similarity(s, t):

    set1 = set(s.split())
    set2 = set(t.split())
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0

def _sequence_matcher(s, t):

    return SequenceMatcher(None, s, t).ratio()
