import json
import time
import os
import hashlib
from typing import Dict, Any, List, Tuple
from llm_starter_code import get_response

# --- Caching Functions ---

CACHE_FILE = 'llm_pairwise_cache.json'

def load_cache() -> Dict[str, Any]:
    """Loads the LLM pairwise comparison cache from disk if it exists."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Cache file {CACHE_FILE} is corrupted. Starting fresh.")
            return {}
    return {}

def save_cache(cache_data: Dict[str, Any]):
    """Saves the given cache data to disk."""
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache_data, f, indent=2)


# --- Pairwise Comparison Function ---

def compare_postings(resume_text: str, posting_a_text: str, posting_b_text: str, 
                     cache: Dict[str, Any], comparison_count: List[int]) -> int:
    """
    Compares two job postings against a resume using an LLM.
    Returns: 1 if posting_a is better, -1 if posting_b is better, 0 if equal
    
    Args:
        comparison_count: A list with one element used as a mutable counter
    """
    
    # 1. Create a unique, stable key for caching (order-independent)
    pair_key_forward = hashlib.sha256((resume_text + posting_a_text + posting_b_text).encode('utf-8')).hexdigest()
    pair_key_reverse = hashlib.sha256((resume_text + posting_b_text + posting_a_text).encode('utf-8')).hexdigest()
    
    # 2. Check cache (both directions)
    if pair_key_forward in cache:
        print("  > Cache HIT (forward).")
        return cache[pair_key_forward]
    
    if pair_key_reverse in cache:
        print("  > Cache HIT (reverse).")
        # Reverse the result since we're asking in opposite order
        return -cache[pair_key_reverse]
    
    # Increment comparison counter
    comparison_count[0] += 1
    print(f"  > Cache MISS. API call #{comparison_count[0]}...")
    
    # 3. Define the prompt for pairwise comparison
    prompt = f"""
You are an expert technical recruiter. Your task is to compare TWO job postings and determine which one is a better match for a candidate's RESUME.

Consider:
- Skills alignment (most important)
- Experience level match (very important - be strict about seniority)
- Domain/industry fit
- Role responsibilities alignment
- Career trajectory fit

Respond *only* with a JSON object in a markdown code block:
```json
{{
    "reasoning": "Brief 1-2 sentence explanation of your choice.",
    "choice": "A" or "B" or "EQUAL"
}}
```

---[RESUME]---
{resume_text}
---[END RESUME]---

---[POSTING A]---
{posting_a_text}
---[END POSTING A]---

---[POSTING B]---
{posting_b_text}
---[END POSTING B]---

Your JSON response:
"""

    # 4. Call the LLM
    model_to_use = "models/gemini-2.0-flash"
    response_str = get_response(model_to_use, prompt, max_tokens=256, temperature=0.0)
    
    # 5. Rate limiting delay
    print("  > API call complete. Waiting 1 seconds...")
    print("  > Wait complete.")

    # 6. Parse the response
    try:
        json_str = response_str.split("```json")[1].split("```")[0].strip()
        data = json.loads(json_str)
        choice = data.get("choice", "EQUAL").upper()
        
        # Convert choice to numeric result
        if choice == "A":
            result = 1
        elif choice == "B":
            result = -1
        else:  # EQUAL
            result = 0
        
        # 7. Update cache and save
        cache[pair_key_forward] = result
        save_cache(cache)
        
        return result
        
    except Exception as e:
        print(f"  > ERROR: Failed to parse LLM response: {e}")
        print(f"  > Response was: {response_str}")
        # Return 0 (equal) on failure to avoid biasing results
        return 0


# --- Merge Sort with Pairwise Comparisons ---

def merge_sort_pairwise(resume_text: str, postings_with_indices: List[Tuple[int, str]], 
                        cache: Dict[str, Any], comparison_count: List[int]) -> List[Tuple[int, str]]:
    """
    Sorts postings using merge sort with LLM pairwise comparisons.
    This is O(n log n) comparisons - optimal for comparison-based sorting.
    
    Args:
        resume_text: The candidate's resume
        postings_with_indices: List of (original_index, posting_text) tuples
        cache: Cache dictionary for API results
        comparison_count: Mutable counter for tracking API calls
    
    Returns:
        Sorted list of (original_index, posting_text) tuples (best to worst)
    """
    # Base case
    if len(postings_with_indices) <= 1:
        return postings_with_indices
    
    # Divide
    mid = len(postings_with_indices) // 2
    left = merge_sort_pairwise(resume_text, postings_with_indices[:mid], cache, comparison_count)
    right = merge_sort_pairwise(resume_text, postings_with_indices[mid:], cache, comparison_count)
    
    # Conquer (merge with pairwise comparisons)
    return merge_with_comparisons(resume_text, left, right, cache, comparison_count)


def merge_with_comparisons(resume_text: str, 
                          left: List[Tuple[int, str]], 
                          right: List[Tuple[int, str]],
                          cache: Dict[str, Any],
                          comparison_count: List[int]) -> List[Tuple[int, str]]:
    """
    Merges two sorted lists using pairwise comparisons.
    """
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        left_idx, left_text = left[i]
        right_idx, right_text = right[j]
        
        # Compare: 1 means left is better, -1 means right is better
        comparison = compare_postings(resume_text, left_text, right_text, cache, comparison_count)
        
        if comparison >= 0:  # left is better or equal
            result.append(left[i])
            i += 1
        else:  # right is better
            result.append(right[j])
            j += 1
    
    # Append remaining elements
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result


# --- Main Ranking Function ---

def pairwise_rank_to_scores(resume_text: str, postings_texts: List[str], 
                           cache: Dict[str, Any]) -> List[float]:
    """
    Ranks postings using merge sort with pairwise comparisons.
    
    Complexity: O(n log n) comparisons
    - For 50 postings: ~282 comparisons (50 * log2(50) * 1.15)
    - For 100 postings: ~664 comparisons
    
    This is the theoretical minimum for comparison-based sorting!
    
    Args:
        resume_text: The candidate's resume
        postings_texts: List of all posting texts
        cache: Cache dictionary for API results
    
    Returns:
        List of scores (0-1 normalized by rank) for each posting
    """
    n = len(postings_texts)
    
    print(f"\nPerforming merge sort with pairwise comparisons...")
    print(f"Expected comparisons: ~{int(n * (n.bit_length() - 1) * 1.15)} (theoretical: {n} * log2({n}))")
    
    # Track number of API calls made
    comparison_count = [0]
    
    # Create list of (original_index, posting_text) tuples
    postings_with_indices = list(enumerate(postings_texts))
    
    # Sort using merge sort with pairwise comparisons
    sorted_postings = merge_sort_pairwise(resume_text, postings_with_indices, cache, comparison_count)
    
    print(f"\nTotal comparisons made: {comparison_count[0]}")
    
    # Convert sorted order to scores
    # Best posting gets score close to 1.0, worst gets score close to 0.0
    scores = [0.0] * n
    for rank, (original_idx, _) in enumerate(sorted_postings):
        # Rank 0 (best) -> score = 1.0
        # Rank n-1 (worst) -> score = 0.0
        # Use (n - 1 - rank) / (n - 1) for smooth distribution
        if n > 1:
            scores[original_idx] = (n - 1 - rank) / (n - 1)
        else:
            scores[original_idx] = 1.0
    
    return scores


# --- Backward compatibility ---

def get_llm_score(resume_text: str, posting_text: str, cache: Dict[str, Any]) -> float:
    """
    Legacy interface: scores a single posting.
    This is less efficient and less accurate than pairwise ranking.
    For best results, use pairwise_rank_to_scores() instead.
    """
    print("Warning: get_llm_score() is deprecated. Use pairwise_rank_to_scores() for better results.")
    
    # We can't do proper pairwise without other postings, so fall back to basic scoring
    hash_key = hashlib.sha256((resume_text + posting_text).encode('utf-8')).hexdigest()
    
    if hash_key in cache:
        return cache[hash_key]
    
    prompt = f"""
You are an expert technical recruiter. Rate this job posting match for the candidate from 0-100.

---[RESUME]---
{resume_text}
---[END RESUME]---

---[POSTING]---
{posting_text}
---[END POSTING]---

Respond with JSON:
```json
{{"score": <0-100>}}
```
"""
    
    model_to_use = "models/gemini-2.5-flash"
    response_str = get_response(model_to_use, prompt, max_tokens=256, temperature=0.0)
    time.sleep(1)
    
    try:
        json_str = response_str.split("```json")[1].split("```")[0].strip()
        data = json.loads(json_str)
        score = float(data.get("score", 50)) / 100.0  # Normalize to 0-1
        cache[hash_key] = score
        save_cache(cache)
        return score
    except:
        return 0.5