import re
import random
from time import sleep
from recommender import get_random_resumes, phase1_recommend
from llm_starter_code import get_response
from database import get_postings_by_ids
from embed import embed_resume_text, embed_posting, _safe_join
from embed_stage2 import embed_text, doc_sim_score
from index import compute_l2_distance
from llm_scorer import load_cache, pairwise_rank_to_scores
import numpy as np
import json
from scipy.stats import rankdata

# --- Tunable Parameters ---

N_CANDIDATES = 6 # Population Size: Number of candidate resumes in each generation
N_ITERATIONS = 3 # Generations: Number of evolution iterations to run
N_ELITE = 2 # Elitism: Number of top candidates to keep directly in the next generation
TOURNAMENT_SIZE = 3 # Tournament Size: Number of candidates in a selection tournament
MUTATION_RATE = 0.20 # Mutation Rate: Probability that a new child will be mutated
LLM_MODEL = "models/gemini-2.0-flash" # LLM Model: The model used for scoring and mutation

# Fitness calculation method: 'simple', 'phase2', or 'advanced'
FITNESS_METHOD = 'advanced'  # 'simple' = Phase 1 only, 'phase2' = Phase 1 + Phase 2, 'advanced' = Full ensemble

# --- Global cache for LLM pairwise comparisons ---
LLM_CACHE = None

def initialize_llm_cache():
    """Load the LLM cache once at startup"""
    global LLM_CACHE
    if LLM_CACHE is None:
        print("[Fitness] Loading LLM cache...")
        LLM_CACHE = load_cache()
        print(f"[Fitness] Loaded {len(LLM_CACHE)} cached comparisons")

# --- Prompt Templates ---
MUTATION_PROMPT_TEMPLATE = """
**Role:** AI Resume Editor (Mutation Operator)

**Task:** Make *one small, targeted improvement* to the provided resume to increase its "fit score" for the target job postings.

**Base Resume:**
---
{resume_text}
---

**Target Job Postings (for context):**
---
{job_context}
---

**Instructions:**
1.  Review the resume and the target jobs.
2.  Identify *one* small change to make. Examples:
    - Rephrase a bullet point in the "Experience" section to use stronger action verbs and metrics.
    - Slightly tweak the "Summary" to include keywords from the job postings.
    - Adjust the ordering or emphasis of a skill in the "Skills" section.
3.  Apply this single change to the resume.
4.  Return the *entire, updated* resume text. **Do not** provide any explanation, preamble, or commentary. Output *only* the full resume.
"""

CROSSOVER_PROMPT_TEMPLATE = """
**Role:** AI Resume Editor (Crossover Operator)

**Task:** Create a new, stronger "child" resume by combining the *best* parts of two "parent" resumes. The child resume must be coherent and effective.

**Parent Resume A:**
---
{resume_a}
---

**Parent Resume B:**
---
{resume_b}
---

**Target Job Postings (for context):**
---
{job_context}
---

**Instructions:**
1.  Analyze both parent resumes and the target job context.
2.  Identify the strongest *sections* (e.g., Summary, a specific Experience entry, Skills section) from each parent.
3.  Construct a *new, single, coherent resume* by merging these best parts. For example, you might take the Summary from Parent A, the Experience from Parent B, and the Skills section from Parent A.
4.  Ensure the final resume is well-formatted and does not duplicate information.
5.  Return the *entire, updated* child resume. **Do not** provide any explanation, preamble, or commentary. Output *only* the full resume.
"""

def _format_postings_for_prompt(postings: list) -> str:
    """Creates a concise summary of job postings for the prompts."""
    context = ""
    for i, posting in enumerate(postings):
        context += f"**Job {i+1}: {posting.title}**\n"
        desc = posting.skills_desc if posting.skills_desc else posting.description
        context += f"Key Skills/Desc: {desc[:250]}...\n\n"
    return context

# --- Advanced Fitness Calculation Functions ---

def normalize_scores_rank(scores_array):
    """
    Rank-based normalization (convert to percentile ranks)
    Returns array normalized to [0, 1]
    """
    scores_clean = scores_array.copy()
    scores_clean[scores_clean == -np.inf] = -999.0
    
    ranks = rankdata(scores_clean, method='average')
    
    if len(ranks) > 1:
        normalized = (ranks - 1) / (len(ranks) - 1)
    else:
        normalized = np.ones_like(ranks)
    
    return normalized

def calculate_fitness_simple(resume_text: str, postings: list) -> float:
    """
    Simple fitness: Phase 1 embeddings only (L2 distance).
    Fastest but least accurate.
    """
    total_score = 0
    num_postings = len(postings)

    if num_postings == 0:
        return 0.0

    resume_embedding = embed_resume_text(resume_text)
    
    for posting in postings:
        posting_embedding = np.array(json.loads(embed_posting(posting)), dtype='float32')
        similarity = 1 - compute_l2_distance(resume_embedding, posting_embedding)
        total_score += similarity
    
    return total_score / num_postings

def calculate_fitness_phase2(resume_text: str, postings: list) -> float:
    """
    Phase 2 fitness: Uses doc_sim_score from embed_stage2.
    Better semantic understanding than Phase 1.
    """
    total_score = 0
    num_postings = len(postings)

    if num_postings == 0:
        return 0.0

    resume_embedding = embed_text(resume_text)
    
    for posting in postings:
        posting_embedding = embed_text(_safe_join([
            posting.title,
            posting.company_name,
            posting.skills_desc,
            posting.description,
            posting.formatted_experience_level,
            posting.location
        ]))
        
        similarity = doc_sim_score(resume_embedding, posting_embedding)
        total_score += similarity
    
    return total_score / num_postings

def calculate_fitness_advanced(resume_text: str, postings: list) -> float:
    """
    Advanced fitness: Ensemble of Phase 1, Phase 2, and LLM pairwise ranking.
    Uses the best performing method from anselbench: WAvg_rank_80-10-10
    
    Most accurate but slower (especially without cache).
    """
    num_postings = len(postings)
    if num_postings == 0:
        return 0.0
    
    # Initialize LLM cache if needed
    initialize_llm_cache()
    
    # --- 1. Calculate Phase 1 scores (embedding similarity) ---
    phase1_scores = np.zeros(num_postings)
    resume_embedding_p1 = embed_resume_text(resume_text)
    
    for i, posting in enumerate(postings):
        posting_embedding = np.array(json.loads(embed_posting(posting)), dtype='float32')
        similarity = 1 - compute_l2_distance(resume_embedding_p1, posting_embedding)
        phase1_scores[i] = similarity
    
    # --- 2. Calculate Phase 2 scores (doc_sim_score) ---
    phase2_scores = np.zeros(num_postings)
    resume_embedding_p2 = embed_text(resume_text)
    
    for i, posting in enumerate(postings):
        posting_embedding = embed_text(_safe_join([
            posting.title,
            posting.company_name,
            posting.skills_desc,
            posting.description,
            posting.formatted_experience_level,
            posting.location
        ]))
        
        similarity = doc_sim_score(resume_embedding_p2, posting_embedding)
        phase2_scores[i] = similarity
    
    # --- 3. Calculate LLM pairwise ranking scores ---
    postings_texts = []
    for posting in postings:
        posting_text = _safe_join([
            f"Title: {posting.title}",
            f"Company: {posting.company_name}",
            f"Location: {posting.location}",
            f"Experience Level: {posting.formatted_experience_level}",
            f"Skills: {posting.skills_desc}",
            f"Description: {posting.description}"
        ])
        postings_texts.append(posting_text)
    
    llm_scores = np.array(pairwise_rank_to_scores(
        resume_text=resume_text,
        postings_texts=postings_texts,
        cache=LLM_CACHE
    ))
    
    # --- 4. Ensemble with rank-based normalization (80-10-10) ---
    norm_p1 = normalize_scores_rank(phase1_scores)
    norm_p2 = normalize_scores_rank(phase2_scores)
    norm_llm = normalize_scores_rank(llm_scores)
    
    # Weighted average: 80% Phase 1, 10% Phase 2, 10% LLM
    ensemble_scores = 0.8 * norm_p1 + 0.1 * norm_p2 + 0.1 * norm_llm
    
    # Return average ensemble score across all postings
    return float(np.mean(ensemble_scores))

def calculate_fitness(resume_text: str, postings: list) -> float:
    """
    Main fitness function that dispatches to the appropriate method.
    """
    print(f"   Scoring resume (method: {FITNESS_METHOD}, hash: {hash(resume_text) % 100000})...")
    
    if FITNESS_METHOD == 'simple':
        fitness = calculate_fitness_simple(resume_text, postings)
    elif FITNESS_METHOD == 'phase2':
        fitness = calculate_fitness_phase2(resume_text, postings)
    elif FITNESS_METHOD == 'advanced':
        fitness = calculate_fitness_advanced(resume_text, postings)
    else:
        raise ValueError(f"Unknown fitness method: {FITNESS_METHOD}")
    
    print(f"   -> Fitness: {fitness:.4f}")
    return fitness

# --- Genetic Operators ---

def mutate_resume(base_resume_text: str, postings: list) -> str:
    """
    Uses the LLM to create a "mutated" version of the base resume.
    """
    print("    Mutating resume...")
    job_context = _format_postings_for_prompt(postings)
    prompt = MUTATION_PROMPT_TEMPLATE.format(
        resume_text=base_resume_text,
        job_context=job_context
    )

    try:
        mutated_resume = get_response(LLM_MODEL, prompt, logging=False)
        if len(mutated_resume) < 0.5 * len(base_resume_text):
             print(f"   [!] Mutation failed, response too short. Re-using parent.")
             return base_resume_text
        return mutated_resume
    except Exception as e:
        print(f"   [!] LLM call failed for mutation: {e}. Re-using parent.")
        sleep(1)
        return base_resume_text

def crossover_resumes(resume_a: str, resume_b: str, postings: list) -> str:
    """
    Uses the LLM to create a "child" resume by combining two parents.
    """
    print("    Crossover...", end=" ")
    job_context = _format_postings_for_prompt(postings)
    prompt = CROSSOVER_PROMPT_TEMPLATE.format(
        resume_a=resume_a,
        resume_b=resume_b,
        job_context=job_context
    )

    try:
        child_resume = get_response(LLM_MODEL, prompt, logging=False)
        if len(child_resume) < 0.5 * min(len(resume_a), len(resume_b)):
            print(f"   [!] Crossover failed, response too short. Using Parent A.")
            return resume_a
        print("Done.")
        return child_resume
    except Exception as e:
        print(f"   [!] LLM call failed for crossover: {e}. Using Parent A.")
        sleep(1)
        return resume_a

def _select_parent(population: list) -> dict:
    """
    Selects a parent from the population using tournament selection.
    """
    tournament = random.sample(population, TOURNAMENT_SIZE)
    tournament.sort(key=lambda x: x['fitness'], reverse=True)
    return tournament[0]

# --- Main Evolution Loop ---

def run_evolution(base_resume: str, target_postings: list):
    """
    Main evolutionary loop.
    """
    print("--- Starting Resume Evolution ---")
    print(f"Fitness Method: {FITNESS_METHOD.upper()}")
    print(f"Population: {N_CANDIDATES} | Generations: {N_ITERATIONS} | Elites: {N_ELITE}")
    print(f"Mutation Rate: {MUTATION_RATE*100}% | Tournament Size: {TOURNAMENT_SIZE}")
    print(f"Targeting {len(target_postings)} Job Postings.")

    population = []

    # --- 1. Initialization (Generation 0) ---
    print("\n--- Generation 0: Initialization ---")
    
    # Add the base resume
    base_fitness = calculate_fitness(base_resume, target_postings)
    population.append({"resume": base_resume, "fitness": base_fitness})

    # Create the rest of the initial population by mutating the base resume
    for i in range(N_CANDIDATES - 1):
        print(f"  Creating initial candidate {i+2}/{N_CANDIDATES}...")
        mutant_resume = mutate_resume(base_resume, target_postings)
        mutant_fitness = calculate_fitness(mutant_resume, target_postings)
        population.append({"resume": mutant_resume, "fitness": mutant_fitness})

    # --- 2. Evolution Loop ---
    for gen in range(1, N_ITERATIONS + 1):
        print(f"\n--- Generation {gen} / {N_ITERATIONS} ---")

        # Sort by fitness (highest first) for elitism and reporting
        population.sort(key=lambda x: x['fitness'], reverse=True)

        # Print stats
        best_fitness = population[0]['fitness']
        avg_fitness = sum(c['fitness'] for c in population) / len(population)
        print(f"  Stats (Gen {gen-1}): Best Fitness={best_fitness:.4f}, Avg Fitness={avg_fitness:.4f}")

        new_population = []

        # **Elitism:** Keep the top N_ELITE candidates
        print(f"  Copying {N_ELITE} elite candidates...")
        new_population.extend(population[:N_ELITE])

        # **Crossover & Mutation:** Fill the rest of the generation
        n_to_generate = N_CANDIDATES - N_ELITE
        print(f"  Generating {n_to_generate} new children via crossover/mutation...")
        
        for i in range(n_to_generate):
            print(f"    Child {i+1}/{n_to_generate}:")
            # Select two parents
            parent_a = _select_parent(population)
            parent_b = _select_parent(population)
            
            # **Crossover:**
            child_resume = crossover_resumes(parent_a['resume'], parent_b['resume'], target_postings)
            
            # **Mutation:**
            if random.random() < MUTATION_RATE:
                child_resume = mutate_resume(child_resume, target_postings)
            
            # Score the new child and add to population
            child_fitness = calculate_fitness(child_resume, target_postings)
            new_population.append({"resume": child_resume, "fitness": child_fitness})

        population = new_population

    # --- 3. Final Result ---
    print("\n--- Evolution Complete ---")
    population.sort(key=lambda x: x['fitness'], reverse=True)
    
    best_candidate = population[0]
    best_fitness = best_candidate['fitness']
    avg_fitness = sum(c['fitness'] for c in population) / len(population)

    print(f"  Final Stats: Best Fitness={best_fitness:.4f}, Avg Fitness={avg_fitness:.4f}")
    
    return best_candidate


if __name__ == "__main__":
    # 1. Setup
    resume = get_random_resumes(n=1)[0]
    scores, ids = phase1_recommend(resume)
    
    # Target postings: top 3 from phase1 results
    target_posting_ids = ids[:3]
    postings = get_postings_by_ids(target_posting_ids)
    
    print(f"Target Posting IDs for optimization: {target_posting_ids}")
    print(f"Base resume (first 100 chars): {resume.text[:100].replace('\n', ' ')}...")
    print("-" * 30)

    # 2. Run the optimization algorithm
    best_result = run_evolution(
        base_resume=resume.text,
        target_postings=postings
    )

    # 3. Print the final, optimized resume
    print("\n" + "="*50)
    print(f"Best Resume Found (Fitness: {best_result['fitness']:.4f})")
    print("="*50)
    print(best_result['resume'])