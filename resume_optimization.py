import re
import random
from time import sleep
from recommender import get_random_resumes, phase1_recommend
from llm_starter_code import get_response
from database import get_postings_by_ids
from embed import embed_resume_text, embed_posting
from index import compute_l2_distance
import numpy as np
import json

# --- Tunable Parameters ---

N_CANDIDATES = 6 # Population Size: Number of candidate resumes in each generation
N_ITERATIONS = 3 # Generations: Number of evolution iterations to run
N_ELITE = 2 # Elitism: Number of top candidates to keep directly in the next generation
TOURNAMENT_SIZE = 3 # Tournament Size: Number of candidates in a selection tournament
MUTATION_RATE = 0.20 # Mutation Rate: Probability that a new child will be mutated
LLM_MODEL = "models/gemini-2.0-flash" # LLM Model: The model used for scoring and mutation

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

def calculate_fitness(resume_text: str, postings: list) -> float:
    """
    Calculates the fitness of a single resume against a list of postings.
    This is the "extensible" fitness function.
    """
    total_score = 0
    num_postings = len(postings)

    if num_postings == 0:
        return 0.0

    print(f"   Scoring resume (hash: {hash(resume_text)})...")

    for posting in postings:
        # Embed both resume and posting for similarity calculation
        resume_embedding = embed_resume_text(resume_text)
        posting_embedding = np.array(json.loads(embed_posting(posting)), dtype='float32')
        similarity = 1 - compute_l2_distance(resume_embedding, posting_embedding)
        # print(f"     - Similarity for '{posting.title}': {similarity:.4f}")
        total_score += similarity
        # Compute similarity (dot product)


        # job_title = posting.title
        # job_description = posting.description
        # job_skills = posting.skills_desc if posting.skills_desc else ""

        # prompt = SCORING_PROMPT_TEMPLATE.format(
        #     job_title=job_title,
        #     job_description=job_description,
        #     job_skills=job_skills,
        #     resume_text=resume_text
        # )

        # try:
        #     response = get_response(LLM_MODEL, prompt, logging=True)
        #     score = _parse_score(response)
        #     print(f"     - Score for '{job_title}': {score}")
        #     total_score += score
        # except Exception as e:
        #     print(f"   [!] LLM call failed for scoring: {e}")
        #     sleep(1)
            
    # Fitness is the average score across all target postings
    average_score = total_score / num_postings
    print(f"   -> Average Fitness: {average_score:.2f}")
    return average_score

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
        mutated_resume = get_response(LLM_MODEL, prompt, logging=True)
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
    print("    Crossover... ", end="")
    job_context = _format_postings_for_prompt(postings)
    prompt = CROSSOVER_PROMPT_TEMPLATE.format(
        resume_a=resume_a,
        resume_b=resume_b,
        job_context=job_context
    )

    try:
        child_resume = get_response(LLM_MODEL, prompt, logging=True)
        if len(child_resume) < 0.5 * min(len(resume_a), len(resume_b)):
            print(f"   [!] Crossover failed, response too short. Using Parent A.")
            return resume_a
        return child_resume
    except Exception as e:
        print(f"   [!] LLM call failed for crossover: {e}. Using Parent A.")
        sleep(1)
        return resume_a

def _select_parent(population: list) -> dict:
    """
    Selects a parent from the population using tournament selection.
    """
    # Pick TOURNAMENT_SIZE individuals at random
    tournament = random.sample(population, TOURNAMENT_SIZE)
    # Return the best individual from that tournament
    tournament.sort(key=lambda x: x['fitness'], reverse=True)
    return tournament[0]

def run_evolution(base_resume: str, target_postings: list):
    """
    Main evolutionary loop.
    """
    print("--- Starting Resume Evolution ---")
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
    for _ in range(N_CANDIDATES - 1):
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
        print(f"  Stats (Gen {gen-1}): Best Fitness={best_fitness:.2f}, Avg Fitness={avg_fitness:.2f}")

        new_population = []

        # **Elitism:** Keep the top N_ELITE candidates
        print(f"  Copying {N_ELITE} elite candidates...")
        new_population.extend(population[:N_ELITE])

        # **Crossover & Mutation:** Fill the rest of the generation
        n_to_generate = N_CANDIDATES - N_ELITE
        print(f"  Generating {n_to_generate} new children via crossover/mutation...")
        
        for _ in range(n_to_generate):
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

    print(f"  Final Stats: Best Fitness={best_fitness:.2f}, Avg Fitness={avg_fitness:.2f}")
    
    return best_candidate


if __name__ == "__main__":
    # 1. Your provided setup code
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
    print(f"Best Resume Found (Fitness: {best_result['fitness']:.2f})")
    print("="*50)
    print(best_result['resume'])