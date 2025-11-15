import json
from database import get_all_resumes, get_resume_candidates 
from typing import List, Dict, Any

def check_reranking_success(resume_id: int) -> dict:
    """
    Checks if Phase 2 re-ranking successfully found a new Top 1 posting 
    with a higher Phase 2 score than the one selected by Phase 1.
    """
    candidates_data = get_resume_candidates(resume_id)
    
    if not candidates_data or not candidates_data.get("candidate_list"):
        # Handle cases where no candidate data exists for the resume ID
        return {"resume_id": resume_id, "success": False, "reason": "No candidate data found."}

    candidate_list = candidates_data["candidate_list"]
    
    # 1. Find Phase 1's Top 1 candidate (based on phase1_score)
    top_p1_candidate = max(candidate_list, key=lambda x: x.get("phase1_score", -1.0))
    
    # 2. Find Phase 2's Top 1 candidate (based on phase2_score)
    top_p2_candidate = max(candidate_list, key=lambda x: x.get("phase2_score", -1.0))
    
    # --- Extract Key Comparison Metrics ---
    
    # A. Phase 2 score of the candidate selected by Phase 2 (The actual highest score)
    score_p2_top1 = top_p2_candidate.get("phase2_score", -1.0)
    
    # B. Phase 2 score of the candidate selected by Phase 1 (The score Phase 2 gave to P1's choice)
    score_p1_top1_by_p2 = top_p1_candidate.get("phase2_score", -1.0)

    # 3. Compare for Re-ranking Success
    # Success if P2's best score is strictly greater than the P2 score of P1's best choice.
    reranking_successful = score_p2_top1 > score_p1_top1_by_p2
    
    # 4. Compile Results
    result = {
        "resume_id": resume_id,
        "success": reranking_successful,
        "Phase1_Top1_ID": top_p1_candidate["id"],
        "Phase2_Top1_ID": top_p2_candidate["id"],
        # Measure improvement based on P2 scores
        "Improvement": score_p2_top1 - score_p1_top1_by_p2
    }
    
    return result

# --- Main function to iterate over all resumes ---
def evaluate_all_resumes():
    """
    Evaluates the effectiveness of Phase 2 re-ranking for all resumes in the database.
    """
    # Fetch all Resume objects from the database
    all_resumes = get_all_resumes()
    
    evaluation_results = []
    
    print(f"Starting evaluation for a total of {len(all_resumes)} resumes.")
    
    successful_count = 0
    total_evaluated = 0
    
    for resume in all_resumes:
        try:
            result = check_reranking_success(resume.id)
            
            # Only count if the candidate data existed (i.e., not a 'reason' dictionary)
            if "reason" not in result:
                evaluation_results.append(result)
                total_evaluated += 1
                
                if result["success"]:
                    successful_count += 1
                
                # Print progress
                status = 'SUCCESS' if result['success'] else 'FAILURE'
                print(f"  - ID {resume.id}: {status} (Improvement: {result['Improvement']:+.4f})")
                
            else:
                 print(f"  - ID {resume.id}: No data available for evaluation.")


        except Exception as e:
            # Handle unexpected database or processing errors
            print(f"  - ID {resume.id}:  ERROR - {e}")

    # --- Final Summary ---
    print("\n" + "="*50)
    print("Final Re-ranking Effectiveness Summary")
    print(f"Total Resumes Evaluated: {total_evaluated} samples")
    print(f"Re-ranking Success Count (P2 Top Score > P1 Top Score): {successful_count} cases")
    
    if total_evaluated > 0:
        success_rate = (successful_count / total_evaluated) * 100
        print(f"**Re-ranking Success Rate: {success_rate:.2f}%")
    print("="*50)

    return evaluation_results

if __name__ == "__main__":
    final_results = evaluate_all_resumes()