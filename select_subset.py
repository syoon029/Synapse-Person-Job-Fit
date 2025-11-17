import json, random
from database import get_all_resumes

def create_resume_subset(n=100, seed=42, out_path="resume_subset.json"):
    resumes = get_all_resumes()
    random.seed(seed)
    random.shuffle(resumes)
    
    ids = [r.id for r in resumes[:n]]
    
    with open(out_path, "w") as f:
        json.dump(ids, f)
    
    print(f"Saved {len(ids)} resume IDs to {out_path}")

if __name__ == "__main__":
    create_resume_subset()
