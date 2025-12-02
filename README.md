# Synapse: Evolving Job-Person Fit​

## What Our Project Does

**Synapse** is an intelligent, two-phase job–candidate matching and resume optimization system designed to bridge the gap between traditional keyword-based job search and the nuanced, contextual reasoning required for real-world hiring.  
Our system integrates:

- **SBERT-based embedding retrieval** for fast, scalable job matching  
- **Semantic reranking** for deeper contextual understanding  
- **RAG-driven explanations** for transparency and user trust  
- **Differential-evolution resume optimization** enabling LLM-guided iterative improvements  

## Motivation

Modern job platforms (e.g., LinkedIn, Indeed) primarily rely on **titles, keywords, or surface-level text similarity**, leading to mismatches such as:

- Candidates receiving irrelevant recommendations that ignore domain expertise  
- Employers struggling to identify applicants with specific skill requirements or cultural fit  
- Hard requirements hidden inside long job descriptions being overlooked  

As shown in our analysis of prior approaches

- **Keyword/embedding-only methods** miss nuances (e.g., seniority, domain specificity)
- **Collaborative filtering** requires proprietary labeled data and suffers from cold-start issues
- **LLM scoring** is expensive, inconsistent, and not natively comparative
- None alone provide **speed + accuracy + interpretability**


### Our Goal  
Build a system that is:

- **Fast** – suitable for real-time interactive search  
- **Effective** – matching candidates with highly relevant job postings  
- **Explainable** – providing transparent, traceable reasoning  

Synapse aims to make job search and resume tuning **data-driven, interpretable**.


## How to start/run our project
### 1.Clone the Repository

```bash
git clone <your-repo-url>
cd SYNAPSE-PERSON-JOB-FIT
```

### 2.Install Dependencies
```
pip install -r requirements.txt
```


### 3.Prepare Environment Variables
```
GEMINI_API_KEY=your_key_here
GEMINI_API_BASE_URL=your_proxy_url
```

### 4.load the postings
To get started, run 
```python infill_commands.py``` to load the postings into the sqllite database and setup faiss.

### 5.Run Resume Optimization
```
python resume_optimization.py
```

This will: Generate mutated resumes, Rank them using Phase 1/Phase 2, Produce the highest-fitness version
### 6.Launch the Streamlit Web UI
```
streamlit run streamlit_ui.py
```

## Future Work

### Additional Evaluation

- Larger dataset with empirical annotations for quantitative evaluation and model tuning  
- User studies and surveys to assess usability, trust, and perceived relevance  

### Model Optimization

- Current performance gains are small; requires larger and more diverse training samples  
- Evaluate the model using additional metrics (e.g., ensemble scores, human expert ratings), not only Phase-2 metrics  

### System Performance Optimization

- Utilize more advanced big-data tooling (e.g., **Apache Spark**) to parallelize embedding and preprocessing bottlenecks  
- Leverage **GPU acceleration** for embedding models and LLM-based components  
- Improve backend scalability to support many concurrent user requests  
- Reduce LLM latency by using faster APIs or self-hosting open-source models on high-performance GPUs  


## DM ME FOR THE CONTENTS OF MY DOTENV, which should be placed at project root directory
## DM ME FOR THE CONTENTS OF postings.csv

To get started, run python infill_commands.py to load the postings into the sqllite database and setup faiss.

To understand the core logic, start with infill_commands and trace where that stuff goes, and then recommender.py, and follow that logic.

As a taxonomy:

linkedin_data/postings.csv: the dataset
database.py: sqllite database schema and connections
embed.py: functions that convert postings and resumes into string embeddings
infill_commands.py: setup script for database and faiss
llm_logging.py: utils for logging previous LLM calls
llm_starter_code.py: example usage of LLM with a query wrapper
recommender.py: core recommendation logic.