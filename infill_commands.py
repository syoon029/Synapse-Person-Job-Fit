from infrastructure.database import init_db, Posting, import_postings_from_csv, import_resumes_from_csv
from infrastructure.index import init_faiss_index

def setup_database():
    """
    Initialize the database and import job postings from a CSV file.
    """
    init_db()
    num_imported = import_postings_from_csv()
    print(f"Imported {num_imported} job postings.")
    num_imported = import_resumes_from_csv()
    print(f"Imported {num_imported} resumes.")

def setup_faiss_index():
    """
    Initialize the FAISS index for job postings.
    """
    init_faiss_index(recompute_embeddings=True)
    print("FAISS index initialized.")

if __name__ == "__main__":
    setup_database()
    setup_faiss_index()