from database import init_db, Posting, import_postings_from_csv

if __name__ == "__main__":
    init_db()
    num_imported = import_postings_from_csv()
    print(f"Imported {num_imported} job postings.")