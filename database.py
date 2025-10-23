import os
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Float, select, Text, UniqueConstraint 
from sqlalchemy.orm import sessionmaker, relationship, Session, declarative_base
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from embed import embed_posting

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///postings.db")
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "faiss_index.index")

Base = declarative_base()


class Posting(Base):
    """SQLAlchemy model for a job posting. Fields are inferred from the CSV headers.

    Primary key: id (autoincrement). Unique constraint on job_id if available.
    """
    __tablename__ = "postings"
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String, unique=True, index=True, nullable=True)
    company_name = Column(String, nullable=True)
    title = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    max_salary = Column(Float, nullable=True)
    pay_period = Column(String, nullable=True)
    location = Column(String, nullable=True)
    company_id = Column(String, nullable=True)
    views = Column(Integer, nullable=True)
    med_salary = Column(Float, nullable=True)
    min_salary = Column(Float, nullable=True)
    formatted_work_type = Column(String, nullable=True)
    applies = Column(Integer, nullable=True)
    original_listed_time = Column(String, nullable=True)
    remote_allowed = Column(String, nullable=True)
    job_posting_url = Column(String, nullable=True)
    application_url = Column(String, nullable=True)
    application_type = Column(String, nullable=True)
    expiry = Column(String, nullable=True)
    closed_time = Column(String, nullable=True)
    formatted_experience_level = Column(String, nullable=True)
    skills_desc = Column(Text, nullable=True)
    listed_time = Column(String, nullable=True)
    posting_domain = Column(String, nullable=True)
    sponsored = Column(String, nullable=True)
    work_type = Column(String, nullable=True)
    currency = Column(String, nullable=True)
    compensation_type = Column(String, nullable=True)
    normalized_salary = Column(Float, nullable=True)
    zip_code = Column(String, nullable=True)
    fips = Column(String, nullable=True)
    embedding = Column(Text, nullable=True)  # Store embedding as string representation of array

    def __repr__(self) -> str:  # pragma: no cover - small helper
        return f"<Posting(id={self.id} job_id={self.job_id} title={self.title})>"


def _parse_float(val):
    try:
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        v = str(val).strip()
        if v == "":
            return None
        return float(v.replace(",", ""))
    except Exception:
        return None


def _parse_int(val):
    try:
        if val is None:
            return None
        if isinstance(val, int):
            return val
        v = str(val).strip()
        if v == "":
            return None
        return int(float(v))
    except Exception:
        return None


def import_postings_from_csv(csv_path: str = "linkedin_data/postings.csv", commit_every: int = 200, chunksize: int = 1000):
    """Import postings from the CSV into the database.

    - Reads the CSV in chunks (pandas) to avoid large memory usage.
    - Converts numeric fields and attempts simple infill for `normalized_salary` when missing.
    - Commits rows in batches (commit_every) for efficiency.
    """
    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError("pandas is required to import CSV files. Install it in your environment.") from e

    sess = Session()
    inserted = 0
    try:
        for chunk in pd.read_csv(csv_path, chunksize=chunksize, dtype=str, na_values=["", "NA", "None"]):
            records = []
            for _, row in chunk.iterrows():
                # map CSV columns to model fields; use .get to avoid KeyError
                job_id = row.get("job_id") if "job_id" in row.index else None
                # handle numeric conversions
                min_salary = _parse_float(row.get("min_salary") if "min_salary" in row.index else None)
                med_salary = _parse_float(row.get("med_salary") if "med_salary" in row.index else None)
                max_salary = _parse_float(row.get("max_salary") if "max_salary" in row.index else None)
                views = _parse_int(row.get("views") if "views" in row.index else None)
                applies = _parse_int(row.get("applies") if "applies" in row.index else None)

                # compute normalized salary: prefer med, else average of min/max, else None
                normalized_salary = None
                if med_salary is not None:
                    normalized_salary = med_salary
                elif min_salary is not None and max_salary is not None:
                    normalized_salary = (min_salary + max_salary) / 2.0
                elif min_salary is not None:
                    normalized_salary = min_salary
                elif max_salary is not None:
                    normalized_salary = max_salary

                p = Posting(
                    job_id=job_id,
                    company_name=row.get("company_name") if "company_name" in row.index else None,
                    title=row.get("title") if "title" in row.index else None,
                    description=row.get("description") if "description" in row.index else None,
                    max_salary=max_salary,
                    pay_period=row.get("pay_period") if "pay_period" in row.index else None,
                    location=row.get("location") if "location" in row.index else None,
                    company_id=row.get("company_id") if "company_id" in row.index else None,
                    views=views,
                    med_salary=med_salary,
                    min_salary=min_salary,
                    formatted_work_type=row.get("formatted_work_type") if "formatted_work_type" in row.index else None,
                    applies=applies,
                    original_listed_time=row.get("original_listed_time") if "original_listed_time" in row.index else None,
                    remote_allowed=row.get("remote_allowed") if "remote_allowed" in row.index else None,
                    job_posting_url=row.get("job_posting_url") if "job_posting_url" in row.index else None,
                    application_url=row.get("application_url") if "application_url" in row.index else None,
                    application_type=row.get("application_type") if "application_type" in row.index else None,
                    expiry=row.get("expiry") if "expiry" in row.index else None,
                    closed_time=row.get("closed_time") if "closed_time" in row.index else None,
                    formatted_experience_level=row.get("formatted_experience_level") if "formatted_experience_level" in row.index else None,
                    skills_desc=row.get("skills_desc") if "skills_desc" in row.index else None,
                    listed_time=row.get("listed_time") if "listed_time" in row.index else None,
                    posting_domain=row.get("posting_domain") if "posting_domain" in row.index else None,
                    sponsored=row.get("sponsored") if "sponsored" in row.index else None,
                    work_type=row.get("work_type") if "work_type" in row.index else None,
                    currency=row.get("currency") if "currency" in row.index else None,
                    compensation_type=row.get("compensation_type") if "compensation_type" in row.index else None,
                    normalized_salary=normalized_salary,
                    zip_code=row.get("zip_code") if "zip_code" in row.index else None,
                    fips=row.get("fips") if "fips" in row.index else None,
                )
                records.append(p)

            for obj in records:
                try:
                    sess.add(obj)
                except Exception:
                    sess.rollback()
                    continue
                inserted += 1
                if inserted % commit_every == 0:
                    try:
                        sess.commit()
                    except Exception:
                        sess.rollback()
            # commit remaining in this chunk
            try:
                sess.commit()
            except Exception:
                sess.rollback()
        return inserted
    finally:
        sess.close()
    
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

def init_db():
    Base.metadata.create_all(engine)

def add_posting(posting : Posting):
    session = Session()
    try:
        session.add(posting)
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_all_postings(embedded_only: bool = False):
    session = Session()
    try:
        if embedded_only:
            postings = session.query(Posting).filter(Posting.embedding.isnot(None)).all()
        else:
            postings = session.query(Posting).all()
        return postings
    finally:
        session.close()

def embed_all_postings(batch_size: int = 100, commit_every: int = 50):
    """Compute and store embeddings for all postings that don't have them.
    
    Args:
        batch_size: How many posts to load at once (memory efficiency).
        commit_every: How often to commit changes to the database.
    """
    session = Session()
    try:
        # Process posts that don't have embeddings yet
        processed = 0
        while True:
            # Get a batch of posts without embeddings
            posts = (
                session.query(Posting)
                .filter(Posting.embedding.is_(None))
                .limit(batch_size)
                .all()
            )
            if not posts:
                break  # No more posts to process

            for i, posting in enumerate(posts, 1):
                try:
                    # Get embedding as a string
                    embedding = embed_posting(posting)
                    if embedding is not None:
                        # Store as string representation
                        posting.embedding = embedding
                        processed += 1
                        
                        # Commit periodically
                        if processed % commit_every == 0:
                            try:
                                session.commit()
                                print(f"Processed {processed} embeddings...")
                            except Exception as e:
                                print(f"Warning: commit failed at {processed}: {e}")
                                session.rollback()
                except Exception as e:
                    print(f"Warning: embedding failed for posting {posting.id}: {e}")
                    continue

            # Commit any remaining in this batch
            try:
                session.commit()
            except Exception as e:
                print(f"Warning: final commit failed: {e}")
                session.rollback()

        print(f"Finished! Processed {processed} embeddings total.")
        return processed
    finally:
        session.close()

def get_posting_by_id(posting_id: int):
    session = Session()
    try:
        posting = session.query(Posting).filter(Posting.id == posting_id).first()
        return posting
    finally:
        session.close()

def get_postings_by_ids(posting_ids: list[int]):
    session = Session()
    try:
        postings = session.query(Posting).filter(Posting.id.in_(posting_ids)).all()
        return postings
    finally:
        session.close()
