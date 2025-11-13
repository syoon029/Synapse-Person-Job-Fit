# open resume_data/resume.csv which has fields: id, string, html, category

# create a small data sample with only id and string fields, and save it to resume_data/sample_resume.csv
# 10 randomly selected resumes

import pandas as pd
import csv
def create_sample_resume(input_file='resume_data/resume.csv', output_file='resume_data/sample_resume.csv', sample_size=10):
    print("Creating sample resume data...")
    # Read the original resume data
    df = pd.read_csv(input_file)

    # Randomly sample the specified number of resumes
    sample_df = df.sample(n=sample_size, random_state=42)[['ID', 'Resume_str']]

    # Save the sampled data to a new CSV file
    sample_df.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)

    print(f"Sampled {sample_size} resumes and saved to {output_file}")

def read_ansel_resume(file_path='resume_data/ansel_resume.csv'):
    print("Reading sample resume data...")
    df = pd.read_csv(file_path)
    resumes = df.to_dict(orient='records')
    print(f"Read {len(resumes)} resumes from {file_path}")
    return resumes


def read_ansel_postings(file_path='linkedin_data/ansel_postings.csv'):
    print("Reading sample job postings data...")
    # schema: "posting.title","posting.company_name","posting.skills_desc","posting.description","posting.formatted_experience_level","posting.location"

    df = pd.read_csv(file_path)
    postings = df.to_dict(orient='records')
    print(f"Read {len(postings)} postings from {file_path}")

    # print the first 50 characters of each field of each of the postings, each posting in one line, ignore newlines inside of posting description
    for posting in postings:
        print(posting)
        print(f"Title: {posting['posting.title'][:50]} | Company: {posting['posting.company_name'][:50]} | Skills: {posting['posting.skills_desc']} | Description: {posting['posting.description'][:50]} | Experience Level: {posting['posting.formatted_experience_level'][:50]} | Location: {posting['posting.location'][:50]}")

if __name__ == "__main__":
    b = read_ansel_postings()
    # a = read_ansel_resume()
    # print(a)
    # create_sample_resume()




