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





