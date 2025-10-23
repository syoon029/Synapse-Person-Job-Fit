# Synapse_6220

Getting started

Create a pyenv with requirements.txt (and activate it)

create a virtual environment named .venv:

python -m venv .venv
run the activate script
.\.venv\Scripts\{activate.bat, Activate.ps1, activate.fish, varies by terminal}

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

deactivate when done


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