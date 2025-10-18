import os
import time
from datetime import datetime, UTC
from dotenv import load_dotenv
from openai import OpenAI
from llm_logging import _append_jsonl, read_logged_responses

# Load environment variables from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = OpenAI(api_key =GEMINI_API_KEY)
client.base_url = os.getenv("GEMINI_API_BASE_URL") # 'https://generativelanguage.googleapis.com/v1beta/openai

# Define the models and hyperparameters to test using LiteLLM model strings
def get_response(model_name : str, prompt : str,  # main parameters
                 max_tokens = None, temperature = None, top_p = None, # optional hyperparameters
                 logging: bool = True, log_file: str = "llm_history.jsonl") -> str:
    """
    Get a response from the specified model.

    Args:
        model_name (str): The name of the model to use.
        -   Possible values: "models/gemini-1.5-flash", "models/gemini-2.0-flash", "models/gemini-2.5-pro"
        prompt (str): The user's prompt.
        max_tokens (int, optional): The maximum number of tokens to generate.
        temperature (float, optional): The temperature for sampling.
        top_p (float, optional): The nucleus sampling probability.
        logging (bool, optional): Whether to log errors and retries. Defaults to True.

    Returns:
        str: The response from the model, or an error message beginning with "ERROR:" if the request fails.
    """
    messages = [{"role": "user", "content": prompt}]
    
    api_params = {}
    if max_tokens is not None:
        api_params["max_tokens"] = max_tokens
    if temperature is not None:
        api_params["temperature"] = temperature
    if top_p is not None:
        api_params["top_p"] = top_p

    retries = 3
    delay = 5 # seconds
    for i in range(retries):
        try:
            response = client.chat.completions.create(model=model_name, messages=messages, **api_params)
            result = response.choices[0].message.content or "ERROR: NO_RESPONSE_FROM_MODEL"
            # Log the prompt/response pair (append as JSON line)
            if logging:
                try:
                    _append_jsonl(log_file, {
                        "timestamp": datetime.now(UTC).isoformat() + "Z",
                        "model": model_name,
                        "prompt": prompt,
                        "response": result,
                        "error": isinstance(result, str) and result.startswith("ERROR:") == True
                    })
                except Exception as e:
                    # Logging should not interrupt main flow; print a warning
                    print(e)
                    print(f"    Warning: failed to write log to {log_file}")
            return result
        except Exception as e:
            print(f"    An API error occurred with {model_name}: {e}")
            if i < retries - 1:
                print(f"    Retrying in {delay * (i + 1)} seconds...")
                time.sleep(delay * (i + 1))
            else:
                err = f"ERROR: Failed after {retries} retries. Last error: {e}"
                if logging:
                    try:
                        _append_jsonl(log_file, {
                            "timestamp": datetime.now(UTC).isoformat() + "Z",
                            "model": model_name,
                            "prompt": prompt,
                            "response": err,
                            "error": True
                        })
                    except Exception:
                        print(f"    Warning: failed to write log to {log_file}")
                return err
    return f"ERROR: Model {model_name} failed to respond after multiple retries."

def list_models():
    """List available models from the Gemini API."""
    try:
        from google import genai
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        models = gemini_client.models.list()
        print([model.name for model in models])
    except Exception as e:
        print(f"Error listing models: {e}")

if __name__ == "__main__":
    # Simple demo: if an API key is present, make one request and show recent logs.
    if GEMINI_API_KEY:
        # list_models()

        print("GEMINI_API_KEY found — running a single demo request (will be logged)...")
        demo_prompt = "Say hello in one sentence."
        resp = get_response("models/gemini-2.0-flash", demo_prompt, logging=True)
        print("Model response:", resp)
        print("Last 5 log entries:")
        for entry in read_logged_responses("llm_history.jsonl", max_entries=5):
            print(entry)
    else:
        print("GEMINI_API_KEY not set. Set it in the environment or .env to run the demo.")
