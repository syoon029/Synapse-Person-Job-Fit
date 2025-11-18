import os
import json
def _append_jsonl(path: str, obj: dict) -> None:
    """
    Append a single JSON object as a line to `path` (JSON Lines / NDJSON).

    - Creates parent directory if needed.
    - Uses utf-8 and ensures we don't load the whole file into memory.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    # Ensure we write a single JSON object per line
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_logged_responses(path: str, max_entries: int | None = None) -> list:
    """
    Read a JSONL log file produced by `_append_jsonl` and return a list of dicts.

    - Skips empty lines and continues past JSON-decoding errors (prints a warning).
    - `max_entries` can be used to limit how many entries are read (from the start).
    """
    results = []
    if not os.path.exists(path):
        return results
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if max_entries is not None and i >= max_entries:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                results.append(obj)
            except json.JSONDecodeError:
                print(f"Warning: failed to decode JSON line {i+1} in {path}; skipping")
                continue
    return results


def stream_logged_responses(path: str):
    """Generator that yields decoded JSON objects from a JSONL log file.
    Useful for large files where you don't want to load everything into memory.
    """
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: failed to decode JSON line {i+1} in {path}; skipping")
                continue