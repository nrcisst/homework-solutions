"""Download only the pinned JSONL data, without executing dataset loading code."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION = "f8fa04aff083aa22c6878d005b4655e34a4cbd85"
DOMAINS = ["fake_news", "job_scams", "phishing", "political_statements",
           "product_reviews", "sms", "twitter_rumours"]
ROLES = ["train", "validation", "test"]
ROOT = Path(__file__).resolve().parent


def download(data_dir=None):
    data_dir = Path(data_dir or ROOT / "data")
    manifest_path = ROOT / "data_manifest.json"
    expected = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    files = [f"{domain}/{role}.jsonl" for domain in DOMAINS for role in ROLES]

    def one(relative):
        path = data_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            url = f"https://huggingface.co/datasets/redasers/difraud/resolve/{REVISION}/{relative}"
            last_error = None
            for _ in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=180) as response:
                        content = response.read()
                    path.write_bytes(content)
                    break
                except Exception as exc:
                    last_error = exc
            else:
                raise RuntimeError(f"Could not download {relative}") from last_error
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if relative in expected and expected[relative]["sha256"] != digest:
            raise ValueError(f"Checksum mismatch: {relative}")
        print(f"Ready: {relative}", flush=True)
        return relative, {"sha256": digest, "bytes": path.stat().st_size}

    with ThreadPoolExecutor(max_workers=4) as pool:
        manifest = dict(pool.map(one, files))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return data_dir


if __name__ == "__main__":
    download()
