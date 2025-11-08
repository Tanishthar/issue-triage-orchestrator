import os
import json
import uuid
from datetime import datetime
from typing import Dict

DRYPR_DIR = "packages/tools/_dry_prs"
os.makedirs(DRYPR_DIR, exist_ok=True)

def _new_pr_id() -> str:
    return uuid.uuid4().hex[:8]

def create_dry_pr(title: str, body: str, branch_name: str, files: Dict[str, str]) -> Dict:
    """
    Create a local dry-run PR artifact.
    - title: PR title
    - body: PR body/description
    - branch_name: branch name (for demo only)
    - files: dict mapping file path -> file contents (relative to repo root)
    Returns a dict with pr_id, pr_path, pr_url.
    """
    pr_id = _new_pr_id()
    ts = datetime.utcnow().isoformat()
    pr_folder = os.path.join(DRYPR_DIR, f"pr_{pr_id}")
    os.makedirs(pr_folder, exist_ok=True)

    # Create a patch folder that mirrors repo structure
    patch_folder = os.path.join(pr_folder, "patch")
    os.makedirs(patch_folder, exist_ok=True)

    for relpath, content in files.items():
        # safe path join (prevent path traversal)
        safe_rel = relpath.lstrip("/").replace("..", "")
        out_path = os.path.join(patch_folder, safe_rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)

    meta = {
        "pr_id": pr_id,
        "title": title,
        "body": body,
        "branch": branch_name,
        "created_at": ts,
        "file_count": len(files),
    }
    with open(os.path.join(pr_folder, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    pr_url = f"dry://pr/{pr_id}"
    return {"pr_id": pr_id, "pr_path": pr_folder, "pr_url": pr_url, "meta": meta}
