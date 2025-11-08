# tests/test_pr_flow.py
import os
from packages.tools.github_mock import DRYPR_DIR
from packages.tools.github_mock import create_dry_pr

def test_create_dry_pr_creates_folder(tmp_path):
    # create a small dry pr
    files = {"tests/test_dummy.py": "def test_dummy(): assert 1 == 2"}
    res = create_dry_pr("title", "body", "branch", files)
    assert "pr_id" in res
    pr_path = res["pr_path"]
    assert os.path.exists(pr_path)
    patch_folder = os.path.join(pr_path, "patch")
    assert os.path.exists(os.path.join(patch_folder, "tests", "test_dummy.py"))
