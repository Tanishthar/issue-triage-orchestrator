#!/usr/bin/env python3
"""
Demonstrable LLM Agent Run Script

This script demonstrates a real run with the LLM agent:
- LLM classifies severity
- LLM extracts repro steps
- LLM proposes fix with tool-driven reasoning
- Generates failing test
- Creates dry PR artifact

Run with: python scripts/run_llm_demo.py
"""

import os
import sys
import json
import requests
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
OUT_DIR = Path(__file__).parent.parent / "sample_runs"
DRYPRS_DIR = Path(__file__).parent.parent / "packages" / "tools" / "_dry_prs"
METRICS_DIR = Path(__file__).parent.parent / "metrics"

# Ensure directories exist
OUT_DIR.mkdir(parents=True, exist_ok=True)
DRYPRS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def timestamp():
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def run_demo():
    """Run a demonstrable LLM agent workflow"""
    
    # Sample issue for demonstration
    demo_issue = {
        "repo_url": "https://github.com/n8n-io/n8n",
        "issue_text": "The application crashes on startup when the database connection fails. Error: 'Connection refused'. This happens when the database service is not running.",
        "readme_file_path": None  # Will auto-detect
    }
    
    run_id = f"llm-demo-{timestamp()}"
    run_dir = OUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"LLM Agent Demo Run: {run_id}")
    print(f"{'='*60}\n")
    print(f"Repository: {demo_issue['repo_url']}")
    print(f"Issue: {demo_issue['issue_text']}\n")
    
    # Save input
    with open(run_dir / "input.json", "w") as f:
        json.dump(demo_issue, f, indent=2)
    
    # POST to /start
    print("Sending request to orchestrator...")
    try:
        response = requests.post(
            f"{API_BASE}/start",
            json=demo_issue,
            headers={"Content-Type": "application/json"},
            timeout=300  # 5 minute timeout for LLM calls
        )
        response.raise_for_status()
        result = response.json()
        
        print("✓ Request successful\n")
        
        # Save response
        with open(run_dir / "response.json", "w") as f:
            json.dump(result, f, indent=2)
        
        # Print summary
        print("Results Summary:")
        print(f"  Status: {result.get('final_state', {}).get('status', 'unknown')}")
        print(f"  Severity: {result.get('final_state', {}).get('severity', 'unknown')}")
        print(f"  LLM Mode: {result.get('llm_mode', 'unknown')}")
        print(f"  PR URL: {result.get('pr_result', {}).get('pr_url', 'N/A')}")
        print(f"  Executor Check: {result.get('executor_check', {}).get('message', 'N/A')}")
        
        if result.get('final_state', {}).get('repro_steps'):
            print(f"\n  Repro Steps:")
            for i, step in enumerate(result['final_state']['repro_steps'], 1):
                print(f"    {i}. {step}")
        
        # Copy artifacts
        print("\nCopying artifacts...")
        
        # Copy dry PRs
        if DRYPRS_DIR.exists():
            import shutil
            dest_prs = run_dir / "_dry_prs"
            if dest_prs.exists():
                shutil.rmtree(dest_prs)
            shutil.copytree(DRYPRS_DIR, dest_prs)
            print(f"  ✓ Copied dry PRs to {dest_prs}")
        
        # Copy metrics
        metrics_file = METRICS_DIR / "metrics.json"
        if metrics_file.exists():
            shutil.copy2(metrics_file, run_dir / "metrics.json")
            print(f"  ✓ Copied metrics")
        
        # Copy step logs
        logs_file = METRICS_DIR / "step_logs.json"
        if logs_file.exists():
            shutil.copy2(logs_file, run_dir / "step_logs.json")
            print(f"  ✓ Copied step logs")
        
        print(f"\n✓ Demo run completed!")
        print(f"  Artifacts saved to: {run_dir}")
        print(f"  View logs: {run_dir / 'step_logs.json'}")
        print(f"  View PR: {run_dir / '_dry_prs'}")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"  Error detail: {json.dumps(error_detail, indent=2)}")
            except:
                print(f"  Error text: {e.response.text[:500]}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_demo()
    sys.exit(0 if success else 1)


