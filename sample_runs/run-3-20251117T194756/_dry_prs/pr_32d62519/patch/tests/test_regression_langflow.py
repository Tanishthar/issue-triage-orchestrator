import os
import json
import tempfile
import pytest

# Import the library's public API – adjust the import path to match the actual repo structure
# For this example we assume the repo exposes `Agent` and a `run_multi_agent` helper after the fix.
# If the actual names differ, the test will need to be updated accordingly.
from multi_agent_framework import Agent, run_multi_agent, SharedStorage

@pytest.fixture
def temp_storage_file():
    """Create a temporary JSON file to act as shared storage and clean it up after the test."""
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    # Initialise with an empty JSON object
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"_meta": "init"}, f)
    yield path
    os.remove(path)

def test_main_agent_reads_background_output(temp_storage_file):
    """Verify that a background agent can write to shared storage and a main agent can read it.

    Steps:
    1. Create a SharedStorage instance pointing at the temporary JSON file.
    2. Define a background agent whose prompt simply returns a known string and stores it.
    3. Define a main agent whose prompt reads the stored value and echoes it back.
    4. Run the orchestration helper.
    5. Assert that the final response from the main agent contains the background output.
    """
    storage = SharedStorage(file_path=temp_storage_file)

    # Background agent: generates a fixed result and stores it under the key "bg_result"
    bg_prompt = "You are a background worker. Return the string 'background_success' and store it."
    bg_agent = Agent(
        name="background",
        system_prompt=bg_prompt,
        storage=storage,
        # The framework is expected to call a provided `post_process` hook after generation.
        post_process=lambda result, agent: agent.store("bg_result", result)
    )

    # Main agent: reads the key "bg_result" from storage and incorporates it into its reply.
    main_prompt = (
        "You are the main orchestrator. Read the value stored under 'bg_result' "
        "from shared storage and reply with: 'Integrated: {bg_result}'."
    )
    main_agent = Agent(
        name="main",
        system_prompt=main_prompt,
        storage=storage,
        # The framework should replace `{bg_result}` with the loaded value before generation.
        pre_process=lambda agent: agent.inject_context({"bg_result": agent.load("bg_result")})
    )

    # Run the orchestration – background runs first, then main.
    final_response = run_multi_agent(main_agent=main_agent, background_agent=bg_agent, storage=storage)

    # The expected integrated string
    expected_substring = "Integrated: background_success"
    assert expected_substring in final_response, f"Expected '{expected_substring}' in response, got: {final_response}"
