# LLM Agent Integration Documentation

## Overview

The Issue Triage Orchestrator now uses a **real LLM agent** with **Ollama support** for tool-driven reasoning capabilities. This replaces the previous mock-based severity classification and fix proposal logic.

## Supported Models

### Ollama Models
- **ollama:llama3.1** (default) - Fast, capable local model
- ollama:llama3.2 - Latest Llama model
- ollama:mistral - Mistral model via Ollama
- ollama:qwen2.5 - Qwen model via Ollama
- Any other model available in your Ollama installation

### Model Selection
Users can select the model via the frontend dropdown. The system automatically:
- Detects Ollama models based on the "ollama:" prefix
- Connects to local Ollama instance (default: http://127.0.0.1:11434)
- Falls back to mock mode if Ollama is not running or unavailable

## Architecture

### Components

1. **LLM Agent** (`packages/agents/llm_agent.py`)
   - Core agent class with tool calling support
   - Handles Ollama API interactions
   - Falls back to mock mode if Ollama unavailable
   - Supports structured reasoning loops

2. **Tool Wrappers** (`packages/agents/tool_wrappers.py`)
   - Wraps existing tools (HTTP fetcher, vector store, executor)
   - Provides structured interfaces for LLM tool calling
   - Logs all tool invocations

3. **Orchestrator Integration** (`apps/orchestrator/main.py`)
   - Uses LLM agent for severity classification
   - Uses LLM agent for repro step extraction
   - Uses LLM agent for fix proposal
   - Integrates tool-driven reasoning for complex issues

## Prompts

### Severity Classification

```
You are an expert at triaging software issues. Classify the severity of this GitHub issue.

Repository: {repo_url}
Issue: {issue_text}

Classify the severity as one of: low, medium, high, critical

Consider:
- Impact on users
- Frequency of occurrence
- Workaround availability
- Security implications

Respond with ONLY the severity level (low, medium, high, or critical), nothing else.
```

**Output**: Single severity level string

### Repro Step Extraction

```
Extract clear reproduction steps from this GitHub issue. Return them as a numbered list.

Issue: {issue_text}

Return ONLY a numbered list of steps (1. step one, 2. step two, etc.). If no clear steps are provided, infer reasonable steps based on the issue description.
```

**Output**: Numbered list of steps (parsed into array)

### Fix Proposal

```
You are an expert software engineer. Analyze this GitHub issue and propose a fix.

Repository Issue:
{issue_text}

Severity: {severity}
Repro Steps:
{numbered_repro_steps}
{context_info}

Your task:
1. Analyze the root cause hypothesis
2. Propose a fix sketch (high-level approach)
3. Generate a failing pytest test that reproduces the issue

Return your response in this exact JSON format:
{
  "fix_sketch": "Detailed explanation of the root cause and proposed fix approach...",
  "test_code": "Complete pytest test code that reproduces the issue (should fail initially)"
}

The test_code should be valid Python pytest code that can be syntax-checked.
```

**Output**: JSON with `fix_sketch` and `test_code` fields

### Tool-Driven Reasoning

```
Analyze this GitHub issue and propose a fix:

Repository: {repo_url}
Issue: {issue_text}
Severity: {severity}
Repro Steps: {repro_steps}

You can use tools to:
1. Search the documentation for relevant context
2. Fetch additional information if needed
3. Check code syntax

After gathering context, propose a fix sketch and generate a failing pytest test.
```

**Output**: Natural language reasoning with optional tool calls

## Tool Calling

### Available Tools

1. **fetch_url**
   - **Purpose**: Fetch content from URLs
   - **Use case**: Get README files, documentation
   - **Parameters**: `url` (string)
   - **Returns**: `{url, status, text, from_cache, text_length}`

2. **search_documentation**
   - **Purpose**: Search vector store for relevant context
   - **Use case**: Find relevant documentation about the repository
   - **Parameters**: `query` (string), `top_k` (integer, default: 3)
   - **Returns**: `{query, results: [{doc_id, score, text_preview}]}`

3. **check_code_syntax**
   - **Purpose**: Validate Python code syntax
   - **Use case**: Check generated test code before proposing
   - **Parameters**: `code` (string)
   - **Returns**: `{ok: bool, message: str, code_length: int}`

### Tool Calling Flow

1. LLM receives query with available tools
2. LLM decides to call tool(s) based on context
3. Tool is executed with provided arguments
4. Tool result is returned to LLM
5. LLM continues reasoning with tool results
6. Process repeats until LLM provides final answer (max 3-5 iterations)

## Safety Rails

### Budget Protection
- **Token limits**: Max 2000 tokens per LLM call
- **Max iterations**: 3-5 iterations for tool reasoning loops
- **Timeout**: 5-minute timeout for API calls
- **Model selection**: Defaults to `ollama:llama3.1` (local, no API costs)

### Error Handling
- **API failures**: Graceful fallback to mock mode
- **Tool errors**: Logged but don't crash the system
- **JSON parsing errors**: Fallback to text extraction
- **Timeout errors**: Return partial results

### Security
- **No code execution**: Only syntax checking, never executes code
- **URL validation**: HTTP fetcher respects robots.txt
- **Input sanitization**: All inputs truncated to safe lengths
- **Error message truncation**: Prevents information leakage

## Mock Mode

When Ollama is not running or unavailable, the system runs in **mock mode**:

- Uses heuristic-based severity classification
- Uses simple pattern matching for repro steps
- Uses template-based fix proposals
- All tool calls still work (they're real tools, just no LLM)

This allows development and testing without requiring Ollama to be running.

## Logging & Observability

All LLM interactions and tool usage are comprehensively logged:

- **Tool usage**: `tool_usage` - Logs "Used tool: [tool_name]" before each tool execution
- **Tool calls**: `tool_call` - Logs tool name and arguments
- **Tool results**: `tool_result` - Logs successful tool execution
- **Tool errors**: `tool_error` - Logs tool failures
- **LLM reasoning**: `llm_reasoning` - Logs reasoning iterations
- **LLM responses**: `llm_classify`, `llm_extract_repro`, `llm_propose_fix`
- **Tool call counts**: Real-time count of tools called during reasoning

**Available Tools (4 total)**:
1. `fetch_url` - Fetches content from URLs (READMEs, documentation)
2. `search_documentation` - Searches vector store for relevant context
3. `check_code_syntax` - Validates Python code syntax
4. Direct vector store search (used during README indexing)

Check `metrics/step_logs.json` for complete execution traces. All tool invocations are logged with step name `tool_usage` for easy filtering.

## Example Run

See `scripts/run_llm_demo.py` for a complete demonstrable run that shows:
- LLM severity classification
- LLM repro step extraction
- Tool-driven reasoning (if README available)
- Fix proposal with test generation
- Full artifact capture

## Configuration

### Environment Variables

- `OLLAMA_BASE_URL`: Ollama base URL (optional, defaults to http://127.0.0.1:11434)
- `DEFAULT_LLM_MODEL`: Default Ollama model to use (optional, defaults to ollama:llama3.1)
  - Examples: `ollama:llama3.1`, `ollama:llama3.2`, `ollama:mistral`, `ollama:gpt-oss:120b-cloud`
  - For Next.js frontend, use `NEXT_PUBLIC_DEFAULT_LLM_MODEL` with the same value
- Model selection can also be done via the frontend UI or API request parameter (overrides env var)

### Code Configuration

In `packages/agents/llm_agent.py`:
- `max_tokens`: 2000 (adjustable)
- `temperature`: 0.7 (adjustable)
- `max_iterations`: 3-5 for tool reasoning (adjustable)

## Future Enhancements

1. **Additional Ollama models**: Support for more Ollama models
2. **Streaming responses**: Real-time updates for better UX
3. **Performance tracking**: Track token usage and response times
4. **Prompt optimization**: A/B test different prompts
5. **Tool result caching**: Cache tool results to reduce network calls
6. **Fine-tuning**: Fine-tune on issue triage datasets

## Troubleshooting

### LLM not working
- Check Ollama is running: `curl http://127.0.0.1:11434/api/tags`
- Check `OLLAMA_BASE_URL` is set correctly if using non-default location
- Verify the model is available: `ollama list`
- System will fall back to mock mode automatically

### Tool calls failing
- Check tool logs in `metrics/step_logs.json`
- Verify tool implementations are correct
- Check for network connectivity issues

### Performance issues
- Use `ollama:llama3.1` (default, good balance)
- Reduce `max_tokens` in agent config
- Reduce `max_iterations` for tool reasoning
- Enable HTTP fetcher caching


