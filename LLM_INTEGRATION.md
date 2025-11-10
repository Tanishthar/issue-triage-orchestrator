# LLM Agent Integration Documentation

## Overview

The Issue Triage Orchestrator now uses a **real LLM agent** with **multi-model support** (OpenAI GPT-4o-mini and Google Gemini models) with tool-driven reasoning capabilities. This replaces the previous mock-based severity classification and fix proposal logic.

## Supported Models

### OpenAI Models
- **gpt-4o-mini** (default) - Cost-effective, fast
- gpt-4o - More capable
- gpt-4-turbo - High performance
- gpt-3.5-turbo - Legacy option

### Google Gemini Models
- **gemini-2.5-flash** - Latest flash model (primary)
- gemini-2.0-flash-exp - Experimental flash model (fallback)
- gemini-1.5-pro - More capable, slower

### Model Selection
Users can select the model via the frontend dropdown. The system automatically:
- Detects the provider (OpenAI vs Gemini) based on model name
- Uses the appropriate API key (OPENAI_API_KEY or GEMINI_API_KEY)
- Falls back to mock mode if API key is missing

## Architecture

### Components

1. **LLM Agent** (`packages/agents/llm_agent.py`)
   - Core agent class with tool calling support
   - Handles OpenAI API interactions
   - Falls back to mock mode if API key unavailable
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
- **Model selection**: Defaults to `gpt-4o-mini` (cost-effective)

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

When `OPENAI_API_KEY` is not set or OpenAI is unavailable, the system runs in **mock mode**:

- Uses heuristic-based severity classification
- Uses simple pattern matching for repro steps
- Uses template-based fix proposals
- All tool calls still work (they're real tools, just no LLM)

This allows development and testing without API costs.

## Logging & Observability

All LLM interactions are logged:

- **Tool calls**: `tool_call` - Logs tool name and arguments
- **Tool results**: `tool_result` - Logs successful tool execution
- **Tool errors**: `tool_error` - Logs tool failures
- **LLM reasoning**: `llm_reasoning` - Logs reasoning iterations
- **LLM responses**: `llm_classify`, `llm_extract_repro`, `llm_propose_fix`

Check `metrics/step_logs.json` for complete execution traces.

## Example Run

See `scripts/run_llm_demo.py` for a complete demonstrable run that shows:
- LLM severity classification
- LLM repro step extraction
- Tool-driven reasoning (if README available)
- Fix proposal with test generation
- Full artifact capture

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (optional, enables real OpenAI LLM mode)
- `GEMINI_API_KEY`: Your Google Gemini API key (optional, enables real Gemini LLM mode)
- Model selection is done via the frontend UI or API request parameter

### Code Configuration

In `packages/agents/llm_agent.py`:
- `max_tokens`: 2000 (adjustable)
- `temperature`: 0.7 (adjustable)
- `max_iterations`: 3-5 for tool reasoning (adjustable)

## Future Enhancements

1. **Multi-model support**: Add Claude, local models (Ollama)
2. **Streaming responses**: Real-time updates for better UX
3. **Cost tracking**: Track token usage and costs
4. **Prompt optimization**: A/B test different prompts
5. **Tool result caching**: Cache tool results to reduce API calls
6. **Fine-tuning**: Fine-tune on issue triage datasets

## Troubleshooting

### LLM not working
- Check `OPENAI_API_KEY` is set
- Check API key is valid
- Check network connectivity
- System will fall back to mock mode automatically

### Tool calls failing
- Check tool logs in `metrics/step_logs.json`
- Verify tool implementations are correct
- Check for rate limiting

### High costs
- Use `gpt-4o-mini` (default, cost-effective)
- Reduce `max_tokens` in agent config
- Reduce `max_iterations` for tool reasoning
- Enable HTTP fetcher caching


