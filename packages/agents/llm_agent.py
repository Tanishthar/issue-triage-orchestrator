"""
LLM Agent with Tool-Driven Reasoning

This module implements an LLM agent that can:
- Classify issue severity
- Extract repro steps
- Propose fixes
- Generate test code
- Call tools (HTTP fetcher, vector store, executor)

The agent uses a structured approach with tool calling to reason about issues.
"""

import os
import json
import re
import textwrap
import time
import socket
import asyncio
from typing import Dict, List, Optional, Any, Callable, Tuple
from packages.agents.reliability import log_step
from packages.agents.fix_proposer import extract_repro_steps as mock_extract_repro_steps, propose_fix_sketch as mock_propose_fix_sketch
from packages.tools.mock_tool import mock_classify_issue

# Try to import numpy for type checking
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

def safe_json_serialize(obj: Any) -> Any:
    """Recursively convert numpy arrays and other non-serializable types to JSON-serializable formats"""
    if NUMPY_AVAILABLE and isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: safe_json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_json_serialize(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        # Handle custom objects
        try:
            return str(obj)
        except:
            return repr(obj)
    else:
        try:
            json.dumps(obj)  # Test if it's already serializable
            return obj
        except (TypeError, ValueError):
            return str(obj)

# Try to import httpx for Ollama API calls
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    log_step("llm_agent", "httpx not available, Ollama support disabled")

# Only Ollama is supported - no other model providers


class Tool:
    """Represents a callable tool with metadata"""
    def __init__(self, name: str, description: str, func: Callable, parameters: Dict[str, Any]):
        self.name = name
        self.description = description
        self.func = func
        self.parameters = parameters
    
    def call(self, **kwargs) -> Any:
        """Call the tool with given arguments"""
        # Safely serialize kwargs for logging
        safe_kwargs = safe_json_serialize(kwargs)
        log_step("tool_call", f"Calling tool: {self.name} with args: {json.dumps(safe_kwargs, default=str)}")
        try:
            result = self.func(**kwargs)
            log_step("tool_result", f"Tool {self.name} succeeded: {str(result)[:200]}")
            return result
        except Exception as e:
            error_msg = str(e)[:500]
            log_step("tool_error", f"Tool {self.name} failed: {error_msg}")
            raise


class LLMAgent:
    """LLM Agent with tool calling capabilities - supports Ollama only"""
    
    # Supported models
    OLLAMA_MODELS = ["ollama:gpt-oss:120b-cloud"]  # Format: ollama:model_name
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, use_mock: bool = False, ollama_base_url: str = "http://127.0.0.1:11434"):
        # Get model from parameter, environment variable, or default
        if model is None:
            model = os.getenv("DEFAULT_LLM_MODEL", "ollama:llama3.1")
        self.model = model
        self.provider = self._detect_provider(model)
        self.tools: Dict[str, Tool] = {}
        # Use 127.0.0.1 instead of localhost for better Windows compatibility
        self.ollama_base_url = ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        
        # Initialize based on provider
        if use_mock:
            self.use_mock = True
            self.ollama_model = None
            log_step("llm_agent", "Running in mock mode (forced)")
        elif self.provider == "ollama":
            self.use_mock = not HTTPX_AVAILABLE
            if not self.use_mock:
                # Extract model name (remove "ollama:" prefix)
                self.ollama_model = model.replace("ollama:", "")
                # Note: Connection test moved to _verify_ollama_connection() which is called before actual API calls
                # This avoids async/sync blocking issues in FastAPI contexts
                log_step("llm_agent", f"Initialized Ollama client with model: {self.ollama_model} (base_url: {self.ollama_base_url})")
            else:
                self.ollama_model = None
                log_step("llm_agent", "Running in mock mode (httpx not available for Ollama)")
        else:
            self.use_mock = True
            log_step("llm_agent", f"Unknown model {model}, running in mock mode. Only Ollama models are supported.")
    
    def _detect_provider(self, model: str) -> str:
        """Detect which provider to use based on model name - only Ollama is supported"""
        if model in self.OLLAMA_MODELS or model.startswith("ollama:"):
            return "ollama"
        else:
            return "unknown"
    
    def register_tool(self, tool: Tool):
        """Register a tool for the agent to use"""
        self.tools[tool.name] = tool
        log_step("llm_agent", f"Registered tool: {tool.name}")
    
    def _get_tool_schemas(self) -> List[Dict]:
        """Convert registered tools to function calling format (OpenAI-compatible)"""
        schemas = []
        for tool in self.tools.values():
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            })
        return schemas
    
    async def _call_llm(self, messages: List[Dict], tools: Optional[List[Dict]] = None) -> Dict:
        """Call the LLM API (Ollama only) with fallback to mock on errors"""
        if self.use_mock:
            # Mock response for testing
            return self._mock_llm_response(messages)

        try:
            if self.provider == "ollama":
                return await self._call_ollama_async(messages, tools)
            else:
                log_step("llm_fallback", f"Unsupported provider {self.provider}, falling back to mock mode")
                return self._mock_llm_response(messages)
        except Exception as e:
            log_step("llm_fallback", f"LLM API call failed, falling back to mock mode. Error: {str(e)[:200]}")

            # Fall back to mock response instead of raising
            return self._mock_llm_response(messages)
    
    def _test_port_connectivity(self, host: str, port: int, timeout: float = 2.0) -> bool:
        """Test if a port is reachable using socket (lower level than httpx)"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception as e:
            log_step("llm_agent", f"Socket test failed for {host}:{port}: {str(e)[:100]}")
            return False

    def _verify_ollama_connection(self) -> bool:
        """Verify Ollama connection before making API calls with fallback to host.docker.internal"""
        try:
            with httpx.Client(timeout=5.0) as client:
                test_response = client.get(f"{self.ollama_base_url}/api/tags")
                test_response.raise_for_status()
                return True
        except Exception as e:
            error_msg = str(e).lower()
            if "connection" in error_msg or "refused" in error_msg or "111" in error_msg:
                # Try alternative URL (127.0.0.1 <-> localhost)
                alt_url = self.ollama_base_url.replace("127.0.0.1", "localhost") if "127.0.0.1" in self.ollama_base_url else self.ollama_base_url.replace("localhost", "127.0.0.1")
                try:
                    with httpx.Client(timeout=5.0) as client:
                        test_response = client.get(f"{alt_url}/api/tags")
                        test_response.raise_for_status()
                    self.ollama_base_url = alt_url
                    log_step("llm_agent", f"Switched to alternative Ollama URL: {self.ollama_base_url}")
                    return True
                except:
                    # Fallback to host.docker.internal if localhost/127.0.0.1 both fail
                    docker_host_url = "http://host.docker.internal:11434"
                    try:
                        with httpx.Client(timeout=5.0) as client:
                            test_response = client.get(f"{docker_host_url}/api/tags")
                            test_response.raise_for_status()
                        self.ollama_base_url = docker_host_url
                        log_step("llm_agent", f"Switched to host.docker.internal Ollama URL: {self.ollama_base_url}")
                        return True
                    except:
                        return False
            return False
    
    async def _call_ollama_async(self, messages: List[Dict], tools: Optional[List[Dict]] = None) -> Dict:
        """Call Ollama API with function calling support using AsyncClient"""
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                # Convert messages format for Ollama
                # Ollama expects a list of message objects with role and content
                ollama_messages = []
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role in ["user", "assistant", "system"]:
                        ollama_messages.append({"role": role, "content": content})
                
                # Prepare request payload
                payload = {
                    "model": self.ollama_model,
                    "messages": ollama_messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                    }
                }
                
                # Add tools if available (Ollama supports function calling in OpenAI-compatible format)
                if tools:
                    # Ollama expects tools in OpenAI format
                    payload["tools"] = tools
                    # Some Ollama models may need tool_choice parameter
                    payload["tool_choice"] = "auto"
                    tool_names = [t.get("function", {}).get("name", "unknown") for t in tools]
                    log_step("llm_agent", f"Passing {len(tools)} tools to Ollama for function calling: {tool_names}")
                else:
                    log_step("llm_agent", "No tools available for this LLM call")
                
                # Make API call to Ollama
                # Use asyncio.to_thread() to run synchronous httpx.Client in thread pool
                # This works better on Windows where AsyncClient can have event loop issues
                log_step("llm_agent", f"Making API call to {self.ollama_base_url}/api/chat")

                def _sync_http_call():
                    """Synchronous HTTP call wrapped for thread execution"""
                    with httpx.Client(
                        timeout=httpx.Timeout(60.0, connect=10.0),
                        follow_redirects=True,
                        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                        trust_env=False  # Don't use proxy settings for local requests
                    ) as client:
                        response = client.post(
                            f"{self.ollama_base_url}/api/chat",
                            json=payload
                        )
                        response.raise_for_status()
                        return response.json()

                # Run synchronous call in thread pool (non-blocking for async context)
                result = await asyncio.to_thread(_sync_http_call)

                # Parse Ollama response
                message = result.get("message", {})
                content = message.get("content", "")

                # Extract tool calls if present
                tool_calls = []
                if "tool_calls" in message:
                    tool_calls = message["tool_calls"]
                    log_step("llm_agent", f"Found {len(tool_calls)} tool calls in message")
                elif "tool_calls" in result:
                    tool_calls = result["tool_calls"]
                    log_step("llm_agent", f"Found {len(tool_calls)} tool calls in result")
                else:
                    log_step("llm_agent", "No tool calls in Ollama response")

                # Convert Ollama tool calls to OpenAI format if needed
                formatted_tool_calls = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        formatted_tool_calls.append({
                            "id": tc.get("id", "call_unknown"),
                            "type": "function",
                            "function": {
                                "name": tc.get("function", {}).get("name", tc.get("name", "unknown")),
                                "arguments": json.dumps(tc.get("function", {}).get("arguments", tc.get("arguments", {})))
                            }
                        })

                return {
                    "content": content,
                    "tool_calls": formatted_tool_calls,
                    "finish_reason": result.get("done", True) and "stop" or "length"
                }
            except httpx.HTTPError as e:
                error_msg = str(e).lower()
                # Log detailed error information
                log_step("llm_error", f"httpx.HTTPError in attempt {attempt + 1}/{max_retries}: {type(e).__name__}: {str(e)[:300]}")
                # Check for connection errors (Ollama not running)
                if "connection" in error_msg or "refused" in error_msg or "111" in error_msg:
                    # Try fallback URLs before retrying
                    fallback_urls = []
                    # Always try 127.0.0.1 and localhost as fallbacks
                    if "127.0.0.1" not in self.ollama_base_url:
                        fallback_urls.append("http://127.0.0.1:11434")
                    if "localhost" not in self.ollama_base_url:
                        fallback_urls.append("http://localhost:11434")
                    # Always try host.docker.internal as final fallback
                    if "host.docker.internal" not in self.ollama_base_url:
                        fallback_urls.append("http://host.docker.internal:11434")
                    
                    if fallback_urls:
                        log_step("llm_agent", f"Trying fallback URLs: {fallback_urls}")
                    for fallback_url in fallback_urls:
                        try:
                            log_step("llm_agent", f"Testing fallback URL: {fallback_url}")
                            # Quick test connection
                            with httpx.Client(timeout=5.0) as test_client:
                                test_response = test_client.get(f"{fallback_url}/api/tags")
                                test_response.raise_for_status()
                            # Switch to fallback URL
                            self.ollama_base_url = fallback_url
                            log_step("llm_agent", f"Successfully switched to fallback Ollama URL: {self.ollama_base_url}")
                            # Retry immediately with new URL
                            continue
                        except Exception as fallback_error:
                            log_step("llm_agent", f"Fallback URL {fallback_url} failed: {str(fallback_error)[:100]}")
                            pass
                    
                    if attempt < max_retries - 1:
                        # Retry with exponential backoff
                        wait_time = retry_delay * (2 ** attempt)
                        log_step("llm_agent", f"Ollama connection failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        # Note: Using async sleep for proper async behavior
                        continue
                    else:
                        raise Exception(f"Ollama connection failed after {max_retries} attempts. Is Ollama running at {self.ollama_base_url}? Try: curl {self.ollama_base_url}/api/tags. Error: {str(e)[:200]}")
                raise Exception(f"Ollama API error: {str(e)[:200]}")
            except Exception as e:
                error_msg = str(e).lower()
                # Log detailed error information
                log_step("llm_error", f"Exception in attempt {attempt + 1}/{max_retries}: {type(e).__name__}: {str(e)[:300]}")
                import traceback
                log_step("llm_error", f"Traceback: {traceback.format_exc()[:500]}")
                if "connection" in error_msg or "refused" in error_msg or "111" in error_msg:
                    # Try fallback URLs before retrying
                    fallback_urls = []
                    # Always try 127.0.0.1 and localhost as fallbacks
                    if "127.0.0.1" not in self.ollama_base_url:
                        fallback_urls.append("http://127.0.0.1:11434")
                    if "localhost" not in self.ollama_base_url:
                        fallback_urls.append("http://localhost:11434")
                    # Always try host.docker.internal as final fallback
                    if "host.docker.internal" not in self.ollama_base_url:
                        fallback_urls.append("http://host.docker.internal:11434")
                    
                    if fallback_urls:
                        log_step("llm_agent", f"Trying fallback URLs: {fallback_urls}")
                    for fallback_url in fallback_urls:
                        try:
                            log_step("llm_agent", f"Testing fallback URL: {fallback_url}")
                            # Quick test connection
                            with httpx.Client(timeout=5.0) as test_client:
                                test_response = test_client.get(f"{fallback_url}/api/tags")
                                test_response.raise_for_status()
                            # Switch to fallback URL
                            self.ollama_base_url = fallback_url
                            log_step("llm_agent", f"Successfully switched to fallback Ollama URL: {self.ollama_base_url}")
                            # Retry immediately with new URL
                            continue
                        except Exception as fallback_error:
                            log_step("llm_agent", f"Fallback URL {fallback_url} failed: {str(fallback_error)[:100]}")
                            pass
                    
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        log_step("llm_agent", f"Ollama connection error (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        # Note: Using async sleep for proper async behavior
                        continue
                # Re-raise with context
                raise Exception(f"Ollama API call failed: {str(e)[:200]}")
        
        # Should not reach here, but just in case
        raise Exception(f"Ollama API call failed after {max_retries} retries")
    
    def _mock_llm_response(self, messages: List[Dict]) -> Dict:
        """Mock LLM response for testing without API"""
        last_message = messages[-1]["content"] if messages else ""
        
        # Simple heuristics for mock responses
        if "classify" in last_message.lower() or "severity" in last_message.lower():
            return {
                "content": "Based on the issue description, I classify this as **high** severity because it involves a crash on startup.",
                "tool_calls": [],
                "finish_reason": "stop"
            }
        elif "repro" in last_message.lower() or "steps" in last_message.lower():
            return {
                "content": "1. Start the application\n2. Navigate to the problematic feature\n3. Observe the crash/error",
                "tool_calls": [],
                "finish_reason": "stop"
            }
        elif "fix" in last_message.lower() or "propose" in last_message.lower():
            return {
                "content": "The issue appears to be caused by unhandled exception handling. I recommend adding try-catch blocks and input validation.",
                "tool_calls": [],
                "finish_reason": "stop"
            }
        else:
            return {
                "content": "I've analyzed the issue and prepared a response.",
                "tool_calls": [],
                "finish_reason": "stop"
            }
    
    async def classify_severity(self, issue_text: str, repo_url: str = "") -> str:
        """Classify issue severity using LLM, with fallback to mock classifier"""
        try:
            prompt = f"""You are an expert at triaging software issues. Classify the severity of this GitHub issue.

Repository: {repo_url}
Issue: {issue_text}

Classify the severity as one of: low, medium, high, critical

Consider:
- Impact on users
- Frequency of occurrence
- Workaround availability
- Security implications

Respond with ONLY the severity level (low, medium, high, or critical), nothing else."""

            messages = [{"role": "user", "content": prompt}]
            response = await self._call_llm(messages)
            
            severity = response["content"].strip().lower()
            # Extract severity if it's in a sentence
            for level in ["critical", "high", "medium", "low"]:
                if level in severity:
                    severity = level
                    break
            
            if severity not in ["low", "medium", "high", "critical"]:
                severity = "medium"  # Default fallback
            
            log_step("llm_classify", f"Classified severity: {severity}")
            return severity
        except Exception as e:
            log_step("llm_fallback", f"Severity classification failed, using mock classifier: {str(e)[:200]}")
            # Fallback to mock classifier
            return mock_classify_issue(issue_text)
    
    async def extract_repro_steps(self, issue_text: str) -> List[str]:
        """Extract reproduction steps from issue text, with fallback to mock extractor"""
        try:
            prompt = f"""Extract clear reproduction steps from this GitHub issue. Return them as a numbered list.

Issue: {issue_text}

Return ONLY a numbered list of steps (1. step one, 2. step two, etc.). If no clear steps are provided, infer reasonable steps based on the issue description."""

            messages = [{"role": "user", "content": prompt}]
            response = await self._call_llm(messages)
            
            content = response["content"]
            # Parse numbered list
            steps = []
            for line in content.split("\n"):
                line = line.strip()
                if not line:
                    continue
                # Remove numbering (1., 2., etc.)
                line = re.sub(r'^\d+[\.\)]\s*', '', line)
                if line:
                    steps.append(line)
            
            if not steps:
                # Fallback: split by sentences
                steps = [s.strip() for s in content.split(".") if s.strip()][:5]
            
            log_step("llm_extract_repro", f"Extracted {len(steps)} repro steps")
            return steps[:5]  # Limit to 5 steps
        except Exception as e:
            log_step("llm_fallback", f"Repro step extraction failed, using mock extractor: {str(e)[:200]}")
            # Fallback to mock extractor
            return mock_extract_repro_steps(issue_text)
    
    async def propose_fix(self, issue_text: str, severity: str, repro_steps: List[str],
                   repo_readme: str = "", context: Optional[Dict] = None) -> Tuple[str, str]:
        """Propose a fix sketch and generate failing test code, with fallback to mock proposer"""
        try:
            # Build context from tools if available
            context_info = ""
            if context:
                if context.get("readme_fetched"):
                    context_info += f"\nRepository README (first 2000 chars): {repo_readme[:2000]}"
                if context.get("vector_search_results"):
                    context_info += f"\nRelevant code context: {context['vector_search_results']}"
            
            prompt = f"""You are an expert software engineer. Analyze this GitHub issue and propose a fix.

Repository Issue:
{issue_text}

Severity: {severity}
Repro Steps:
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(repro_steps))}
{context_info}

Your task:
1. Analyze the root cause hypothesis
2. Propose a fix sketch (high-level approach)
3. Generate a failing pytest test that reproduces the issue

CRITICAL: You MUST return ONLY valid JSON in this exact format (no other text before or after):
{{
  "fix_sketch": "Detailed explanation of the root cause and proposed fix approach...",
  "test_code": "import pytest\\n\\ndef test_regression():\\n    # Test code here\\n    assert False"
}}

IMPORTANT REQUIREMENTS:
- The test_code field MUST contain complete, valid Python pytest code
- The test_code must be a string with escaped newlines (\\n) - do NOT use actual newlines in JSON strings
- Do NOT use placeholders like "See above" or "TODO" - write actual test code
- The test should attempt to reproduce the issue described
- Example test_code format: "import pytest\\n\\ndef test_issue():\\n    # actual test implementation\\n    assert some_condition == expected_value"
- ALL newlines in JSON string values MUST be escaped as \\n (not actual newline characters)
- The JSON must be valid and parseable - use \\n for newlines, not actual line breaks"""
            
            # Use system message to enforce JSON format
            messages = [
                {"role": "system", "content": "You are a helpful assistant that always responds with valid JSON. When asked to generate test code, you must provide complete, executable Python pytest code, not placeholders."},
                {"role": "user", "content": prompt}
            ]
            response = await self._call_llm(messages)

            content = response["content"]
            
            # Try to parse JSON from response
            try:
                # Clean up the content - remove any leading/trailing whitespace
                content = content.strip()
                
                # Extract JSON if wrapped in markdown code blocks
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
                if json_match:
                    content = json_match.group(1)
                else:
                    # Try to find JSON object directly - find the first { and match to the last }
                    # This is a simple approach; for complex nested JSON, we'll rely on json.loads to validate
                    start_idx = content.find('{')
                    if start_idx >= 0:
                        # Find matching closing brace by counting braces
                        brace_count = 0
                        end_idx = start_idx
                        for i in range(start_idx, len(content)):
                            if content[i] == '{':
                                brace_count += 1
                            elif content[i] == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    end_idx = i
                                    break
                        if brace_count == 0:
                            content = content[start_idx:end_idx + 1]
                
                # Try to parse JSON - if it fails due to control characters, try to fix them
                try:
                    result = json.loads(content)
                except json.JSONDecodeError as json_err:
                    # If JSON parsing fails due to control characters, try to escape them
                    if "control character" in str(json_err).lower():
                        log_step("llm_parse_error", f"JSON has control characters, attempting to fix: {str(json_err)[:200]}")
                        # Escape control characters in string values
                        # This is a simple approach: find string values and escape newlines/tabs
                        fixed_content = ""
                        in_string = False
                        escape_next = False
                        i = 0
                        while i < len(content):
                            char = content[i]
                            if escape_next:
                                fixed_content += char
                                escape_next = False
                            elif char == '\\':
                                fixed_content += char
                                escape_next = True
                            elif char == '"' and not escape_next:
                                in_string = not in_string
                                fixed_content += char
                            elif in_string:
                                # Inside a string - escape control characters
                                if char == '\n':
                                    fixed_content += '\\n'
                                elif char == '\r':
                                    fixed_content += '\\r'
                                elif char == '\t':
                                    fixed_content += '\\t'
                                elif ord(char) < 32:  # Other control characters
                                    fixed_content += f'\\u{ord(char):04x}'
                                else:
                                    fixed_content += char
                            else:
                                fixed_content += char
                            i += 1
                        content = fixed_content
                        result = json.loads(content)
                    else:
                        raise
                fix_sketch = result.get("fix_sketch", "Unable to generate fix sketch.")
                test_code = result.get("test_code", "")
                
                # Validate test_code - check if it looks like actual Python code
                if not test_code or len(test_code.strip()) < 20:
                    raise ValueError("test_code is too short or empty")
                
                # Check for common placeholder phrases
                placeholder_phrases = [
                    "see above", "see below", "todo", "placeholder", 
                    "implement", "add code", "write test", "test code here"
                ]
                test_code_lower = test_code.lower()
                if any(phrase in test_code_lower for phrase in placeholder_phrases) and "def test" not in test_code_lower:
                    raise ValueError(f"test_code appears to contain placeholder text: {test_code[:100]}")
                
                # Ensure test_code has at least a function definition
                if "def test" not in test_code and "def " not in test_code:
                    raise ValueError("test_code does not contain a function definition")
                
                # Unescape newlines if they're escaped
                if "\\n" in test_code:
                    test_code = test_code.replace("\\n", "\n")
                
            except Exception as e:
                log_step("llm_parse_error", f"Failed to parse LLM response as JSON: {str(e)[:200]}")
                log_step("llm_parse_error", f"Raw response content (first 500 chars): {content[:500]}")
                
                # Fallback: generate a proper test based on the issue
                fix_sketch = content[:1000] if len(content) > 100 else "Fix sketch could not be generated."
                
                # Generate a more realistic test based on the issue description
                test_code = f"""import pytest

def test_regression_{severity}_{hash(issue_text) % 10000}():
    \"\"\"Failing test for issue: {issue_text[:80]}
    
    Repro steps:
{chr(10).join(f'    # {i+1}. {step}' for i, step in enumerate(repro_steps[:3]))}
    \"\"\"
    # This test should reproduce the issue
    # TODO: Replace with actual implementation based on the issue
    assert False, "Test needs implementation to reproduce: {issue_text[:100]}"
"""
            
            log_step("llm_propose_fix", f"Generated fix sketch ({len(fix_sketch)} chars) and test code ({len(test_code)} chars)")
            return fix_sketch, test_code
        except Exception as e:
            log_step("llm_fallback", f"Fix proposal failed, using mock proposer: {str(e)[:200]}")
            # Fallback to mock proposer
            return mock_propose_fix_sketch(severity, repro_steps, repo_readme)
    
    async def reason_with_tools(self, query: str, max_iterations: int = 5) -> Dict:
        """
        Main reasoning loop: LLM can call tools, observe results, and continue reasoning.
        Returns final answer and tool call history.
        """
        messages = [{"role": "system", "content": "You are a helpful assistant that can use tools to gather information and reason about software issues."}]
        messages.append({"role": "user", "content": query})
        
        tool_call_history = []
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            log_step("llm_reasoning", f"Iteration {iteration}: Processing query")
            
            # Get tool schemas
            tool_schemas = self._get_tool_schemas()
            if tool_schemas:
                log_step("llm_reasoning", f"Available tools for LLM: {[t.get('function', {}).get('name', 'unknown') for t in tool_schemas]}")

            # Call LLM
            response = await self._call_llm(messages, tools=tool_schemas if tool_schemas else None)
            
            # Log if LLM returned tool calls
            if response.get("tool_calls"):
                log_step("llm_reasoning", f"LLM requested {len(response.get('tool_calls', []))} tool calls")
            else:
                log_step("llm_reasoning", "LLM did not request any tool calls")
            
            # Add assistant message
            assistant_msg = {"role": "assistant", "content": response.get("content")}
            if response.get("tool_calls"):
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.get("id", f"call_{iteration}_{i}"),
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"]
                        }
                    }
                    for i, tc in enumerate(response["tool_calls"])
                ]
            messages.append(assistant_msg)
            
            # If no tool calls, we're done
            if not response.get("tool_calls") or response.get("finish_reason") == "stop":
                log_step("llm_reasoning", f"Completed reasoning after {iteration} iterations")
                return {
                    "answer": response.get("content", ""),
                    "tool_calls": tool_call_history,
                    "iterations": iteration
                }
            
            # Execute tool calls
            for tool_call in response["tool_calls"]:
                func_name = tool_call["function"]["name"]
                func_args = json.loads(tool_call["function"]["arguments"])
                
                if func_name not in self.tools:
                    error_msg = f"Unknown tool: {func_name}"
                    log_step("llm_error", error_msg)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id"),
                        "name": func_name,
                        "content": json.dumps({"error": error_msg})
                    })
                    continue
                
                # Call the tool
                try:
                    log_step("tool_usage", f"Used tool: {func_name}")
                    tool = self.tools[func_name]
                    result = tool.call(**func_args)
                    
                    # Log tool call
                    tool_call_history.append({
                        "tool": func_name,
                        "args": func_args,
                        "result": str(result)[:500]  # Truncate long results
                    })
                    
                    # Add tool result to messages - safely serialize result
                    safe_result = safe_json_serialize(result)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id"),
                        "name": func_name,
                        "content": json.dumps(safe_result, default=str)
                    })
                except Exception as e:
                    error_msg = str(e)[:500]
                    log_step("llm_tool_error", f"Tool {func_name} error: {error_msg}")
                    tool_call_history.append({
                        "tool": func_name,
                        "args": func_args,
                        "error": error_msg
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id"),
                        "name": func_name,
                        "content": json.dumps({"error": error_msg})
                    })
        
        # Max iterations reached
        log_step("llm_reasoning", f"Reached max iterations ({max_iterations})")
        return {
            "answer": messages[-1].get("content", ""),
            "tool_calls": tool_call_history,
            "iterations": iteration,
            "truncated": True
        }


