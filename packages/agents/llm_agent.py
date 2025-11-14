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

# Try to import httpx for Ollama API calls
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    log_step("llm_agent", "httpx not available, Ollama support disabled")

# Try to import OpenAI, fall back to mock if not available
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    log_step("llm_agent", "OpenAI not available, will use mock mode")

# Try to import Google Gemini, fall back if not available
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    log_step("llm_agent", "Gemini not available")


class Tool:
    """Represents a callable tool with metadata"""
    def __init__(self, name: str, description: str, func: Callable, parameters: Dict[str, Any]):
        self.name = name
        self.description = description
        self.func = func
        self.parameters = parameters
    
    def call(self, **kwargs) -> Any:
        """Call the tool with given arguments"""
        log_step("tool_call", f"Calling tool: {self.name} with args: {json.dumps(kwargs, default=str)}")
        try:
            result = self.func(**kwargs)
            log_step("tool_result", f"Tool {self.name} succeeded: {str(result)[:200]}")
            return result
        except Exception as e:
            error_msg = str(e)[:500]
            log_step("tool_error", f"Tool {self.name} failed: {error_msg}")
            raise


class LLMAgent:
    """LLM Agent with tool calling capabilities - supports OpenAI, Gemini, and Ollama"""
    
    # Supported models
    OPENAI_MODELS = ["gpt-4o-mini"]
    GEMINI_MODELS = ["gemini-2.5-flash"]
    OLLAMA_MODELS = ["ollama:llama3.1"]  # Format: ollama:model_name
    
    def __init__(self, api_key: Optional[str] = None, model: str = "ollama:llama3.1", use_mock: bool = False, ollama_base_url: str = "http://127.0.0.1:11434"):
        self.model = model
        self.provider = self._detect_provider(model)
        self.tools: Dict[str, Tool] = {}
        # Use 127.0.0.1 instead of localhost for better Windows compatibility
        self.ollama_base_url = ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        
        # Initialize based on provider
        if use_mock:
            self.use_mock = True
            self.client = None
            self.gemini_client = None
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
        elif self.provider == "openai":
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.use_mock = not OPENAI_AVAILABLE or not self.api_key
            if not self.use_mock:
                openai.api_key = self.api_key
                self.client = openai.OpenAI(api_key=self.api_key)
                log_step("llm_agent", f"Initialized OpenAI client with model: {model}")
            else:
                log_step("llm_agent", "Running in mock mode (OpenAI not available or no API key)")
        elif self.provider == "gemini":
            self.api_key = api_key or os.getenv("GEMINI_API_KEY")
            self.use_mock = not GEMINI_AVAILABLE or not self.api_key
            if not self.use_mock:
                genai.configure(api_key=self.api_key)
                # Map user-friendly model names to actual Google API model names
                # Google API model names: gemini-2.0-flash-exp, gemini-1.5-pro, gemini-pro
                # Note: gemini-2.5-flash doesn't exist yet, using gemini-2.0-flash-exp as closest match
                model_mapping = {
                    "gemini-2.5-flash": "gemini-2.0-flash-exp",  # Map 2.5 to 2.0 experimental (latest available)
                    "gemini-2.0-flash-exp": "gemini-2.0-flash-exp",
                    "gemini-1.5-pro": "gemini-1.5-pro",
                    "gemini-1.5-flash": "gemini-2.0-flash-exp",  # 1.5-flash deprecated, use 2.0
                }
                model_name = model_mapping.get(model, model)  # Use mapping or original if not found
                self.gemini_client = genai.GenerativeModel(model_name)
                log_step("llm_agent", f"Initialized Gemini client with model: {model_name} (requested: {model})")
            else:
                self.gemini_client = None
                log_step("llm_agent", "Running in mock mode (Gemini not available or no API key)")
        else:
            self.use_mock = True
            log_step("llm_agent", f"Unknown model {model}, running in mock mode")
    
    def _detect_provider(self, model: str) -> str:
        """Detect which provider to use based on model name"""
        if model in self.OPENAI_MODELS or model.startswith("gpt-"):
            return "openai"
        elif model in self.GEMINI_MODELS or model.startswith("gemini-"):
            return "gemini"
        elif model in self.OLLAMA_MODELS or model.startswith("ollama:"):
            return "ollama"
        else:
            return "unknown"
    
    def register_tool(self, tool: Tool):
        """Register a tool for the agent to use"""
        self.tools[tool.name] = tool
        log_step("llm_agent", f"Registered tool: {tool.name}")
    
    def _get_tool_schemas(self) -> List[Dict]:
        """Convert registered tools to OpenAI function calling format"""
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
        """Call the LLM API (OpenAI, Gemini, or Ollama) with fallback to mock on errors"""
        if self.use_mock:
            # Mock response for testing
            return self._mock_llm_response(messages)

        try:
            if self.provider == "openai":
                return self._call_openai(messages, tools)
            elif self.provider == "gemini":
                return self._call_gemini(messages, tools)
            elif self.provider == "ollama":
                return await self._call_ollama_async(messages, tools)
            else:
                return self._mock_llm_response(messages)
        except Exception as e:
            error_msg = str(e).lower()
            # Check for quota/rate limit errors
            is_quota_error = any(keyword in error_msg for keyword in [
                "quota", "rate limit", "429", "insufficient_quota",
                "billing", "exceeded", "limit exceeded"
            ])

            if is_quota_error:
                log_step("llm_fallback", f"Quota/rate limit exceeded, falling back to mock mode. Error: {str(e)[:200]}")
            else:
                log_step("llm_fallback", f"LLM API call failed, falling back to mock mode. Error: {str(e)[:200]}")

            # Fall back to mock response instead of raising
            return self._mock_llm_response(messages)
    
    def _call_openai(self, messages: List[Dict], tools: Optional[List[Dict]] = None) -> Dict:
        """Call OpenAI API with error handling"""
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2000,
            }
            
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            
            response = self.client.chat.completions.create(**kwargs)
            return {
                "content": response.choices[0].message.content,
                "tool_calls": response.choices[0].message.tool_calls or [],
                "finish_reason": response.choices[0].finish_reason
            }
        except Exception as e:
            error_msg = str(e).lower()
            # Re-raise with context for quota detection
            if "quota" in error_msg or "rate limit" in error_msg or "429" in error_msg or "insufficient_quota" in error_msg:
                raise Exception(f"OpenAI quota/rate limit exceeded: {str(e)[:200]}")
            raise
    
    def _call_gemini(self, messages: List[Dict], tools: Optional[List[Dict]] = None) -> Dict:
        """Call Gemini API"""
        # Convert messages format for Gemini
        # Gemini uses a different format - combine system/user messages
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(content)
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        full_prompt = "\n".join(prompt_parts)
        
        # For now, Gemini tool calling is simplified (Gemini 2.0 supports function calling)
        # We'll use a simpler approach: include tool descriptions in the prompt
        if tools:
            tool_descriptions = []
            for tool in tools:
                func = tool.get("function", {})
                tool_descriptions.append(
                    f"- {func.get('name')}: {func.get('description')}"
                )
            full_prompt += f"\n\nAvailable tools:\n" + "\n".join(tool_descriptions)
            full_prompt += "\n\nYou can use these tools by describing what you want to do. The system will execute the appropriate tool calls."
        
        try:
            response = self.gemini_client.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=2000,
                )
            )
        except Exception as e:
            error_msg = str(e).lower()
            # Check for quota/rate limit errors
            if "quota" in error_msg or "rate limit" in error_msg or "429" in error_msg or "exceeded" in error_msg:
                raise Exception(f"Gemini quota/rate limit exceeded: {str(e)[:200]}")
            # Provide helpful error message if model not found
            if "404" in error_msg or "not found" in error_msg:
                log_step("llm_error", f"Model {self.model} not found. Available Gemini models: gemini-2.0-flash-exp, gemini-1.5-pro")
                raise Exception(f"Model {self.model} is not available. Try: gemini-2.0-flash-exp or gemini-1.5-pro")
            raise
        
        # Parse response
        content = response.text if hasattr(response, 'text') else str(response)
        
        # Gemini doesn't have native tool calling in the same way, so we return empty tool_calls
        # In a production system, you'd parse the response for tool call requests
        return {
            "content": content,
            "tool_calls": [],  # Gemini tool calling would need custom parsing
            "finish_reason": "stop"
        }
    
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
                elif "tool_calls" in result:
                    tool_calls = result["tool_calls"]

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

            # Call LLM
            response = await self._call_llm(messages, tools=tool_schemas if tool_schemas else None)
            
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
                    tool = self.tools[func_name]
                    result = tool.call(**func_args)
                    
                    # Log tool call
                    tool_call_history.append({
                        "tool": func_name,
                        "args": func_args,
                        "result": str(result)[:500]  # Truncate long results
                    })
                    
                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id"),
                        "name": func_name,
                        "content": json.dumps(result, default=str)
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


