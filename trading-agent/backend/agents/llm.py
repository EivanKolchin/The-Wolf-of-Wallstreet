import asyncio
import json
import urllib.request
import urllib.error
import logging
from backend.core.config import settings

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, provider: str, anthropic_key: str, gemini_key: str, ollama_model: str = "llama3"):
        self.provider = (provider or "ollama").lower()
        self.primary_ollama_model = ollama_model
        self.current_ollama_model = ollama_model
        self.fallback_ollama_model = settings.OLLAMA_FALLBACK_MODEL.strip()
        self.downgrade_time = 0
        self._pull_attempted_models: set[str] = set()

        self.anthropic_client = None
        self.gemini_model_haiku = None
        self.gemini_model_sonnet = None

        if "gemini" in self.provider and gemini_key and "your_" not in gemini_key:
            try:
                # Suppress the warning by using the intended package if possible, or suppressing warnings
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    import google.generativeai as genai
                
                genai.configure(api_key=gemini_key)
                self.gemini_model_haiku = genai.GenerativeModel('gemini-1.5-flash')
                self.gemini_model_sonnet = genai.GenerativeModel('gemini-1.5-pro')
            except ImportError:
                logger.error("google-generativeai not installed, cannot use Gemini")

        if ("anthropic" in self.provider or "claude" in self.provider) and anthropic_key and "your_" not in anthropic_key:
            try:
                from anthropic import AsyncAnthropic
                self.anthropic_client = AsyncAnthropic(api_key=anthropic_key)
            except ImportError:
                logger.error("anthropic package not installed")

    def _push_system_alert(self, message: str):
        import os, datetime
        try:
            cache_file = os.path.join(os.getcwd(), "raw_news_cache.json")
            existing = []
            if os.path.exists(cache_file):
                with open(cache_file, "r") as f:
                    existing = json.load(f)
                    
            existing.insert(0, {
                "headline": "⚡ SYSTEM ALERT: " + message,
                "source": "RESOURCE MANAGER",
                "time": datetime.datetime.utcnow().isoformat()
            })
            with open(cache_file, "w") as f:
                json.dump(existing[:20], f)
        except: pass

    def _sync_llm_state(self, active: bool = False):
        """Synchronizes LLM status to a shared file for the FastAPI frontend to read/write."""
        import os
        state_file = os.path.join(os.getcwd(), "llm_state.json")
        try:
            if os.path.exists(state_file):
                with open(state_file, "r") as f:
                    state = json.load(f)
                    # Check if frontend requested a force revert
                    if state.get("force_revert", False) and self.current_ollama_model != self.primary_ollama_model:
                        print(f"[Resource Manager] User forced immediate revert to {self.primary_ollama_model}!")
                        self.current_ollama_model = self.primary_ollama_model
                        self.downgrade_time = 0
            
            # Write current actual state
            with open(state_file, "w") as f:
                json.dump({
                    "primary_model": self.primary_ollama_model,
                    "current_model": self.current_ollama_model,
                    "downgrade_time": self.downgrade_time,
                    "is_overloaded": self.current_ollama_model != self.primary_ollama_model,
                    "force_revert": False
                }, f)
        except Exception: pass

    async def _call_ollama(self, prompt: str) -> str:
        """Call local Ollama instance with auto-downgrade failover protection if hardware gets overloaded."""
        import time
        url = "http://localhost:11434/api/generate"
        
        self._sync_llm_state()
        
        # Check if we should attempt to recover back up to the primary model (cooldown: 2 mins)
        if self.current_ollama_model != self.primary_ollama_model:
            if time.time() - self.downgrade_time > 120:
                print(f"[Resource Manager] Cooldown finished. Attempting to restore primary {self.primary_ollama_model} engine.")
                self._push_system_alert(f"Cooldown completed. Testing memory resources with primary {self.primary_ollama_model} model again...")
                self.current_ollama_model = self.primary_ollama_model
                self.downgrade_time = 0
                self._sync_llm_state()

        safe_prompt = prompt + "\n\nIMPORTANT: Provide ONLY valid JSON. No conversational text."
        
        def build_request():
            data = {
                "model": self.current_ollama_model, 
                "prompt": safe_prompt, 
                "stream": False,
                "options": {
                    "num_ctx": 4096
                }
            }
            return urllib.request.Request(url, data=json.dumps(data).encode('utf-8'),
                                        headers={'Content-Type': 'application/json'})

        def _model_name_matches(installed: str, target: str) -> bool:
            i = (installed or "").strip().lower()
            t = (target or "").strip().lower()
            if not i or not t:
                return False
            if i == t:
                return True
            if ":" not in t and i == f"{t}:latest":
                return True
            if t.endswith(":latest") and i == t[:-7]:
                return True
            # Covers richer variant tags such as llama3.2:1b-instruct-* for target llama3.2:1b
            if i.startswith(f"{t}-") or i.startswith(f"{t}:"):
                return True
            return False

        def _is_model_installed(target_model: str) -> bool:
            try:
                tags_req = urllib.request.Request("http://127.0.0.1:11434/api/tags")
                with urllib.request.urlopen(tags_req, timeout=5) as tags_response:
                    tags_data = json.loads(tags_response.read().decode("utf-8"))
                for model_obj in tags_data.get("models", []):
                    if _model_name_matches(model_obj.get("name", ""), target_model):
                        return True
            except Exception:
                return False
            return False


        def fetch(retry_on_404=True):
            req = build_request()
            try:
                with urllib.request.urlopen(req, timeout=120) as response:
                    return json.loads(response.read().decode())['response']
            except urllib.error.HTTPError as e:
                error_body = e.read().decode('utf-8', errors='ignore')
                
                # Check for OOM/Hardware overload triggers
                if e.code == 500 and ("terminated" in error_body or "memory" in error_body.lower() or "oom" in error_body.lower() or "failed to allocate" in error_body.lower()):
                    if self.current_ollama_model == self.primary_ollama_model and self.fallback_ollama_model:
                        print(f"[Ollama Critical] VRAM overload on {self.primary_ollama_model}! Fallback to {self.fallback_ollama_model} triggered.")
                        self.current_ollama_model = self.fallback_ollama_model
                        self.downgrade_time = time.time()
                        self._sync_llm_state()
                        self._push_system_alert(f"Critical memory overload. The primary engine ({self.primary_ollama_model}) crashed the LLM handler. Auto-downgrading to lightweight {self.fallback_ollama_model} parameters to keep data flowing. Will re-attempt the {self.primary_ollama_model} engine in 2 minutes.")
                        return fetch(retry_on_404=True) # Retry immediately with fallback model
                    print("[Ollama Critical] Memory overload detected and no fallback model configured.")
                    return ""

                if e.code == 404 and retry_on_404:
                    normalized_model = self.current_ollama_model.strip().lower()
                    if _is_model_installed(normalized_model):
                        print(f"[Ollama Info] Model '{self.current_ollama_model}' appears installed via tags, skipping pull and retrying once.")
                        return fetch(retry_on_404=False)

                    if not settings.AUTO_PULL_OLLAMA_MODELS:
                        print(
                            f"[Ollama Info] Model '{self.current_ollama_model}' is missing. "
                            "Auto-pull disabled (AUTO_PULL_OLLAMA_MODELS=false). "
                            f"Run: ollama pull {self.current_ollama_model}"
                        )
                        return ""
                    if normalized_model in self._pull_attempted_models:
                        print(f"[Ollama Info] Pull for '{self.current_ollama_model}' already attempted in this session. Skipping repeated pull.")
                        return ""
                    print(f"[Ollama Info] Model '{self.current_ollama_model}' not found locally. Auto-downloading...")
                    try:
                        self._pull_attempted_models.add(normalized_model)
                        pull_req = urllib.request.Request("http://localhost:11434/api/pull",
                            data=json.dumps({"name": self.current_ollama_model, "stream": False}).encode('utf-8'),
                            headers={'Content-Type': 'application/json'})
                        urllib.request.urlopen(pull_req, timeout=600)
                        print(f"[Ollama Info] Successfully downloaded '{self.current_ollama_model}'.")
                        return fetch(retry_on_404=False)
                    except Exception as pe:
                        print(f"[Ollama Auto-Pull Error] {pe}")
                        return ""
                else:
                    print(f"[Ollama Error] HTTP Status {e.code}: {e.reason} - Details: {error_body}")
                    return ""
            except Exception as e:
                print(f"[Ollama Error] {e}")
                return ""
        return await asyncio.to_thread(fetch)

    def _compress_context(self, prompt: str, max_chars: int = 12000) -> str:
        """Truncate or compress overly long context to avoid open-source model forgetting."""
        if len(prompt) <= max_chars:
            return prompt
        # Naive compression: keep start (instructions) and end (latest data)
        keep_start = max_chars // 2
        keep_end = max_chars // 2
        return prompt[:keep_start] + "\n\n...[TRUNCATED DATA]...\n\n" + prompt[-keep_end:]

    async def generate_text(self, prompt: str, tier: str = "haiku", max_tokens: int = 300) -> str:
        prompt = self._compress_context(prompt)
        
        # Determine routing based on provider
        use_ollama = False
        use_gemini = False
        use_claude = False
        
        if self.provider == "ollama":
            use_ollama = True
        elif self.provider == "hybrid_gemini":
            use_ollama = True if tier == "haiku" else False
            use_gemini = not use_ollama
        elif self.provider == "hybrid_claude":
            use_ollama = True if tier == "haiku" else False
            use_claude = not use_ollama
        elif self.provider == "gemini":
            use_gemini = True
        else: # defaults to anthropic/claude
            use_claude = True

        if use_ollama:
            res = await self._call_ollama(prompt)
            # Basic validation check for JSON
            if "{" in res and "}" in res:
                return res
            # If Ollama failed completely or returned garbage, fallback if hybrid
            if self.provider == "hybrid_gemini":
                use_gemini = True
            elif self.provider == "hybrid_claude":
                use_claude = True
            else:
                return res # user is fully ollama, return best effort

        if use_gemini and self.gemini_model_haiku:
            # Attempt to use 2.5 models as 1.5 may be deprecated
            model = self.gemini_model_sonnet if tier == "sonnet" else self.gemini_model_haiku
            try:
                response = await model.generate_content_async(prompt)
                return response.text
            except Exception as e:
                import google.generativeai as genai
                if "404" in str(e):
                    # Fallback models for 2026+
                    fallback_model_name = 'gemini-2.5-pro' if tier == "sonnet" else 'gemini-2.5-flash'
                    fallback_model = genai.GenerativeModel(fallback_model_name)
                    response = await fallback_model.generate_content_async(prompt)
                    return response.text
                if "429" in str(e):
                    logger.error("Gemini Rate Limit Exceeded")
                    if "hybrid" in self.provider:
                         return await self._call_ollama(prompt)
                    return ""
                raise e
        elif use_claude and self.anthropic_client:
            anthropic_model = "claude-3-5-sonnet-20241022" if tier == "sonnet" else "claude-3-haiku-20240307"
            response = await self.anthropic_client.messages.create(
                model=anthropic_model,
                max_tokens=max_tokens,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        else:
            # Fallback natively to OLLAMA when all other keys are missing or invalid
            return await self._call_ollama(prompt)
