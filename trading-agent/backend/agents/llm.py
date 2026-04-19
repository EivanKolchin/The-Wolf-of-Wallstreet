import asyncio
import json
import urllib.request
import urllib.error
import logging

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, provider: str, anthropic_key: str, gemini_key: str, ollama_model: str = "llama3"):
        self.provider = (provider or "ollama").lower()
        self.ollama_model = ollama_model
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

    async def _call_ollama(self, prompt: str) -> str:
        """Call local Ollama instance with optional JSON format enforcement (grammar) if needed."""
        url = "http://localhost:11434/api/generate"
        # Provide structural prompt hint to reduce hallucinations
        safe_prompt = prompt + "\n\nIMPORTANT: Provide ONLY valid JSON. No conversational text."
        
        # Adding options to prevent VRAM spikes (Out Of Memory causing 500 Internal Server Errors)
        data = {
            "model": self.ollama_model, 
            "prompt": safe_prompt, 
            "stream": False,
            "options": {
                "num_ctx": 4096
            }
        }
        
        req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'),
                                    headers={'Content-Type': 'application/json'})
        def fetch(retry_on_404=True):
            try:
                with urllib.request.urlopen(req, timeout=120) as response:
                    return json.loads(response.read().decode())['response']
            except urllib.error.HTTPError as e:
                error_body = e.read().decode('utf-8', errors='ignore')
                if e.code == 404 and retry_on_404:
                    print(f"[Ollama Info] Model '{self.ollama_model}' not found locally. Auto-downloading (this may take a few minutes)...")
                    try:
                        pull_req = urllib.request.Request("http://localhost:11434/api/pull",
                            data=json.dumps({"name": self.ollama_model, "stream": False}).encode('utf-8'),
                            headers={'Content-Type': 'application/json'})
                        urllib.request.urlopen(pull_req, timeout=600)  # Up to 10 mins for auto-pull
                        print(f"[Ollama Info] Successfully downloaded '{self.ollama_model}'. Resuming operation...")
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
