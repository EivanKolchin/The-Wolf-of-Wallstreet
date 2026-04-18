import asyncio
import json
import urllib.request
import google.generativeai as genai
from anthropic import AsyncAnthropic

class LLMService:
    def __init__(self, provider: str, anthropic_key: str, gemini_key: str, ollama_model: str = "llama3"):
        self.provider = provider.lower()
        self.ollama_model = ollama_model
        self.anthropic_client = None
        self.gemini_model_haiku = None
        self.gemini_model_sonnet = None

        if "gemini" in self.provider and gemini_key:
            genai.configure(api_key=gemini_key)
            self.gemini_model_haiku = genai.GenerativeModel('gemini-1.5-flash')
            self.gemini_model_sonnet = genai.GenerativeModel('gemini-1.5-pro')
            
        if "anthropic" in self.provider or "claude" in self.provider:
            if anthropic_key:
                self.anthropic_client = AsyncAnthropic(api_key=anthropic_key)

    async def _call_ollama(self, prompt: str) -> str:
        """Call local Ollama instance with optional JSON format enforcement (grammar) if needed."""
        url = "http://localhost:11434/api/generate"
        # Provide structural prompt hint to reduce hallucinations
        safe_prompt = prompt + "\n\nIMPORTANT: Provide ONLY valid JSON. No conversational text."
        data = {"model": self.ollama_model, "prompt": safe_prompt, "stream": False, "format": "json"}
        
        req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'),
                                    headers={'Content-Type': 'application/json'})
        def fetch():
            try:
                with urllib.request.urlopen(req, timeout=30) as response:
                    return json.loads(response.read().decode())['response']
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

        if use_gemini:
            # Attempt to use 2.5 models as 1.5 may be deprecated
            model = self.gemini_model_sonnet if tier == "sonnet" else self.gemini_model_haiku
            try:
                response = await model.generate_content_async(prompt)
                return response.text
            except Exception as e:
                if "404" in str(e):
                    # Fallback models for 2026+
                    fallback_model_name = 'gemini-2.5-pro' if tier == "sonnet" else 'gemini-2.5-flash'
                    fallback_model = genai.GenerativeModel(fallback_model_name)
                    response = await fallback_model.generate_content_async(prompt)
                    return response.text
                raise e
        else:
            anthropic_model = "claude-3-5-sonnet-20241022" if tier == "sonnet" else "claude-3-haiku-20240307"
            response = await self.anthropic_client.messages.create(
                model=anthropic_model,
                max_tokens=max_tokens,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
