import asyncio
import google.generativeai as genai
from anthropic import AsyncAnthropic

class LLMService:
    def __init__(self, provider: str, anthropic_key: str, gemini_key: str):
        self.provider = provider.lower()
        self.anthropic_client = None
        self.gemini_model_haiku = None
        self.gemini_model_sonnet = None

        if self.provider == "gemini" and gemini_key:
            genai.configure(api_key=gemini_key)
            self.gemini_model_haiku = genai.GenerativeModel('gemini-1.5-flash')
            self.gemini_model_sonnet = genai.GenerativeModel('gemini-1.5-pro')
        else:
            self.anthropic_client = AsyncAnthropic(api_key=anthropic_key)

    async def generate_text(self, prompt: str, tier: str = "haiku", max_tokens: int = 300) -> str:
        if self.provider == "gemini":
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
