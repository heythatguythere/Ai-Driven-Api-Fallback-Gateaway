import asyncio
import random
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any

app = FastAPI(
    title="AI-Driven API Fallback Gateway",
    description="Enterprise middleware for LLM routing and fault tolerance.",
    version="1.0.0"
)

class GatewayRequest(BaseModel):
    prompt: str = Field(..., description="The user's input text for the LLM.")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Creativity threshold.")

PRIMARY_PROVIDER = "Groq"
SECONDARY_PROVIDER = "Ollama-Local"

@app.get("/health")
async def system_health():
    return {"status": "online", "message": "Gateway is operational."}

def adapt_payload(provider: str, data: GatewayRequest) -> Dict[str, Any]:
    if provider == PRIMARY_PROVIDER:
        return {
            "model": "mixtral-8x7b-32768",
            "messages": [{"role": "user", "content": data.prompt}],
            "temperature": data.temperature
        }
    elif provider == SECONDARY_PROVIDER:
        return {
            "model": "llama3",
            "prompt": data.prompt,
            "options": {"temperature": data.temperature},
            "stream": False
        }
    raise ValueError(f"Unsupported provider: {provider}")

async def execute_request_with_backoff(provider: str, payload: dict, max_retries: int = 2) -> dict:
    base_delay = 1.0
    async with httpx.AsyncClient() as client:
        for attempt in range(max_retries + 1):
            try:
                print(f"\n[Gateway Engine] Target: {provider} | Attempt {attempt + 1}/{max_retries + 1}")
                if provider == PRIMARY_PROVIDER and attempt < 5:
                    print(f" [Simulated Error] {PRIMARY_PROVIDER} returned HTTP Status 429.")
                    raise httpx.HTTPStatusError(
                        "Rate limit hit", 
                        request=httpx.Request("POST", "https://api.groq.com"), 
                        response=httpx.Response(429)
                    )
                await asyncio.sleep(0.4) 
                return {
                    "provider_used": provider,
                    "status": "success",
                    "payload_sent": payload,
                    "text": f"Response generated successfully by {provider} proxy layer."
                }
            except httpx.HTTPStatusError as e:
                if e.response.status_code in [429, 503]:
                    if attempt == max_retries:
                        raise e 
                    delay = base_delay * (2 ** attempt)
                    jitter = random.uniform(0, 0.4)
                    total_delay = delay + jitter
                    print(f" [Retry Logic] Backing off. Retrying in {total_delay:.2f} seconds...")
                    await asyncio.sleep(total_delay)
                else:
                    raise e

@app.post("/v1/chat/completions")
async def handle_gateway_request(request_data: GatewayRequest):
    """Core Middleware Routing Interceptor"""
    try:
        print("\n--- [INCOMING REQUEST INTERCEPTED] ---")
        primary_payload = adapt_payload(PRIMARY_PROVIDER, request_data)
        result = await execute_request_with_backoff(PRIMARY_PROVIDER, primary_payload)
        return result
        
    except httpx.HTTPStatusError as e:
        print(f"\n[CRITICAL FAULT] {PRIMARY_PROVIDER} exhausted all retries. Error: {e.response.status_code}")
        print(f"[FAILOVER TRIGGERED] Diverting traffic dynamically to {SECONDARY_PROVIDER}...")
        
        try:
            fallback_payload = adapt_payload(SECONDARY_PROVIDER, request_data)
            result = await execute_request_with_backoff(SECONDARY_PROVIDER, fallback_payload, max_retries=1)
            return result
        except Exception as fallback_error:
            raise HTTPException(
                status_code=500, 
                detail=f"Complete System Failure. Both primary and fallback AI providers failed: {str(fallback_error)}"
            )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
