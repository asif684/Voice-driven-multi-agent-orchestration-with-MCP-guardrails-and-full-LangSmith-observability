#!/usr/bin/env python3
"""Quick test for the research_tool (requires TAVILY_API_KEY in .env)."""
import os
from dotenv import load_dotenv

load_dotenv()

# Check Tavily key
tavily_key = os.getenv("TAVILY_API_KEY")
if not tavily_key or tavily_key.startswith("tvly-..."):
    print("[ERROR] TAVILY_API_KEY not set in .env")
    print("   Get one at https://app.tavily.com and add to .env")
    exit(1)

print(f"[OK] TAVILY_API_KEY found: {tavily_key[:10]}...")

# Now test the research tool
from voice_notion_agent.tools.research_tool import research_topic

print("\n[TEST] Testing research_topic('pros and cons of server-side rendering')...")
try:
    result = research_topic.invoke({"topic": "pros and cons of server-side rendering"})
    print(f"\n[OK] Research result ({len(result)} chars):\n")
    print(result)
except Exception as e:
    print(f"[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()