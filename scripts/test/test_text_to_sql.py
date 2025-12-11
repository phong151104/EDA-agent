"""
Interactive Text-to-SQL Test.

Run: python scripts/test/test_text_to_sql.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


async def main():
    from src.mcp.tools import TextToSQL
    
    print("\n" + "=" * 60)
    print("  TEXT-TO-SQL INTERACTIVE TEST")
    print("=" * 60)
    print("  Commands:")
    print("    'quit' or 'q' - Exit")
    print("    'reuse' - Reuse last session")
    print("    'new' - Force new session")
    print()
    
    tool = TextToSQL()
    last_session_id = None
    use_last_session = False
    
    while True:
        try:
            prompt = input("📝 Prompt: ").strip()
            
            if not prompt:
                continue
            
            if prompt.lower() in ['quit', 'q', 'exit']:
                print("👋 Goodbye!")
                break
            
            if prompt.lower() == 'reuse':
                if last_session_id:
                    use_last_session = True
                    print(f"  ✅ Will reuse session: {last_session_id}")
                else:
                    print("  ⚠️ No previous session.")
                continue
            
            if prompt.lower() == 'new':
                use_last_session = False
                print("  ✅ Will create new session")
                continue
            
            print("  ⏳ Generating SQL...")
            
            session_id = last_session_id if use_last_session else None
            result = await tool.generate(prompt, session_id=session_id)
            
            if result.success:
                last_session_id = result.session_id
                print(f"\n  ✅ Session: {result.session_id}")
                print(f"  📊 Tables: {', '.join(result.tables_used or [])}")
                print(f"\n  📄 SQL:")
                print("  " + "-" * 50)
                for line in result.sql.split("\n"):
                    print(f"  {line}")
                print("  " + "-" * 50)
            else:
                print(f"  ❌ Error: {result.error}")
            
            print()
            use_last_session = False
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"  ❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
