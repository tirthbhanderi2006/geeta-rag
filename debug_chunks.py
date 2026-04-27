"""Test that Dharmakshetra retrieval now works correctly."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from utils import retrieve_chunks

print("=== Query: 'What is Dharmakshetra?' ===\n")
results = retrieve_chunks("What is Dharmakshetra?", top_k=5, persist_dir="chroma_db")

for i, r in enumerate(results):
    print(f"[{i+1}] Page {r['page_num']} | Strategy: {r['strategy']} | Score: {r['score']}")
    # Show ASCII-safe preview
    safe_text = r['text'][:200].encode('ascii', errors='replace').decode('ascii')
    print(f"    Text: {safe_text}")
    print()

pages_seen = set(r['page_num'] for r in results)
print(f"Pages represented: {sorted(pages_seen)} ({len(pages_seen)} unique pages)")

if 1 in pages_seen:
    print("SUCCESS: Page 1 (Dharmakshetra definition) IS in results!")
else:
    print("FAIL: Page 1 is missing from results")
