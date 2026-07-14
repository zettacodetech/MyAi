"""MiniGPT (Transformer) ni corpus.txt da o'qitib gpt.pkl ga saqlaydi."""
import sys, time
from transformer import MiniGPT

CORPUS = sys.argv[1] if len(sys.argv) > 1 else "corpus.txt"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 4000

text = open(CORPUS, encoding="utf-8").read()
print(f"Korpus: {len(text)} belgi")

g = MiniGPT(block_size=24, n_embd=48)
g.build_vocab(text)
print(f"Vocab: {g.vocab_size} | parametrlar: transformer (attention + mlp)")
print(f"O'qitish ({STEPS} qadam)...")

t0 = time.time()
g.train(text, steps=STEPS, batch_size=16, lr=2e-3, log_every=250)
print(f"O'qitish tugadi: {time.time()-t0:.1f}s")

g.save("gpt.pkl")
print("Saqlandi: gpt.pkl")
print("\n--- NAMUNALAR (Transformer) ---")
for temp in (0.5, 0.7, 0.9):
    print(f"[temp={temp}] {g.generate('salom', n=140, temperature=temp, seed=1)!r}")
