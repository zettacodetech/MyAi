"""MyAI ni corpus.txt da o'qitib, model.pkl ga saqlaydi."""
import sys, time
from model import MyAI

CORPUS = sys.argv[1] if len(sys.argv) > 1 else "corpus.txt"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 8000

text = open(CORPUS, encoding="utf-8").read()
print(f"Korpus: {len(text)} belgi")

ai = MyAI(block_size=8, n_embd=24, n_hidden=128)
ai.build_vocab(text)
print(f"Vocab: {ai.vocab_size} belgi -> {''.join(ai.itos[i] for i in range(ai.vocab_size))!r}")
print(f"O'qitish boshlandi ({STEPS} qadam)...")

t0 = time.time()
ai.train(text, steps=STEPS, batch_size=64, lr=0.1, log_every=1000)
print(f"O'qitish tugadi: {time.time()-t0:.1f}s")

ai.save("model.pkl")
print("Model saqlandi: model.pkl")
print("\n--- NAMUNALAR ---")
for temp in (0.6, 0.8, 1.0):
    print(f"[temp={temp}] {ai.generate('salom', n=120, temperature=temp, seed=1)!r}")
