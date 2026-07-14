"""
MyAI - 0 dan yozilgan neyron til modeli (character-level neural language model).

Hech qanday tayyor ML kutubxona (PyTorch/TensorFlow) ISHLATILMAYDI.
Faqat numpy (matritsa hisoblari uchun). Forward pass, backpropagation
(gradientlar) va o'qitish - hammasi qo'lda, 0 dan yozilgan.

Arxitektura (Bengio 2003 uslubidagi MLP til modeli):
    kontekst (oxirgi N belgi) -> embedding -> yashirin qatlam (tanh) -> softmax -> keyingi belgi
"""

import numpy as np
import json
import pickle
from pathlib import Path


class MyAI:
    def __init__(self, vocab_size=0, block_size=8, n_embd=24, n_hidden=128, seed=1337):
        self.block_size = block_size      # nechta oldingi belgiga qarab bashorat qilamiz
        self.n_embd = n_embd              # har bir belgi vektori o'lchami
        self.n_hidden = n_hidden          # yashirin qatlam neyronlari
        self.stoi = {}                    # belgi -> raqam
        self.itos = {}                    # raqam -> belgi
        self.vocab_size = vocab_size
        self._rng = np.random.RandomState(seed)
        if vocab_size:
            self._init_params()

    def _init_params(self):
        r = self._rng
        V, E, H, B = self.vocab_size, self.n_embd, self.n_hidden, self.block_size
        # Kichik tasodifiy og'irliklar (Kaiming-ga yaqin masshtab)
        self.C  = r.randn(V, E) * 0.1                      # embedding jadvali
        self.W1 = r.randn(B * E, H) * (5/3) / (B*E)**0.5   # tanh uchun gain
        self.b1 = np.zeros(H)
        self.W2 = r.randn(H, V) * 0.01
        self.b2 = np.zeros(V)

    # ------------------------------------------------------------------ #
    #  Matn <-> raqamlar
    # ------------------------------------------------------------------ #
    def build_vocab(self, text):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)
        self._init_params()

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def _make_dataset(self, text):
        """Matndan (kontekst -> keyingi belgi) juftliklarini tayyorlaydi."""
        data = self.encode(text)
        X, Y = [], []
        ctx = [0] * self.block_size
        for idx in data:
            X.append(ctx)
            Y.append(idx)
            ctx = ctx[1:] + [idx]
        return np.array(X), np.array(Y)

    # ------------------------------------------------------------------ #
    #  Forward + Backward (0 dan, qo'lda)
    # ------------------------------------------------------------------ #
    def _forward(self, X):
        emb = self.C[X]                              # (B, block, E)
        x = emb.reshape(X.shape[0], -1)              # (B, block*E)
        hpre = x @ self.W1 + self.b1                 # (B, H)
        h = np.tanh(hpre)                            # (B, H)
        logits = h @ self.W2 + self.b2               # (B, V)
        cache = (X, emb, x, h, logits)
        return logits, cache

    def _loss(self, logits, Y):
        # Barqaror softmax + cross-entropy
        logits = logits - logits.max(axis=1, keepdims=True)
        counts = np.exp(logits)
        probs = counts / counts.sum(axis=1, keepdims=True)
        n = Y.shape[0]
        loss = -np.log(probs[np.arange(n), Y] + 1e-9).mean()
        return loss, probs

    def _backward(self, cache, probs, Y, lr):
        X, emb, x, h, logits = cache
        n = Y.shape[0]

        # softmax+CE gradienti
        dlogits = probs.copy()
        dlogits[np.arange(n), Y] -= 1
        dlogits /= n

        dW2 = h.T @ dlogits
        db2 = dlogits.sum(0)
        dh = dlogits @ self.W2.T
        dhpre = dh * (1 - h**2)              # tanh hosilasi
        dW1 = x.T @ dhpre
        db1 = dhpre.sum(0)
        dx = dhpre @ self.W1.T               # (B, block*E)
        demb = dx.reshape(emb.shape)         # (B, block, E)

        # embedding jadvaliga gradientni tarqatamiz
        dC = np.zeros_like(self.C)
        np.add.at(dC, X, demb)

        # SGD yangilanish
        self.W1 -= lr * dW1; self.b1 -= lr * db1
        self.W2 -= lr * dW2; self.b2 -= lr * db2
        self.C  -= lr * dC

    # ------------------------------------------------------------------ #
    #  O'qitish
    # ------------------------------------------------------------------ #
    def train(self, text, steps=3000, batch_size=64, lr=0.1, log_every=200):
        X, Y = self._make_dataset(text)
        nvals = X.shape[0]
        r = self._rng
        history = []
        for step in range(1, steps + 1):
            # minibatch
            ix = r.randint(0, nvals, size=batch_size)
            xb, yb = X[ix], Y[ix]
            logits, cache = self._forward(xb)
            loss, probs = self._loss(logits, yb)
            # lr ni oxirida biroz kamaytiramiz
            cur_lr = lr if step < steps * 0.8 else lr * 0.1
            self._backward(cache, probs, yb, cur_lr)
            if step % log_every == 0 or step == 1:
                history.append((step, float(loss)))
                print(f"  qadam {step:5d}/{steps}  loss={loss:.4f}")
        return history

    # ------------------------------------------------------------------ #
    #  Generatsiya (matn yaratish)
    # ------------------------------------------------------------------ #
    def generate(self, prompt="", n=200, temperature=1.0, seed=None):
        rng = np.random.RandomState(seed) if seed is not None else self._rng
        ctx = [0] * self.block_size
        out = []
        # prompt bilan kontekstni to'ldiramiz
        for ch in prompt:
            if ch in self.stoi:
                ctx = ctx[1:] + [self.stoi[ch]]
                out.append(self.stoi[ch])
        for _ in range(n):
            X = np.array([ctx])
            logits, _ = self._forward(X)
            logits = logits[0] / max(temperature, 1e-3)
            logits -= logits.max()
            p = np.exp(logits); p /= p.sum()
            idx = rng.choice(self.vocab_size, p=p)
            out.append(idx)
            ctx = ctx[1:] + [idx]
        return self.decode(out)

    # ------------------------------------------------------------------ #
    #  Saqlash / yuklash
    # ------------------------------------------------------------------ #
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "block_size": self.block_size, "n_embd": self.n_embd,
                "n_hidden": self.n_hidden, "vocab_size": self.vocab_size,
                "stoi": self.stoi, "itos": self.itos,
                "C": self.C, "W1": self.W1, "b1": self.b1,
                "W2": self.W2, "b2": self.b2,
            }, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        m = cls(vocab_size=0)
        m.block_size = d["block_size"]; m.n_embd = d["n_embd"]
        m.n_hidden = d["n_hidden"]; m.vocab_size = d["vocab_size"]
        m.stoi = d["stoi"]; m.itos = d["itos"]
        m.C = d["C"]; m.W1 = d["W1"]; m.b1 = d["b1"]
        m.W2 = d["W2"]; m.b2 = d["b2"]
        return m


if __name__ == "__main__":
    # Kichik o'z-o'zini test (XOR-ga o'xshash: takrorlanuvchi matnni o'rganadimi?)
    text = "salom dunyo. " * 200
    ai = MyAI(block_size=4, n_embd=8, n_hidden=32)
    ai.build_vocab(text)
    print("Vocab:", ai.vocab_size, "belgi")
    ai.train(text, steps=1000, log_every=200)
    print("Namuna:", repr(ai.generate("salom", n=40, seed=1)))
