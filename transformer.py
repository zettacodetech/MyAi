"""
MiniGPT - 0 dan yozilgan Transformer (self-attention).
autograd.py (o'zimizning avtomatik differensiallash) ustiga qurilgan.
PyTorch/TensorFlow YO'Q. Bu GPT'ning haqiqiy yuragi - e'tibor (attention) mexanizmi.
"""
import numpy as np
import pickle
from pathlib import Path
from autograd import Tensor, embed, cross_entropy


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params = params; self.lr = lr
        self.b1 = b1; self.b2 = b2; self.eps = eps
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.b1*self.m[i] + (1-self.b1)*g
            self.v[i] = self.b2*self.v[i] + (1-self.b2)*(g*g)
            mh = self.m[i]/(1-self.b1**self.t)
            vh = self.v[i]/(1-self.b2**self.t)
            p.data -= self.lr * mh/(np.sqrt(vh)+self.eps)
    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)


def layernorm(x, g, b):
    mu = x.mean(axis=-1, keepdims=True)
    xm = x - mu
    var = (xm*xm).mean(axis=-1, keepdims=True)
    return xm/((var + 1e-5).sqrt()) * g + b


class MiniGPT:
    def __init__(self, vocab_size=0, block_size=24, n_embd=48, seed=1337):
        self.block_size = block_size
        self.n_embd = n_embd
        self.vocab_size = vocab_size
        self.stoi = {}; self.itos = {}
        self._rng = np.random.RandomState(seed)
        if vocab_size:
            self._init_params()

    def _p(self, *shape, scale=None):
        scale = scale if scale else (1.0/np.sqrt(shape[-1]))
        return Tensor(self._rng.randn(*shape) * scale)

    def _init_params(self):
        V, C = self.vocab_size, self.n_embd
        T = self.block_size
        self.tok_emb = Tensor(self._rng.randn(V, C) * 0.02)
        self.pos_emb = Tensor(self._rng.randn(T, C) * 0.02)
        # attention
        self.Wq = self._p(C, C); self.Wk = self._p(C, C)
        self.Wv = self._p(C, C); self.Wo = self._p(C, C)
        self.g1 = Tensor(np.ones(C)); self.b1 = Tensor(np.zeros(C))
        # mlp
        self.W1 = self._p(C, 4*C); self.bm1 = Tensor(np.zeros(4*C))
        self.W2 = self._p(4*C, C); self.bm2 = Tensor(np.zeros(C))
        self.g2 = Tensor(np.ones(C)); self.b2 = Tensor(np.zeros(C))
        # head
        self.gf = Tensor(np.ones(C)); self.bf = Tensor(np.zeros(C))
        self.Wh = self._p(C, V); self.bh = Tensor(np.zeros(V))
        # causal mask (T,T): diagonal ustidagilar -inf
        m = np.triu(np.ones((T, T)) * -1e9, k=1)
        self._mask = m

    def params(self):
        return [self.tok_emb, self.pos_emb, self.Wq, self.Wk, self.Wv, self.Wo,
                self.g1, self.b1, self.W1, self.bm1, self.W2, self.bm2,
                self.g2, self.b2, self.gf, self.bf, self.Wh, self.bh]

    # ---------- vocab ----------
    def build_vocab(self, text):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)
        self._init_params()

    def encode(self, s): return [self.stoi[c] for c in s if c in self.stoi]
    def decode(self, ids): return "".join(self.itos[i] for i in ids)

    # ---------- forward ----------
    def forward(self, X):
        """X: (B,T) int. logits (B,T,V) qaytaradi."""
        B, T = X.shape
        C = self.n_embd
        tok = embed(self.tok_emb, X)                    # (B,T,C)
        pos = embed(self.pos_emb, np.arange(T))         # (T,C)
        x = tok + pos
        # --- attention blok ---
        xn = layernorm(x, self.g1, self.b1)
        q = xn @ self.Wq; k = xn @ self.Wk; v = xn @ self.Wv
        att = (q @ k.transpose(-1, -2)) * (1.0/np.sqrt(C))   # (B,T,T)
        att = att + self._mask[:T, :T]
        att = att.softmax(axis=-1)
        out = att @ v                                   # (B,T,C)
        out = out @ self.Wo
        x = x + out
        # --- mlp blok ---
        xn2 = layernorm(x, self.g2, self.b2)
        h = (xn2 @ self.W1 + self.bm1).gelu()
        m = h @ self.W2 + self.bm2
        x = x + m
        # --- head ---
        xf = layernorm(x, self.gf, self.bf)
        logits = xf @ self.Wh + self.bh                 # (B,T,V)
        return logits

    # ---------- train ----------
    def train(self, text, steps=3000, batch_size=16, lr=3e-3, log_every=200):
        data = np.array(self.encode(text))
        T = self.block_size
        opt = Adam(self.params(), lr=lr)
        n = len(data) - T - 1
        for step in range(1, steps+1):
            ix = self._rng.randint(0, n, size=batch_size)
            X = np.stack([data[i:i+T] for i in ix])
            Y = np.stack([data[i+1:i+T+1] for i in ix])
            logits = self.forward(X)
            loss = cross_entropy(logits.reshape(batch_size*T, self.vocab_size),
                                 Y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % log_every == 0 or step == 1:
                print(f"  qadam {step:5d}/{steps}  loss={loss.data:.4f}")
        return self

    # ---------- generate ----------
    def generate(self, prompt="", n=200, temperature=1.0, seed=None):
        rng = np.random.RandomState(seed) if seed is not None else self._rng
        ids = self.encode(prompt) or [0]
        for _ in range(n):
            ctx = ids[-self.block_size:]
            X = np.array([ctx])
            logits = self.forward(X).data[0, -1]        # oxirgi pozitsiya
            logits = logits/max(temperature, 1e-3)
            logits -= logits.max()
            p = np.exp(logits); p /= p.sum()
            idx = int(rng.choice(self.vocab_size, p=p))
            ids.append(idx)
        return self.decode(ids)

    # ---------- saqlash ----------
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {"block_size": self.block_size, "n_embd": self.n_embd,
                 "vocab_size": self.vocab_size, "stoi": self.stoi, "itos": self.itos,
                 "params": [p.data for p in self.params()], "mask": self._mask}
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            s = pickle.load(f)
        m = cls(vocab_size=0)
        m.block_size = s["block_size"]; m.n_embd = s["n_embd"]
        m.vocab_size = s["vocab_size"]; m.stoi = s["stoi"]; m.itos = s["itos"]
        m._rng = np.random.RandomState(0)
        m._init_params()
        for p, d in zip(m.params(), s["params"]):
            p.data = d
        m._mask = s["mask"]
        return m


if __name__ == "__main__":
    # overfit sinovi: kichik matnni yodlab olsinmi?
    txt = "salom dunyo, men transformer. "
    g = MiniGPT(block_size=16, n_embd=32)
    g.build_vocab(txt)
    print("Vocab:", g.vocab_size)
    g.train(txt*20, steps=400, batch_size=8, lr=5e-3, log_every=100)
    print("Namuna:", repr(g.generate("salom", n=30, seed=1)))
