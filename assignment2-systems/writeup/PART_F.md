# Part F — Analyzing Parallelism Strategies (written calculations)

First principles used throughout: FP16 ⇒ 2 bytes/element; matmul \((A,B)\times(B,C)\) costs \(2ABC\) FLOPs;
ring all-reduce of \(S\) bytes on \(N\) devices takes \(2\frac{N-1}{N}\frac{S}{W}\) seconds; ring all-gather /
reduce-scatter each take \(\frac{N-1}{N}\frac{S}{W}\). Computation time is FLOPs\(/C\). We are
**communication-bottlenecked** when \(T_\mathrm{comm} > T_\mathrm{comp}\) (no free overlap left).

FFN under study (handout §8.2):
\[
\mathbf{x}_1=\mathbf{x}W_1,\;
\mathbf{x}_2=\mathbf{x}W_2,\;
\mathbf{z}=f(\mathbf{x}_1)*\mathbf{x}_2,\;
\mathbf{y}=\mathbf{z}W_3
\]
with \(\mathbf{x}\in\mathbb{R}^{B\times D}\), \(W_1,W_2\in\mathbb{R}^{D\times D_\mathrm{FF}}\), \(W_3\in\mathbb{R}^{D_\mathrm{FF}\times D}\).

---

## F.1 `alternate_ring_all_reduce` (1 pt)

**Answer.** \(\displaystyle T = (N-1)\,\frac{S}{W}\).

**Justification.** The algorithm runs \(N-1\) ring steps; each step each device **sends a full size-\(S\)
tensor** to its neighbor (egress bandwidth \(W\)). Unlike reduce-scatter+all-gather (which ships
chunks of size \(S/N\)), this circulating-full-tensor schedule never shrinks the payload, so latency
is linear in \(N-1\), not \(\frac{N-1}{N}\).

(Compare to the handout’s standard ring all-reduce \(2\frac{N-1}{N}\frac{S}{W}\), which is ~\(N/2\times\) faster for large \(N\).)

---

## F.2 `data_parallel_calcs` (3 pts)

Local batch \(B'=B/N_\mathrm{DP}\). Ignore non-matmul ops. FP16.

### (a) Backward FLOPs

**Answer.** \(\displaystyle 12\,\frac{B}{N_\mathrm{DP}}\,D\,D_\mathrm{FF}\).

**Justification.** Full-batch FFN backward has three activation matmuls and three weight-gradient
matmuls, each \(2BDD_\mathrm{FF}\), totaling \(12BDD_\mathrm{FF}\); under DP each rank sees batch \(B/N_\mathrm{DP}\).

(Explicitly: \(d\mathbf{z}=\mathbf{dy}W_3^\top\), \(d\mathbf{x}=d\mathbf{x}_1 W_1^\top+d\mathbf{x}_2 W_2^\top\),
\(dW_3=\mathbf{z}^\top\mathbf{dy}\), \(dW_2=\mathbf{x}^\top d\mathbf{x}_2\), \(dW_1=\mathbf{x}^\top d\mathbf{x}_1\).)

### (b) Backward communication time

**Answer.** \(\displaystyle T_\mathrm{comm}=12\,\frac{N_\mathrm{DP}-1}{N_\mathrm{DP}}\,\frac{D\,D_\mathrm{FF}}{W}\).

**Justification.** All-reduce the three weight gradients (\(3DD_\mathrm{FF}\) FP16 elements ⇒ \(6DD_\mathrm{FF}\) bytes)
with ring cost \(2\frac{N-1}{N}\frac{S}{W}\).

### (c) Max \(N_\mathrm{DP}\) before comm-bound

Require \(T_\mathrm{comm}<T_\mathrm{comp}\):
\[
12\frac{N-1}{N}\frac{DD_\mathrm{FF}}{W}
\;<\;
12\frac{B}{N}\frac{DD_\mathrm{FF}}{C}
\quad\Rightarrow\quad
N-1 < \frac{BW}{C}.
\]

**Answer.** \(\displaystyle N_\mathrm{DP} < 1 + \frac{BW}{C}\).
(For large \(N\), ≈ \(N_\mathrm{DP} < BW/C\).)

---

## F.3 `fsdp_calcs` (3 pts)

Same FP16 setting; replace \(N_\mathrm{DP}\) by \(N_\mathrm{FSDP}\).

### (a) FLOPs

- **Backward:** \(\displaystyle 12\frac{B}{N_\mathrm{FSDP}}DD_\mathrm{FF}\) — same local matmuls as DP.
- **Forward:** \(\displaystyle 6\frac{B}{N_\mathrm{FSDP}}DD_\mathrm{FF}\) — three matmuls × \(2B'DD_\mathrm{FF}\).

### (b) Communication time

Each weight is \(2DD_\mathrm{FF}\) bytes. AG or RS of one weight: \(\frac{N-1}{N}\frac{2DD_\mathrm{FF}}{W}\).

- **Forward:** 3× all-gather weights
  \(\displaystyle T_\mathrm{fwd}=6\frac{N_\mathrm{FSDP}-1}{N_\mathrm{FSDP}}\frac{DD_\mathrm{FF}}{W}\).
- **Backward:** 3× AG weights + 3× RS grads
  \(\displaystyle T_\mathrm{bwd}=12\frac{N_\mathrm{FSDP}-1}{N_\mathrm{FSDP}}\frac{DD_\mathrm{FF}}{W}\).

### (c) Scaling limits

Same algebra as DP for both passes (fwd and bwd factors cancel equally):

**Answer.** \(\displaystyle N_\mathrm{FSDP} < 1 + \frac{BW}{C}\) (backward and forward).

---

## F.4 `tp_calcs` (4 pts)

Megatron-style: \(W_1,W_2\) **column-parallel** \((D, D_\mathrm{FF}/N_\mathrm{TP})\); \(W_3\) **row-parallel**
\((D_\mathrm{FF}/N_\mathrm{TP}, D)\). Forward ends with `all-reduce` on \(\mathbf{y}\).

### (a) Backward equations (per rank \(i\))

Upstream \(\mathbf{dy}\) is the **full** \((B,D)\) gradient (backward of sum-all-reduce copies \(\mathbf{dy}\) to every rank).

\[
\begin{aligned}
dW_3^{(i)} &= {\mathbf{z}^{(i)}}^\top \mathbf{dy},\\
d\mathbf{z}^{(i)} &= \mathbf{dy}\,{W_3^{(i)}}^\top,\\
d\mathbf{x}_2^{(i)} &= d\mathbf{z}^{(i)} * f(\mathbf{x}_1^{(i)}),\\
d\mathbf{x}_1^{(i)} &= d\mathbf{z}^{(i)} * f'(\mathbf{x}_1^{(i)}) * \mathbf{x}_2^{(i)},\\
dW_1^{(i)} &= \mathbf{x}^\top d\mathbf{x}_1^{(i)},\\
dW_2^{(i)} &= \mathbf{x}^\top d\mathbf{x}_2^{(i)},\\
d\mathbf{x}_\mathrm{partial}^{(i)} &= d\mathbf{x}_1^{(i)}{W_1^{(i)}}^\top + d\mathbf{x}_2^{(i)}{W_2^{(i)}}^\top,\\
d\mathbf{x} &= \mathrm{all\text{-}reduce}\big(\{d\mathbf{x}_\mathrm{partial}^{(i)}\}\big).
\end{aligned}
\]

### (b) FLOPs

Each rank’s matmuls use the \(1/N_\mathrm{TP}\) shard of \(D_\mathrm{FF}\):

- **Forward:** \(\displaystyle 6\frac{BDD_\mathrm{FF}}{N_\mathrm{TP}}\)
- **Backward:** \(\displaystyle 12\frac{BDD_\mathrm{FF}}{N_\mathrm{TP}}\)

### (c) Communication time

One AR of activations size \(BD\) FP16 = \(2BD\) bytes each way (fwd \(\mathbf{y}\), bwd \(d\mathbf{x}\)):

\[
T = 2\frac{N_\mathrm{TP}-1}{N_\mathrm{TP}}\frac{2BD}{W}
= 4\frac{N_\mathrm{TP}-1}{N_\mathrm{TP}}\frac{BD}{W}
\]
for **both** forward and backward.

### (d) Scaling limits

Backward \(T_\mathrm{comm}<T_\mathrm{comp}\):
\[
4\frac{N-1}{N}\frac{BD}{W} < 12\frac{BDD_\mathrm{FF}}{NC}
\;\Rightarrow\;
N_\mathrm{TP} < 1 + \frac{3D_\mathrm{FF}W}{C}.
\]

Forward:
\[
4\frac{N-1}{N}\frac{BD}{W} < 6\frac{BDD_\mathrm{FF}}{NC}
\;\Rightarrow\;
N_\mathrm{TP} < 1 + \frac{3D_\mathrm{FF}W}{2C}.
\]

---

## F.5 `fsdp_tp_calcs` (6 pts) — 2D parallelism

Grid: \(N=N_\mathrm{TP}N_\mathrm{FSDP}\). Local batch \(B/N_\mathrm{FSDP}\); weights further sharded by TP.

### (a) Forward FLOPs

**Answer.** \(\displaystyle 6\frac{BDD_\mathrm{FF}}{N_\mathrm{FSDP}\,N_\mathrm{TP}} = 6\frac{BDD_\mathrm{FF}}{N}\).

### (b) Forward communication (axes overlapped)

- FSDP axis: 3× AG of TP-sharded weights (each full-TP-shard is \(2DD_\mathrm{FF}/N_\mathrm{TP}\) bytes):
  \(\displaystyle T_\mathrm{FSDP}=6\frac{N_\mathrm{FSDP}-1}{N_\mathrm{FSDP}}\frac{DD_\mathrm{FF}}{N_\mathrm{TP}\,W}\).
- TP axis: AR of batch-sharded activations \((B/N_\mathrm{FSDP}, D)\):
  \(\displaystyle T_\mathrm{TP}=4\frac{N_\mathrm{TP}-1}{N_\mathrm{TP}}\frac{BD}{N_\mathrm{FSDP}\,W}\).

**Answer.** \(\displaystyle T_\mathrm{comm}=\max(T_\mathrm{FSDP},\,T_\mathrm{TP})\) (axes overlapped).

### (c) Max \(N\) with axis overlap (optimal split)

Approx \(\frac{N-1}{N}\approx 1\). Balance \(T_\mathrm{FSDP}=T_\mathrm{TP}\):
\[
\frac{6DD_\mathrm{FF}}{N_\mathrm{TP}W}=\frac{4BD}{N_\mathrm{FSDP}W}
\;\Rightarrow\;
\frac{N_\mathrm{FSDP}}{N_\mathrm{TP}}=\frac{2B}{3D_\mathrm{FF}}.
\]
Compute-bound also needs \(N_\mathrm{FSDP}\le BW/C\) and \(N_\mathrm{TP}\le \tfrac{3}{2}D_\mathrm{FF}W/C\), which bind
**simultaneously** at that ratio. Therefore

**Answer.** \(\displaystyle N = N_\mathrm{TP}N_\mathrm{FSDP} \;<\; \frac{3}{2}\,B\,D_\mathrm{FF}\,\Big(\frac{W}{C}\Big)^2\).

### (d) Same but axes **cannot** overlap (\(T_\mathrm{comm}=T_\mathrm{FSDP}+T_\mathrm{TP}\))

Minimize \(6DD_\mathrm{FF}/(N_\mathrm{TP}W)+4BD/(N_\mathrm{FSDP}W)\) subject to \(N_\mathrm{FSDP}N_\mathrm{TP}=N\).
Optimum at \(N_\mathrm{TP}=\sqrt{3D_\mathrm{FF}N/(2B)}\), giving
\[
T_\mathrm{min}=\frac{4\sqrt{6}\,D\sqrt{B D_\mathrm{FF}}}{W\sqrt{N}}.
\]
Set \(T_\mathrm{min}<T_\mathrm{comp}=6BDD_\mathrm{FF}/(NC)\):

**Answer.** \(\displaystyle N \;<\; \frac{3}{8}\,B\,D_\mathrm{FF}\,\Big(\frac{W}{C}\Big)^2\).

(Without overlap you get only \(\frac{3}{8}/\frac{3}{2}=\frac{1}{4}\) the device count of the overlapped case — overlapping the two mesh axes is worth a **4×** scale-out factor here.)

---

## Quick reference

| Strategy | Comm-bound when roughly |
|----------|-------------------------|
| DP / FSDP | \(N \gtrsim BW/C\) |
| TP (bwd) | \(N_\mathrm{TP} \gtrsim 3 D_\mathrm{FF} W/C\) |
| FSDP+TP, overlap | \(N \gtrsim \tfrac{3}{2} B D_\mathrm{FF}(W/C)^2\) |
| FSDP+TP, no overlap | \(N \gtrsim \tfrac{3}{8} B D_\mathrm{FF}(W/C)^2\) |
