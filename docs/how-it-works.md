# How it works

This page explains the ideas behind quantum-adapter without assuming any quantum background.

## The problem: change a big model cheaply

A language model is mostly big grids of numbers (weight matrices). Fine-tuning changes them. Adapter methods keep the original grid `W` frozen and learn a small correction `dW`, so the layer computes `x @ W + x @ dW`. LoRA stores `dW` as two thin matrices. Here, a circuit produces `dW`.

```
x --> [ W (frozen) ] --+
                       +--> output
x --> [ dW (learned) ] +
```

## Amplitudes: a tower of 2^n numbers

A circuit on `n` qubits describes a list of `2^n` numbers called amplitudes. Ten qubits give 1,024 numbers; fourteen give 16,384. The list is the circuit's output, and a handful of angles decides what it contains.

```
n = 3 qubits  ->  8 amplitudes

 0.5 |#####
 0.1 |#
-0.3 |###        (negative values point down)
 ...
```

This is why a circuit is compact: few angles, many output numbers.

## Gates: RY and CNOT / CZ

- **RY(angle)** rotates one qubit. It is the trainable dial: turning it shifts weight between the qubit's two states.
- **CNOT / CZ** link two qubits, so that what happens to one depends on the other. Without them each qubit would be independent and the output numbers could not be correlated.

```
q0 --RY(a)--*--------      RY layer, then a ring of CNOTs,
q1 --RY(b)--X--*-----      repeated `depth` times
q2 --RY(c)-----X-----
```

The `amplitude` circuit is close to Qiskit's `RealAmplitudes`, with one difference: it has no closing RY layer and keeps the last CNOT layer instead. That last layer is a fixed reordering of the amplitudes: it changes which amplitude lands in which weight, not what the circuit can express. (`unitary` does end with an RY layer.)

All gates here are real-valued, so the amplitudes are real numbers that can be used directly as weights. A gate sequence is also exact to simulate on a normal computer, with gradients computed by ordinary backpropagation.

## From dials to dW, per method

```
amplitude:  angles -> amplitudes -> reshape into A (in x r) and B (r x out) -> dW = A B * scale
unitary:    angles -> orthogonal matrices U and V -> dW = U diag(lambda) V^T
tensor:     split hidden size into axes (e.g. 768 = 6 x 4 x 4 x 8) -> small tensors on pairs of axes
```

- `amplitude` reads LoRA-shaped factors straight from the state. A single trainable scale multiplies `B`.
- `unitary` (Quantum-PEFT) builds orthogonal matrices from circuits and trains only a short diagonal between them. Sizes that are not powers of two use the next power of two, take the needed rows, and re-orthonormalise with a QR step.
- `tensor` (QuanTA) treats the hidden vector as a grid and applies trainable tensors to pairs of its axes, like two-qubit gates. The number of axes is `tensor_axes` (default 4), and each size is split into factors as close to equal as possible: 768 becomes 6 x 4 x 4 x 8 and 2304 becomes 6 x 8 x 6 x 8. It is quantum-inspired: the tensors are not unitary, so there is no circuit to run on hardware.

## Why the neutral start matters

If a new adapter changed the model's output at step 0, training would begin from a damaged model. Every method here starts with `dW = 0` (for `amplitude` the scale starts at 0; for `unitary` the diagonal does; for `tensor` the trainable tensors start equal to a frozen copy, which is subtracted). The adapted model gives exactly the original outputs at step 0, and tests check this.

```
step 0:  adapted model == base model
step N:  adapted model == base model + learned correction
```

## Why merging makes inference free

`x @ W + x @ dW` equals `x @ (W + dW)`. After training, `merge_and_unload()` adds `dW` into `W` once and removes the adapter. The result is a plain transformers model, with the same speed and size as the original.

```
before:  x -> W and dW -> sum        after merge:  x -> (W + dW)
```

## What the classical control tests

Fewer trainable numbers can mean a worse result for a boring reason: less capacity. To test whether the circuit itself adds anything, the benchmark trains a classical adapter with the same number of trainable parameters (a fixed random projection for `amplitude` and `unitary`; a LoRA of nearest size for `tensor`). In the preliminary SST-2 run the control matched the quantum adapters, so whether the circuit beats an equal-size classical generator is still open; the full benchmark tests it and reports both.

```
same size, circuit-made dW   vs   same size, classical dW
         accuracy?                       accuracy?
```

## Why quantum adapters need longer training

A possible reason, which we have not tested: the weights of ΔW come from a few circuit angles, so each step moves ΔW through a fixed, shared set of angles rather than through independent entries as in LoRA. What we observed is that learning was slower: train about 2x longer than LoRA (6 epochs where LoRA uses 3) with a learning rate near 1e-2. Depth 20 was worse in one test (3 epochs), and adapting more layers did not help. These are observations from one task, not a proof of the cause; see [experiments.md](experiments.md).
