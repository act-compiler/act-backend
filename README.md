# ACT Generator: Compiler Backend

This repository contains the source code and documentation for `act-backend`, a compiler backend generator in the [ACT Ecosystem](https://github.com/act-compiler/act) that automatically generates compiler backends from ISA specifications written in [TAIDL](https://github.com/act-compiler/taidl).

For more details about the ACT Ecosystem, refer to the top-level repository: [act-compiler/act](https://github.com/act-compiler/act).

## TAIDL: Tensor Accelerator ISA Definition Language

TAIDL is a domain-specific language designed to define instruction set architectures (ISAs) for tensor accelerators. It is published at [MICRO 2025](https://doi.org/10.1145/3725843.3756075).
TAIDL not only standardizes the way tensor accelerator ISAs are specified but also enables automated generation of tools such as test oracles (functional simulators) and compiler backends, significantly reducing the effort required to develop and maintain these components.

## Compiler Backend Generation

`act-backend` is one of the tool generators in the [ACT Ecosystem](https://github.com/act-compiler/act) that consumes TAIDL specifications and emits out a compiler backend instantaneously.
The generated compiler backend translates high-level tensor kernels in [XLA-HLO IR](https://openxla.org/xla) into low-level machine code for the target tensor accelerator defined in TAIDL.
Therefore, it can be directly integrated into existing ML compilers like [XLA](https://www.tensorflow.org/xla), enabling seamless support for tensor accelerators in popular machine learning frameworks like JAX and PyTorch with minimal engineering effort.

The compiler backend has provable guarantees of soundness (i.e., it will never generate invalid machine code) and completeness (i.e., it can generate machine code for all valid programs).
Furthermore, the generated compiler backend supports integration with cost models to enable performance-aware code generation.
The generated compiler backend is agnostic to the choice of the cost model and can be easily adapted to work with different cost models like cycle-accurate simulators, analytical models, or ML-based models.

Details on automatically generating compiler backends from TAIDL definitions is present in our [arXiv release](https://arxiv.org/abs/2510.09932).
