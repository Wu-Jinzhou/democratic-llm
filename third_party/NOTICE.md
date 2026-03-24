# Third-Party Code

This directory contains `stratification.py`, derived from the Sortition Foundation implementation
used in:

Bailey Flanigan, Paul Gölz, Anupam Gupta, Brett Hennig, Ariel D. Procaccia.
Fair Algorithms for Selecting Citizens' Assemblies. (2021).

License: GNU GPLv3. See `third_party/LICENSE-GPLv3`.

This directory also contains a minimal vendored subset of OvertonBench code under
`third_party/overtonbench/`, adapted from:

Elinor Poole-Dayan, Jiayi Wu, Taylor Sorensen, Jiaxin Pei, Michiel A. Bakker.
Benchmarking Overton Pluralism in LLMs. arXiv:2512.01351.

Vendored components are limited to prompt templates, few-shot prompt assembly, Gemini
rating helpers, and the unadjusted/weighted OvertonScore computations used by this repo's
evaluation pipeline.
