# Neurosymbolic Variable Autonomy — Safety-VA-RL

This repository contains experimental code for studying **neurosymbolic variable autonomy**
and **safety-aware reinforcement learning** in continuous-control tasks using
[Safety-Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium).

This project is developed as part of the **PRIM (Research and Innovation Master Project)**  
within the **MSc in Engineering program at Télécom Paris**.

The work is primarily inspired by the paper:

> **Negotiating Control: Neurosymbolic Variable Autonomy**  
> Georgios Bakirtzis, Manolis Chiou, Andreas Theodorou  
> https://arxiv.org/abs/2407.16254

---

## Research goal

**In what ways can symbolic logic contribute to improving learning efficiency and operational
safety in a transparent variable-autonomy negotiation framework?**

This project explores how symbolic reasoning layers can:

- Improve safety in reinforcement learning  
- Enable interpretable autonomy switching  
- Support meaningful negotiation between human and autonomous control  

Demonstrating improvements in safety and a functioning negotiation layer is essential
for interpretability, which in turn can increase understanding and trust in
variable-autonomy systems.

---

## Repository structure

```text
safety-va-rl/
├── src/        # Core library: trainers, evaluation, sweeps, wrappers
├── scripts/    # User-facing entry points (train / evaluate / sweep)
├── cluster/    # Optional SLURM job scripts (examples)
├── sweeps/     # Sweep databases and summaries (results, not required)
├── pyproject.toml
├── uv.lock
└── README.md
