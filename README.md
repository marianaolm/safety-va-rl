# Safety-VA-RL

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

This project explores how **symbolic reasoning layers** can:

- Improve safety in reinforcement learning
- Enable interpretable autonomy switching
- Support meaningful negotiation between human and autonomous control

Demonstrating improvements in safety and a functioning negotiation layer is essential
for **interpretability**, which can increase **understanding and trust** in
variable-autonomy systems.

---

## Repository structure

```text
safety-va-rl/
├── src/        # trainers, evaluation, sweeps, wrappers
├── scripts/    # executable entry points (train / evaluate / sweep)
├── cluster/    # example SLURM scripts used on an HPC cluster
├── sweeps/    
├── pyproject.toml
├── uv.lock
└── README.md
```

## Environment setup (uv)
This project uses [uv](https://github.com/astral-sh/uv) for dependency management and reproducibility

### 1. Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone the repository
```bash
git clone git@github.com:marianaolm/safety-va-rl.git
cd safety-va-rl
```

### 3. Create the environment and install dependencies
```bash
uv sync
```
This will:
* Create a local virtual environment (`./venv`)
* Install all dependencies pinned in `uv.lock`

---
## Contributors
<table>
  <tr>
    <td align="center"><a href="https://github.com/mariddc"><img src="https://avatars.githubusercontent.com/u/88353514?v=4" width="100px;" alt=""/><br /><sub><b>Mariana Dutra</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/marianaolm"><img src="https://avatars.githubusercontent.com/u/90351165?v=4" width="100px;" alt=""/><br /><sub><b>Mariana Olm</b></sub></a><br /></td>
  </tr>
</table>
