# Hierarchical Demand Forecasting

## Overview
Demand forecasting project that compares three hierarchical reconciliation approaches (Bottom-Up, Top-Down, Middle-Out) across a 4-level product hierarchy: SKU -> Category -> Plant -> Total. Demonstrates how to maintain forecast coherence (parts sum to whole) across organizational levels.

**Built by:** Nithin Kumar Kokkisa — Senior Demand Planner with 12+ years at HPCL managing 180,000 MTPA facility.

---

## Business Problem
Companies forecast at multiple levels simultaneously: SKUs for production, categories for procurement, plants for logistics, and totals for finance. When each level is forecast independently, they DON'T add up — creating conflicts between planning teams. This project solves the coherence problem using hierarchical reconciliation.

## Hierarchy Structure
```
Total Company
  |-- Plant North
  |     |-- Category A: SKU_A1, SKU_A2, SKU_A3
  |     |-- Category B: SKU_B1, SKU_B2
  |-- Plant South
        |-- Category A: SKU_A4, SKU_A5
        |-- Category B: SKU_B3, SKU_B4, SKU_B5
```
**17 total series** across 4 levels (1 Total + 2 Plants + 4 Categories + 10 SKUs)

## Three Approaches Compared

| Approach | Method | Best At | Weakness |
|----------|--------|---------|----------|
| **Bottom-Up** | Forecast SKUs, sum up | SKU-level accuracy | Noise accumulates at top |
| **Top-Down** | Forecast total, allocate down | Total-level accuracy | Loses granular patterns |
| **Middle-Out** | Forecast category, reconcile both ways | Balanced across levels | Requires choosing middle |

## Key Concepts
- Hierarchical coherence (parts must sum to whole)
- Proportion-based allocation (historical shares for top-down)
- Optimal reconciliation (MinT — statistical combination)
- MAPE comparison across approaches AND hierarchy levels

## Tools & Technologies
- **Python** (Pandas, NumPy, Matplotlib, Seaborn)

---

## About
Part of a **30-project data analytics portfolio**. See [GitHub profile](https://github.com/Kokkisa) for the full portfolio.
