# IAI_HW_02

## Project Description

This repository contains Homework 02 for an Introduction to Artificial Intelligence course.  
The project models a competitive **two-robot warehouse domain** and focuses on implementing and evaluating decision-making agents under time constraints.

Each robot acts in turns on a 5x5 grid, trying to maximize its score (credit) by collecting packages and delivering them to their destinations, while managing limited battery and optional charging actions.

## Main Goal

The main objective is to design and compare AI agents with different search strategies:

- Baseline random agent
- Baseline greedy agent
- Improved greedy agent with a richer heuristic
- Minimax agent with iterative deepening
- Alpha-Beta agent with pruning
- Expectimax agent with weighted stochastic opponent modeling

## Repository Structure

- `/home/runner/work/IAI_HW_02/IAI_HW_02/main.py`  
  Command-line runner for single matches and tournament mode.
- `/home/runner/work/IAI_HW_02/IAI_HW_02/WarehouseEnv.py`  
  Environment implementation: game state, legal actions, transitions, rewards, and rendering.
- `/home/runner/work/IAI_HW_02/IAI_HW_02/Agent.py`  
  Base agent API and baseline agents (`AgentRandom`, `AgentGreedy`).
- `/home/runner/work/IAI_HW_02/IAI_HW_02/submission.py`  
  Homework implementations: improved heuristic and search-based agents.
- `/home/runner/work/IAI_HW_02/IAI_HW_02/icons/`  
  Assets for optional graphical (pygame) visualization.
- `/home/runner/work/IAI_HW_02/IAI_HW_02/HW2_IntroToAI.docx`  
  Assignment document.

## Game Mechanics (High Level)

- Two robots compete over multiple turns.
- Robots can move, pick up packages, drop off packages, charge, or park.
- Battery decreases on movement and can be replenished using accumulated credit at charging stations.
- Delivering a package grants credit based on Manhattan distance (scaled by environment reward settings).
- The game ends when no robot can continue or when the step budget is exhausted.

## Running the Project

Run from `/home/runner/work/IAI_HW_02/IAI_HW_02`:

```bash
python main.py <agent0> <agent1> [options]
```

Supported agent names:

- `random`
- `greedy`
- `greedyImproved`
- `minimax`
- `alphabeta`
- `expectimax`
- `hardcoded`

Useful options:

- `-t, --time_limit` – per-turn time limit in seconds
- `-s, --seed` – random seed for environment generation
- `-c, --count_steps` – number of turns per robot
- `--console_print` – print board states in terminal
- `--screen_print` – show pygame visualization
- `--tournament` – run 100 games and print aggregate results

## Learning Focus

This homework emphasizes:

- Heuristic design in adversarial domains
- Deterministic and stochastic game-tree search
- Trade-offs between search depth, quality, and runtime constraints
- Empirical comparison between agent strategies
