# Amazon Robotics AI Speed Hackathon 🤖

**4th Place · Team JOM · 2026**

Built a deterministic fleet-routing system for an Amazon Robotics warehouse simulation under a strict hackathon time limit. The planner coordinates multiple drive units, assigns pods, avoids congestion, and adapts routes around aisle and station capacity constraints.

## What I demonstrated 🚀

- Algorithm design and weighted graph search
- Multi-agent coordination and task assignment
- Congestion-aware routing and collision avoidance
- Simulation-driven testing and performance optimization
- Fast, collaborative problem-solving under pressure

## Technical approach 🛠️

The solution uses Python's standard library to run a rolling-horizon fleet planner at every simulation step. It combines global task planning, capacity-aware pathfinding, delivery sequencing, and deterministic tie-breaking to move pods efficiently without overloading shared routes.

All six provided scenarios deliver **100% of their pods**.

## Run locally

```bash
python3 -m pip install -e .
python3 scripts/run_game.py test_cases/level3/test_case_6.json
```

📌 [View the event recap and results on LinkedIn](https://www.linkedin.com/posts/ubc-amazonrobotics-ubcsciencecoop-ugcPost-7506479644846575616-Omyp/)
