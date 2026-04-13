# External Planning Tools

## Fast Downward

Classical planner for solvability checking.

- **Repo**: https://github.com/aibasel/downward
- **Docs**: https://www.fast-downward.org/

### Setup (WSL/Linux)

```bash
# From WSL:
bash /mnt/c/Expo/Research/PDDL/tools/setup_planning_tools.sh
```

### Manual Build

```bash
git clone https://github.com/aibasel/downward.git /opt/downward
cd /opt/downward
python3 build.py
```

## VAL (Plan Validator)

Validates that a plan achieves the goal from the initial state.

- **Repo**: https://github.com/KCL-Planning/VAL

### Manual Build

```bash
git clone https://github.com/KCL-Planning/VAL.git /opt/VAL
cd /opt/VAL
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Environment Variables

Set in `.env` or shell profile:

```bash
# If running natively (Linux/WSL Python)
export FAST_DOWNWARD_PATH=/opt/downward/fast-downward.py
export VAL_PATH=/opt/VAL/build/bin/Validate

# If calling from Windows Python into WSL
export WSL_FAST_DOWNWARD_PATH=/opt/downward/fast-downward.py
export WSL_VAL_PATH=/opt/VAL/build/bin/Validate
```

## Notes

- Fast Downward and VAL are optional for initial development. The equivalence
  evaluation (Planetarium) works without them.
- Solvability checks add confidence but are not required for semantic
  equivalence evaluation.
- These tools are most useful in the full VCSR pipeline for planner-based
  filtering of candidates.
