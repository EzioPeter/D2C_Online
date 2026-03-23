---
name: add-d2c-algorithm
description: Add a new RL or imitation-learning algorithm to the D2C-XJTU repository end to end. Use this skill when the user asks to新增算法, add an algorithm, implement a new agent, port a paper into D2C, or automate algorithm integration while following docs/tutorials/developer.rst and the repository's existing patterns.
---

# Add D2C Algorithm

Use this skill when the task is to add a new algorithm to this repository, not just discuss it.

The authoritative guide is [docs/tutorials/developer.rst](../../docs/tutorials/developer.rst). Treat that file as the contract. This skill turns that contract into an execution workflow for Codex.

## Required repo anchors

Read these files before editing:

- [docs/tutorials/developer.rst](../../docs/tutorials/developer.rst)
- [d2c/models/base.py](../../d2c/models/base.py)
- [d2c/models/__init__.py](../../d2c/models/__init__.py)
- One close reference algorithm in the same family:
  - offline actor-critic: [d2c/models/model_free/td3_bc.py](../../d2c/models/model_free/td3_bc.py)
  - soft actor-critic or online RL: [d2c/models/model_free/sac.py](../../d2c/models/model_free/sac.py)
  - on-policy: [d2c/models/model_free/ppo.py](../../d2c/models/model_free/ppo.py)
  - imitation: [d2c/models/imitation/bc.py](../../d2c/models/imitation/bc.py) or [d2c/models/imitation/dmil.py](../../d2c/models/imitation/dmil.py)
- Matching test and demo references:
  - [test/models/model_free/test_td3_bc.py](../../test/models/model_free/test_td3_bc.py)
  - [example/benchmark/demo_td3_bc.py](../../example/benchmark/demo_td3_bc.py)
  - [example/benchmark/config/model_config.json5](../../example/benchmark/config/model_config.json5)
  - [README.md](../../README.md)

If the user does not specify the algorithm family, infer it from the paper/method. If that inference is uncertain, pause and ask a single concise question before writing code.

## Non-negotiable requirements

Every implementation must satisfy all of the following:

- Inherit from `BaseAgent`.
- Place the algorithm file under the correct subfolder of `d2c/models`.
- Name the agent class `XxxAgent`.
- Implement or override the methods required by the chosen pattern, including `_get_modules`, `_build_fns`, `_build_optimizers`, `_optimize_step`, `_build_test_policies`, `save`, and `restore`.
- Create an `AgentModule` subclass inheriting `BaseAgentModule`.
- Register the algorithm in [d2c/models/__init__.py](../../d2c/models/__init__.py) and `AGENT_MODULES_DICT`.
- Add the algorithm config block to [example/benchmark/config/model_config.json5](../../example/benchmark/config/model_config.json5).
- Add or update a benchmark demo under `example/benchmark/demo_<algo>.py`.
- Add unit tests under the appropriate `test/models/...` folder.
- If new network classes, policy classes, replay buffer utilities, or helpers are introduced, add tests in their corresponding test folders.
- Add the algorithm name to [README.md](../../README.md) under supported algorithms.
- Follow PEP8, add docstrings, and add type annotations for public functions and methods.

Do not stop after creating only the core algorithm file. A partial integration is considered incomplete.

## Input contract

Before implementation, extract or infer these items from the user request:

- algorithm name in snake_case, for example `cql`
- public class name, for example `CQLAgent`
- algorithm family and target folder, for example `model_free`
- training mode: offline, online, on-policy, model-based, or imitation
- required networks and optimizers
- whether the algorithm needs new utilities outside the agent file
- minimal benchmark demo target

If the user only provides a paper or method name, derive a reasonable snake_case module name and say what you chose in the final summary.

## Execution workflow

1. Inspect the closest existing implementation and copy its structure, not just its API names.
2. Choose the target file path under `d2c/models/<family>/<algo>.py`.
3. Implement the agent class with repository-consistent naming, docstrings, and type annotations.
4. Implement `AgentModule` and wire every trainable module through `_build_fns`.
5. Build optimizers from `self._optimizers` using `d2c.utils.utils.get_optimizer`.
6. Implement loss builders and `_optimize_*` helpers as needed.
7. Implement `_optimize_step` to match the algorithm's update schedule and target-network updates.
8. Implement `_build_test_policies`, `save`, and `restore`.
9. Register the agent in [d2c/models/__init__.py](../../d2c/models/__init__.py).
10. Add the config block in [example/benchmark/config/model_config.json5](../../example/benchmark/config/model_config.json5).
11. Add the benchmark demo in `example/benchmark/demo_<algo>.py`, following the nearest existing demo for the same training regime.
12. Add unit tests, using the nearest existing algorithm test as a template.
13. If auxiliary modules were added, add their tests too.
14. Update [README.md](../../README.md).
15. Run the smallest relevant verification you can afford locally and report what was or was not run.

## Family-specific guidance

### Offline actor-critic

Use [d2c/models/model_free/td3_bc.py](../../d2c/models/model_free/td3_bc.py) as the default skeleton.

Expected pieces usually include:

- critic networks and target critics
- actor network and target actor if needed
- batch sampling from offline replay via `_get_train_batch`
- deterministic or stochastic evaluation policy in `_build_test_policies`

### Online off-policy

Use [d2c/models/model_free/sac.py](../../d2c/models/model_free/sac.py) plus its trainer/demo style as the skeleton.

Pay attention to:

- environment interaction inside `_get_train_batch`
- replay buffer ownership
- learning starts, rollout schedule, and target update cadence

### On-policy

Use `ppo.py` and the corresponding trainer flow as the skeleton.

Pay attention to:

- trajectory collection
- policy/value update schedule
- any specialized buffer or transition structure

### Imitation / discriminator-based methods

Use `bc.py` or `dmil.py` as the nearest reference.

Pay attention to:

- whether testing policy is deterministic
- whether additional modules such as discriminators or encoders need separate tests

## File checklist

Unless clearly not applicable, expect to touch these locations:

- `d2c/models/<family>/<algo>.py`
- `d2c/models/__init__.py`
- `example/benchmark/config/model_config.json5`
- `example/benchmark/demo_<algo>.py`
- `test/models/<family>/test_<algo>.py`
- `README.md`

Sometimes also:

- `d2c/utils/networks.py`
- `d2c/utils/policies.py`
- `d2c/networks_and_utils_for_agent/*.py`
- `example/benchmark/run_<algo>.sh`
- `example/benchmark/results/<algo>/...`
- other test files for any newly introduced shared module

## Quality bar

When implementing, prefer repository consistency over paper-perfect abstraction. Match the local coding style and existing agent patterns.

Before finishing, check for these common misses:

- forgot to import/register the new agent in `d2c/models/__init__.py`
- class name does not end with `Agent`
- config key name does not match `model.model_name`
- missing `save` and `restore`
- missing test policy in `_test_policies['main']`
- demo script still points to the wrong algorithm name
- README not updated
- docstrings or type annotations missing
- new helper modules added without tests

## Validation

Run focused verification after editing. Prefer the smallest meaningful command first.

Typical checks:

- run the new unit test file only
- run a short smoke test of the new demo with reduced train steps if feasible
- if you only changed registration/config wiring, add a lightweight import or construction check

If full benchmark training is too expensive, say so explicitly and leave a concrete note about what remains unverified.

## Final response contract

When using this skill, the final response should include:

- what algorithm was added and which family/folder was chosen
- which files were created or updated
- what verification was run
- any remaining risks, especially around training stability or unrun benchmark evaluation

