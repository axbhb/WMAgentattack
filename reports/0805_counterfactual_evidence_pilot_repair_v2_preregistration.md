# Counterfactual evidence execution pilot: nonsemantic repair v2

Date: 2026-08-05

Status: preregistered after collector crash and before any repair execution.

## Frozen failure evidence

Remote Slurm job `6344` retained 23 of 24 canonical outcomes and failed one row with `KeyError: get_current_day`. An independent full-manifest coverage audit found 24 required ledger tools: all 12 observed-prefix replay tools were covered, while `get_current_day` was the only uncovered candidate tool. The failed archive at `fixed_v1` remains immutable and is recorded as a collector crash, not a scientific result.

## Sole permitted repair

Add one outcome-label-blind `VALUE` adapter that records the sandbox-returned current day as a time-scoped `calendar_context` fact. Add a pre-execution coverage assertion over every selected candidate and every observed prefix-replay tool. No candidate, argument binding, task, state, suite/difficulty cell, seed, manifest row, outcome threshold, training gate, or execution budget changes.

The corrected run must use the existing manifest SHA256 `a7e99a9c821757d200c6e943f0a22c764a22564a956897b68e384c7a25229569`, exactly 24 queries, two fresh replicas, 48 counterfactual calls, 36 observed prefix-replay calls, and 84 total AgentDojo sandbox tool calls. The new immutable archive is `fixed_v2`.

This consumes one of the three preregistered nonsemantic repair attempts. If the collector passes but the already frozen readiness thresholds fail, the data are retained and every model, attack, and Dreamer run remains blocked.
