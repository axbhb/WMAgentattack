"""Frozen task-level paired evaluation of clean protocol recovery."""
from collections import Counter
from .clean_pairing import paired_episode_seed


def make_plan(tasks, seeds, arms):
    if len(tasks) != 20 or len(set(tasks)) != 20 or len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("Expected twenty distinct tasks and three distinct seeds")
    if len(set(arms)) != len(arms) or "strict" not in arms:
        raise ValueError("Missing or duplicate control arm")
    rows = []
    for si, seed in enumerate(seeds):
        for ti, task in enumerate(tasks):
            shift = (si + ti) % len(arms)
            for arm in arms[shift:] + arms[:shift]:
                rows.append({"arm": arm, "task": task, "run_seed": seed,
                             "episode_seed": paired_episode_seed(seed, task, tasks)})
    return rows


def task_signflip_p(count_differences):
    """Exact one-sided sign randomization, one independent unit per task."""
    observed = sum(count_differences)
    distribution = Counter({0: 1})
    for difference in count_differences:
        if not difference:
            continue
        updated = Counter()
        for value, count in distribution.items():
            updated[value + abs(difference)] += count
            updated[value - abs(difference)] += count
        distribution = updated
    return sum(count for value, count in distribution.items() if value >= observed) / sum(distribution.values())


def evaluate(results, tasks, seeds, arms, threshold, read_artifacts, prior=None):
    expected = {(a, t, s) for a in arms for t in tasks for s in seeds}
    keys = [(r.get("arm"), r.get("task"), r.get("run_seed")) for r in results]
    integrity = {"complete_keys": len(keys) == len(expected) and len(set(keys)) == len(keys) and set(keys) == expected,
                 "all_completed": all(r.get("status") == "completed" for r in results),
                 "binary_utility": all(type(r.get("utility")) is bool for r in results),
                 "no_network_attempts": all(r.get("blocked_network_attempts") == 0 for r in results)}
    if not all(integrity.values()):
        return {"decision": "INVALID_PROTOCOL_RECOVERY", "integrity": integrity, "scientific_result": False}
    by_key = dict(zip(keys, results))
    artifact_cache = {}
    alignment, raw_match, bounded_retry = True, True, True
    for key, row in by_key.items():
        raw, diag = read_artifacts(row)
        artifact_cache[key] = diag
        alignment &= row["episode_seed"] == paired_episode_seed(row["run_seed"], row["task"], tasks)
        raw_match &= raw.get("utility") == row["utility"] and not raw.get("error")
        assistant_count = sum(m["role"] == "assistant" for m in raw["messages"])
        parsed_calls = sum(len(m.get("tool_calls") or []) for m in raw["messages"] if m["role"] == "assistant")
        extra = len(diag["events"]) - assistant_count
        bounded_retry &= extra in ([0, 1] if row["arm"] == "syntax_retry" else [0])
        raw_match &= row["parsed_tool_calls"] == parsed_calls and bool(diag["events"]) and bool(diag["first_prompt_ids"])
    first_input_match, first_generation_match = True, True
    for task in tasks:
        for seed in seeds:
            base = artifact_cache[("strict", task, seed)]
            for arm in arms:
                item = artifact_cache[(arm, task, seed)]
                first_input_match &= item["first_prompt_ids"] == base["first_prompt_ids"]
                first_generation_match &= item["events"][0]["completion"] == base["events"][0]["completion"]
    integrity.update({"seed_alignment": alignment, "raw_records_match": raw_match,
                      "bounded_retry": bounded_retry, "paired_first_inputs_identical": first_input_match,
                      "paired_first_completions_identical": first_generation_match})
    if not all(integrity.values()):
        return {"decision": "INVALID_PROTOCOL_RECOVERY", "integrity": integrity, "scientific_result": False}
    metrics = {}
    for arm in arms:
        subset = [by_key[(arm, t, s)] for t in tasks for s in seeds]
        stable = [t for t in tasks if sum(by_key[(arm, t, s)]["utility"] for s in seeds) >= 2]
        suites = Counter(t.split("|")[0] for t in stable)
        suite_rates = {suite: sum(r["utility"] for r in subset if r["task"].startswith(suite + "|")) / sum(r["task"].startswith(suite + "|") for r in subset) for suite in {t.split("|")[0] for t in tasks}}
        events = [e for t in tasks for s in seeds for e in artifact_cache[(arm, t, s)]["events"]]
        metrics[arm] = {"successes": sum(r["utility"] for r in subset), "episodes": len(subset),
                        "utility": sum(r["utility"] for r in subset) / len(subset),
                        "stable_tasks": stable, "stable_count": len(stable), "stable_by_suite": dict(suites),
                        "suite_utility": suite_rates, "zero_call_failures": sum(not r["utility"] and r["parsed_tool_calls"] == 0 for r in subset),
                        "generation_calls": len(events), "output_tokens": sum(e["output_tokens"] for e in events)}
    comparisons = {}
    for arm in arms:
        if arm == "strict": continue
        differences = [sum(int(by_key[(arm, t, s)]["utility"]) - int(by_key[("strict", t, s)]["utility"]) for s in seeds) for t in tasks]
        gain = sum(differences) / (len(tasks) * len(seeds))
        regressions = sum(by_key[("strict", t, s)]["utility"] and by_key[("strict", t, s)]["parsed_tool_calls"] == 0 and not by_key[(arm, t, s)]["utility"] for t in tasks for s in seeds)
        degradation = max(0.0, max(metrics["strict"]["suite_utility"][suite] - rate for suite, rate in metrics[arm]["suite_utility"].items()))
        pvalue = task_signflip_p(differences)
        checks = {"gain": gain + 1e-12 >= threshold["minimum_gain"],
                  "improved_tasks": sum(d > 0 for d in differences) >= threshold["minimum_improved_tasks"],
                  "paired_task_p": pvalue <= threshold["maximum_task_signflip_p"],
                  "suite_noninferiority": degradation <= threshold["maximum_suite_degradation"] + 1e-12,
                  "preserve_successful_tool_free": regressions <= threshold["maximum_regressions_on_successful_tool_free_strict"],
                  "stable_eligibility": metrics[arm]["stable_count"] >= threshold["minimum_stable_tasks"],
                  "suite_eligibility": sum(n >= 2 for n in metrics[arm]["stable_by_suite"].values()) >= threshold["minimum_suites_with_two_stable_tasks"]}
        if prior is not None:
            overlap = len(set(metrics[arm]["stable_tasks"]) & set(prior["metrics"][arm]["stable_tasks"]))
            checks["stable_confirmation_overlap"] = overlap >= threshold["minimum_stable_task_overlap_with_v39"]
        comparisons[arm] = {"passed": all(checks.values()), "checks": checks, "gain": gain,
                             "task_signflip_p": pvalue, "positive_tasks": sum(d > 0 for d in differences),
                             "negative_tasks": sum(d < 0 for d in differences), "per_task_success_count_difference": dict(zip(tasks, differences)),
                             "tool_free_regressions": regressions, "maximum_suite_degradation": degradation}
    selected = next((a for a in ["syntax", "syntax_retry"] if comparisons.get(a, {}).get("passed")), None)
    return {"decision": "GO_PROTOCOL_RECOVERY" if selected else "NO_GO_PROTOCOL_RECOVERY",
            "scientific_result": True, "selected_arm": selected, "integrity": integrity,
            "metrics": metrics, "comparisons": comparisons, "new_attack_episodes": 0, "model_fits": 0}
