"""Read-only diagnostic of within-goal support; never selects executable attacks."""
from collections import defaultdict
from itertools import combinations
import json

import numpy as np

from .attack_conditioned_ranker import structured_attack_features, task_name, forbidden_feature_keys
from .comparison_reward_policy import four_cell_counts_from_label_group, posterior_preference_probability


def align_fixed_goals(manifest_rows, labels, expected):
    ids = [str(row["row_id"]) for row in manifest_rows]
    attacks = [row for row in labels if row.get("source_kind") == "attack"]
    label_ids = [str(row["row_id"]) for row in attacks]
    if len(set(ids)) != len(ids) or len(set(label_ids)) != len(label_ids):
        raise ValueError("duplicate candidate or label")
    if set(ids) != set(label_ids) or len(ids) != expected["candidates"]:
        raise ValueError("candidate alignment mismatch")
    by_id = {str(row["row_id"]): row for row in attacks}
    rows = []
    for action in manifest_rows:
        task = task_name(action)
        goal = str(action["injection_task_id"])
        if not goal or action.get("base_pair_id") != f"{task}|{goal}":
            raise ValueError("goal identity mismatch")
        counts = four_cell_counts_from_label_group(by_id[str(action["row_id"])])
        if sum(counts) != expected["trials_per_candidate"]:
            raise ValueError("trial count mismatch")
        features = structured_attack_features(action, include_family=False)
        if forbidden_feature_keys(features):
            raise ValueError("forbidden feature")
        rows.append({"row_id": str(action["row_id"]), "task": task,
                     "goal": goal, "counts": counts, "features": features,
                     "empirical_p11": counts[3] / sum(counts)})
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["task"]][row["goal"]].append(row)
    if len(grouped) != expected["tasks"]:
        raise ValueError("task count mismatch")
    if any(len(goals) != expected["goals_per_task"] for goals in grouped.values()):
        raise ValueError("goal count mismatch")
    if any(len(block) != expected["variants_per_goal"] for goals in grouped.values() for block in goals.values()):
        raise ValueError("variant count mismatch")
    return rows


def fixed_goal_audit(rows, posterior, gate):
    tasks = defaultdict(list)
    for row in rows:
        tasks[row["task"]].append(row)
    support = {"within_goal_reward_pairs": 0, "cross_goal_reward_pairs": 0,
               "within_goal_p11_pairs": 0, "contrasting_feature_collisions": 0,
               "within_goal_reward_only_pairs": 0}
    task_details, goal_details = [], []
    suite_tasks = defaultdict(int)
    for task, candidates in sorted(tasks.items()):
        goals = defaultdict(list)
        informative = set()
        for row in candidates:
            goals[row["goal"]].append(row)
        pairs_by_goal = defaultdict(int)
        for left, right in combinations(candidates, 2):
            kwargs = dict(left_id=left["row_id"], right_id=right["row_id"],
                          draws=posterior["draws"], seed=posterior["seed"],
                          dirichlet_prior=posterior["dirichlet_prior"])
            reward_prob = posterior_preference_probability(left["counts"], right["counts"], **kwargs)
            reward_confident = abs(reward_prob - 0.5) >= posterior["confidence_gap"]
            same_goal = left["goal"] == right["goal"]
            if reward_confident:
                support["within_goal_reward_pairs" if same_goal else "cross_goal_reward_pairs"] += 1
            if not same_goal:
                continue
            p11_prob = posterior_preference_probability(left["counts"], right["counts"],
                                                        reward_weights=[0, 0, 0, 1], **kwargs)
            p11_confident = (left["counts"][3] != right["counts"][3]
                             and abs(p11_prob - 0.5) >= posterior["confidence_gap"])
            if p11_confident:
                support["within_goal_p11_pairs"] += 1
                informative.add(left["goal"])
                pairs_by_goal[left["goal"]] += 1
                # Direct dictionary equality, not a content digest.
                support["contrasting_feature_collisions"] += int(left["features"] == right["features"])
            if reward_confident and not p11_confident:
                support["within_goal_reward_only_pairs"] += 1
        gaps, randoms, oracles = [], [], []
        for goal, block in sorted(goals.items()):
            values = [row["empirical_p11"] for row in block]
            oracle, uniform = max(values), float(np.mean(values))
            gaps.append(oracle - uniform)
            randoms.append(uniform)
            oracles.append(oracle)
            goal_details.append({"task": task, "goal": goal, "p11_success_counts": [row["counts"][3] for row in block],
                                 "confident_p11_pairs": pairs_by_goal[goal], "informative": goal in informative,
                                 "oracle_p11": oracle, "uniform_p11": uniform})
        if informative:
            suite_tasks[task.split("|", 1)[0]] += 1
        task_details.append({"task": task, "informative_goals": len(informative),
                             "oracle_random_gap": float(np.mean(gaps)),
                             "oracle_p11": float(np.mean(oracles)), "uniform_p11": float(np.mean(randoms))})
    if not task_details:
        raise ValueError("empty task audit")
    support.update({
        "tasks": len(task_details), "goals": len(goal_details),
        "informative_tasks": sum(t["informative_goals"] > 0 for t in task_details),
        "informative_goals": sum(t["informative_goals"] for t in task_details),
        "informative_tasks_by_suite": dict(suite_tasks),
        "suites_with_two_informative_tasks": sum(n >= 2 for n in suite_tasks.values()),
        "task_macro_oracle_random_p11_gap": float(np.mean([t["oracle_random_gap"] for t in task_details])),
        "task_macro_uniform_p11": float(np.mean([t["uniform_p11"] for t in task_details])),
        "task_macro_oracle_p11": float(np.mean([t["oracle_p11"] for t in task_details])),
        "feature_collision_fraction": support["contrasting_feature_collisions"] / max(1, support["within_goal_p11_pairs"]),
    })
    checks = {
        "informative_tasks": support["informative_tasks"] >= gate["minimum_informative_tasks"],
        "informative_goals": support["informative_goals"] >= gate["minimum_informative_goals"],
        "oracle_gap": support["task_macro_oracle_random_p11_gap"] >= gate["minimum_task_macro_oracle_random_p11_gap"],
        "feature_distinguishability": support["feature_collision_fraction"] <= gate["maximum_feature_collision_fraction"],
        "suite_coverage": support["suites_with_two_informative_tasks"] >= gate["minimum_suites_with_two_informative_tasks"],
    }
    return {"decision": "GO_FIXED_GOAL_SUPPORT_V36" if all(checks.values()) else "NO_GO_FIXED_GOAL_SUPPORT_V36",
            "checks": checks, "metrics": support, "per_task": task_details, "per_goal": goal_details,
            "model_fits": 0, "new_episodes": 0, "runtime_failures": 0,
            "inference_scope": "descriptive archive audit; independent posterior approximation; in-sample oracle is optimistic"}
