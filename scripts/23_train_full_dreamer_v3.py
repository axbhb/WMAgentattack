"""Train the full SheepRL DreamerV3 backend on AgentDojo trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.full_dreamer_v3 import FullDreamerV3Config, FullSheepRLDreamerV3
from wmagentattack.io_utils import read_jsonl
from wmagentattack.schema import StepRecord


def _steps(path: Path) -> list[StepRecord]:
    return [StepRecord.model_validate(row) for row in read_jsonl(path)]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _step_provenance(path: Path, steps: list[StepRecord]) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": _file_sha256(path),
        "step_count": len(steps),
        "trajectory_count": len({step.trajectory_id for step in steps}),
        "multiseed_group_count": len(
            {
                getattr(step, "multiseed_group_id", None)
                for step in steps
                if getattr(step, "multiseed_group_id", None) is not None
            }
        ),
        "user_task_count": len({(step.domain, step.task_id) for step in steps}),
        "user_tasks_by_domain": {
            domain: len(
                {
                    step.task_id
                    for step in steps
                    if step.domain == domain
                }
            )
            for domain in sorted({step.domain for step in steps})
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--obs-dim", type=int, default=768)
    parser.add_argument(
        "--observation-feature-mode",
        choices=["hash", "precomputed"],
        default="hash",
    )
    parser.add_argument("--observation-feature-path", type=Path)
    parser.add_argument("--dense-units", type=int, default=256)
    parser.add_argument("--recurrent-state-size", type=int, default=256)
    parser.add_argument("--stochastic-size", type=int, default=32)
    parser.add_argument("--discrete-size", type=int, default=32)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--actor-layers", type=int, default=2)
    parser.add_argument("--critic-layers", type=int, default=2)
    parser.add_argument("--head-layers", type=int, default=1)
    parser.add_argument("--reward-bins", type=int, default=255)
    parser.add_argument("--imagination-horizon", type=int, default=5)
    parser.add_argument("--imagination-batch-size", type=int, default=256)
    parser.add_argument("--world-learning-rate", type=float, default=3e-4)
    parser.add_argument("--actor-learning-rate", type=float, default=8e-5)
    parser.add_argument("--critic-learning-rate", type=float, default=8e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lmbda", type=float, default=0.95)
    parser.add_argument("--target-critic-tau", type=float, default=0.02)
    parser.add_argument("--entropy-scale", type=float, default=3e-4)
    parser.add_argument("--behavior-cloning-scale", type=float, default=1.0)
    parser.add_argument("--risk-reward-scale", type=float, default=1.0)
    parser.add_argument("--risk-reward-binary-mix", type=float, default=1.0)
    parser.add_argument("--utility-reward-scale", type=float, default=1.0)
    parser.add_argument("--utility-reward-binary-mix", type=float, default=0.0)
    parser.add_argument("--target-skill-reward-scale", type=float, default=0.25)
    parser.add_argument("--risk-loss-scale", type=float, default=1.0)
    parser.add_argument("--binary-risk-loss-scale", type=float, default=1.0)
    parser.add_argument("--soft-risk-loss-scale", type=float, default=0.0)
    parser.add_argument("--risk-final-step-only", action="store_true")
    parser.add_argument(
        "--group-risk-calibration-loss-scale", type=float, default=0.0
    )
    parser.add_argument(
        "--group-risk-calibration-detach-latent", action="store_true"
    )
    parser.add_argument(
        "--grouped-risk-calibration-batches", action="store_true"
    )
    parser.add_argument("--utility-loss-scale", type=float, default=1.0)
    parser.add_argument("--binary-utility-loss-scale", type=float, default=0.0)
    parser.add_argument("--soft-utility-loss-scale", type=float, default=1.0)
    parser.add_argument("--utility-ranking-loss-scale", type=float, default=0.0)
    parser.add_argument("--utility-ranking-margin", type=float, default=0.2)
    parser.add_argument("--utility-ranking-detach-latent", action="store_true")
    parser.add_argument("--ranking-pairs-per-batch", type=int, default=0)
    parser.add_argument(
        "--group-utility-calibration-loss-scale", type=float, default=0.0
    )
    parser.add_argument(
        "--group-utility-calibration-detach-latent", action="store_true"
    )
    parser.add_argument(
        "--group-utility-ranking-loss-scale", type=float, default=0.0
    )
    parser.add_argument(
        "--group-utility-ranking-detach-latent", action="store_true"
    )
    parser.add_argument(
        "--group-utility-min-target-gap", type=float, default=0.1
    )
    parser.add_argument("--group-utility-pairs-per-task", type=int, default=8)
    parser.add_argument("--grouped-utility-batches", action="store_true")
    parser.add_argument("--group-utility-head-only-updates", action="store_true")
    parser.add_argument("--configuration-value-head", action="store_true")
    parser.add_argument(
        "--group-value-calibration-loss-scale", type=float, default=0.0
    )
    parser.add_argument(
        "--group-value-ranking-loss-scale", type=float, default=0.0
    )
    parser.add_argument("--group-value-min-target-gap", type=float, default=0.1)
    parser.add_argument("--group-value-pairs-per-task", type=int, default=8)
    parser.add_argument("--group-value-head-only-updates", action="store_true")
    parser.add_argument("--preservation-loss-scale", type=float, default=1.0)
    parser.add_argument("--candidate-loss-scale", type=float, default=0.25)
    parser.add_argument(
        "--validation-risk-mode",
        choices=["continuous", "binary"],
        default="binary",
    )
    parser.add_argument(
        "--validation-utility-mode",
        choices=["continuous", "binary"],
        default="continuous",
    )
    parser.add_argument(
        "--validation-aggregation",
        choices=["step", "multiseed_group"],
        default="step",
    )
    parser.add_argument(
        "--validation-group-step",
        choices=["first", "final"],
        default="final",
    )
    parser.add_argument(
        "--checkpoint-objective",
        choices=[
            "validation_objective",
            "grouped_configuration_value_brier",
        ],
        default="validation_objective",
    )
    parser.add_argument("--grouped-ranking-batches", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    train_steps = _steps(args.train)
    val_steps = _steps(args.val)
    config = FullDreamerV3Config(
        obs_dim=args.obs_dim,
        observation_feature_mode=args.observation_feature_mode,
        observation_feature_path=(
            str(args.observation_feature_path.resolve())
            if args.observation_feature_path is not None
            else None
        ),
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        dense_units=args.dense_units,
        recurrent_state_size=args.recurrent_state_size,
        stochastic_size=args.stochastic_size,
        discrete_size=args.discrete_size,
        actor_layers=args.actor_layers,
        critic_layers=args.critic_layers,
        head_layers=args.head_layers,
        reward_bins=args.reward_bins,
        world_learning_rate=args.world_learning_rate,
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        imagination_horizon=args.imagination_horizon,
        imagination_batch_size=args.imagination_batch_size,
        gamma=args.gamma,
        lmbda=args.lmbda,
        target_critic_tau=args.target_critic_tau,
        entropy_scale=args.entropy_scale,
        behavior_cloning_scale=args.behavior_cloning_scale,
        risk_reward_scale=args.risk_reward_scale,
        risk_reward_binary_mix=args.risk_reward_binary_mix,
        utility_reward_scale=args.utility_reward_scale,
        utility_reward_binary_mix=args.utility_reward_binary_mix,
        target_skill_reward_scale=args.target_skill_reward_scale,
        risk_loss_scale=args.risk_loss_scale,
        binary_risk_loss_scale=args.binary_risk_loss_scale,
        soft_risk_loss_scale=args.soft_risk_loss_scale,
        risk_final_step_only=args.risk_final_step_only,
        group_risk_calibration_loss_scale=(
            args.group_risk_calibration_loss_scale
        ),
        group_risk_calibration_detach_latent=(
            args.group_risk_calibration_detach_latent
        ),
        grouped_risk_calibration_batches=(
            args.grouped_risk_calibration_batches
        ),
        utility_loss_scale=args.utility_loss_scale,
        binary_utility_loss_scale=args.binary_utility_loss_scale,
        soft_utility_loss_scale=args.soft_utility_loss_scale,
        utility_ranking_loss_scale=args.utility_ranking_loss_scale,
        utility_ranking_margin=args.utility_ranking_margin,
        utility_ranking_detach_latent=args.utility_ranking_detach_latent,
        ranking_pairs_per_batch=args.ranking_pairs_per_batch,
        group_utility_calibration_loss_scale=(
            args.group_utility_calibration_loss_scale
        ),
        group_utility_calibration_detach_latent=(
            args.group_utility_calibration_detach_latent
        ),
        group_utility_ranking_loss_scale=args.group_utility_ranking_loss_scale,
        group_utility_ranking_detach_latent=(
            args.group_utility_ranking_detach_latent
        ),
        group_utility_min_target_gap=args.group_utility_min_target_gap,
        group_utility_pairs_per_task=args.group_utility_pairs_per_task,
        grouped_utility_batches=args.grouped_utility_batches,
        group_utility_head_only_updates=args.group_utility_head_only_updates,
        configuration_value_head_enabled=args.configuration_value_head,
        group_value_calibration_loss_scale=(
            args.group_value_calibration_loss_scale
        ),
        group_value_ranking_loss_scale=args.group_value_ranking_loss_scale,
        group_value_min_target_gap=args.group_value_min_target_gap,
        group_value_pairs_per_task=args.group_value_pairs_per_task,
        group_value_head_only_updates=args.group_value_head_only_updates,
        preservation_loss_scale=args.preservation_loss_scale,
        candidate_loss_scale=args.candidate_loss_scale,
        validation_risk_mode=args.validation_risk_mode,
        validation_utility_mode=args.validation_utility_mode,
        validation_aggregation=args.validation_aggregation,
        validation_group_step=args.validation_group_step,
        checkpoint_objective=args.checkpoint_objective,
        grouped_ranking_batches=args.grouped_ranking_batches,
        seed=args.seed,
        device=args.device,
    )
    model = FullSheepRLDreamerV3(config).fit(
        train_steps,
        val_steps=val_steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    model.save(args.model_out)
    data_provenance = {
        "split_unit": "suite_and_user_task_id",
        "train": _step_provenance(args.train, train_steps),
        "validation": _step_provenance(args.val, val_steps),
        "task_overlap": len(
            {(step.domain, step.task_id) for step in train_steps}
            & {(step.domain, step.task_id) for step in val_steps}
        ),
    }
    if args.observation_feature_path is not None:
        data_provenance["observation_features"] = {
            "mode": args.observation_feature_mode,
            "path": str(args.observation_feature_path.resolve()),
            "sha256": _file_sha256(args.observation_feature_path),
        }
    metadata_path = args.model_out / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["data_provenance"] = data_provenance
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    result = {
        "backend": "sheeprl_full_dreamer_v3_offline",
        "train_steps": len(train_steps),
        "val_steps": len(val_steps),
        "model_out": str(args.model_out.resolve()),
        "best_epoch": model.best_epoch,
        "model_info": model.model_info(),
        "data_provenance": data_provenance,
        "training_history": model.training_history,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
