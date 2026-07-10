"""Train the full SheepRL DreamerV3 backend on AgentDojo trajectories."""

from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--obs-dim", type=int, default=768)
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
    parser.add_argument("--utility-reward-scale", type=float, default=1.0)
    parser.add_argument("--target-skill-reward-scale", type=float, default=0.25)
    parser.add_argument("--risk-loss-scale", type=float, default=1.0)
    parser.add_argument("--utility-loss-scale", type=float, default=1.0)
    parser.add_argument("--preservation-loss-scale", type=float, default=1.0)
    parser.add_argument("--candidate-loss-scale", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    train_steps = _steps(args.train)
    val_steps = _steps(args.val)
    config = FullDreamerV3Config(
        obs_dim=args.obs_dim,
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
        utility_reward_scale=args.utility_reward_scale,
        target_skill_reward_scale=args.target_skill_reward_scale,
        risk_loss_scale=args.risk_loss_scale,
        utility_loss_scale=args.utility_loss_scale,
        preservation_loss_scale=args.preservation_loss_scale,
        candidate_loss_scale=args.candidate_loss_scale,
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
    result = {
        "backend": "sheeprl_full_dreamer_v3_offline",
        "train_steps": len(train_steps),
        "val_steps": len(val_steps),
        "model_out": str(args.model_out.resolve()),
        "best_epoch": model.best_epoch,
        "model_info": model.model_info(),
        "training_history": model.training_history,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
