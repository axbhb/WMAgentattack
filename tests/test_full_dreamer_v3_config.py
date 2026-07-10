from wmagentattack.full_dreamer_v3 import FullDreamerV3Config


def test_full_dreamer_defaults_include_behavior_learning():
    config = FullDreamerV3Config()
    assert config.imagination_horizon > 0
    assert config.actor_learning_rate > 0
    assert config.critic_learning_rate > 0
    assert config.target_critic_tau > 0
    assert config.behavior_cloning_scale > 0
    assert config.stochastic_size * config.discrete_size > 0
