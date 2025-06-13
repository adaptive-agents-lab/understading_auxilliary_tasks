import jax.numpy as jnp

from aux_tasks.model_utils.model_rollout import (
    evaluate_trajectory,
    generate_latent_rollout_from_actions,
    generate_latent_rollout_from_policy,
)


def test_rollout():
    t1 = generate_latent_rollout_from_policy(
        jnp.array([0.0]),
        None,
        lambda s, x, a: (x + 1 + a, x),
        None,
        lambda _, x: 2 * x,
        4,
    )
    t2 = generate_latent_rollout_from_actions(
        jnp.array([0.0]), None, lambda s, x, a: (x + 1 + a, x), t1["action"]
    )
    print(t1)
    print(t2)
    target = evaluate_trajectory(
        t1, None, lambda _, s, a: (3 * s, 2 * s), None, lambda _, x: x, 0.9, True
    )
    print(target)
    target = evaluate_trajectory(
        t1, None, lambda _, s, a: (3 * s, 2 * s), None, lambda _, x: x, 0.9, False
    )
    print(target)


if __name__ == "__main__":
    test_rollout()
