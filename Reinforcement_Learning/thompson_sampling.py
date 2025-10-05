import numpy as np


class ThompsonSamplingBernoulli:
    """
    Thompson Sampling for Bernoulli multi-armed bandit.
    Maintains Beta(alpha, beta) posteriors for each arm.
    reward should be 0 or 1.
    """
    def __init__(self, n_arms, prior_alpha=1.0, prior_beta=1.0, rng=None):
        self.n_arms = int(n_arms)
        self.alpha = np.full(self.n_arms, prior_alpha, dtype=float)  # successes + prior
        self.beta = np.full(self.n_arms, prior_beta, dtype=float)    # failures + prior
        self.rng = np.random.default_rng() if rng is None else rng

    def select_arm(self):
        """Sample theta for each arm from Beta(alpha, beta) and return argmax."""
        samples = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, chosen_arm, reward):
        """Update posterior for chosen_arm. reward must be 0 or 1."""
        if reward not in (0, 1):
            raise ValueError("Reward must be 0 or 1 for Bernoulli bandit.")
        self.alpha[chosen_arm] += reward
        self.beta[chosen_arm] += (1 - reward)

    def estimated_means(self):
        """Posterior mean estimate for each arm = alpha/(alpha+beta)."""
        return self.alpha / (self.alpha + self.beta)


def simulate(true_ps, n_rounds=10000, seed=42, prior_alpha=1.0, prior_beta=1.0):
    """
    Simulate Thompson Sampling on a Bernoulli bandit.
    Args:
      true_ps: list or array of true success probabilities for each arm (values in [0,1]).
      n_rounds: number of total pulls to simulate.
    Returns:
      dict with cumulative reward, pulls per arm, final posterior estimates, regret.
    """
    rng = np.random.default_rng(seed)
    n_arms = len(true_ps)
    ts = ThompsonSamplingBernoulli(n_arms, prior_alpha, prior_beta, rng=rng)

    pulls = np.zeros(n_arms, dtype=int)
    rewards = np.zeros(n_rounds, dtype=int)
    chosen_arms = np.zeros(n_rounds, dtype=int)

    for t in range(n_rounds):
        arm = ts.select_arm()
        # Simulate Bernoulli reward using true probability of the chosen arm
        reward = 1 if rng.random() < true_ps[arm] else 0
        ts.update(arm, reward)

        pulls[arm] += 1
        rewards[t] = reward
        chosen_arms[t] = arm

    cumulative_reward = rewards.sum()
    best_prob = max(true_ps)
    optimal_total = best_prob * n_rounds
    regret = optimal_total - cumulative_reward  # expected regret wrt always pulling best arm

    return {
        "cumulative_reward": int(cumulative_reward),
        "pulls_per_arm": pulls,
        "final_posterior_means": ts.estimated_means(),
        "regret": float(regret),
        "chosen_arms": chosen_arms,
        "true_ps": np.array(true_ps)
    }



# Example usage
true_probabilities = [0.05, 0.10, 0.08, 0.20, 0.12]  # five arms, arm 3 (index 3) is best (0.20)
rounds = 5000

result = simulate(true_probabilities, n_rounds=rounds, seed=123, prior_alpha=1.0, prior_beta=1.0)

print("True probs:         ", result["true_ps"])
print("Pulls per arm:      ", result["pulls_per_arm"])
print("Posterior mean est.:", np.round(result["final_posterior_means"], 4))
print("Cumulative reward:  ", result["cumulative_reward"])
print("Simple regret (opt - actual): {:.2f}".format(result["regret"]))
