"""Hierarchical rollout coordinator for belief selection and LL sampling."""

import random
from typing import Callable, Dict, List


class HierarchicalRolloutCoordinator:
    """Coordinates hierarchical high-level selection and low-level generation."""

    def __init__(
        self,
        softmax_selector: Callable[[Dict[str, float]], str],
        regret_selector: Callable[[Dict[str, float], Dict[str, float], float], str],
        template_name_selector: Callable[..., str],
        noise_filter: Callable[[str], bool],
    ) -> None:
        self.softmax_selector = softmax_selector
        self.regret_selector = regret_selector
        self.template_name_selector = template_name_selector
        self.noise_filter = noise_filter

    def precompute_low_level_actions(self, episodes, ll_agent, config) -> None:
        eligible = [
            ep for ep in episodes
            if ep.is_active and not ep.done_from_env and ep.turn > 0 and not ep.is_frozen
        ]
        if not eligible:
            return

        if config.ll_candidate_rerank:
            return

        belief_contexts = []
        histories = []
        target_episodes = []
        chunk_size = config.batch_generation_chunk_size
        epsilon_to_use = getattr(config, "_current_epsilon", config.epsilon)
        filter_noise = config.contrastive_ablation_mode == "noise_in_candidates"

        for episode in eligible:
            q_values = getattr(episode, "_q_values", {})
            high_level_candidates = [c.summary for c in episode.belief_state.candidates]
            valid_candidates = [c for c in high_level_candidates if c != "[SKIP]"]
            if filter_noise:
                valid_candidates = [c for c in valid_candidates if not self.noise_filter(c)]

            valid_q_values = {k: v for k, v in q_values.items() if k != "[SKIP]"}
            if filter_noise:
                valid_q_values = {k: v for k, v in valid_q_values.items() if not self.noise_filter(k)}
            if not valid_candidates:
                continue

            selected_belief = self._select_high_level_belief(
                valid_candidates=valid_candidates,
                valid_q_values=valid_q_values,
                episode=episode,
                config=config,
                epsilon=epsilon_to_use,
            )
            ll_history = episode.belief_state.history
            if not ll_history:
                continue

            belief_contexts.append(selected_belief)
            histories.append(ll_history)
            target_episodes.append(episode)
            episode._preselected_belief = selected_belief

        if not target_episodes:
            return

        baseline_mode = config.baseline_mode
        template_name = self.template_name_selector(
            config,
            baseline_mode=baseline_mode,
            episode=target_episodes[0],
        )
        actions = ll_agent.generate_action_batch(
            belief_contexts=belief_contexts,
            histories=histories,
            temperature=0.7,
            belief_only=config.ll_action_belief_only if not baseline_mode else False,
            chunk_size=chunk_size,
            template_name=template_name,
        )
        for ep, action in zip(target_episodes, actions):
            ep._precomputed_agent_response = action
            if hasattr(ep.env, "env_state") and hasattr(ep.env.env_state, "ll_agent_outputs"):
                ep.env.env_state.ll_agent_outputs.append(action)

    def _select_high_level_belief(self, valid_candidates, valid_q_values, episode, config, epsilon):
        if config.random_belief_selection:
            return random.choice(valid_candidates)
        if config.raw_judge_belief_selection:
            ranked = [c for c in valid_candidates if c in valid_q_values]
            if not ranked:
                raise RuntimeError(
                    f"raw_judge_belief_selection: no scored candidates among valid={valid_candidates}"
                )
            best = max(valid_q_values[c] for c in ranked)
            tops = [c for c in ranked if valid_q_values[c] == best]
            return random.choice(tops)
        if not valid_q_values:
            return random.choice(valid_candidates)
        if random.random() < epsilon:
            return random.choice(valid_candidates)
        regret_values = getattr(episode, "_regret_values", None)
        if config.use_regret_critic and regret_values and all(k in regret_values for k in valid_q_values):
            beta = config.regret_critic_beta
            return self.regret_selector(valid_q_values, regret_values, beta)
        return self.softmax_selector(valid_q_values)
