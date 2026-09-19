"""Base configuration class for the LLM Context Belief Framework."""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BaseConfig:
    """Type-safe configuration with validation."""
    
    # Model settings
    model_name: str = "HuggingFaceH4/zephyr-7b-beta"
    user_model_name: Optional[str] = None  # If None, uses same model as model_name
    device: str = "cuda"
    use_bf16: bool = True
    
    # GPT agent settings (for using OpenAI API for agents instead of local model)
    use_gpt_for_agents: bool = False  # If True, use GPT API for high-level and low-level agents
    use_gpt_for_user: bool = False  # If True, use GPT API for user agent (in addition to agents)
    use_gpt_for_patient: bool = False  # If True, CARES/WJB patient sim via CMU/OpenAI gateway (no local user model)
    use_gpt_for_judge: bool = False  # If True, fulfillment/assistance judges via CMU/OpenAI gateway
    gpt_agent_model: str = "gpt-4o-mini"  # GPT model to use for agents (e.g., "gpt-4o-mini", "gpt-4o")
    gpt_judge_model: Optional[str] = None  # Judge/guard API model; defaults to gpt_agent_model when unset
    gpt_patient_model: Optional[str] = None  # Patient/attacker API model; defaults to gpt_agent_model when unset
    gpt_reasoning_effort: Optional[str] = None  # GPT reasoning effort for gpt-5 family (e.g., "none", "medium")
    gpt_agent_api_key: Optional[str] = None  # OpenAI API key (if None, uses OPENAI_API_KEY env var)
    hl_enable_thinking: Optional[bool] = None  # If None, defaults to True for Qwen3.5 and False otherwise
    ll_enable_thinking: Optional[bool] = None  # If None, defaults to False
    
    # Agent hyperparameters
    n_candidates: int = 5
    n_ll_candidates: int = 5  # K LL candidates per selected belief; >1 enables token critic selection
    n_next_state_candidates: int = 3  # Fewer candidates for next-state Q-value computation (faster)
    belief_gen_temperature: float = 0.7
    max_tokens: int = 256  
    epsilon: float = 0.1  # For epsilon-greedy action selection
    belief_gen_template: str = "belief_generation"  # Template name for belief generation (options: belief_generation, belief_generation_v2, belief_generation_v3, belief_generation_v4, belief_generation_v5)
    
    # Belief update parameters
    dpo_temperature: float = 1.0  # Fixed DPO temperature
    
    # Value function parameters
    discount_factor: float = 0.9
    learning_rate: float = 1e-4
    entropy_coef: float = 0.1
    # Stabilization (target nets, capacity, schedules)
    critic_target_tau: float = 0.0  # >0 enables Polyak target V/Q heads
    mlp_width_mult: float = 1.0  # 2.0 = 2x mid-layer width
    # Explicit MLP hidden widths (excluding the final scalar). None = use mlp_width_mult.
    # Ablations: [2048] → H→2048→1; [4096, 512] → H→4096→512→1.
    critic_mlp_dims: Optional[List[int]] = None
    critic_lora_r: int = 0  # 0 = frozen backbone only; >0 LoRA on last critic_lora_layers
    critic_lora_alpha: int = 16
    critic_lora_layers: int = 4
    critic_lora_lr: float = 2e-5
    normalize_td_targets: bool = False
    reward_norm_momentum: float = 0.99
    reward_norm_clip: float = 10.0
    use_behavior_snapshot: bool = False
    behavior_refresh_episodes: int = 50
    match_epsilon_candidates: bool = False  # if True, no dual-template merge (i.i.d. π^ref)
    critic_warmup_updates: int = 0
    critic_head_anneal_end: float = 0.3
    critic_lora_anneal_end: float = 0.1
    critic_anneal_total_updates: int = 10000
    
    # Marginal token rewards parameters
    use_marginal_token_rewards: bool = False  # Enable marginal token reward computation
    num_masked_marginal_tokens: int = 5  # Number of top-k tokens to process
    pairwise_marginal_masking: bool = True  # Enable pairwise masking
    max_marginal_token_positions: int = 10  # Memory limit for token processing
    use_task_reward: bool = True  # Include instruction-following reward
    reward_model_type: Optional[str] = "llamaguard"  # "llamaguard", "shieldgemma", "skywork", or None (None = use environment reward)
    reward_model_name: Optional[str] = None  # Path to reward model (if None, uses sequence critic)
    marginal_reward_chunk_size: int = 4  # Chunk size for processing masked versions
    token_level_reward_schema: str = "marginal"  # "marginal" | "uniform" | "decayed" (token critic targets)
    token_reward_decay_gamma: Optional[float] = None  # For "decayed"; None = use discount_factor

    # Contrastive learning parameters
    contrastive_coef: float = 0.0  # Coefficient for contrastive loss (0.0 = disabled)
    contrastive_ablation_mode: str = "none"  # One of: "crossed_data", "random_noise", "noise_in_candidates", "crossed_data_in_candidates", "none"

    # Regret critic (adversarial robustness) - for cares/wildjailbreak
    use_regret_critic: bool = False  # Enable three-critic: value, min-value, regret
    # If True, execute argmax_b Q(s,b) (epsilon-greedy). Skips regret-softmax / softmax.
    hl_greedy_q: bool = False
    # If True, online r_nom uses paper DSR/GCR prompts (harmless + assist / helpful).
    paper_dsr_reward: bool = False
    regret_critic_beta: float = 0.2  # High-level policy: softmax over (1-beta)*Q - beta*Q^regret
    regret_zero_sum_targets: bool = True  # Enforce zero-sum regret targets
    # min_q_over_states: Q_min(s,a) target = min_{s̃} Q(s̃, a) with action held fixed
    # (worst state for this action — not min over candidate actions at fixed s).
    regret_min_target_mode: str = "min_q_over_states"
    
    # Low-level action generation mode
    ll_action_belief_only: bool = True  # If True, LL actions depend only on belief P(a|b). If False, use P(a|o,b) (old mode for ablations)
    defender_backend: str = "standard"  # "standard" | "smoothllm" | "tpo" | "nbf"

    # SmoothLLM defender hyperparameters
    pert_type: str = "RandomSwapPerturbation"
    pert_pct: float = 10.0
    num_copies: int = 8
    smooth_batch_size: int = 8

    # TPO defender hyperparameters
    tpo_sample_size: int = 8
    tpo_max_iters: int = 3
    tpo_temperature: float = 0.7
    tpo_reward_model: str = "heuristic"
    tpo_mode: str = "tpo"  # "tpo" | "revision" | "bon"

    # Neural Barrier Function (Hu et al. TMLR 2026) filter. Used with defender_backend="nbf".
    nbf_model_path: Optional[str] = None  # HuggingFace hanjianghu/NBF-LLM models_best_nbf_released.pth
    nbf_threshold: float = 0.001  # η; safe iff safety index < -η
    nbf_embedder_name: str = "sentence-transformers/all-mpnet-base-v2"
    nbf_device: str = "cpu"  # keep mpnet off the Zephyr GPU
    
    # Baseline mode (skip high-level belief generation, use only conversation history)
    baseline_mode: bool = False  # If True, skip high-level agent and DST, use only conversation history for low-level actions
    
    # Random belief selection (skip Q-value computation, always randomly select from candidates)
    random_belief_selection: bool = False  # If True, always randomly selects from candidates (no Q-value computation, no value function)

    # Oracle HL: skip π^ref sampling and use the dataset ground-truth goal as the sole inferred belief.
    ground_truth_belief_selection: bool = False

    # Offline two-head LL token critic (`harm_head` + `follow_head`). When set, LL rerank
    # uses this checkpoint instead of ValueFunction.token_critic_head.
    ll_token_critic_path: Optional[str] = None

    # Always-on intent hypothesis: skip π^ref sampling and use one fixed belief.
    # "none" = sample K candidates; "antagonistic" / "benign_misrepresented" = static baselines.
    static_belief_mode: str = "none"

    # HL critic ablation: score each HL candidate with R(user, b) via env judge/guard on
    # (last user utterance, belief text) — same input space as Q(o,b), no LL expansion.
    # Pick argmax; then generate a single LL reply from the selected belief. No value function.
    raw_judge_belief_selection: bool = False

    # LL critic ablation: HL selection still uses the HL critic; generate K LL candidates,
    # score each with R(user, a) (env judge/guard), pick argmax. Requires n_ll_candidates > 1
    # and a loaded value function for HL scoring.
    raw_judge_ll_selection: bool = False
    # When True (and n_ll_candidates > 1), generate K LL candidates and select via token critic
    # (or via raw_judge_ll_selection scores). Off by default so cares/wjb keep single-LL generation
    # even though n_ll_candidates defaults may be unused for those environments.
    ll_candidate_rerank: bool = False

    # Hierarchical setup
    use_hierarchical_agent: bool = False
    high_level_policy_type: str = "belief_candidates"  # "belief_candidates" | "freeform"
    critic_only_training: bool = False
    freeform_n_instructions: int = 5
    freeform_temperature: float = 0.8
    freeform_top_p: float = 0.9
    freeform_max_new_tokens: int = 96
    freeform_iterative_candidate_generation: bool = False
    freeform_per_instruction_max_new_tokens: int = 96
    freeform_iterative_max_attempts_per_candidate: int = 3
    hierarchical_rejection_sampling: bool = True
    hierarchical_rejection_candidates: int = 3
    
    # Training settings
    max_turns: int = 20
    batch_size: int = 16
    episode_batch_size: int = 1  # Number of episodes to process in parallel (all steps batched)
    vitabench_judge_batch_size: int = 16  # Number of VitaBench episodes to judge together with local evaluator
    evaluation_interval: int = 100  # Episodes between evaluation runs
    transition_prob_chunk_size: int = 8  # Chunk size for transition probability computation to prevent OOM
    q_value_chunk_size: int = 16  # Chunk size for Q-value computation to prevent OOM
    batch_generation_chunk_size: int = 16  # Chunk size for batch text generation to prevent OOM
    n_action_candidates: int = 5  # Number of candidate low-level actions to generate per belief for P(a|b) normalization
    output_dir: str = "outputs"
    
    # Online Q-learning settings
    online_batch_size: int = 8  # Batch size for online updates (from replay buffer)
    replay_buffer_size: int = 1024  # Size of replay buffer for online mode
    epsilon_decay_rate: float = 0.995  # Epsilon decay per episode (exponential)
    epsilon_min: float = 0.05  # Minimum epsilon value
    online_update_frequency: int = 1  # Update Q-function every N turns (1 = every turn, 2 = every other turn, etc.)
    
    # Data settings
    data_path: Optional[str] = None
    persona_dir: str = "src/prompts/personas"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    evaluation_sample_size: Optional[int] = None  # Subsample size for evaluation (None = use all)
    max_dialogues: Optional[int] = None  # Maximum number of dialogues to use (None = use all, for faster training set to e.g. 2000)
    
    # Environment settings
    environment_type: str = "multiwoz_offline"  # "multiwoz_offline", "multiwoz_online", "vitabench", "salesagent", "userbench", "cares", "wildjailbreak", "redbench", "harmbench"
    language: Optional[str] = None  # Language for VitaBench (e.g., "english")
    task_config: Dict = field(default_factory=dict)

    # CARES-18K settings
    cares_split: str = "train"
    cares_use_lead_in: bool = True
    cares_online: bool = True  # Use PatientAgent (online) vs synthetic lead-in (offline)

    # WildJailbreak settings
    wildjailbreak_split: str = "train"
    wildjailbreak_data_types: Optional[List[str]] = None  # None = all 4; else e.g. ["vanilla_harmful", "adversarial_benign"]
    wildjailbreak_use_lead_in: bool = True
    wildjailbreak_online: bool = True  # Use PatientAgent (online) vs synthetic lead-in (offline)

    # RedBench settings
    redbench_split: str = "train"
    redbench_subsets: Optional[List[str]] = None  # None = all subsets in knoveleng/redbench
    redbench_max_examples: Optional[int] = None  # None = all rows across selected subsets
    redbench_mapping_mode: str = "category"  # "category" | "source" | "category_with_source_override"
    redbench_refusal_sources: Optional[List[str]] = None  # None = default refusal sources (CoCoNot, ORBench, SGXSTest, XSTest)
    redbench_use_lead_in: bool = True
    redbench_online: bool = True  # Use PatientAgent (online) vs synthetic lead-in (offline)

    # HarmBench settings
    harmbench_split: str = "train"
    harmbench_max_examples: Optional[int] = None  # None = all rows in RedBench HarmBench subset
    harmbench_use_lead_in: bool = True
    harmbench_online: bool = True  # Use PatientAgent (online) vs synthetic lead-in (offline)
    
    # VitaBench LLM settings
    vitabench_llm_user: Optional[str] = None  # LLM model for VitaBench user simulator (None = use default)
    vitabench_llm_evaluator: Optional[str] = None  # LLM model for VitaBench evaluator/judge (None = use default)
    vitabench_llm_args_user: Optional[Dict] = None  # Additional args for user LLM
    vitabench_llm_args_evaluator: Optional[Dict] = None  # Additional args for evaluator LLM
    vitabench_minibatch_size: int = 8  # Minibatch size for local VitaBench batch_generate
    openai_api_key: Optional[str] = None  # OpenAI API key (if None, uses OPENAI_API_KEY env var)
    
    # Checkpoint settings
    checkpoint_path: Optional[str] = None  # Path to pretrained Q-network checkpoint
    
    # Debug settings
    debug: bool = False
    
    def __post_init__(self) -> None:
        if self.hl_enable_thinking not in (True, False):
            self.hl_enable_thinking = self.model_name.startswith("Qwen/Qwen3.5")
        if self.ll_enable_thinking not in (True, False):
            self.ll_enable_thinking = False
        if self.evaluation_interval <= 0:
            raise ValueError("evaluation_interval must be a positive integer")
        valid_ablation_modes = ["crossed_data", "random_noise", "noise_in_candidates", "crossed_data_in_candidates", "none"]
        if self.contrastive_ablation_mode not in valid_ablation_modes:
            raise ValueError(f"contrastive_ablation_mode must be one of {valid_ablation_modes}, got {self.contrastive_ablation_mode}")
        valid_token_reward_schemas = ["marginal", "uniform", "decayed"]
        if self.token_level_reward_schema not in valid_token_reward_schemas:
            raise ValueError(
                f"token_level_reward_schema must be one of {valid_token_reward_schemas}, got {self.token_level_reward_schema!r}"
            )
        valid_backends = ["standard", "smoothllm", "tpo", "nbf"]
        if self.defender_backend not in valid_backends:
            raise ValueError(f"defender_backend must be one of {valid_backends}, got {self.defender_backend}")
        if self.defender_backend == "nbf":
            if not self.nbf_model_path:
                raise ValueError("defender_backend='nbf' requires nbf_model_path.")
            if self.nbf_threshold < 0:
                raise ValueError(f"nbf_threshold must be >= 0, got {self.nbf_threshold}")
            if not self.baseline_mode:
                raise ValueError("defender_backend='nbf' is a non-critic baseline; set baseline_mode=True.")
        valid_static_modes = ["none", "antagonistic", "benign_misrepresented"]
        if self.static_belief_mode not in valid_static_modes:
            raise ValueError(
                f"static_belief_mode must be one of {valid_static_modes}, got {self.static_belief_mode}"
            )
        static_on = self.static_belief_mode != "none"
        if static_on and self.baseline_mode:
            raise ValueError("static_belief_mode requires baseline_mode=False so the fixed belief reaches the LL agent.")
        if static_on and self.random_belief_selection:
            raise ValueError("static_belief_mode and random_belief_selection are mutually exclusive.")
        if static_on and self.raw_judge_belief_selection:
            raise ValueError("static_belief_mode and raw_judge_belief_selection are mutually exclusive.")
        if static_on and self.raw_judge_ll_selection:
            raise ValueError("static_belief_mode is a non-critic baseline; set raw_judge_ll_selection=False.")
        if static_on and self.use_regret_critic:
            raise ValueError("static_belief_mode is a non-critic baseline; set use_regret_critic=False.")
        if static_on and self.defender_backend != "standard":
            raise ValueError("static_belief_mode requires defender_backend='standard'.")
        if self.ll_token_critic_path == "":
            self.ll_token_critic_path = None
        if self.ground_truth_belief_selection and self.random_belief_selection:
            raise ValueError(
                "ground_truth_belief_selection and random_belief_selection are mutually exclusive."
            )
        if self.ground_truth_belief_selection and self.baseline_mode:
            raise ValueError(
                "ground_truth_belief_selection requires the HL belief to reach the LL agent; set baseline_mode=False."
            )
        if self.ground_truth_belief_selection and self.static_belief_mode != "none":
            raise ValueError(
                "ground_truth_belief_selection and static_belief_mode are mutually exclusive."
            )
        if self.ground_truth_belief_selection and self.raw_judge_belief_selection:
            raise ValueError(
                "ground_truth_belief_selection and raw_judge_belief_selection are mutually exclusive."
            )
        if self.ground_truth_belief_selection and self.use_regret_critic:
            raise ValueError(
                "ground_truth_belief_selection is a non-critic baseline; set use_regret_critic=False."
            )
        if self.ground_truth_belief_selection and self.defender_backend != "standard":
            raise ValueError("ground_truth_belief_selection requires defender_backend='standard'.")
        if self.ground_truth_belief_selection and self.environment_type not in (
            "cares",
            "wildjailbreak",
            "redbench",
            "harmbench",
        ):
            raise ValueError(
                "ground_truth_belief_selection only supports cares/wildjailbreak/redbench/harmbench."
            )
        if self.critic_mlp_dims is not None:
            if not self.critic_mlp_dims:
                raise ValueError("critic_mlp_dims must be a non-empty list of positive integers.")
            for width in self.critic_mlp_dims:
                if int(width) <= 0:
                    raise ValueError(f"critic_mlp_dims entries must be positive, got {self.critic_mlp_dims}")
            self.critic_mlp_dims = [int(w) for w in self.critic_mlp_dims]
        valid_high_level_policy_types = ["belief_candidates", "freeform"]
        if self.high_level_policy_type not in valid_high_level_policy_types:
            raise ValueError(
                f"high_level_policy_type must be one of {valid_high_level_policy_types}, got {self.high_level_policy_type}"
            )
        if self.use_hierarchical_agent and self.high_level_policy_type != "freeform":
            raise ValueError(
                "use_hierarchical_agent requires high_level_policy_type='freeform' for the hierarchical setup."
            )
        if self.use_hierarchical_agent and not self.critic_only_training:
            raise ValueError(
                "use_hierarchical_agent requires critic_only_training=True in this codebase."
            )
        if self.use_hierarchical_agent and self.use_gpt_for_agents:
            raise ValueError(
                "use_hierarchical_agent requires local high-level and low-level agents. Set use_gpt_for_agents=False."
            )
        if self.freeform_n_instructions <= 0:
            raise ValueError("freeform_n_instructions must be positive.")
        if self.freeform_per_instruction_max_new_tokens <= 0:
            raise ValueError("freeform_per_instruction_max_new_tokens must be positive.")
        if self.hierarchical_rejection_candidates <= 0:
            raise ValueError("hierarchical_rejection_candidates must be positive.")
        if self.n_ll_candidates < 1:
            raise ValueError("n_ll_candidates must be >= 1.")
        if self.n_candidates < 1:
            raise ValueError("n_candidates must be >= 1.")
        adversarial_envs = ("cares", "wildjailbreak", "redbench", "harmbench")
        if self.defender_backend == "nbf" and self.environment_type not in adversarial_envs:
            raise ValueError(
                "defender_backend='nbf' only supports cares/wildjailbreak/redbench/harmbench."
            )
        if self.static_belief_mode != "none" and self.environment_type not in adversarial_envs:
            raise ValueError(
                "static_belief_mode only supports cares/wildjailbreak/redbench/harmbench."
            )
        if self.raw_judge_belief_selection and self.raw_judge_ll_selection:
            raise ValueError(
                "raw_judge_belief_selection and raw_judge_ll_selection are mutually exclusive "
                "(separate HL vs LL critic ablations)."
            )
        if self.raw_judge_belief_selection and self.random_belief_selection:
            raise ValueError(
                "raw_judge_belief_selection and random_belief_selection are mutually exclusive."
            )
        if self.raw_judge_belief_selection and self.use_regret_critic:
            raise ValueError(
                "raw_judge_belief_selection is a non-critic baseline; set use_regret_critic=False."
            )
        if self.raw_judge_belief_selection and self.baseline_mode:
            raise ValueError(
                "raw_judge_belief_selection requires HL candidate generation; set baseline_mode=False."
            )
        if self.raw_judge_belief_selection and self.environment_type not in adversarial_envs:
            raise ValueError(
                "raw_judge_belief_selection only supports cares/wildjailbreak/redbench/harmbench."
            )
        if self.raw_judge_ll_selection and self.random_belief_selection:
            raise ValueError(
                "raw_judge_ll_selection requires critic-based HL selection; set random_belief_selection=False."
            )
        if self.raw_judge_ll_selection and self.baseline_mode:
            raise ValueError(
                "raw_judge_ll_selection requires HL candidate generation; set baseline_mode=False."
            )
        if self.raw_judge_ll_selection and self.n_ll_candidates <= 1:
            raise ValueError(
                "raw_judge_ll_selection requires n_ll_candidates > 1 (LL candidate pool to rerank)."
            )
        if self.raw_judge_ll_selection:
            self.ll_candidate_rerank = True
        if self.ll_candidate_rerank and self.n_ll_candidates <= 1:
            raise ValueError(
                "ll_candidate_rerank requires n_ll_candidates > 1."
            )
        if self.ll_token_critic_path is not None:
            if self.n_ll_candidates <= 1:
                raise ValueError(
                    "ll_token_critic_path requires n_ll_candidates > 1."
                )
            self.ll_candidate_rerank = True
            if self.environment_type not in adversarial_envs:
                raise ValueError(
                    "ll_token_critic_path only supports cares/wildjailbreak/redbench/harmbench."
                )
        if self.raw_judge_ll_selection and self.environment_type not in adversarial_envs:
            raise ValueError(
                "raw_judge_ll_selection only supports cares/wildjailbreak/redbench/harmbench."
            )
        if self.ll_candidate_rerank and self.defender_backend != "standard":
            raise ValueError(
                "ll_candidate_rerank / raw_judge_ll_selection require defender_backend='standard'."
            )
        valid_regret_min_target_modes = ["min_q_over_states"]
        if self.regret_min_target_mode not in valid_regret_min_target_modes:
            raise ValueError(
                f"regret_min_target_mode must be one of {valid_regret_min_target_modes}, got {self.regret_min_target_mode}"
            )
        if self.use_regret_critic and not self.regret_zero_sum_targets:
            raise ValueError(
                "use_regret_critic requires regret_zero_sum_targets=True for zero-sum alignment."
            )
        if self.use_regret_critic and self.regret_min_target_mode != "min_q_over_states":
            raise ValueError(
                "use_regret_critic requires regret_min_target_mode='min_q_over_states'."
            )
        if not (0.0 <= self.regret_critic_beta <= 1.0):
            raise ValueError(
                f"regret_critic_beta must be in [0, 1], got {self.regret_critic_beta}."
            )
        if self.hl_greedy_q and self.random_belief_selection:
            raise ValueError("hl_greedy_q and random_belief_selection are mutually exclusive.")
        if self.hl_greedy_q and self.raw_judge_belief_selection:
            raise ValueError("hl_greedy_q and raw_judge_belief_selection are mutually exclusive.")
        if self.hl_greedy_q and self.ground_truth_belief_selection:
            raise ValueError("hl_greedy_q and ground_truth_belief_selection are mutually exclusive.")
        if self.paper_dsr_reward and not self.use_gpt_for_judge:
            raise ValueError("paper_dsr_reward requires use_gpt_for_judge=True.")
        if self.paper_dsr_reward and self.reward_model_type != "api":
            raise ValueError("paper_dsr_reward requires reward_model_type='api'.")
        valid_redbench_mapping_modes = [
            "category",
            "source",
            "category_with_source_override",
        ]
        if self.redbench_mapping_mode not in valid_redbench_mapping_modes:
            raise ValueError(
                f"redbench_mapping_mode must be one of {valid_redbench_mapping_modes}, got {self.redbench_mapping_mode}"
            )
    
    @classmethod
    def from_json(cls, config_path: str) -> "BaseConfig":
        """Load configuration from JSON file."""
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_json(self, config_path: str) -> None:
        """Save all public dataclass fields so eval round-trips match training configs."""
        payload = {key: value for key, value in self.__dict__.items() if not key.startswith("_")}
        with open(config_path, "w") as f:
            json.dump(payload, f, indent=2)

