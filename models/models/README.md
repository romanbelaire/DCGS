# Two-head LL token critic

CARES: `models/cares_ll_token_critic/ll_token_critic.pt`  
WildJailbreak: `models/wildjailbreak_ll_token_critic/ll_token_critic.pt`


```python
from src.value.ll_token_critic import LLTokenCritic

critic = LLTokenCritic.from_checkpoint(
    "../models/cares_ll_token_critic/ll_token_critic.pt",
    device="cuda:0",
    backbone=zephyr,
)
scores = critic.score_actions(observation, selected_belief, candidate_replies)
best = max(scores, key=scores.get)
```
