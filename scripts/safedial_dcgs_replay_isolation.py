"""Opt-in replay isolation for source-pinned SafeDial main-policy runs.

The original replay returns references to journal results. Upstream belief
retries replace entries in the returned texts list, modifying the evidence being
audited. Give each replay a private trace; retain every original validation.
This module does not change generation, saved traces, manifests, or source pins.
"""
import copy
from contextlib import contextmanager
from unittest.mock import patch

import safedial_dcgs_wildjailbreak as adapter
import safedial_dcgs_run_state as state


@contextmanager
def isolated_replay():
    original = adapter.replay_trace

    def replay(row, turn, seed, config, tokenizer, events, allow_incomplete=False):
        return original(row, turn, seed, config, tokenizer,
                        copy.deepcopy(events), allow_incomplete=allow_incomplete)

    # validate_turn resolves adapter.replay_trace; failure/interruption audits
    # use the reference imported into the run-state module.
    with patch.object(adapter, "replay_trace", replay), patch.object(state, "replay_trace", replay):
        yield
