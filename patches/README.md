# patches/

`catk_action_hooks.patch` modifies [NVlabs/catk](https://github.com/NVlabs/catk) at commit `d23886761fc5b5628c5973148c40284452745745` (`src/smart/modules/agent_decoder.py`). It moves the two action-specific steps of `SMARTAgentDecoder.inference()` into overridable methods, `sample_next_action()` and `embed_next_action()`, so a different action head can plug into the unchanged rollout loop. `just clone` applies it.

The patch contains catk code and is a derivative work of catk. It is distributed under the NVIDIA Source Code License for CAT-K, included here as [`LICENSE-catk`](LICENSE-catk), which limits use to non-commercial research or evaluation. This notice does not change the terms of the rest of the repository.
