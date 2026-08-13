# One-Page Results Summary

## Experiment purpose
Compare different KL-control strategies during RL fine-tuning of `Qwen2.5-0.5B-Instruct` on the MATH500-style validation split. The goal was to see whether adaptive β control improves final validation correctness over fixed or zero KL regularization.

## Approach
- Single-seed campaign using the same training setup for every run.
- Sequential comparison of 5 controller conditions.
- All runs used the same model, dataset, 3 epochs, and configuration except for the KL controller.
- Each controller produced a final test score on the same validation split.

## Controller conditions
- `zero_kl`: no KL regularization (β = 0).
- `fixed`: constant KL coefficient, β = 0.001.
- `rule`: heuristic adaptive KL controller that increases β when KL is above target and decreases it when KL is below target.
- `mlp`: learned stateless neural controller that predicts β from the current state features.
- `lstm`: learned recurrent neural controller that predicts β using past state history.

## Role of β
β is the coefficient for the KL divergence penalty in the actor loss:

`loss = policy_gradient_loss + β · KL(policy ‖ reference)`

- A larger β keeps the policy closer to the reference model and prevents drift.
- A smaller β lets the policy explore more but risks collapse or degenerate behavior.
- Adaptive β is intended to balance stability and learning dynamically during training.

## Basic results
- `rule`: **0.34879**
- `mlp`: **0.342742**
- `lstm`: **0.334677** (evaluated from a 96% completed checkpoint after an OOM crash near the end)
- `fixed`: **0.324597**
- `zero_kl`: **0.3246**

## Key conclusion
- Adaptive KL control improved performance over fixed and zero KL.
- The rule-based adaptive controller achieved the highest score.
- The MLP learned controller also improved over fixed β.
- The LSTM controller was lower in this run and was affected by a runtime OOM near the final step.
- β acts as the critical regularization strength that controls the trade-off between staying close to the reference and allowing policy improvement.

## Notes
- The LSTM result is from a checkpoint at step 1500 because the training container crashed near step 1527 due to CUDA out-of-memory.
- The fixed and zero-KL scores are essentially tied, indicating the chosen constant β was too weak to provide meaningful extra regularization.
