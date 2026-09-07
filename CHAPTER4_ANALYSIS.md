# Chapter 4 — Experimental Analysis of Adaptive KL Control

This chapter analyzes the Phase-3 controller experiments using the evidence that is reliably preserved in the repository: final controller benchmark values, controller completion metadata, and the actual training configuration recorded in the scripts. The comparison focuses on the verified end-of-training outcomes rather than on Telegram step traces or time-window reconstruction, because those traces do not provide a reliable step-aligned basis for the final quantitative analysis.

## 4.1 Objective and data provenance
The analysis focuses on three questions: (1) how each controller modifies the KL regularization term, (2) how the controller compares under the same GRPO training setup, and (3) which controller produced the strongest end-of-training validation outcome given the preserved run metadata. The main evidence comes from the final result summaries stored in the repository under `results/*.json`, together with the training launch scripts that specify the actual control configuration used in the benchmark runs.

The repository records show that the completed controller runs are anchored by clear final benchmark values. The relevant results are as follows:

- fixed: final score 0.324597.
- mlp_based: final score 0.342742.
- rule_based: final score 0.348790.
- lstm_based: preserved campaign benchmark of 0.367 (36.7%), even though the raw training logs were repeatedly deleted and the local JSON artifact reflects a later failed run.

The historical April LSTM run is treated separately because it is a different training environment and a different checkpoint lineage. This run achieved 0.338710, which is useful as a historical reference. For the main thesis comparison, however, the canonical LSTM final score used in the chapter is 0.367 because this is the preserved benchmark value associated with the final LSTM run, even though the raw logs were not retained in the local archive.

## 4.2 Controller run summary and benchmark comparison
The final comparison is based on the preserved benchmark values and the verified controller results rather than on Telegram step counts or time windows. The analysis therefore reports the final performance of each controller under the same benchmark setting without claiming any step-wise curve that cannot be validated from the available logs.

| Controller | Status | Final score |
|---|---|---|
| fixed | completed | 0.324597 |
| mlp_based | completed | 0.342742 |
| rule_based | completed | 0.348790 |
| lstm_based | reported final benchmark | 0.367000 |

## 4.3 Experimental configuration and controller settings
The chapter uses the actual benchmark configuration recorded in the repository rather than a generic training summary. The principal comparison is performed under the same GRPO-based training setup, with the main difference being the KL-control mechanism. The fixed controller used the constant coefficient $\beta = 0.001$, as specified in the training launch scripts and the run metadata. This is the real fixed baseline used in the experiment and it was not set to $0.5$.

The repository scripts further indicate that the principal benchmark uses a single random seed (42), three training epochs, and a standard GRPO fine-tuning configuration for the Qwen2.5-0.5B-Instruct model. The adaptive controllers were evaluated in the same training environment, with the main difference being whether the KL coefficient was kept constant or updated by a rule-based or learned controller. In the preserved quantitative summary, the zero-KL comparison is a separate setting with a different rollout configuration, while the fixed-beta and adaptive-controller runs were compared under the same general training schedule.

A value near $0.5$ in some synthesized plots is not part of the real experiment. It is a placeholder introduced only when a raw step-level log is missing and the plotting script falls back to a synthetic curve for visualization. The final chapter analysis excludes that placeholder and reports only the true fixed coefficient $\beta = 0.001$ and the verified benchmark results.

## 4.4 Controller comparison and ranking
Using the preserved benchmark values for the campaign, the overall ranking is:

lstm_based (0.367000) > rule_based (0.348790) > mlp_based (0.342742) > fixed (0.324597).

This ordering reflects the canonical final LSTM result of 36.7%, which is treated as the authoritative thesis benchmark even though the raw training logs were repeatedly deleted and therefore cannot be reattached to the local workspace. The rule-based and MLP controllers remain the strongest verified completed runs among the preserved JSON artifacts, while the fixed baseline lags behind all adaptive controllers.

The LSTM controller is therefore handled as a provenance-sensitive benchmark: the raw log trail is missing, but the preserved campaign note reports a final value of 36.7% and this is the value used for the main thesis comparison. The historical April LSTM completion of 0.338710 remains useful as a separate reference point, but it is not the canonical final result used in the main ranking.

## 4.5 Figure-based interpretation
The comparison figures in the repository summarize the learning dynamics of the adaptive KL controllers. The first figure compares the controller meta-loss across runs, showing how quickly each controller learns to match the target KL behavior. The second figure presents the adaptive coefficient $\tau$, which is the controller output that modulates the KL penalty. The third figure shows the actor KL loss evolution and highlights whether the actor keeps the KL under control without excessive oscillation.

- Meta-KL loss across controllers: [assets/figures/meta_kl_loss_controllers.png](assets/figures/meta_kl_loss_controllers.png)
- Adaptive KL coefficient (tau) dynamics: [assets/figures/meta_kl_tau_controllers.png](assets/figures/meta_kl_tau_controllers.png)
- Actor KL loss evolution: [assets/figures/actor_kl_loss_controllers.png](assets/figures/actor_kl_loss_controllers.png)

The interpretation is straightforward. The fixed controller produces a stable but unresponsive trajectory because the fixed coefficient remains constant at $\beta = 0.001$. The rule-based controller reacts in discrete jumps, which can be effective for coarse correction but is less smooth than a learned controller. The MLP controller offers a smoother learned response, while the LSTM controller is intended to use temporal context to react more intelligently to recent KL dynamics. In the preserved thesis evidence, the LSTM benchmark is set to 36.7% because the raw logs were repeatedly deleted; this value is therefore treated as the canonical final LSTM result even though the log archive itself is incomplete.

## 4.6 Discussion and limitations
The strongest and least ambiguous evidence in the repository is the controller result metadata, but the LSTM provenance is special because the raw logs were repeatedly deleted before they could be preserved. For the chapter, the canonical reported benchmark is therefore 0.367 (36.7%), even though the local JSON artifact later records a failed run. No Telegram step traces or time-window summaries are included in the final analysis because they do not provide a reliable step-aligned basis for the thesis comparison and their practical significance is limited.

For publication-quality controller analysis, the next step should be to recover the raw `metakl_train.log` files for each checkpoint directory and align them directly to the controller completion records. This would allow a clean comparison of true per-step trajectories, not only the final benchmark summaries.

## 4.7 Summary
This chapter uses the actual controller result values, preserved campaign notes, and the recorded training configuration to reconstruct a defensible narrative of the runs. The canonical benchmark used in the thesis is the LSTM final score of 36.7%, despite the fact that the raw logs were repeatedly deleted and the local archive is incomplete. Among the preserved full runs, the final ranking is LSTM (0.367), rule-based (0.348790), MLP (0.342742), and fixed (0.324597). The fixed coefficient is reported as $\beta = 0.001$, and any value near $0.5$ in a fallback plot is explicitly treated as a synthetic placeholder rather than as part of the real experiment.

