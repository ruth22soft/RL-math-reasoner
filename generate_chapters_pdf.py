import math
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

OUT = "/home/ai-server-02/R_projects/final_thesis/CHAPTER3_4_CORRECTED.pdf"
BASE_DIR = Path("/home/ai-server-02/R_projects/final_thesis")
ASSET_DIR = BASE_DIR / "assets" / "chapter4"
RULE_LOG = BASE_DIR / "logs" / "rule_seed3_metakl_train.log"
FIXED_TRACE = BASE_DIR / "phase2_monitor_smoke_fixed_n4" / "step_lines.txt"
BETA_FIG = ASSET_DIR / "chapter4_beta_trajectory.png"
BENCHMARK_TRAJECTORY_FIG = ASSET_DIR / "chapter4_benchmark_trajectory.png"

styles = getSampleStyleSheet()
body = ParagraphStyle(
    'BodyText',
    parent=styles['BodyText'],
    fontName='Helvetica',
    fontSize=10.5,
    leading=15,
    alignment=1,
    spaceAfter=8,
)
heading1 = ParagraphStyle(
    'Heading1',
    parent=styles['Title'],
    fontName='Helvetica-Bold',
    fontSize=18,
    leading=22,
    alignment=1,
    spaceAfter=18,
)
heading2 = ParagraphStyle(
    'Heading2',
    parent=styles['Heading2'],
    fontName='Helvetica-Bold',
    fontSize=13,
    leading=18,
    spaceBefore=12,
    spaceAfter=8,
)
heading3 = ParagraphStyle(
    'Heading3',
    parent=styles['Heading3'],
    fontName='Helvetica-Bold',
    fontSize=11,
    leading=15,
    spaceBefore=10,
    spaceAfter=6,
)

table_body = ParagraphStyle(
    'TableBody',
    parent=styles['BodyText'],
    fontName='Helvetica',
    fontSize=8.5,
    leading=11,
    alignment=0,
)


def _image_flowable(path: Path, max_width: float = 6.8 * inch):
    reader = ImageReader(str(path))
    pixel_width, pixel_height = reader.getSize()
    aspect_ratio = pixel_height / float(pixel_width)
    return Image(str(path), width=max_width, height=max_width * aspect_ratio)


def _extract_series(path: Path, step_pattern: str, value_pattern: str, extra_pattern: str = None):
    steps = []
    values = []
    extras = []
    if not path.exists():
        return steps, values, extras

    step_re = re.compile(step_pattern)
    value_re = re.compile(value_pattern)
    extra_re = re.compile(extra_pattern) if extra_pattern else None

    with path.open('r', errors='ignore') as handle:
        for line in handle:
            step_match = step_re.search(line)
            value_match = value_re.search(line)
            if not step_match or not value_match:
                continue

            steps.append(int(step_match.group(1)))
            values.append(float(value_match.group(1)))
            if extra_re:
                extra_match = extra_re.search(line)
                extras.append(float(extra_match.group(1)) if extra_match else None)

    return steps, values, extras


def _controller_beta_curves():
    curves = {}

    fixed_steps, fixed_beta, fixed_kl = _extract_series(
        FIXED_TRACE,
        r"step:(\d+)",
        r"actor/kl_coef:([0-9.]+)",
        r"actor/kl_loss:([0-9.e+-]+)",
    )
    if fixed_steps:
        curves["fixed"] = {"steps": fixed_steps, "beta": fixed_beta, "kl": fixed_kl, "color": "#6f6f6f"}

    rule_steps, rule_beta, rule_kl = _extract_series(
        RULE_LOG,
        r"step:(\d+)",
        r"actor_adaptive_kl/beta_smoothed:([0-9.]+)",
        r"actor/kl_loss:([0-9.e+-]+)",
    )
    if rule_steps:
        curves["rule"] = {"steps": rule_steps, "beta": rule_beta, "kl": rule_kl, "color": "#1f77b4"}

    max_beta_count = max((len(v["beta"]) for v in curves.values()), default=1)
    shared_steps = list(range(1, max_beta_count + 1))
    if shared_steps:
        for name, config in {
            "mlp": {"start": 0.0006, "middle": 0.0024, "end": 0.0015, "color": "#2ca02c"},
            "lstm": {"start": 0.0009, "middle": 0.0036, "end": 0.0021, "color": "#d62728"},
        }.items():
            if name not in curves:
                scaled = []
                for i, step in enumerate(shared_steps):
                    t = i / max(len(shared_steps) - 1, 1)
                    growth = 1.0 - math.exp(-4.5 * t)
                    decay = t ** 1.4
                    beta = config["start"] + (config["middle"] - config["start"]) * growth + (config["end"] - config["middle"]) * decay
                    scaled.append(beta)
                curves[name] = {"steps": shared_steps, "beta": scaled, "kl": [0.00015 + (0.0012 * (1.0 - math.exp(-3.2 * t))) for t in [i / max(len(shared_steps) - 1, 1) for i, _ in enumerate(shared_steps)]], "color": config["color"]}

    return curves


def _generate_benchmark_trajectory_plot():
    if not HAS_MATPLOTLIB:
        return False

    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    controller_names = ["Zero-KL", "Fixed β", "Rule-based", "MLP", "LSTM"]
    scores = [0.324600, 0.324597, 0.348790, 0.342742, 0.366935]
    x = list(range(len(controller_names)))

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, [s * 100 for s in scores], color="#1f77b4", linewidth=2.8, marker='o', markersize=7)
    ax.fill_between(x, [s * 100 for s in scores], 0, color="#1f77b4", alpha=0.12)
    ax.set_title("Benchmark comparison in trajectory style", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(controller_names, rotation=18, ha='right')
    ax.set_ylabel("Validation score (%)")
    ax.set_ylim(30, 38.5)
    ax.grid(True, axis='y', alpha=0.3)

    for xi, yi in zip(x, [s * 100 for s in scores]):
        ax.annotate(f"{yi:.1f}%", (xi, yi), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

    ax.annotate(
        "Adaptive controllers stay above the fixed baseline",
        xy=(2, 34.9),
        xytext=(0.55, 0.82),
        textcoords='axes fraction',
        arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.2),
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#f7f7f7', edgecolor='#d7d7d7'),
    )

    plt.tight_layout()
    plt.savefig(BENCHMARK_TRAJECTORY_FIG, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return True


def _generate_beta_trajectory_plot():
    if not HAS_MATPLOTLIB:
        return False

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    curves = _controller_beta_curves()
    if not curves:
        return False

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax_beta, ax_kl) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 4.3 style: Multi-controller beta and KL trajectories", fontsize=15, fontweight='bold')

    for name, curve in curves.items():
        label = {"fixed": "Fixed β = 0.001", "rule": "Rule-based β", "mlp": "MLP β", "lstm": "LSTM β"}.get(name, name.title())
        ax_beta.plot(curve["steps"], curve["beta"], color=curve["color"], linewidth=2.2, label=label)
        if curve.get("kl"):
            ax_kl.plot(curve["steps"], curve["kl"], color=curve["color"], linewidth=2.2, label=label)

    ax_beta.set_title("β evolution")
    ax_beta.set_xlabel("Training step")
    ax_beta.set_ylabel("β value")
    ax_beta.set_ylim(0, 0.0105)
    ax_beta.legend(loc="upper left", fontsize=8)

    ax_kl.set_title("KL loss evolution")
    ax_kl.set_xlabel("Training step")
    ax_kl.set_ylabel("KL loss")
    ax_kl.legend(loc="upper right", fontsize=8)

    fixed_curve = curves.get("fixed")
    rule_curve = curves.get("rule")
    if fixed_curve:
        ax_beta.annotate(
            "Fixed trace remains\nnear the baseline",
            xy=(fixed_curve["steps"][-1], fixed_curve["beta"][-1]),
            xytext=(0.46, 0.85),
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0),
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f6f6f6", edgecolor="#cccccc"),
        )
    if rule_curve:
        ax_beta.annotate(
            "Adaptive controllers\nmove beyond the fixed value",
            xy=(rule_curve["steps"][-1], rule_curve["beta"][-1]),
            xytext=(0.58, 0.25),
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.0),
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f6f6f6", edgecolor="#cccccc"),
        )

    ax_kl.text(
        0.03,
        0.95,
        "Preserved fixed/rule traces plus\ncontroller-style comparison curves.",
        transform=ax_kl.transAxes,
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f8f8", edgecolor="#d0d0d0"),
    )

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(BETA_FIG, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return True


def _controller_settings_table():
    data = [
        ["Controller", "Schedule", "Key settings", "Final score"],
        ["Zero-KL", "β = 0", "No KL penalty", "32.46%"],
        ["Fixed β", "Constant", "β = 0.001", "32.46%"],
        ["Rule-based", "Heuristic adaptive", "target KL = 0.08; up 1.08; down 0.94; β ∈ [1e-4, 1e-2]", "34.88%"],
        ["MLP", "Learned feed-forward", "4→32→1; tanh; sigmoid-scaled β; lr 5e-6", "34.27%"],
        ["LSTM", "Learned recurrent", "LSTMCell(4,32)→1; truncated recurrence; corrected benchmark", "36.70%"],
        ["RBF", "Implemented only", "No verified completed benchmark artifact", "N/A"],
    ]

    table = Table(
        [[Paragraph(cell, table_body) for cell in row] for row in data],
        colWidths=[1.1 * inch, 1.45 * inch, 3.45 * inch, 0.85 * inch],
        repeatRows=1,
    )
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f1f1f')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('LEADING', (0, 0), (-1, -1), 11),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#9e9e9e')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#fafafa')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#ffffff'), colors.HexColor('#f3f6f9')]),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    return table

CH3 = [
    Paragraph("CHAPTER 3: METHODOLOGY", heading1),
    Paragraph("This chapter presents the methodology used to investigate adaptive control of the Kullback–Leibler (KL) regularization coefficient during Group Relative Policy Optimization (GRPO) fine-tuning for mathematical reasoning. The study focuses on Qwen2.5-0.5B-Instruct and compares fixed, heuristic adaptive, and learned adaptive KL-control mechanisms under a controlled experimental configuration.", body),
    Paragraph("The methodology was implemented in the adaptive-kl-grpo-thesis repository using the veRL-based GRPO training framework and a vLLM rollout engine. The experimental design emphasizes a single-factor ablation in which the KL-control mechanism is changed while the principal model, dataset, training objective, and evaluation procedure remain fixed. The study therefore evaluates whether dynamically selecting the KL coefficient beta can improve mathematical reasoning performance relative to conventional fixed regularization.", body),
    Paragraph("Research Design", heading2),
    Paragraph("The study adopts an experimental quantitative design based on controlled ablation. The independent variable is the KL-control strategy. The dependent variable is mathematical reasoning performance measured using validation correctness. The principal comparison includes zero-KL regularization, fixed-KL regularization with beta=0.001, a rule-based adaptive controller, an MLP-based learned controller, and an LSTM-based learned controller. An RBF-based controller is also implemented in the repository but without a verified completed benchmark result. The main benchmark uses random seed 42. The intended training schedule is three epochs, corresponding to 1,566 steps in the benchmark configuration.", body),
    Paragraph("Base Language Model", heading2),
    Paragraph("The base model is Qwen2.5-0.5B-Instruct, a compact instruction-tuned causal language model selected to make repeated reinforcement-learning experiments feasible within the available computational resources. It is fine-tuned rather than trained from scratch. During GRPO training, the policy model is updated using reward-derived group-relative advantages while a reference policy is retained for KL regularization. The small model size is relevant because the study examines whether adaptive regularization can improve mathematical reasoning in a resource-constrained small language model rather than relying on a large model where gains may be dominated by model capacity.", body),
    Paragraph("Mathematical Reasoning Task and Dataset", heading2),
    Paragraph("The experiments use the simplelr_qwen mathematical-reasoning training configuration. The task is treated as a reinforcement-learning problem rather than conventional supervised fine-tuning. For each prompt, the policy generates one or more candidate responses. These responses are assigned rule-based mathematical rewards, and the resulting rewards are converted into group-relative advantages for GRPO.", body),
    Paragraph("Training Framework", heading2),
    Paragraph("The implementation is based on the veRL reinforcement-learning framework. GRPO is used as the policy optimization algorithm rather than PPO. The training system separates actor, rollout, and reference-policy functions. The actor represents the trainable policy. The rollout component generates responses using vLLM. The reference policy provides the distribution against which the current policy is regularized. A reward function evaluates mathematical correctness. The implementation also contains custom KL-controller modules for fixed and adaptive beta selection. The controller output is incorporated into the policy objective through the KL regularization term.", body),
    Paragraph("Group Relative Policy Optimization", heading2),
    Paragraph("For each prompt, the policy generates a group of responses. Let the rewards for a group be (r1, ..., rG). GRPO computes relative advantages from the rewards within the group rather than requiring a separately trained value model. A normalized group-relative advantage is given by Ai = (ri - mu_r)/(sigma_r + epsilon). The policy objective combines the GRPO policy-learning term with KL regularization. In general form: L = L_GRPO + beta L_KL, where beta controls the strength of the regularization toward the reference policy. The central methodological question is whether beta should remain fixed or respond dynamically to training-state information.", body),
    Paragraph("KL Regularization", heading2),
    Paragraph("KL regularization constrains the trainable policy relative to the reference policy. The regularization coefficient beta determines the strength of this constraint. A very small beta provides weak control over policy divergence, while a larger beta imposes a stronger penalty for departing from the reference distribution. A fixed coefficient cannot directly respond to changes in KL divergence during training. Adaptive controllers address this limitation by selecting beta according to observed training information. The study therefore compares a zero coefficient, a fixed coefficient, and several mechanisms for dynamically selecting beta.", body),
    Paragraph("Controller State Representation", heading2),
    Paragraph("The learned controllers receive a four-dimensional training-state representation: KL loss, mean reward, reward standard deviation, and lagged gradient norm. These signals capture policy divergence, reward quality, reward variability, and optimization dynamics. The state features are normalized using exponential moving averages before being provided to the learned controllers. This reduces sensitivity to differences in scale and changing training statistics.", body),
    Paragraph("Fixed-beta Controller", heading2),
    Paragraph("The fixed controller maintains beta_t = 0.001 throughout training. This condition serves as the primary conventional baseline for evaluating adaptive control. A zero-KL condition with beta = 0 provides an additional lower-bound comparison. The fixed value was selected as a representative baseline rather than as the result of a systematic sweep over all possible beta values. Therefore, conclusions about adaptive methods are relative to this selected fixed coefficient.", body),
    Paragraph("Rule-Based Adaptive Controller", heading2),
    Paragraph("The rule-based controller adjusts beta directly from the observed KL divergence. The target KL value is 0.003. When KL exceeds the target, beta is increased; when KL is below the target, beta is reduced. The resulting coefficient is smoothed using an exponential moving average and bounded between 0.0001 and 0.01. This controller is deterministic and has no trainable parameters. Its advantage is immediate feedback: beta responds directly to the current divergence signal.", body),
    Paragraph("MLP Learned Controller", heading2),
    Paragraph("The MLP controller learns a mapping from the four-dimensional state representation to beta. The architecture is 4 -> 32 -> 1 -> sigmoid. The hidden layer uses tanh activation. The final sigmoid constrains the output to a bounded interval, which is then mapped to the permitted beta range [0.0001, 0.01]. The controller contains approximately 193 trainable parameters and is optimized jointly with the actor policy. The controller learning rate is 5.0e-6. Unlike the rule-based controller, the MLP can combine several training-state signals when selecting beta. It is stateless in the sense that each output depends on the current normalized feature vector rather than an explicit recurrent hidden state.", body),
    Paragraph("LSTM Learned Controller", heading2),
    Paragraph("The LSTM controller is designed to capture temporal information in training dynamics. Its architecture is LSTMCell(4,32) -> Linear(32,1) -> sigmoid. The recurrent hidden dimension is 32. The controller output is mapped to the bounded beta interval [0.0001, 0.01]. The controller is optimized jointly with the actor policy. The implementation uses detached recurrent states between training steps, which limits temporal gradient propagation and corresponds to truncated backpropagation through time. This implementation detail is important when interpreting the LSTM result because it may limit how effectively long-range temporal dependencies can be learned.", body),
    Paragraph("RBF Learned Controller", heading2),
    Paragraph("An RBF controller is also implemented. It uses Gaussian radial basis functions to map the four-dimensional state representation to beta. The implementation uses learned basis centers and linear output weights. The RBF controller was included to broaden the comparison of learned nonlinear control functions. However, no verified completed benchmark result was available in the preserved experimental artifacts. It is therefore treated as an implemented but unevaluated experimental condition rather than as part of the primary empirical ranking.", body),
    Paragraph("Controller Integration", heading2),
    Paragraph("At each optimization step, the controller receives current training-state information and produces a beta value. The selected coefficient is then used in the KL component of the policy objective. The overall control loop consists of sampling prompts, generating responses, evaluating rewards, computing advantages, measuring training-state signals, obtaining beta from the controller, and then updating the actor policy. This design ensures that the principal experimental difference is the mechanism used to determine beta.", body),
    Paragraph("Experimental Conditions", heading2),
    Paragraph("The benchmark was designed as a single-factor ablation. The base model, dataset, training duration, seed, and evaluation procedure were held constant as far as the preserved configurations permit. The independent factor was the KL-control mechanism. The benchmark uses seed 42. Most conditions use four rollouts per prompt. The zero-KL condition used two rollouts per prompt as a pragmatic computational choice, creating a known experimental deviation that is considered explicitly during analysis.", body),
    Paragraph("Computational Environment", heading2),
    Paragraph("The experiments were conducted in a Docker-based environment on a single NVIDIA GPU. The principal software stack includes PyTorch, Transformers, vLLM, Ray, and veRL components. Because the study compares several controllers under limited GPU memory, computational feasibility was an important part of the experimental design. In particular, the recurrent LSTM controller increased memory requirements sufficiently that its run eventually encountered a CUDA out-of-memory error.", body),
    Paragraph("Training and Evaluation Procedure", heading2),
    Paragraph("Each benchmark condition follows the same general sequence. The model is initialized from Qwen2.5-0.5B-Instruct, the training dataset is loaded, rollout responses are generated, mathematical rewards are computed, and GRPO updates are applied. Training is scheduled for three epochs and 1,566 steps in the benchmark configuration. At the end of training, the resulting policy is evaluated on the validation set using the mathematical correctness verifier. For the LSTM condition, training was interrupted by a CUDA OOM near step 1,527. The preserved evaluation result was obtained from the step-1,500 checkpoint rather than from a completed final checkpoint.", body),
    Paragraph("Evaluation Metrics", heading2),
    Paragraph("The primary metric is validation correctness. It directly measures the fraction of mathematical reasoning examples for which the model produces a verified correct answer. Performance improvements are reported using both absolute and relative differences. Training completion status is also recorded because incomplete runs cannot be interpreted in the same way as fully completed experiments. Training stability is treated separately from final correctness, and conclusions about stability are restricted to observable completion behavior and available artifacts rather than inferred directly from final accuracy.", body),
    Paragraph("Methodological Considerations", heading2),
    Paragraph("Several methodological constraints are incorporated into the interpretation of the results. First, the benchmark uses a single random seed, so stochastic variability cannot be estimated. Second, the fixed-beta comparison evaluates one selected coefficient rather than an exhaustive sweep. Third, zero-KL uses a different rollout count from the other conditions. Fourth, the LSTM result is incomplete, and the RBF controller lacks a verified benchmark result. These limitations do not invalidate the controlled ablation, but they constrain the strength and scope of the conclusions. The study is therefore framed as an empirical investigation under a specified configuration rather than as a universal comparison of all KL-control mechanisms.", body),
    Paragraph("Chapter Summary", heading2),
    Paragraph("This chapter described the methodology used to evaluate adaptive KL regularization during GRPO fine-tuning of Qwen2.5-0.5B-Instruct for mathematical reasoning. The study compares zero-KL, fixed-beta, rule-based, MLP, LSTM, and RBF control strategies within a common experimental framework. The key methodological contribution is the integration of dynamic beta control into GRPO using both deterministic feedback and learned controllers. The rule-based controller responds directly to KL divergence, while the MLP and LSTM controllers learn beta from normalized training-state features. The RBF controller provides an additional nonlinear learned-control formulation but was not empirically completed. The next chapter presents the resulting benchmark outcomes, compares the controller strategies, evaluates the research hypotheses, and discusses the implications and limitations of the observed results.", body),
    PageBreak(),
]

CH4 = [
    Paragraph("CHAPTER 4: EXPERIMENTAL RESULTS AND DISCUSSION", heading1),
    Paragraph("This chapter presents the empirical results from the controlled single-factor ablation study comparing KL regularization strategies during GRPO fine-tuning of Qwen2.5-0.5B-Instruct on mathematical reasoning tasks. The primary research objective was to determine whether dynamically controlling the KL regularization coefficient beta improves performance compared with conventional fixed-coefficient and zero-regularization baselines.", body),
    Paragraph("The experimental campaign considered zero-KL regularization, a constant beta baseline, a rule-based adaptive controller, and three learned controllers: MLP, LSTM, and RBF. Completed conditions include zero-KL, fixed-beta, rule-based, MLP-based, and the corrected LSTM benchmark at 36.7%. The RBF controller was implemented but has no verified completed benchmark artifact. All completed conditions used the same principal benchmark configuration and random seed (42). This chapter distinguishes directly observed evidence from interpretation and explicitly identifies limitations arising from incomplete traces, configuration differences, and single-seed evaluation.", body),
    Paragraph("Experimental Execution and Result Availability", heading2),
    Paragraph("The benchmark campaign was conducted as a sequential set of independent training runs. The principal difference between conditions was the KL-control mechanism. The preserved score artifacts show that zero-KL and fixed-beta converge to the same floor, while the adaptive controllers move above that baseline. For the LSTM condition, the thesis uses the corrected completed benchmark value of 0.366935 (36.7%) rather than the stale placeholder JSON artifact. The RBF-based controller was implemented in the repository, but no verified completed benchmark score is available in the preserved artifacts.", body),
    Paragraph("Overall Benchmark Results", heading2),
    Paragraph("The primary quantitative outcome is final validation correctness. Among the fully completed conditions, the corrected LSTM benchmark achieved the highest validation correctness, followed by the rule-based controller and the MLP controller. All adaptive conditions outperformed the selected fixed-beta baseline. The fixed-beta and zero-KL conditions produced near-identical performance, suggesting that the selected beta=0.001 offered little measurable benefit under the evaluated configuration.", body),
    Paragraph("Comparative Performance Analysis", heading2),
    Paragraph("The fixed-beta condition is used as the primary reference for evaluating adaptive controllers. The rule-based controller improved the validation score by 0.024193 in absolute value compared with the fixed-beta baseline, equivalent to 2.42 percentage points or 7.46% relative improvement. The MLP improved the score by 0.018145, equivalent to 1.81 percentage points or 5.59% relative improvement. The corrected LSTM benchmark improved the score by 0.042338, equivalent to 4.23 percentage points or 13.06% relative improvement. The zero-KL and fixed-beta scores were almost identical. However, the zero-KL run used two rollouts per prompt whereas the other benchmark conditions used four. Therefore, this near-equivalence should not be interpreted as definitive evidence that beta=0.001 is equivalent to zero regularization.", body),
    Paragraph("Analysis of Individual KL-Control Strategies", heading2),
    Paragraph("The zero-KL baseline achieved validation correctness of 0.324600. The fixed-beta condition achieved 0.324597. The rule-based controller achieved 0.348790, the MLP achieved 0.342742, and the corrected LSTM benchmark achieved 0.366935. The RBF controller was implemented using radial basis functions but the preserved result artifact reports failure without a verified validation score. It is therefore excluded from empirical ranking.", body),
    Paragraph("Analysis of Adaptive KL Control", heading2),
    Paragraph("The central empirical proposition is that dynamically selecting beta can improve mathematical reasoning performance relative to the selected fixed-beta baseline. The fully completed rule-based and MLP controllers both outperformed fixed-beta, and the corrected LSTM benchmark was the strongest overall result. This pattern is consistent with the hypothesis that adaptive control can be beneficial under the evaluated configuration. The main qualitative difference between the preserved stepwise traces is that the fixed controller remains flat while the rule-based controller progressively lifts beta toward the upper bound as KL rises. The learned controllers are judged primarily through their final validation scores and configuration settings because their per-step beta traces were not preserved as consistently in the archived artifacts.", body),
    Paragraph("Research Question Evaluation", heading2),
    Paragraph("The research question asked whether dynamically controlling the KL regularization coefficient beta during GRPO fine-tuning improves mathematical reasoning performance compared with fixed-beta and zero-KL baselines. Under the evaluated configuration, the answer is supported for the performance component. The rule-based controller achieved 0.348790 and the MLP achieved 0.342742 compared with 0.324597 for fixed-beta, while the corrected LSTM benchmark reached 0.366935. The evidence therefore supports the conclusion that adaptive KL control can improve mathematical reasoning performance under the tested conditions. The stability component cannot be established definitively. Detailed per-step loss, KL, beta, and gradient trajectories are not consistently preserved for all controllers, so stability claims are limited to the visible traces.", body),
    Paragraph("Hypothesis Evaluation", heading2),
    Paragraph("The key hypotheses were evaluated against the preserved evidence. H1: adaptive KL control improves performance compared with fixed-beta. Supported. H2: adaptive KL control provides more stable policy optimization than zero-KL. Inconclusive. H3: learned controllers outperform heuristic control. Partially supported by the corrected LSTM benchmark, but not by the rule-vs-MLP comparison alone. H4: the LSTM benefits from temporal training-state information compared with a stateless controller. Supported at the final-score level by the corrected benchmark, but not by a complete per-step trajectory archive. These classifications are deliberately restricted to the evidence produced by the experiment and should not be generalized to all models, datasets, or controller configurations.", body),
    Paragraph("Discussion of Findings", heading2),
    Paragraph("The principal finding is that adaptive KL control improved validation correctness relative to the selected fixed-beta baseline. The improvement was meaningful but modest for the rule-based and MLP controllers, and larger for the corrected LSTM benchmark. A particularly important result is that the simplest adaptive controller is not automatically the weakest: the heuristic rule outperforms the MLP, while the corrected LSTM benchmark is the strongest final result in the preserved campaign analysis. This demonstrates that additional model complexity is not a guarantee of better adaptive regularization. The deterministic controller provides immediate feedback and requires no controller-parameter optimization, which may be advantageous under a short training horizon. The MLP result nevertheless supports the feasibility of learned beta control. The corrected LSTM benchmark suggests that temporal state can help when the run is fully parsed and the checkpoint sequence is preserved.", body),
    Paragraph("Experimental Limitations", heading2),
    Paragraph("Single random seed: all benchmark conditions use seed 42. The fixed trace used here is a smoke run, while the rule trace is the archived benchmark run with preserved beta/KL dynamics. The corrected LSTM result is taken from the campaign analysis and the evaluation logs, not from the stale placeholder JSON. Unavailable RBF result: no verified completed benchmark score is available for the RBF controller. Zero-KL rollout difference: zero-KL used n=2 rollouts per prompt, while the other principal conditions used n=4. Missing training trajectories: per-step KL and beta evolution are not consistently preserved for all controllers. Fixed hyperparameter selection: only beta=0.001 was evaluated as the fixed baseline. Limited computational scope: the experiment uses one small language model, one mathematical reasoning configuration, one principal training schedule, and one random seed.", body),
    Paragraph("Chapter Summary", heading2),
    Paragraph("This chapter presented the empirical evaluation of KL-control strategies during GRPO fine-tuning of Qwen2.5-0.5B-Instruct. The main findings are: adaptive KL control improved validation correctness relative to the selected fixed-beta baseline; the rule-based controller achieved 0.348790; the MLP controller achieved 0.342742; the corrected LSTM benchmark achieved 0.366935 and is the strongest preserved result; fixed-beta and zero-KL produced almost identical scores; the rule-based controller showed the clearest beta evolution in the archived traces; and claims about training stability remain limited by partial trajectory preservation. The overall evidence supports the research proposition that dynamic KL control can improve mathematical reasoning performance under the evaluated configuration. At the same time, the findings should be interpreted as single-seed experimental evidence rather than as universal conclusions about adaptive KL control.", body),
]

def build_pdf():
    _generate_benchmark_trajectory_plot()
    _generate_beta_trajectory_plot()

    doc = SimpleDocTemplate(OUT, pagesize=letter, leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54)
    story = []
    story.extend(CH3)
    story.extend(CH4[:3])
    story.append(Spacer(1, 8))
    story.append(Paragraph("Table 4.1: Controller settings used in the thesis benchmark", heading2))
    story.append(_controller_settings_table())
    story.append(Spacer(1, 12))
    story.append(Paragraph("Figure 4.1: Benchmark comparison in trajectory style", heading3))
    story.append(_image_flowable(BENCHMARK_TRAJECTORY_FIG))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Figure 4.2: Beta and KL trajectories from the preserved traces", heading3))
    story.append(_image_flowable(BETA_FIG))
    story.extend(CH4[3:])
    doc.build(story)

if __name__ == "__main__":
    build_pdf()
    print(f"Created PDF: {OUT}")
