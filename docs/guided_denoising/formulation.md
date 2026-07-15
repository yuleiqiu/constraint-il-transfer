# Guided Denoising Formulation

## 1. Problem Setup

We first train a diffusion policy from an imitation-learning dataset:

$$
\mathcal{D}_{\mathrm{train}}
=
\{(o_i, A_i)\}_{i=1}^{N},
$$

where $o_i$ is an observation and $A_i$ is an action chunk. The learned policy
defines an action prior:

$$
p_{\theta}(A \mid o).
$$

At deployment time, the environment may provide additional information that was
not available during training. We represent this information by a cost:

$$
C\bigl(F(o,A), e^{\star}\bigr),
$$

where $F$ maps a candidate action chunk to the space in which the deployment
condition is evaluated. For example, $F$ may predict the physical trajectory
induced by the action chunk. Lower cost indicates better agreement with the
deployment-time condition.

## 2. Guided Target Distribution

Instead of retraining the policy, we use the deployment cost to reweight the
imitation prior:

$$
\boxed{
p_{\mathrm{guided}}(A \mid o,e^{\star})
\propto
p_{\theta}(A \mid o)
\exp\left(
-\lambda C\bigl(F(o,A),e^{\star}\bigr)
\right)
}
$$

The first term preserves actions preferred by the imitation policy, while the
cost term favors actions compatible with the deployment environment. The
guidance strength $\lambda$ controls this trade-off.

The corresponding clean-action score has the conceptual form

$$
\nabla_A \log p_{\mathrm{guided}}
=
\nabla_A \log p_{\theta}(A \mid o)
-
\lambda \nabla_A C\bigl(F(o,A),e^{\star}\bigr).
$$

## 3. Guidance During Denoising

Let $A_t$ denote the noisy action chunk at diffusion step $t$, and let
$\hat{A}_0^{(t)}$ be the policy's current prediction of the clean action:

$$
\hat{A}_0^{(t)}
=
\operatorname{Pred}_{\theta}(A_t,o,t).
$$

We evaluate the deployment cost on this predicted clean action and
differentiate it through the denoising prediction:

$$
g_t
=
\nabla_{A_t}
C\left(F\left(o,\hat{A}_0^{(t)}\right),e^{\star}\right).
$$

A guided denoising step can then be written schematically as

$$
\boxed{
A_{t-1}
=
\operatorname{DenoiseStep}_{\theta}(A_t,o,t)
-
\eta_t g_t
}
$$

where $\eta_t$ is a time-dependent guidance scale. Starting from
$A_T \sim \mathcal{N}(0,I)$, this update is repeated until a clean action chunk
$A_0$ is obtained.

This process should be understood as an approximate sampler for the guided
target distribution, rather than as a separate optimization performed after
diffusion sampling.

## 4. Receding-Horizon Execution

At each control cycle, the policy:

1. observes the current environment;
2. generates an action chunk through guided denoising;
3. executes only the first action or a short prefix; and
4. replans from the next observation.

This allows the guidance signal to be updated as the environment changes.

## 5. Key Assumptions

The formulation relies on three main assumptions:

- the deployment cost gives a meaningful preference over candidate actions;
- the map $F$ is sufficiently accurate in the region visited during denoising;
- the guidance schedule is strong enough to affect sampling without destroying
  the imitation prior.

In particular, inaccurate action-to-trajectory mapping produces inaccurate
guidance gradients even when the cost itself is well designed. A prediction
space that is directly executable and directly constrained can reduce this
source of error.

## 6. Research Direction

The central question is:

> Can a pretrained diffusion imitation policy adapt to new deployment-time
> conditions by modifying only its denoising process?

This formulation treats the deployment signal generically as a cost, without
requiring a predefined taxonomy of failure modes.
