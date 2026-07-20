# CAPE: Context-Aware Diffusion Policy Via Proximal Mode Expansion for Collision Avoidance

Rui Heng Yang<sup>∗†</sup>, Xuan Zhao<sup>∗†</sup>, Leo Maxime Brunswic<sup>†</sup>, Montgomery Alban<sup>¶</sup> Mateo Clemente<sup>†</sup>, Tongtong Cao<sup>‡</sup>, Jun Jin<sup>§</sup>, Amir Rasouli<sup>†</sup>

Abstract— In robotics, diffusion models can capture multimodal trajectories from demonstrations, making them a transformative approach in imitation learning. However, achieving optimal performance following this regiment requires a largescale dataset, which is costly to obtain, especially for challenging tasks, such as collision avoidance. In those tasks, generalization at test time demands coverage of many obstacles types and their spatial configurations, which are impractical to acquire purely via data. Recent works ease this burden with trainingfree guidance by injecting environmental context at inference, however, it only works when paired with a sufficiently diverse training dataset that yields a conditional trajectory distribution with rich multimodal coverage. To remedy this problem, we propose Context-Aware diffusion policy via Proximal mode Expansion (CAPE), a framework that expands trajectory distribution modes with context-aware prior and guidance at inference via a novel prior-seeded iterative guided refinement procedure. The framework generates an initial trajectory plan and executes a short prefix trajectory, and then the remaining trajectory segment is perturbed to an intermediate noise level, forming a trajectory prior. Such a prior is context-aware and preserves task intent. Repeating the process with contextaware guided denoising iteratively expands mode support to allow finding smoother, less collision-prone trajectories. For collision avoidance, CAPE expands trajectory distribution modes with collision-aware context, enabling the sampling of collision-free trajectories in previously unseen environments while maintaining goal consistency. We evaluate CAPE on diverse manipulation tasks in cluttered unseen simulated and real-world settings and show up to 26% and 80% higher success rates respectively compared to SOTA methods, demonstrating better generalization to unseen environments.

## I. INTRODUCTION

Diffusion models have achieved remarkable success in generative tasks, such as image synthesis, video generation, and text-to-image translation, owing to their ability to model complex multimodal distributions [1], [2], [3]. Building on this success, recent work has adopted them for robotic control, leveraging their multimodal sampling to model diverse trajectory distributions from demonstrations [4], [5], [6], [7]. Unlike language and vision, where large and standardized datasets enable broad generalization, robotics lacks comparable resources. Demonstrations are typically specific to a platform, task, or environment, making them expensive to collect and even more so to scale. As a result, diffusion policies for robot control are typically trained on narrowly distributed datasets, hence struggle to generalize reliably to

![](images/4b2683b6ac1dfdbcf0c34f022c948d1ad3836285958d82bd0fca8f3a75e2b1a5.jpg)

Fig. 1: Overview of the proposed method. Priors derived from previous iterations are incorporated to expand the support of trajectory modes, thereby facilitating the generation of trajectories that are more context-aware.

novel objects, configurations, and real-world scenarios due to insufficient modes for diverse trajectory sampling.

The aforementioned limitation is particularly significant in challenging scenarios involving collision avoidance, where capturing diverse trajectory modes is essential. A single goal configuration can admit multiple feasible paths depending on obstacle placement and grasp orientation, representing rich trajectory modalities [8]. Capturing the full range of such variations from data alone is impractical: simulation is computationally expensive and prone to sim-to-real gaps, while exhaustive real-world collection is infeasible.

A common strategy to mitigate limited training data is to apply training-free guidance during inference, steering the diffusion process toward context-aware, task-relevant modes [9], [10]. Guidance leverages the learned trajectory diversity to bias sampling toward modalities underrepresented in training data, such as collision-free trajectories. However, this approach involves a brittle trade-off: weak guidance may be insufficient to prevent unsafe trajectories, while strong signals risk distorting the learnt distribution, leading to degraded performance and unrealistic trajectories [10], [11].

To address incomplete trajectory modality and the tradeoffs inherent in training-free guidance, we propose Context-Aware diffusion policy via Proximal mode Expansion (CAPE). CAPE expands mode support iteratively with a context-aware prior and guidance at inference time (Figure 2). After executing a short prefix of the trajectory,

CAPE perturbs the remaining segment to an intermediate noise level, constructing a trajectory prior. Using this prior with context-aware guidance, the trajectory modes are expanded to include context-relevant regions. More precisely, the prior preserves the previously expanded mode support and task intent, while the guided refinement process further broadens its distributional support. This procedure yields an increasingly context-aware trajectory distribution, enabling better generalization. For collision avoidance, CAPE yields collision-aware trajectories without the brittle guidance finetuning or large-scale data collection. CAPE is also applicable to other contexts, provided that a measurable guidance objective can be defined. Unlike prior-free approaches [9], [12], CAPE couples the prior with training-free for better generalization in unseen environments by producing contextaware and goal-consistent trajectories.

In summary, our contributions are as follows:

• The prior is essential for preserving expanded contextaware modes and ensuring consistency with the original task objective, while iterative guided expansion further enlarges the mode support. By jointly preserving taskconsistent modes and expanding distributional support, CAPE produces trajectories that generalize more effectively to unseen contexts, for example, environments with previously unseen obstacles.

• Our evaluation of CAPE for collision-avoidance across diverse manipulation tasks in unseen cluttered environments demonstrates up to 80% higher real-world success rates, demonstrating its robustness and adaptability without the need for tedious hyperparameter tuning.

## II. RELATED WORKS

## A. Diffusion policies for robot control

Diffusion models have gained significant attention in robotics for addressing key challenges in imitation learning, such as multimodal trajectory generation in high-dimensional action spaces [4], [5], [6], [10], [13]. Approaches like Diffusion Behavior Cloning [10], Diffusion Policy [4], and its extension to 3D visual inputs [6] showcase how diffusion can generate full action sequences conditioned on robot observations and environment context. However, limited trajectory diversity in the training data restricts distributional mode support, causing poor generalization in unseen scenarios, particularly cluttered environments with new obstacles.

## B. Guidance mechanisms for collision-free diffusion control

Classifier guidance [14] uses pretrained classifiers to steer the denoising process and is used in diffusion-based policies for collision avoidance [15], [16], [17]. While effective, this approach requires wide trajectory modality coverage. In fact, APEX [15] collects 500k collision-free trajectories across diverse start-goal configurations and obstacle layouts, ensuring broad workspace coverage for generalization.

Classifier-free guidance (CFG) [18], adopted in [19], [20], interpolates between conditional and unconditional scores. It avoids a separate classifier, but requires evaluating the model twice per diffusion step, increasing latency and computational overhead that may be safety-critical in timesensitive applications. This method also struggles to handle novel obstacles and novel configurations.

Training-free, loss-based guidance methods [21] apply a cost function directly at inference time, without any additional supervision. MPD [12] leverages this to adapt to novel environments, but its performance is highly sensitive to a single guidance-weight hyperparameter, often requiring environment-specific tuning. RA-DP [9] and Lan-o3dp [22] further omit task constraints from their guidance, resulting in task non-completion when guidance dominates. While such guidance methods help steer the sampling towards underrepresented but safer modes, an improper application degrades performance: overly strong guidance pushes samples offdistribution, leading to poor generalization [10], [11], [8], [23], while weak guidance fails to prevent collisions. Hence, training-free guidance requires sufficient multimodal trajectory distributions to steer sampling toward underrepresented modes while remaining within the distribution [24].

## C. Prior-guided Initialization in Diffusion Motion Planning

Prior-guided initialization offers a principled alternative to Gaussian-noise sampling. Denoising Diffusion Bridge Models [25] replace the noise source with structured priors and target distributions, and NaviBridger [26] shows that initializing from an informative action prior improves generalization and downstream performance in visual navigation. In motion planning, READ [27] retrieves context-relevant expert trajectories to define start and goal states. RealDrive [28] interpolates between retrieved demonstrations and current observations to warm-start planning. These retrieval-based designs depend on curated offline data and effective matching, limiting their applicability in novel scenes. A complementary line employs diffusion as a seed trajectory generator. PRESTO [29] and DiffusionSeeder [30] generate trajectory candidates through diffusion and refine them via classical optimization.

In contrast to previous motion planning work, CAPE constructs a prior by perturbing its previous output and expands its trajectory mode support with context-aware guidance. CAPE does not require offline retrieval or secondary optimization.

## III. METHODOLOGY

## A. Background

Diffusion-based controllers, such as Diffusion Policy [4], are a probabilistic framework for action generation by iteratively denoising noisy action sequences. We follow the same approach in MPD [12], where the diffusion model generates full trajectories τ instead of action chunks directly, capturing both geometric and temporal structures from expert demonstrations. Let $\tau \in \mathbb { R } ^ { N \times d }$ denote a trajectory consisting of N steps in a d-dimensional action space. We assume to have samples from a conditional distribution of trajectories $p ( \tau \mid \mathbf { o } )$ to solve a task. The task context $\mathbf { O } = \left( { \boldsymbol { \ell } } , o \right)$ is defined by language instruction or task description ℓ and observations o (e.g. vision, proprioception) of the environment.

![](images/d9f4cbb229fc21b42c99bedc2120b863563b30431dd5199f78c29068cc2e080e.jpg)  
Fig. 2: An overview of the proposed framework: A diffusion model is trained to learn pick-and-place using data from an empty scene. The model uses this skill at inference time in cluttered environments. During inference, the task description and robot observations are sent to the model, and 3D point clouds are used to generate collision-aware guidance signals. Initial planning: The noisy trajectory is sampled from Gaussian distribution. Prior-Seeded Guided Iterative Refinement: After executing a short prefix trajectory, the remaining trajectory is perturbed with an intermediate noise level $\delta ,$ forming a prior. The prior preserves task intent and previously expanded mode support, which is further iteratively expanded with collision-aware guidance, until task completion.

We employ a Denoising Diffusion Probabilistic Model (DDPM) [31] to learn the conditional trajectory distribution $p _ { t } ( \tau \mid \mathbf { O } )$ , where the index $t \in \{ 0 , 1 , \ldots , T \}$ denotes the noise levels.

The reverse diffusion distribution $q _ { t } ( \tau ~ | ~ \mathbf { O } ; \theta )$ parameterized by θ, constructs a Markov chain starting from $\tau _ { T } ~ \sim ~ \mathcal { N } ( 0 , I )$ . At each reverse step, the model denoises by computing

$$
\tau_ {t - 1} = \frac {1}{\sqrt {\alpha_ {t}}} \left(\tau_ {t} - \frac {1 - \alpha_ {t}}{\sqrt {1 - \overline {{\alpha}} _ {t}}} \epsilon_ {\theta} (\tau_ {t}, t, \mathbf {O})\right) + \sigma_ {t} z,
$$

where $\begin{array} { r } { z \sim \mathcal { N } ( 0 , I ) , \alpha _ { t } = 1 - \beta _ { t } , \overline { { \alpha } } _ { t } = \prod _ { k = 1 } ^ { t } \alpha _ { k } . } \end{array}$ , and $\epsilon _ { \theta }$ denotes the neural network’s learned noise predictor. In our work, $\epsilon _ { \theta }$ is parameterized by a U-Net [32] conditioned on embeddings of O via cross-attention [33]. The noise model $\epsilon _ { \theta }$ is trained using a dataset of samples from $p ( \tau | \mathbf { O } )$ .

Training-free guidance enables steering the diffusion sampling process during inference without retraining. Specifically, we define a guidance function $\mathcal { L } _ { \mathrm { g u i d } } ( \tau , t , \mathbf { O } )$ that evaluates a noised trajectory $\tau _ { t }$ at noise level t given the task context O. During sampling, this guidance is incorporated via its gradient $\nabla _ { \tau _ { t } } \mathcal { L } _ { \mathrm { g u i d } }$ , modifying the denoising step as follows:

$$
\begin{array}{r} \tau_ {t - 1} = \frac {1}{\sqrt {\alpha_ {t}}} \left(\tau_ {t} - \frac {1 - \alpha_ {t}}{\sqrt {1 - \overline {{\alpha}} _ {t}}}   \epsilon_ {\theta} (\tau_ {t}, t, \mathbf {O})\right) \\ + \lambda   \nabla_ {\tau_ {t}} \mathcal {L} _ {\mathrm{guid}} (\tau_ {t}, t, \mathbf {O}) + \sigma_ {t} z. \end{array}\tag{1}
$$

In our implementation, ${ \mathcal { L } } _ { \mathrm { g u i d } }$ is defined using the signed distance function (SDF) with respect to environment obstacles. This enables us to steer trajectories away from collisions at sampling time, without requiring any additional training [12].

## B. Modality Expansion via Prior-Seeded Iterative Guided Refinement Procedure

This section motivates our iterative refinement procedure, which uses priors constructed from previous trajectories to expand the distribution modes for better generalization.

The Guidance-Denoising Tension: A key difficulty for effective training-free guidance is the anisotropy of learned trajectory distributions. Data scarcity concentrates probability mass around limited modes, creating distributions where context-aware trajectories often reside in low-density regions, on the edges of the support. This creates opposing forces: guidance steers sampling toward context-dependent low-density areas while the denoising process pushes noisy samples $\tau _ { t }$ toward high-density regions of $\dot { \mathbf { \rho } } p ( \tau | \mathbf { O } )$ . Resolving this conflict requires strong guidance to overcome denoising bias toward familiar trajectories, but excessive guidance from high-magnitude signals or constant injection produces offdistribution samples, leading to poor generalization. Guidance strength scheduling could address this but requires extensive task-specific tuning.

Trajectory-Prior Guided Sampling: Our approach addresses this fundamental tension by employing trajectory priors that preserve the distribution’s anisotropic structure while iteratively expanding its mode support in contextually relevant directions, achieving better generalization. Rather than fighting the anisotropy or destroying it with isotropic Gaussian noise, we leverage it by constructing structured priors that guide the expansion process.

We modify the diffusion sampling by first selecting a plausible prior $\widetilde { \tau } _ { t = \delta }$ at intermediate noise level $\delta \in [ 0 , T ]$ constructed from the unused segment of the previously planned trajectory $\widetilde { \tau } _ { t = 0 }$ . Using the remainder trajectory as prior preserves context-aware information relevant to the current scene and task objectives, unlike arbitrary priors that lack this task-specific context. This prior serves as an anchor for proximal modality expansion when combined with context-aware guidance signals.

The key advantage of our prior is preserving the distribu tion’s anisotropic structure while expanding mode support. Unlike pure Gaussian noise sampling, the prior concentrates probability density around structured, task-relevant regions. By Bayes’ theorem, the probability ratio (LHS) between sampling with and without the prior is:

$$
\frac {q _ {0} (\tau | \mathbf {O} , \widetilde {\tau} _ {\delta} , \lambda)}{q _ {0} (\tau | \mathbf {O} , \lambda)} = \frac {q _ {\delta} (\widetilde {\tau} _ {\delta} | \mathbf {O} , \tau , \lambda)}{q _ {\delta} (\widetilde {\tau} _ {\delta} | \mathbf {O} , \lambda)}\tag{2}
$$

![](images/0e0670aedfbc0e522e5521be04bbc8683139e32fead45a54c8f2bee901c9b154.jpg)  
Fig. 3: Trajectory samples under different guidance level in a real planning task without any prior.

The conditional likelihood $q _ { \delta } ( \widetilde { \tau } _ { \delta } | \mathbf { O } , \tau , \lambda )$ of observing the prior given a specific target trajectory τ is high when $\tau$ is near the denoised prior. In contrast, the unconditional likelihood $q _ { \delta } ( \widetilde { \tau } _ { \delta } | \mathbf { O } , \lambda )$ is much smaller since it considers all possible trajectories. This yields a probability ratio greater than 1 in neighborhoods of the denoised prior. This multiplicative scaling preserves anisotropy while concentrating mass around feasible solutions, enabling efficient sampling with weaker guidance signals.

This directly addresses the fundamental tension between guidance and denoising. The structured prior anchors sampling within plausible, task-consistent regions while maintaining the distribution’s meaningful anisotropic patterns. Context-aware guidance then provides targeted corrections toward underrepresented collision-free modalities, enabling controlled mode expansion without the risk of offdistribution sampling.

Figure 3 illustrates the limitations of applying guidance without a structured prior in motion planning. We examine guidance strength values $\lambda \in \{ 0 . 2 , 0 . 5 , 1 . 0 \}$ , sampling three trajectories for each parameter setting. Weak guidance $( \lambda =$ 0.2) fails to sufficiently steer trajectories away from obstacles, leaving samples trapped near collision-prone modes. Medium guidance $( \lambda ~ = ~ 0 . 5 )$ creates conflicting gradients around the central obstacle that pull neighboring waypoints in opposite directions, disrupting trajectory coherence while still resulting in collisions. Strong guidance $( \lambda = 1 . 0 )$ overwhelms the sampling process, generating highly distorted trajectories with excessive curvature that are kinematically infeasible for robot execution.

## C. Algorithms

Unlike previous works, we expand the trajectory distribu tion modes gradually via training-free, prior-seeded iterative guided refinement. We first introduce the notation and the training procedure of our method. We then present the guided denoising for motion planning in Algorithm 1, and describe our two-phase process in Algorithm 2: (i) an initial planning pass proposes a trajectory with weak guidance before executing the trajectory prefix of length m; (ii) an iterative refinement phase re-noises the remaining segment and applies guided denoising before each subsequent trajectory prefix execution. This process repeats until task completion.

Notation: A trajectory τ is discretized into N end-effector waypoints, each represented by a 9-dimensional pose vector $x = [ p , r ]$ , where $p \in \mathbb { R } ^ { 3 }$ is position and $r \in \mathbb { R } ^ { 6 }$ is the continuous 6D rotation representation for stable and discontinuityfree orientation modeling [34]. We define the task description as $\ell = \{ s _ { s } , s _ { g } \}$ for reach, pick and pick-and-place scenarios, where $s _ { s }$ and $s _ { g }$ denote the start and goal states respectively. The observation o comprises robot end-effector states from the previous H time steps, where $H$ is the observation horizon. ℓ and o together form the task context O, guidance strength is $\lambda \geq 0$ . The intermediate noise level δ defines the perturbation applied to the unexecuted trajectory suffix forming the prior $\tau _ { \delta }$ . Guidance is only applied from timestep χ during the guided denoising.

Training: We train on point-to-point motion trajectories of length N collected in obstacle-free scenes, simplifying data collection and reducing cost. This contrasts with previous approaches requiring broad configuration coverage with obstacles [15]. We deliberately collect trajectories from empty scenes to highlight the benefits of our framework: contextaware distributional mode expansion. CAPE has no obstacle information during training, so any collision-awareness emerges from our inference-time expansion. The framework is dataset-agnostic and can also be trained on demonstrations containing obstacles if available. Increasing the multimodality of the initially learned trajectory distribution can yield better performance. Training follows the standard DDPM procedure with the following loss: $\mathcal { L } ( \theta ) = \| \epsilon - \epsilon _ { \theta } ( \tau _ { t } , t ) \| _ { 2 }$

Guided Denoising for Motion Planning: Starting from a noisy trajectory $\tau _ { t }$ with noise perturbation level t, we iteratively denoise with the learned model $\epsilon _ { \theta } .$ For each timestep $t \leq \chi ,$ we apply the context-aware guidance signal $\lambda \nabla _ { \tau _ { t } } \mathcal { L } _ { \mathrm { g u i d } }$ with strength λ. In our collision setting, ${ \mathcal { L } } _ { \mathrm { g u i d } }$ is a collision cost evaluated using the obstacle point cloud $\mathbf { P _ { o b s } }$ This guidance gradually expands the relevant distribution modes, steering the samples away from obstacles while preserving task consistency. To ensure task completion, we enforce boundary conditions at every step by clamping the first and last waypoints to the start $s _ { s }$ and goal $s _ { g }$ states. The procedure outputs a collision-aware trajectory $\tau _ { 0 }$

Context-Aware Policy via Proximal Mode Expansion: The initial planning phase samples a trajectory from a standard Gaussian through guided denoising. The first $m$ steps (prefix trajectory) are executed. The prior-seeded iterative guided refinement phase extracts the unexecuted segment, renoises it to an intermediate noise level $t = \delta$ following the forward noising process (line 11 in Algorithm 2), yielding the prior $\tau _ { \delta } .$ . Starting from $\tau _ { \delta } .$ , a new guided denoising pass is executed, augmenting the context-aware modes of the prior distribution. The prior preserves the previously expanded mode support and task consistency, and further iterative expansion is applied on it. At the end of the refinement phase, a new trajectory $\tau _ { 0 }$ is produced. The controller executes the new trajectory prefix, and phase 2 iterates until task completion.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1: Guided Denoising for Motion Planning
Input : Noisy trajectory $\tau_t$, Noise level $t$, Guidance start step $\chi$, Task context $\mathbf{O}$, Trained diffusion model $\epsilon_\theta$, Obstacle point cloud $\mathbf{P}_{\text{obs}}$, Context-aware guidance function $\mathcal{L}_{\text{guid}}$, Guidance strength $\lambda$, Diffusion schedule parameters $(\alpha_t, \bar{\alpha}_t, \sigma_t)$
for $t = t, \ldots, 1$ do
    $\mu_t = \frac{1}{\sqrt{\alpha_t}} \left( \tau_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(\tau_t | t, \mathbf{O}) \right)$;
    if $t \leq \chi$ then
        // Apply Cost-Based Guidance
        $g = -\lambda \nabla_{\tau_{t-1}} \mathcal{L}_{\text{guid}}(\tau_{t-1} = \mu_t, \mathbf{P}_{\text{obs}})$;
        $\tau_{t-1} = \mu_t + g + \sigma_t z$, where $z \sim \mathcal{N}(0, \mathbf{I})$;
    // Enforce Boundary Constraints
    $\tau_{t-1}[0] = s_s, \tau_{t-1}[H - 1] = s_g$;
Output: Denoised trajectory $\tau_0$
</div>

Collision-Aware Guidance Computation: In our instantiation, the context encodes collision avoidance. Obstacles are represented as a point cloud $\mathbf { P _ { o b s } }$ available at inference time only. We approximate the end-effector as a sphere of radius $r _ { e e f }$ . Given an end-effector position $\boldsymbol { p } \in \mathbb { R } ^ { \bar { 3 } }$ and a set of obstacles’ point clouds, we compute the minimum distance $d ( p )$ between the end-effector and the nearest point in $\mathbf { P _ { o b s } }$ using the Chamfer distance from PyTorch3D [35]. We define a safety distance of ϵ. $\mathbf { P _ { o b s } }$ is only used to generate the training-free guidance during guided denoising. The signeddistance-based guidance cost is then defined as:

$$
\mathcal {L} _ {\text {guid}} (p) = \left\{ \begin{array}{l l} - d (p) + (\epsilon + r _ {e e f}) & \text {if} d (p) \leq \epsilon + r _ {e e f} \\ 0 & \text {if} d (p) > \epsilon + r _ {e e f} \end{array} \right.
$$

## IV. EXPERIMENTS

(3)

Environment setup. We evaluate CAPE across three progressively challenging collision-avoidance settings to test generalization from the expanded context-aware modes. First, a conceptual environment inspired by [36] isolates and visualizes the effects of the structured prior and the iterative guided refinement. Second, realistic simulated tabletop scenes of 3 difficulty levels (Fig. 4) evaluate collision avoidance under two observation regimes: full observations (complete obstacle point clouds) and limited observations (wrist-mounted camera). We generate 20 randomized layouts with 5 random initial pose of the robot. To increase the difficulty of the collision-avoidance task, the end-effector height is constrained to remain within 0.3 m above the tabletop so that it needs to move across the obstacles, and the robotic arm must travel a minimum distance of 0.4 m to reach the target object. Finally, we deploy CAPE in real-world cluttered tabletop scenarios, with observations from front-facing and wrist-mounted RGBD cameras. This progression holds the policy fixed while increasing difficulty, showing that

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 2: Collision-Aware Diffusion Policy Via Proximal Mode Expansion
Input : Trained diffusion model $\epsilon_{\theta}$, Task context O, Obstacle point cloud P_obs, Context-aware guidance function $\mathcal{L}_{\text{guid}}$, Guidance strength $\lambda$, Perturbation noise level $\delta$, Guidance start step $\chi$, Diffusion schedule parameters $(\alpha_t, \bar{\alpha}_t, \sigma_t)$
Initialize: $\tau_0 \leftarrow$ null, task_done $\leftarrow$ false, $k \leftarrow 0$, first_plan $\leftarrow$ true;
while not task_done do
    if first_plan then
        — Initial Planning —;
        $\tau_T[0] = s_s$, $\tau_T[N-1] = s_g$;
        $\tau_T \sim \mathcal{N}(0, \mathbf{I})$;
        $\tau_0^{k=1} \leftarrow$ GuidedDenoising($\tau_T$, $T$, $\chi$, O, $\epsilon_{\theta}$, P_obs, $\mathcal{L}_{\text{guid}}$, $\lambda$, $(\alpha_t, \bar{\alpha}_t, \sigma_t)$);
        first_plan $\leftarrow$ false;
    else
        — Prior-Seeded Iterative Refinement —;
        Update task context O' from environment;
        // Extract remaining trajectory
        $\widetilde{\tau}_0^k = \text{LinearInterpolate}(s'_s, \tau_0^k[m:N-1], s'_g)$;
        // Key step: Perturb trajectory to noise level $t = \delta$ $\widetilde{\tau}_{\delta}^k = \sqrt{\bar{\alpha}_t} \widetilde{\tau}_0^k + \sqrt{1 - \bar{\alpha}_t} \epsilon$;
        $\tau_0^{k+1} \leftarrow$ GuidedDenoising($\widetilde{\tau}_{\delta}^k$, $\delta$, $t_{start}$, O', $\epsilon_{\theta}$, P_obs, $\mathcal{L}_{\text{guid}}$, $\lambda$, $(\alpha_t, \bar{\alpha}_t, \sigma_t)$);
    Execute prefix (first m steps from $\tau_0^{k+1}$);
    $k \leftarrow k + 1$;
    if goal reached or max iterations exceeded then
        task_done $\leftarrow$ true;
</div>

CAPE can expand the trajectory distribution with contextaware modes. All simulations are run in ManiSkill2 [37]. All experiments are executed on a 7-DoF Franka Panda.

Data generation & training. We collect 1 000 training trajectories using an RRT planner [38] in an obstacle-free simulated environment. We augment the dataset by randomly resampling start points along each trajectory while keeping the target object fixed. Like [12], all trajectories are normalized to a fixed length N using linear interpolation for longer sequences and end-padding for shorter ones. The training set contains no obstacle by design to highlight that CAPE can expand mode support with collision-aware guidance at inference time. CAPE remains compatible with datasets that include obstacles. The same policy is used for both simulation and real-world experiments.

Models. We compare our method against an inferencetime guided variant of Diffusion Policy [4] (DP+Guidance), SOTA Motion Planning Diffusion (MPD) [12], which performs one-shot trajectory generation with inference-time guidance, and a variant that adds prior-free refinement to MPD (MPD+Refine) to assess context-aware mode support expansion without the prior. All methods use state-only inputs, the same guidance strength, and the same U-Net backbone [32] as our approach.

![](images/b40f1885cc30c5cc24a59e5f9530a9995dbd7228246121dfb0ab57de0b03ee0b.jpg)  
Fig. 4: Simulated environments with increasing level of difficulty used in the experiments. From left to right: 1-conceptual, 2-environment with 25 small obstacles, 3- environment with 15 small and 2 medium size obstacles, and 4- environment with 25 small and 2 large obstacles.

TABLE I: Key hyperparameters used in our experiments.

<table><tr><td colspan="2">General Settings</td><td colspan="2">Model</td></tr><tr><td>Trajectory Length N</td><td>32</td><td>Variance Schedule</td><td>exponential</td></tr><tr><td>State Dimension d</td><td>9</td><td>Diffusion Steps T</td><td>25</td></tr><tr><td>Batch Size B</td><td>256</td><td>Predict Epsilon</td><td>True</td></tr><tr><td>History Length H</td><td>8</td><td></td><td></td></tr><tr><td colspan="2">Training</td><td colspan="2">Inference</td></tr><tr><td>Learning Rate γ</td><td>1e-4</td><td>Intermediate Noise Level δ</td><td>2</td></tr><tr><td>Training Epochs</td><td>80</td><td>Trajectory Prefix Length m</td><td>2</td></tr><tr><td></td><td></td><td>Guidance Strength λ</td><td>0.2</td></tr><tr><td></td><td></td><td>Guidance Start Step χ</td><td>5</td></tr><tr><td colspan="4">Point Cloud Parameters</td></tr><tr><td colspan="3">Collision Sphere Radius  $r_{eef}$ </td><td>0.08 m</td></tr><tr><td colspan="3">Safety Margin ε</td><td>0.06 m</td></tr></table>

Metrics. We report three core metrics: success rate (SR), the fraction of episodes completed without collision; collision rate (CR), the fraction with any contact with obstacles; and non-completion rate (NCR), the fraction that remain collision-free but fail to reach the goal within the horizon. By construction, $\mathbf { S } \mathbf { R } + \mathbf { C } \mathbf { R } + \mathbf { N } \mathbf { C } \mathbf { R } = 1$ . We highlight NCR to quantify the trade-off between collision avoidance and task completion discussed previously.

## A. Experiment in Simulated Environment

DP+Guidance: This method is highly sensitive to the guidance signal, which can dominate the learnt trajectory distribution, suppressing goal-consistent action chunks. This pushes the robot to get trapped in local areas, unable to complete the task as it aim to avoid collisions. Consequently, tasks remain unfinished in most episodes, shown by a NCR ≥ 79% and $\mathrm { C R } \leq 2 1 \%$

MPD: On easy scenes with full observation, MPD reaches 96% SR. However, performance degrades all the way to 36% SR with increasing clutter and under partial observability due to missing collision-free mode support. First, sam pling trajectories from Gaussian noise provides no collisionawareness, requiring mode expansion from scratch through guidance whose strength is difficult to tune: insufficient guidance results in collisions, while excessive guidance produces unrealistic trajectories. Second, MPD performs one-shot trajectory generation with no further refinement, making it vulnerable to collisions from incomplete observations. Hence, MPD’s trajectory distribution has limited mode support, yielding less diverse samples, constraining its performance in more challenging scenarios.

MPD+Refine: MPD+Refine achieves better performance in limited observation scenarios with a 14% SR increase over MPD, as it continuously incorporates up-to-date collisionaware guidance from the environment. However, it suffers from the same limitation as MPD: each refinement starts from Gaussian noise, making it difficult to sample sufficiently diverse trajectories due to limited modal augmentation.

CAPE: Our framework introduces a prior-seeded guided iterative refinement. The trajectory prior preserves the previous context-aware modal augmentations, while guided denoising further expands them with up-to-date guidance. Figure 5 illustrate the importance of using a prior with guided refinement. This yields the best SR, with up to 40% SR gain over MPD and 26% over MPD+Refine. Additionally, the prior provides a significant computational advantage, increasing the refinement frequency by approximately 4×.

![](images/3748d813342dbca7d085f5ead88de160c75711ac4550796a8e9d5cf625c2e881.jpg)

![](images/cdb9c3668638bf22f99d368b8896e03f94089e98bef72d5435dd366c56e93a08.jpg)  
Fig. 5: 3D visualization of trajectory updates during execution in Env4 under full observation. Without a prior, the trajectory is trapped in clutter regions. With a trajectory prior, repeated guided refinement augments context-aware distributional mode support and increases diversity, so the trajectory progressively shifts out of clutter toward the goal.

## B. Experiments in Real World

![](images/879313c7c465ac174616ffbf66283440bb092a4c7ce61f038ca24d87d6bb39f6.jpg)  
Fig. 6: Real-World Cluttered Environments. Left: Pick-and-Place Cup - The goal is to pick the cup, and place it on the disk rack. Right: Pick Tape - The goal is to pick the tape roll and lift it slightly.

We conducted real-world experiments to benchmark CAPE against MPD and MPD+Refine. We provide quantitative results across two environments, running 5 trials for each method in each environment and reporting SR and CR. Table III summarizes our findings. Additionally, we conducted qualitative comparisons between the previous

TABLE II: Results of the experiments in the simulated environments. Values are reported as $\mathbf { S R ( \hat { \rho } ) } / \mathbf { C R ( \downarrow ) } / \mathbf { N C R ( \downarrow ) }$ . For SR higher value is better and for CR and NCR the lower. Full and Limited refer to the types of observation.

<table><tr><td>ENVIRONMENTS→</td><td>ENV1: CONCEPT</td><td>ENV2: EASY</td><td>ENV3: MEDIUM</td><td>ENV4: HARD</td><td>ENV3: MEDIUM</td><td rowspan="2">REFINE FREQUENCY (Hz)</td></tr><tr><td>POLICY↓</td><td>FULL</td><td>FULL</td><td>FULL</td><td>FULL</td><td>LIMITED</td></tr><tr><td>DP+GUIDANCE</td><td>0.00/0.00/1.00</td><td>0.11/0.00/0.89</td><td>0.00/0.12/0.88</td><td>0.00/0.19/0.81</td><td>0.00/0.21/0.79</td><td>N/A</td></tr><tr><td>MPD</td><td>0.38/0.60/0.02</td><td>0.96/0.02/0.02</td><td>0.66/0.33/0.01</td><td>0.59/0.39/0.02</td><td>0.36/0.64/0.00</td><td>N/A</td></tr><tr><td>MPD+REFINE</td><td>0.54/0.40/0.06</td><td>0.97/0.01/0.02</td><td>0.67/0.32/0.01</td><td>0.63/0.34/0.03</td><td>0.50/0.50/0.00</td><td>4.35</td></tr><tr><td>CAPE(REF+PRIOR)</td><td>0.94/0.02/0.04</td><td>0.98/0.02/0.00</td><td>0.82/0.17/0.01</td><td>0.75/0.21/0.04</td><td>0.76/0.24/0.00</td><td>16.67</td></tr></table>

TABLE III: Results of the real-world experiments reported as SR(↑)/CR(↓). RF stands for refinement frequency.

<table><tr><td>ENVIRONMENTS→</td><td>PICK&amp;PLACE CUP</td><td>PICK TAPE</td><td>RF (Hz)</td></tr><tr><td>MPD</td><td>0.80/0.20</td><td>0.00/1.00</td><td>N/A</td></tr><tr><td>MPD+REFINE</td><td>1.00/0.00</td><td>0.20/0.80</td><td>1.35</td></tr><tr><td>CAPE(REF+PRIOR)</td><td>1.00/0.00</td><td>0.80/0.20</td><td>4.54</td></tr></table>

SOTA (MPD) and our method (CAPE), with videos provided in the supplementary material.

MPD: In environments with relatively complete observations and moderate clutter, MPD performs reasonably well, as shown in Environment 1. This aligns with our simulation findings. However, in Environment 2, where obstacles are both more numerous and partially observable, MPD fails and collides with initially unseen obstacles.

MPD+Refine: This augmentation of MPD performs better at handling unseen obstacles. However, since each refinement starts from Gaussian noise, the learned trajectory distribution lacks sufficient collision-aware mode support, resulting in jerky and erratic movements that often lead to collisions. This is reflected in the poor results (SR of 0.2 and CR of 0.8), which aligns with our simulation findings.

CAPE: Our framework achieves the best results. By using a trajectory prior, we can continuously expand the trajectory distribution modes, enabling stronger generalization through more context-aware sampled trajectories. This is evidenced by an 80% improvement over MPD and 60% over MPD+Refine in Pick Tape. However, CAPE has limitations in extreme clutter scenarios where the distributional mode expansion may be insufficient to sample feasible trajectories. When environments become densely cluttered with complex obstacle configurations, the iterative refinement process may fail to adequately expand the trajectory distribution to cover the narrow solution space required for successful navigation.

## C. Ablation

Sensitivity to Guidance Strength λ: We compare success rate of MPD, MPD+Refine and CAPE at different guidance strengths. As shown in Figure 7, our method with prior is significantly less sensitive to guidance strength and achieves high success rates at very low guidance levels. The improvement gain from guidance is higher in the more challenging partial observable environment. This is, however, not the case for MPD as the guidance plays a little role to improve the success rate in the absence of the refinement mechanism.

![](images/354e9ec042598b70f736ae5d305d18f992a88c715c3cccb26239d0bdcaf06992.jpg)  
Fig. 7: Impact of guidance strength on SR in the environ ments under full / wrist camera observation.

Sensitivity to prefix length m and noise level δ: We perform a sweep over the trajectory prefix length m and the intermediate noise level δ. The ”Noise” column corresponds to the case where no prior is used (i.e., sampling directly from Gaussian noise). We find that the best performance is achieved with frequent replanning (short prefix) and low noise (small δ), corresponding to rapid denoising from a constantly updating prior. This aligns with intuition: frequent updates allow the prior to incorporate fresh environmental cues, improving responsiveness. In contrast, infrequent replanning increases collision risk due to stale context, while higher noise levels degrade the expanded modes and task information. Detailed results are shown in Table IV. Experiments are conducted in simulation on Environment 3 (medium difficulty) under limited observation conditions.

TABLE IV: Results of the experiments with partial observability. Values are reported as SR(↑)/CR(↓). Noise means no prior.

<table><tr><td>m\δ</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>8</td><td>10</td><td>NOISE</td></tr><tr><td>2</td><td>0.76/0.24</td><td>0.72/0.28</td><td>0.71/0.29</td><td>0.68/0.32</td><td>0.61/0.39</td><td>0.56/0.44</td><td>0.53/0.47</td><td>0.53/0.47</td></tr><tr><td>3</td><td>0.71/0.29</td><td>0.70/0.30</td><td>0.69/0.31</td><td>0.64/0.36</td><td>0.66/0.34</td><td>0.57/0.43</td><td>0.53/0.47</td><td>0.50/0.50</td></tr><tr><td>4</td><td>0.65/0.35</td><td>0.69/0.31</td><td>0.68/0.32</td><td>0.62/0.38</td><td>0.58/0.42</td><td>0.55/0.45</td><td>0.54/0.46</td><td>0.51/0.49</td></tr><tr><td>5</td><td>0.62/0.38</td><td>0.61/0.39</td><td>0.64/0.36</td><td>0.61/0.39</td><td>0.60/0.40</td><td>0.54/0.46</td><td>0.52/0.48</td><td>0.52/0.48</td></tr><tr><td>8</td><td>0.55/0.45</td><td>0.56/0.44</td><td>0.56/0.44</td><td>0.56/0.44</td><td>0.52/0.48</td><td>0.53/0.47</td><td>0.52/0.48</td><td>0.49/0.51</td></tr><tr><td>10</td><td>0.53/0.47</td><td>0.55/0.45</td><td>0.55/0.45</td><td>0.54/0.46</td><td>0.53/0.47</td><td>0.51/0.49</td><td>0.50/0.50</td><td>0.50/0.50</td></tr></table>

Sensitivity to Guidance Start Step χ: We conduct an ablation study to determine the optimal guidance start step, denoted by χ in our denoising algorithm. This parameter governs when context-aware guidance begins during the reverse diffusion process. Starting guidance too late provides insufficient collision avoidance, while applying it too early in the entire denoising can over-correct and generate offdistribution trajectories. With fixed prefix length m = 2 and intermediate noise level δ = 2, we find that $\chi = 5$ achieves optimal performance by balancing collision-awareness with trajectory quality. Table V reports results on Environment 3 (medium difficulty) under limited observation conditions.

TABLE V: Guidance start step χ sweep with guidance strength λ 0.2. Results are reported as SR(↑)/CR(↓).

<table><tr><td> $\chi$ </td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td></tr><tr><td>SR</td><td>0.72/0.28</td><td>0.73/0.27</td><td>0.73/0.27</td><td>0.76/0.24</td><td>0.75/0.25</td><td>0.75/0.25</td><td>0.74/0.26</td><td>0.74/0.26</td></tr></table>

## V. CONCLUSION

In this work, we proposed CAPE, a novel diffusion-based planning framework that expands trajectory mode support with context-aware prior and guidance at inference via a prior-seeded iterative guided refinement procedure. CAPE addresses a central limitation of diffusion policies in robotics: their collapse onto narrow trajectory modes due to limited, task-specific demonstrations. Our approach resolves the fundamental tension between guidance and denoising by using trajectory priors that preserve the distribution’s anisotropic structure while iteratively expanding mode support in contextually relevant directions. By constructing structured priors from previously executed trajectory segments and applying context-aware guidance, our method enables effective collision avoidance without the brittleness of pure guidance approaches or the computational burden of extensive data collection. Empirical results across conceptual, simulated, and real-world environments demonstrate that CAPE consistently outperforms state-of-the-art methods, achieving significant improvements in success rates while maintaining trajectory quality in cluttered scenarios.

Despite these advances, our approach has limitations. Our method continuously refines a trajectory prior for contextaware mode expansion; however, if the initial prior is suboptimal, it may be preferable to reinitialize the planning process entirely. We did not address how to determine prior quality in this work. Additionally, while our guidance approach effectively handles end-effector collision avoidance in the tested scenarios, it does not explicitly ensure full-body collision avoidance.

## REFERENCES

[1] D. Epstein, A. Jabri, B. Poole, A. Efros, and A. Holynski, “Diffusion self-guidance for controllable image generation,” in NeurIPS, 2023.

[2] P. Esser, J. Chiu, P. Atighehchian, J. Granskog, and A. Germanidis, “Structure and content-guided video synthesis with diffusion models,” in ICCV, 2023.

[3] Y. Li, H. Wang, Q. Jin, J. Hu, P. Chemerys, Y. Fu, Y. Wang, S. Tulyakov, and J. Ren, “Snapfusion: Text-to-image diffusion model on mobile devices within two seconds,” in NeurIPS, 2023.

[4] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song, “Diffusion policy: Visuomotor policy learning via action diffusion,” The International Journal of Robotics Research, p. 02783649241273668, 2023.

[5] M. Reuss, M. Li, X. Jia, and R. Lioutikov, “Goal conditioned imitation learning using score-based diffusion policies,” in RSS, 2023.

[6] Y. Ze, G. Zhang, K. Zhang, C. Hu, M. Wang, and H. Xu, “3d diffusion policy: Generalizable visuomotor policy learning via simple 3d representations,” in RSS, 2024.

[7] A. Prasad, K. Lin, J. Wu, L. Zhou, and J. Bohg, “Consistency policy: Accelerated visuomotor policies via consistency distillation,” in RSS, 2024.

[8] S. Liu, L. Wu, B. Li, H. Tan, H. Chen, Z. Wang, K. Xu, H. Su, and J. Zhu, “RDT-1b: a diffusion foundation model for bimanual manipulation,” in ICLR, 2025.

[9] X. Ye, R. H. Yang, J. Jin, Y. Li, and A. Rasouli, “Ra-dp: Rapid adaptive diffusion policy for training-free high-frequency robotics replanning,” arXiv preprint arXiv:2503.04051, 2025.

[10] T. Pearce, T. Rashid, A. Kanervisto, D. Bignell, M. Sun, R. Georgescu, S. V. Macua, S. Z. Tan, I. Momennejad, K. Hofmann, and S. Devlin, “Imitating human behaviour with diffusion models,” in ICLR, 2023.

[11] Y. Guo, H. Yuan, Y. Yang, M. Chen, and M. Wang, “Gradient guidance for diffusion models: An optimization perspective,” in NeurIPS, 2024.

[12] J. Carvalho, A. T. Le, P. Kicki, D. Koert, and J. Peters, “Motion planning diffusion: Learning and adapting robot motion planning with diffusion models,” IEEE Transactions on Robotics, pp. 1–20, 2025.

[13] M. Janner, Y. Du, J. Tenenbaum, and S. Levine, “Planning with diffusion for flexible behavior synthesis,” in ICML, 2022.

[14] P. Dhariwal and A. Nichol, “Diffusion models beat gans on image synthesis,” in NeurIPS, 2021.

[15] A. Dastider, H. Fang, and M. Lin, “Apex: Ambidextrous dual-arm robotic manipulation using collision-free generative diffusion models,” in IROS, 2024.

[16] Y. Zheng, R. Liang, K. ZHENG, J. Zheng, L. Mao, J. Li, W. Gu, R. Ai, S. E. Li, X. Zhan, and J. Liu, “Diffusion-based planning for autonomous driving with flexible guidance,” in ICLR, 2025.

[17] H. Lin, X. Huang, T. Phan, D. Hayden, H. Zhang, D. Zhao, S. Srinivasa, E. Wolff, and H. Chen, “Causal composition diffusion model for closed-loop traffic generation,” in CVPR, 2025.

[18] J. Ho and T. Salimans, “Classifier-free diffusion guidance,” in NeurIPSW, 2021.

[19] Y. Luo, C. Sun, J. B. Tenenbaum, and Y. Du, “Potential based diffusion motion planning,” in ICML, 2024.

[20] W. Yu, J. Peng, H. Yang, J. Zhang, Y. Duan, J. Ji, and Y. Zhang, “Ldp: A local diffusion planner for efficient robot navigation and collision avoidance,” in IROS, 2024.

[21] Y. Shen, X. Jiang, Y. Yang, Y. Wang, D. Han, and D. Li, “Understanding and improving training-free loss-based diffusion guidance,” in NeurIPS, 2024.

[22] Q. Feng, H. Li, Z. Zheng, J. Feng, and A. Knoll, “Language-guided object-centric diffusion policy for collision-aware robotic manipulation,” in ICRA, 2025.

[23] P. M. Julbe, J. Nubert, H. Hose, S. Trimpe, and K. J. Kuchenbecker, “Diffusion-based approximate mpc: Fast and consistent imitation of multi-modal action distributions,” arXiv preprint arXiv:2504.04603, 2025.

[24] L. Mao, H. Xu, X. Zhan, W. Zhang, and A. Zhang, “Diffusion-dice: In-sample diffusion guidance for offline reinforcement learning,” in NeurIPS, 2024.

[25] L. Zhou, A. Lou, S. Khanna, and S. Ermon, in ICLR, B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y. Sun, Eds., 2024.

[26] H. Ren, Y. Zeng, Z. Bi, Z. Wan, J. Huang, and H. Cheng, “Prior does matter: Visual navigation via denoising diffusion bridge models,” in CVPR, 2025.

[27] T. Oba, M. Walter, and N. Ukita, “Read: Retrieval-enhanced asymmetric diffusion for motion planning,” in CVPR, 2024.

[28] W. Ding, S. Veer, Y. Chen, Y. Cao, C. Xiao, and M. Pavone, “Realdrive: Retrieval-augmented driving with diffusion models,” arXiv preprint arXiv:2505.24808, 2025.

[29] M. Seo, Y. Cho, Y. Sung, P. Stone, Y. Zhu, and B. Kim, “Presto: Fast motion planning using diffusion models based on key-configuration environment representation,” in ICRA, 2025.

[30] H. Huang, B. Sundaralingam, A. Mousavian, A. Murali, K. Goldberg, and D. Fox, “Diffusionseeder: Seeding motion optimization with diffusion for rapid motion planning,” in CoRL, 2024.

[31] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” in NeurIPS, 2020.

[32] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in MICCAI, 2015.

[33] C.-F. R. Chen, Q. Fan, and R. Panda, “Crossvit: Cross-attention multiscale vision transformer for image classification,” in ICCV, 2021.

[34] Y. Zhou, C. Barnes, J. Lu, J. Yang, and H. Li, “On the continuity of rotation representations in neural networks,” in CVPR, 2019.

[35] N. Ravi, J. Reizenstein, D. Novotny, T. Gordon, W.-Y. Lo, J. Johnson, and G. Gkioxari, “Accelerating 3d deep learning with pytorch3d,” arXiv:2007.08501, 2020.

[36] X. Jia, D. Blessing, X. Jiang, M. Reuss, A. Donat, R. Lioutikov, and G. Neumann, “Towards diverse behaviors: A benchmark for imitation learning with human demonstrations,” in ICLR, 2024.

[37] J. Gu, F. Xiang, X. Li, Z. Ling, X. Liu, T. Mu, Y. Tang, S. Tao, X. Wei, Y. Yao, X. Yuan, P. Xie, Z. Huang, R. Chen, and H. Su, “Maniskill2: A unified benchmark for generalizable manipulation skills,” in ICLR, 2023.

[38] S. LAVALLE, “Rapidly-exploring random trees : a new tool for path planning,” Research Report 9811, 1998.