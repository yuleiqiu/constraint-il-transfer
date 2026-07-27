# Grounding Generative Policies in Physics: Optimization-Guided Diffusion for Robot Control

Sabrina Bodmer <sup>∗</sup> ETH Zurich

René Zurbrügg ∗ ETH Zurich

Tifanny Portela ETH Zurich

Hao Ma ETH Zurich

Alexandre Didier ETH Zurich

Marco Hutter ETH Zurich

Colin Jones <sup>†</sup> EPFL

Melanie Zeilinger <sup>†</sup> ETH Zurich <sup>∗</sup>

Abstract: Diffusion models sample effectively from high-dimensional, multimodal distributions, but their outputs may violate deployment constraints. For task-space robot policies, generated grasps, waypoints, or trajectories can be distributionally valid yet infeasible, violating reachability, collision-avoidance, or closed-loop executability requirements. This embodiment gap limits zero-shot deployment across robots, even when the task-space behavior itself is transferable.

We propose an inference-time optimization framework that couples the behavior generation to physical feasibility by formulating diffusion guidance as a constrained optimization problem. Our key insight is to replace the sampling perturbation in the backward process with an optimized correction, allowing hard constraints or soft penalties to be imposed during sampling without the need to retrain the diffusion model, while keeping samples close to the learned prior.

We evaluate the method on dexterous grasp synthesis with reachability and collisionavoidance constraints, and dynamic manipulation with controller-level trackability constraints. Across settings and robot embodiments, optimization-guided denoising matches the feasibility of projection- and gradient-guidance baselines while better preserving grasp quality, and improving controller-level executability and task success, with task success improving by up to 20pp. on dexterous grasping and 23pp. on visuomotor manipulation over the best baseline.

Keywords: Diffusion Policies, Optimization-Constrained Denoising, Dynamic Feasibility, Robot Manipulation, Cross-Embodiment Adaptation

## 1 Introduction

Diffusion models provide a flexible generative framework for robotics, capturing high-dimensional and multimodal distributions over actions, trajectories, or behaviors [1, 2]. Recent diffusion-based policies increasingly operate in task space rather than a specific robot’s joint or motor space, predicting end-effector poses, waypoints, grasp candidates, or reference trajectories [3, 4, 5]. This abstraction is attractive because it separates high-level behavior generation from robot-specific realization, including morphology, actuation, and low-level control. By providing a common interface, this approach allows policies to be transferred across entirely different embodiments without the costly need to collect new data or retrain models for each platform. However, task-space sampling only ensures training data consistency, not embodiment feasibility; predictions can violate kinematic reachability, joint/torque limits, collision-avoidance, or closed-loop execution constraints, transferring intent but failing on a specific physical robot.

Several lines of work have explored how diffusion-model outputs can be guided toward deployment constraints, including projection layers [6], differentiable barriers [7], constraint-aware training [8], gradient guidance [9], and post-hoc projections [10]. However, these approaches often require retraining, task-specific tuning, or setting-specific constraint formulations, and strong guidance can pull samples away from the learned prior. Cross-embodiment learning aims to share policies across robots through unified action spaces [11, 12] or morphology-aware encoders [13], but typically lacks an explicit inference-time mechanism for ensuring that sampled task-space predictions are executable on the target robot. This leaves open the need for a unified inference-time mechanism that keeps the generative prior fixed while applying minimal, structured corrections for embodiment, environment, and closed-loop execution constraints.

Our method is based on a structural property of DDIM sampling [14]: each reverse update separates the model-predicted denoising direction from a sampling perturbation. This separation provides a principled interface for constrained generation. Rather than modifying the pretrained score model or applying external gradients directly to the sample, we replace the perturbation with an optimization variable and regularize its magnitude, which keeps each step close to the pretrained sampling trajectory. The final denoising step can therefore satisfy user-defined constraints while preserving the behavior encoded by the pretrained prior. In this paper, we instantiate this principle for robot task-space policies, where the constraints encode reachability, collision avoidance, or controllerlevel executability. The resulting optimization-constrained denoising process injects information the diffusion model does not encode, without retraining or fine-tuning the model. As a result, the same task-space generative prior can be transferred zero-shot to different manipulator arms and environments by changing only the objectives of the optimization problem.

Concretely, our contributions are:

• Optimization-constrained denoising: an inference-time framework that replaces the DDIM sampling perturbation with a constrained optimization variable, enabling small, structured corrections to a frozen diffusion prior.

• Unified feasibility interface: a formulation that can be solved either as a constrained nonlinear program or as a differentiable least-squares relaxation, with instantiations for reachability, collision avoidance, and controller-level executability.

• Cross-embodiment generalization without retraining: an evaluation showing that the same task-space diffusion prior can be reused across different simulated manipulator embodiments by changing only the inference-time optimization problem, improving grasp quality, task success, and controller-level executability.

## 2 Related Work

Cross-embodiment adaptation in diffusion policies. Cross-embodiment learning aims to reduce the need for large-scale, embodiment-specific demonstrations. One line of work learns from multi-robot datasets or shared action spaces: Open X-Embodiment aggregates demonstrations across robots to train RT-X models [11], Octo trains a transformer-based generalist policy with a diffusion action head on such aggregated data [15], and UMI collects human demonstrations in an embodiment-agnostic end-effector space [12]. Another line explicitly encodes morphology through hardware-conditioned, graph-based, modular, or transformer-based representations [16, 17, 18, 13]. These methods primarily address policy transfer, leaving embodiment-specific feasibility to be learned implicitly or handled by a downstream controller. UMI-on-Air is closest to our setting, guiding an embodiment-agnostic diffusion policy via low-level tracking costs [9]. In contrast, we treat the diffusion model as a fixed embodiment-agnostic prior and plug embodiment-specific feasibility into each reverse step through a replaceable constrained optimization module.

Constraints in diffusion policies. Diffusion policies generally do not guarantee that generated predictions are executable for a specific embodiment or environment. Existing constraint-aware methods fall into two categories. The first targets environment-level safety or geometric feasibility, such as obstacle avoidance, collision avoidance, or path constraints. Projected diffusion models reformulate sampling as a constrained projection to enforce physical or geometric constraints [6], while other approaches penalize violations during training or employ barrier functions at inference [8, 7]. Compositional methods such as Diffusion-CCSP and potential-based motion planning combine constraint energies to satisfy multiple task and environment constraints jointly [19, 20]. These methods improve task-level feasibility but typically do not capture embodiment dynamics, actuation limits, or controller response. The second category targets dynamic or control feasibility for a specific embodiment. DPCC combines model-based projection and constraint tightening during denoising to generate dynamically feasible control trajectories [10], while UMI-on-Air incorporates low-level controller tracking cost into sampling to favor trajectories that are easier to track [9], and DynaGuide steers a frozen diffusion policy at inference by injecting the gradient of a learned latent dynamics model into each DDIM step, biasing samples toward outcomes that match user-provided goal images [21]. These methods rely on soft guidance: feasibility is encouraged through denoising gradients, but corrections are not explicitly constrained, satisfaction can’t be guaranteed and/or strong guidance can pull samples away from the learned prior. Multi-objective settings further rely on ad-hoc gradient aggregation, which can destabilize sampling when feasibility terms conflict. These limitations are shared by the broader family of gradient-based guidance, including classifier and classifier-free guidance, posterior sampling, and plug-and-play priors [22, 23, 24], in which a cost gradient is added to each denoising step: feasibility is only encouraged rather than enforced, the guidance strength trades feasibility against fidelity, and constraint satisfaction cannot be guaranteed. In contrast, we replace the stochastic perturbation of each reverse step with a constrained optimization variable, so that feasibility is imposed as an explicit hard or soft constraint over the reverse process rather than as a soft gradient, while a regularizer on the correction keeps the sample close to the pretrained prior. This lets us enforce inverse-kinematic, joint-velocity, and controller-level feasibility.

## 3 Background

Our method modifies the reverse sampling process of a pretrained diffusion model. We briefly review the DDIM sampler [14], focusing on the stochastic perturbation that will later be replaced by an optimization variable. A diffusion model [25] corrupts a clean sample $x _ { 0 } \sim p _ { \mathrm { d a t a } }$ as

$$
x _ {k} = \sqrt {\alpha_ {k}}   x _ {0} + \sqrt {1 - \alpha_ {k}}   \varepsilon , \qquad \varepsilon \sim \mathcal {N} (0, I),\tag{1}
$$

and trains a denoising network $\hat { \varepsilon } _ { \boldsymbol { \theta } } ( x _ { k } , k )$ with the standard diffusion loss (Appendix 8.1). Given $x _ { k }$ , DDIM forms the clean-sample estimate $\hat { x } _ { 0 } = ( x _ { k } - \sqrt { 1 - \alpha _ { k } } \hat { \varepsilon } _ { \theta } ( x _ { k } , k ) ) / \sqrt { \alpha _ { k } }$ and performs the reverse update

$$
x _ {k - 1} = \underbrace {\sqrt {\alpha_ {k - 1}} \hat {x} _ {0} + \sqrt {1 - \alpha_ {k - 1} - \sigma_ {k} ^ {2}} \hat {\varepsilon} _ {\theta} (x _ {k} , k)} _ {\mu_ {\theta} (x _ {k}, k)} + \sigma_ {k} \omega , \qquad \omega \sim \mathcal {N} (0, I),\tag{2}
$$

with $\sigma _ { k }$ given in Appendix 8.1. This update separates the deterministic model prediction $\mu _ { \boldsymbol { \theta } } ( \boldsymbol { x } _ { k } , k )$ from the stochastic perturbation ω. We exploit this separation by replacing ω with an optimization variable that enforces feasibility constraints during sampling.

## 4 Method

We present an inference-time framework to optimally guide a pretrained diffusion model. The key idea is to replace the stochastic perturbation in each DDIM reverse step with an optimized correction term. This preserves the nominal denoising direction predicted by the pretrained model, while using the correction term to enforce robot-, environment-, or task-specific feasibility. We use $k = K , \ldots , 0$ to denote the diffusion denoising steps, where the subscripts $k = K$ and $k = 0$ refers to the noisy sample and the clean prediction, respectively. Each diffusion variable $x _ { k }$ denotes an entire candidate output, such as a grasp or a reference trajectory, in which case $x _ { k } = [ x _ { 0 \mid k } , \dots , x _ { N \mid k } ]$ where $n = 0 , \ldots , N$ denotes the physical execution time along that trajectory.

![](images/29d412c794323ad430e4e62b81e44e6d17dc064411b5c3c5a1c509e99ff56a24.jpg)  
Figure 1: Optimization-guided diffusion. A task-space diffusion prior generates an initial reverse denoising step. Instead of sampling the standard DDIM perturbation $\omega _ { k }$ , we replace it with a structured correction δ<sub>k</sub> obtained from a constrained optimization problem. The objective minimizes the perturbation magnitude while incorporating embodiment- and environment-specific costs and constraints, such as $J , \mathcal { X } _ { \mathrm { t a r g e t } } ,$ and ${ \mathcal { X } } _ { \mathrm { i n i t } }$ . By changing only this constraint module, the same frozen prior can be reused across different manipulators.

## 4.1 Optimization-Constrained Denoising

The DDIM reverse update in (2) decomposes into a nominal prediction $\mu _ { \boldsymbol { \theta } } ( \boldsymbol { x } _ { k } , k )$ and a perturbation $\sigma _ { k } \omega$ . We replace the random variable ω with an optimization variable $\delta _ { k }$ , such that

$$
x _ {k - 1} = \mu_ {\theta} (x _ {k}, k) + \sigma_ {k} \delta_ {k}.\tag{3}
$$

The optimized perturbations $\{ \delta _ { k } \} _ { k = 1 } ^ { K }$ guide the reverse trajectory toward feasible outputs while regularizing deviations from the pretrained sampler. Let $J : \mathbb { R } ^ { d }  \mathbb { R } _ { > 0 }$ be a feasibility cost, such as, e.g., a distance to the kinematically reachable set. We formulate optimization-guided sampling as

$$
\begin{array}{l l} \min _ {x _ {K}, \{\delta_ {k} \} _ {k = 1} ^ {K}} & \frac {1}{2} \sum_ {k = 1} ^ {K} \| \delta_ {k} \| _ {2} ^ {2} + \sum_ {k = 0} ^ {K} \beta_ {k} J (x _ {k}) \\ \text {s.t.} & x _ {k - 1} = \mu_ {\theta} (x _ {k}, k) + \sigma_ {k} \delta_ {k}, \quad k = K, \ldots , 1, \\ & x _ {0} \in \mathcal {X} _ {\text {target}}, \\ & x _ {K} \in \mathcal {X} _ {\text {init}}. \end{array}\tag{4}
$$

Problem (4) admits a MAP interpretation in which $\frac { 1 } { 2 } \| \delta _ { k } \| _ { 2 } ^ { 2 }$ is the negative log-density of a standard normal prior on $\delta _ { k }$ and $\beta _ { k } J ( x _ { k } )$ plays the role of a Boltzmann pseudo-likelihood on feasibility; we defer the derivation to Appendix 8.2. The terminal set $\mathcal { X } _ { \mathrm { t a r g e t } }$ is typically chosen as a sublevel set of the feasibility cost, $\mathcal { X } _ { \mathrm { t a r g e t } } = \{ x \in \mathbb { R } ^ { d } : J ( x ) \leq \varepsilon _ { \mathrm { t o l } } \}$ , enforcing the desired tolerance at the clean sample; ${ \mathcal { X } } _ { \mathrm { i n i t } }$ optionally restricts the initial noisy sample, with $\bar { \mathcal { X } } _ { \mathrm { i n i t } } = \mathbb { R } ^ { d }$ by default. The scaling by $\sigma _ { k }$ matches the correction scale to the stochastic term in the original DDIM sampler, so together with the Gaussian prior on $\delta _ { k }$ this encourages small, structured corrections from the learned reverse process. In practice, we evaluate feasibility costs only at denoising states close enough to the data manifold, since the cost gradients are often not meaningful for nearly Gaussian intermediate samples. This is done by increasing the penalty $\beta _ { k }$ on the feasibility cost.

## 4.2 Encoding Feasibility through J

Problem (4) is independent of the concrete feasibility model. Different robot-, environment-, or task-specific requirements can therefore be incorporated by changing J and the associated terminal set $\mathcal { X } _ { \mathrm { t a r g e t } }$ , while keeping the diffusion prior fixed. We consider three instantiations: kinematic reachability, collision avoidance, and controller-level executability. Platform-specific implementations are given in Appendix 8.5.2.

Kinematic Reachability. Kinematic reachability captures whether a task-space output can be realized by the target embodiment within its admissible configuration space. Let $\mathcal { X } _ { \mathrm { I K } } \subseteq \mathbb { R } ^ { d }$ be the set of reachable task-space outputs. We define the ideal reachability cost as

$$
J _ {\mathrm{IK}} (x) = \min _ {y \in \mathcal {X} _ {\mathrm{IK}}} \| x - y \| _ {2}.\tag{5}
$$

This cost vanishes on the reachable set and increases with distance from it. Using $J = J _ { \mathrm { I K } }$ in (4) therefore guides sampling toward outputs that satisfy the kinematic limits of the target embodiment. The corresponding terminal set

$$
\mathcal {X} _ {\mathrm{target}} ^ {\mathrm{IK}} = \left\{x \in \mathbb {R} ^ {d}: J _ {\mathrm{IK}} (x) \leq \varepsilon_ {\mathrm{IK}} \right\}\tag{6}
$$

is a sublevel set of the cost $J _ { \mathrm { I K } }$ and recovers exact reachability in the limit $\varepsilon _ { \mathrm { I K } } \to 0$ . The framework does not assume a particular representation of $\mathcal { X } _ { \mathrm { I K } }$ . The cost may be computed using an analytical IK solver, a distance model, a projection operator, or a learned differentiable surrogate. Alternatively, the cost could also be computed implicitly through a kinematic model constraint, i.e., using additional optimization variables.

Collision Avoidance. A kinematically reachable output must additionally remain collision-free with respect to the environment and the robot itself. We encode collision avoidance through a signed distance function (SDF) $s : \mathbb { R } ^ { m }  \mathbb { R }$ , with $s ( p ) > 0$ on free space and $s ( p ) \leq 0$ inside obstacles. For a trajectory $x _ { k } = [ x _ { 0 \mid k } , \dots , x _ { N \mid k } ]$ , and safety margin $d _ { \mathrm { s a f e } } \geq 0$ , we define

$$
J _ {\mathrm{coll}} (x) = \sum_ {n = 0} ^ {N} \max \bigl (0, d _ {\mathrm{safe}} - s (r _ {n}) \bigr) ^ {2}.\tag{7}
$$

The term $J _ { \mathrm { c o l l } }$ enforces the safety margin: it vanishes in free space beyond $d _ { \mathrm { s a f e } }$ and grows quadratically with the depth of any violation. Using $J = J _ { \mathrm { c o l l } }$ in (4) therefore guides sampling toward collision-free outputs. The framework does not assume a particular representation of s. It may be obtained from an analytical scene description, a precomputed voxel grid, a mesh-based query, or a learned neural SDF; gradient-based solvers additionally require s to be differentiable. Our implementation is described in Appendix 8.5.2.

Controller-Level Executability. Kinematic reachability and collision-avoidance alone do not ensure that a generated reference can be executed by the target robot under its low-level controller. We therefore also consider controller-level executability. Here, each diffusion variable represents a complete task-space reference trajectory, $x _ { k } = [ x _ { 0 \mid k } , \ldots , x _ { N \mid k } ]$ , where $n = 0 , \ldots , N$ indexes physical execution time. Let $\phi ( x ; q _ { 0 } ; \kappa )$ denote the closed-loop trajectory realizable when the robot starts from joint state $q _ { 0 }$ and tracks the reference x using a feasible task space controller κ. We define

$$
J _ {\mathrm{dyn}} (x) = \| x - \phi (x; q _ {0}; \kappa) \| _ {2} ^ {2}.\tag{8}
$$

This cost penalizes references whose commanded trajectory differs from what the closed-loop system can realize. Using $J = J _ { \mathrm { d y n } }$ in (4) therefore guides sampling toward references that are executable by the target controller. The rollout map ϕ depends on the low-level controller $\kappa ,$ reference-hold behavior, control frequency, and horizon. Our framework only requires that the cost can be evaluated or implicitly represented through additional optimization variables. The rollout model used in our dynamic manipulation experiments is described in Appendix 8.5.2.

Differentiable NLS (relaxation). When differentiability or runtime is critical, we relax (4) into a nonlinear least-squares problem over $\boldsymbol { z } = \left\{ x _ { K } , \dots , x _ { 0 } , \delta _ { K } , \dots , \delta _ { 1 } \right\}$ . The reverse-step equalities become squared consistency residuals, and the terminal feasibility constraint is replaced by a hinge penalty $\ddot { \beta _ { \mathrm { t e r m } } } [ J ( x _ { 0 } ) - \varepsilon _ { \mathrm { t o l } } ] _ { + } ^ { 2 }$ . We solve the NLS problem with Theseus [26] in a fixed small number of Levenberg-Marquardt iterations. The relaxation enforces feasibility only softly, but is differentiable and lightweight, which makes it easier to embed in learning-based pipelines. The full objective and weight choices are given in Appendix 8.3.

## 5 Results

We evaluate our method on two settings. The first, dexterous grasping on two manipulator arms with a shared diffusion prior (Section 5.1), demonstrates gain over baselines, transfer across manipulator arms, and composition with collision constraints. The second, image-conditioned dynamic manipulation on two tasks (Section 5.2), tests extension from single-pose synthesis to controller-level trackability of full trajectories.

Implementation Details. The optimization interface in (4) decouples the diffusion prior from the particular feasibility model and numerical solver used at inference time. We use three solver instantiations across our experiments: IPOPT for strict terminal feasibility, Theseus for a differentiable nonlinear least-squares relaxation, and L-BFGS for lightweight online replanning in the image-conditioned manipulation setting. The corresponding feasibility costs, closed-loop rollout approximations, solver settings, and relaxed objectives, as well as the hyperparameters for all baseline methods are given in Appendix 8.5.2.

## 5.1 Dexterous grasping across manipulator arms

We evaluate the framework on dexterous grasp synthesis with a 5-fingered XHand gripper. Each grasp is represented as a 21-dimensional configuration consisting of the wrist pose and 12 finger joints. A single taskspace diffusion model from [27] is used unchanged across all experiments; embodiment-specific information enters only through the feasibility cost $J _ { \mathrm { I K } }$ . We deploy this prior on two manipulator arms with substantially different workspaces and joint limits, the Franka Panda [28] and the Dynaarm [29]. We predict 8 grasps

Table 1: GRASP PREDICTIONS: Comparison of methods across evaluation metrics, averaged over 30 objects, 8 grasps per object, and 5 base poses. DDIM<sup>†</sup> is the raw diffusion model output with no IK checks or projection applied.

<table><tr><td rowspan="2">Arm</td><td rowspan="2">Run</td><td rowspan="2">Success Rate $SR^{(1)}$ [%] ↑</td><td rowspan="2">Kinematics $SR^{IK}$ [%] ↑</td><td colspan="2">Grasp</td></tr><tr><td> $E_{fc} (\times 10^3) \downarrow$ </td><td> $Q_1 (\times 10^{-3}) \uparrow$ </td></tr><tr><td></td><td> $DDIM^\dagger$ </td><td>75.0</td><td>N/A</td><td>0.01</td><td>16.0</td></tr><tr><td rowspan="5">Dynaarm</td><td>DDIM [27]</td><td>27.0</td><td>54.6</td><td>2.37</td><td>6.5</td></tr><tr><td>Projection Guidance</td><td>23.7</td><td>99.9</td><td>0.72</td><td>6.8</td></tr><tr><td>Gradient Guidance</td><td>58.8</td><td>99.3</td><td>0.97</td><td>12.80</td></tr><tr><td>Ours (Theseus)</td><td>63.5</td><td>95.8</td><td>0.26</td><td>14.5</td></tr><tr><td>Ours (IPOPT)</td><td>69.8</td><td>96.4</td><td>0.20</td><td>15.2</td></tr><tr><td rowspan="5">Franka Arm</td><td>DDIM [27]</td><td>33.7</td><td>83.7</td><td>2.02</td><td>7.1</td></tr><tr><td>Projection Guidance</td><td>37.4</td><td>100.0</td><td>1.19</td><td>8.1</td></tr><tr><td>Gradient Guidance</td><td>50.9</td><td>97.9</td><td>1.04</td><td>11.1</td></tr><tr><td>Ours (Theseus)</td><td>61.0</td><td>98.3</td><td>0.23</td><td>14.1</td></tr><tr><td>Ours (IPOPT)</td><td>71.0</td><td>99.8</td><td>0.06</td><td>15.7</td></tr></table>

per object for 30 objects across 5 randomized base poses spanning easy and hard reach configurations for a total of 1200 grasps.

Baselines. DDIM<sup>†</sup> reports the diffusion prior evaluated with no embodiment at all (floating gripper), serving as an upper bound on what the generative model alone can produce when arm-reachability is removed from the problem. DDIM [27] takes the raw diffusion output and solves for the closest IK-reachable grasp pose using cuRobo <sup>2</sup> [30], representing a decoupled snap-to-feasibility baseline. Gradient Guidance [9] applies classifier-style gradient updates of an IK cost along the reverse process; Projection Guidance [10] projects each denoising step onto the IK-feasible set during sampling. Our variants solve either the constrained problem (4) (IPOPT) or its differentiable relaxation (21) (Theseus).

Metrics. $\mathrm { S R } ^ { ( 1 ) }$ reports per-(object, pose) success rates in simulation; $\mathrm { S R ^ { I K } }$ is the fraction of kinematically reachable predictions; $E _ { \mathrm { f c } }$ , and $Q _ { 1 }$ capture force-closure violation, and grasp wrench quality [31], respectively. Full definitions are given in Appendix 8.6.

## 5.1.1 Cross-embodiment performance

Table 1 shows that without guidance, inverse-kinematic feasibility $( \mathrm { S R } ^ { \mathrm { I K } } )$ is low on the tightly constrained Dynaarm workspace (54.6%) and only moderate on the Panda (83.7%). All guidance methods, including both baselines, raise $\mathrm { S R } ^ { \mathrm { I K } }$ above 95%. The differences appear in task success and grasp quality. On task success, projection guidance attains high feasibility but does not convert it into success $( \mathrm { S R } ^ { ( 1 ) }$ of 23.7% and 37.4% on the Dynaarm and Panda), whereas gradient guidance is the strongest baseline (58.8% and 50.9%). Our optimization-constrained approaches reach 63.5–69.8% on the Dynaarm and 61.0–71.0% on the Panda, exceeding gradient guidance by up to 11 points on the Dynaarm and 20 points on the Panda. The pattern carries over to grasp quality: the cuRobo-snap DDIM baseline preserves feasibility by construction but collapses $Q _ { 1 } ~ ( \leq ~ 7 . 1 ~ \mathrm { v s } . \geq ~ 1 4 $ for our methods), and projection guidance similarly degrades $E _ { \mathrm { f c } }$ and $Q _ { 1 } ,$ , consistent with the structural concern that projecting intermediate denoising states drags samples off the grasp manifold before the prediction becomes semantically meaningful. The optimization-constrained approaches, by contrast, maintain $Q _ { 1 }$ values approaching DDIM<sup>†</sup>, showing that enforcing feasibility need not sacrifice grasp quality. The hard-constrained formulation using IPOPT achieves slightly higher success rates compared to the relaxed Theseus formulation, but at the cost of runtime: on the Dynaarm, DDIM denoising takes 79.3 ± 1.3ms per trajectory and gradient and projection guidance add negligible overhead (84.2 ± 0.9 and 84.1 ± 1.9ms), whereas Theseus and IPOPT require $3 . 6 3 \pm 0 . 0 3 \mathrm { s }$ and $5 . 4 7 \pm 0 . 6 \mathrm { s }$ , roughly 46× and 69× slower than DDIM. We further provide evaluations broken down by base pose difficulty in Appendix 8.5.3, Figure 6.

![](images/21aa25af78b311a363461d51e0a92312afffb97e26381a29be0f2b23ee4ccddd.jpg)  
Figure 2: Collision-Aware Grasping. Grasp poses for different environments. IPOPT respects collision constraints while preserving grasp quality; Gradient Guidance degrades the grasp.

## 5.1.2 Composing kinematic and collision constraints

We next add an environment-level collision cost $J _ { \mathrm { c o l l } }$ to the feasibility module and evaluate our method in different constrained environments. The diffusion model and arm configurations remain unchanged; only the feasibility term is extended from $J _ { \mathrm { I K } }$ to include $J _ { \mathrm { c o l l } }$ in the cost and/or constraint formulation. Figure 2 illustrates qualitative failure modes: Gradient Guidance often produces collision-free grasps that fail to grasp the object, whereas our optimization-guided variants preserve stable grasps. Quantitative evaluations, including per-environment results, are provided in Appendix 8.7.

## 5.2 Image-Conditioned Manipulation with a Floating-Gripper Prior

We next test whether optimization-guided denoising can make an image-conditioned floating-gripper policy executable on embodied robot arms. Unlike the grasping experiments, this setting requires steering short-horizon trajectories that are repeatedly replanned from visual observations.

We train a Diffusion Transformer on demonstrations collected with a simulated, modified Hoi! [32] two-finger gripper in the Drawer Opening and Pick-and-Place environments from [33] (Appendix Fig. 4).

Table 2: TRAJECTORY PREDICTIONS: Comparison of methods across evaluation metrics, evaluated on 100 environments with statistics reported over 4 runs. DDIM<sup>†</sup> denotes the raw diffusion model output deployed on a floating gripper without an arm. We compare $( x _ { k } )$ -nudging, which directly shifts the diffusion model output using the cost gradient, (σ )- guidance, which injects the gradient through the DDIM perturbation term $( \sigma _ { k } \omega _ { k } )$ , and our optimization-based guidance.

<table><tr><td rowspan="2">Task</td><td rowspan="2">Method</td><td colspan="2">Dynaarm</td><td colspan="2">Franka</td></tr><tr><td> $SR_{task [ \% ] \uparrow}$ </td><td> $SR_{sub [ \% ] \uparrow}$ </td><td> $SR_{task [ \% ] \uparrow}$ </td><td> $SR_{sub [ \% ] \uparrow}$ </td></tr><tr><td rowspan="5">Drawer</td><td> $\mathsf{DDIM^{\dagger}}$ </td><td> $99.5\pm 1.0$ </td><td> $100.0\pm 0.0$ </td><td> $99.5\pm 1.0$ </td><td> $100.0\pm 0.0$ </td></tr><tr><td>DDIM [27]</td><td> $79.5\pm 1.9$ </td><td> $84.0\pm 0.8$ </td><td> $57.8\pm 2.2$ </td><td> $66.3\pm 3.1$ </td></tr><tr><td>Gradient Guidance ( $\sigma$ )</td><td> $79.8\pm 1.7$ </td><td> $85.0\pm 1.6$ </td><td> $57.8\pm 7.8$ </td><td> $66.8\pm 6.8$ </td></tr><tr><td>Gradient Guidance ( $x_t$ -nudge)</td><td> $79.8\pm 3.3$ </td><td> $83.8\pm 3.0$ </td><td> $58.0\pm 2.4$ </td><td> $67.0\pm 2.8$ </td></tr><tr><td>Ours (L-BFGS)</td><td> $81.8\pm 2.9$ </td><td> $87.3\pm 2.9$ </td><td> $60.0\pm 3.4$ </td><td> $70.0\pm 1.6$ </td></tr><tr><td rowspan="5">Pick and Place</td><td> $\mathsf{DDIM^{\dagger}}$ </td><td> $62.5\pm 6.2$ </td><td> $95.3\pm 2.1$ </td><td> $62.5\pm 6.2$ </td><td> $95.3\pm 2.1$ </td></tr><tr><td>DDIM [27]</td><td> $26.0\pm 2.0$ </td><td> $63.3\pm 3.3$ </td><td> $44.0\pm 4.5$ </td><td> $89.0\pm 3.4$ </td></tr><tr><td>Gradient Guidance ( $\sigma$ )</td><td> $29.3\pm 4.3$ </td><td> $69.3\pm 4.6$ </td><td> $41.0\pm 1.4$ </td><td> $92.3\pm 1.3$ </td></tr><tr><td>Gradient Guidance ( $x_t$ -nudge)</td><td> $26.3\pm 3.4$ </td><td> $63.3\pm 5.0$ </td><td> $43.3\pm 1.7$ </td><td> $90.3\pm 3.8$ </td></tr><tr><td>Ours (L-BFGS)</td><td> $41.8\pm 3.8$ </td><td> $66.0\pm 3.6$ </td><td> $67.0\pm 3.4$ </td><td> $98.3\pm 1.3$ </td></tr></table>

The policy predicts task-space trajectories of horizon length N = 6 from image observations. After execution, a new trajectory is generated based on the updated camera view. At deployment, the policy is grounded to the Dynaarm or Franka Panda through the controller-level executability cost $J _ { \mathrm { d y n } }$ from Section 4.2. The cost compares the predicted reference trajectory to a differentiable approximation of the closed-loop trajectory under the low-level controller. We use a Cartesian impedance controller as it provides passive compliance and admits an efficient differentiable closed-loop approximation. While MPC could improve tracking performance as in [9], solving an MPC problem at each control step would dominate the inference budget. For online replanning, we optimize an additive trajectory offset during the final denoising steps using L-BFGS as a lightweight backend; implementation details are provided in Appendix 8.5.2. We compare against unguided DDIM and two gradient-guidance variants: an $x _ { k }$ -nudging variant similar to Gupta et al. [9], and a sigma-guidance variant that replaces the sampling perturbation ω with a weighted gradient of the cost. Sigma guidance is the natural single-gradient analogue of our method: it occupies the same reverse-step slot that we instead fill with an optimized correction, so the gap between the two isolates the benefit of optimizing the correction rather than merely injecting a gradient. DDIM<sup>†</sup> denotes the raw floating-gripper policy without embodiment-level trackability constraints and serves as an unembodied upper bound. We report task and subtask success rates on both arms in Table 2. On Drawer Opening, the methods are closely matched and the differences fall within run-to-run variation: L-BFGS guidance moves task success from 79.5% to 81.8% on the Dynaarm and from 57.8% to 60.0% on the Franka, comparable to the gradient-guidance baselines, with subtask success at 87.3% and 70.0% respectively. The gains are substantial on Pick and Place, where task success improves from 26.0% to 41.8% on the Dynaarm and from 44.0% to 67.0% on the Franka, while subtask success on the Franka approaches the unembodied upper bound at 98.3%. On this harder task our method clearly outperforms unguided DDIM and both gradient-guidance baselines on both embodiments, while sharing a single floating-gripper prior. These gains show that controller-level guidance corrects visually plausible floating-gripper trajectories that are difficult for the embodied controller to track. The improvement comes at a moderate runtime cost: guided inference takes 0.28–0.32 s compared to 0.02–0.07 s for unguided DDIM. Overall, the results show that the same image-conditioned floating-gripper prior can be deployed on embodied robot arms by changing only the inference-time feasibility module, without retraining the policy.

## 6 Limitations

Our framework has several limitations that suggest natural directions for future work. First, the constrained nonlinear program introduces non-trivial per-sample runtime, exceeding the gradientguidance baselines; the Theseus and L-BFGS relaxations reduce this overhead but replace hard constraint satisfaction with soft penalties. Second, the problem in (4) is generally nonconvex, so neither solver provides global feasibility or optimality guarantees, and we observed occasional convergence failures on poorly conditioned instances. Third, our main evaluation is conducted in simulation; we report preliminary hardware experiments in Appendix 8.7.1 as initial evidence that the same corrections transfer to physical deployment. Finally, our dynamic tracking cost assumes a Cartesian impedance controller and kinematic rollouts, an intentionally lightweight choice that shows the guidance mechanism helps even with a simple low-level controller. Replacing this approximation with a higher-fidelity model, such as an MPC or a differentiable physics simulator like Newton, is a promising direction but introduces additional design and numerical-conditioning challenges.

## 7 Conclusion

We presented an inference-time framework to steer pretrained diffusion policies toward embodiment and task-specific feasibility requirements. The key idea is to replace the reverse-step sampling perturbation with an optimization variable, allowing feasibility constraints to be imposed during denoising while leaving the generative model unchanged. By formulating diffusion guidance as a constrained optimization problem, our approach couples embodiment-agnostic behavior generation to physical feasibility at inference time, enabling a single diffusion prior to be reused across platforms by swapping only the optimization module.

We instantiated this interface for two complementary settings: kinematic reachability in dexterous grasp generation and controller-level trackability in dynamic manipulation. Across two manipulator arms, optimization-guided denoising achieved a better feasibility–success trade-off compared to projection- and gradient-guidance baselines, without retraining the diffusion model. These benefits come at a cost: the approach inherits the computational overhead of inference-time optimization, and the resulting nonconvex problems carry no global optimality guarantees.

Promising directions include reducing the solver runtime, extending the interface to richer contact and force constraints, and integrating optimization-guided denoising with large vision-language-action policies, where embodiment-specific retraining is particularly costly.

## References

[1] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, 44(10-11):1684–1704, 2025.

[2] M. Janner, Y. Du, J. B. Tenenbaum, and S. Levine. Planning with Diffusion for Flexible Behavior Synthesis. arXiv preprint arXiv:2205.09991, 2022.

[3] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, et al. OpenVLA: An Open-Source Vision-Language-Action Model. arXiv preprint arXiv:2406.09246, 2024.

[4] H. Ha, Y. Gao, Z. Fu, J. Tan, and S. Song. UMI on Legs: Making Manipulation Policies Mobile with Manipulation-Centric Whole-body Controllers. arXiv preprint arXiv:2407.10353, 2024.

[5] R. Punamiya, S. Kareer, Z. Liu, J. Citron, R.-Z. Qiu, X. Cai, A. Gavryushin, J. Chen, D. Liconti, L. Y. Zhu, et al. EgoVerse: An Egocentric Human Dataset for Robot Learning from Around the World. arXiv preprint arXiv:2604.07607, 2026.

[6] J. K. Christopher, S. Baek, and F. Fioretto. Constrained Synthesis with Projected Diffusion Models. Advances in Neural Information Processing Systems, 37:89307–89333, 2024.

[7] H. Ma, S. Bodmer, A. Carron, M. Zeilinger, and M. Muehlebach. Constraint-Aware Diffusion Guidance for Robotics: Real-Time Obstacle Avoidance for Autonomous Racing. In Proceedings of the Conference on Robot Learning, pages 1756–1776, 2025.

[8] A. Li, Z. Ding, A. B. Dieng, and R. Beeson. Constraint-Aware Diffusion Models for Trajectory Optimization. In International Conference on Dynamic Data Driven Applications Systems, pages 308–316, 2024.

[9] H. Gupta, X. Guo, H. Ha, C. Pan, M. Cao, D. Lee, S. Scherer, S. Song, and G. Shi. UMI-on-Air: Embodiment-Aware Guidance for Embodiment-Agnostic Visuomotor Policies, 2025.

[10] R. Römer, A. v. Rohr, and A. Schoellig. Diffusion Predictive Control with Constraints. In Proceedings of Machine Learning Research, pages 1–13, 2025.

[11] A. O’Neill, A. Rehman, A. Maddukuri, A. Gupta, A. Padalkar, A. Lee, A. Pooley, A. Gupta, A. Mandlekar, A. Jain, et al. Open X-Embodiment: Robotic Learning Datasets and RT-X Models. In Proceedings of the International Conference on Robotics and Automation, pages 6892–6903, 2024.

[12] C. Chi, Z. Xu, C. Pan, E. Cousineau, B. Burchfiel, S. Feng, R. Tedrake, and S. Song. Universal Manipulation Interface: In-The-Wild Robot Teaching Without In-The-Wild Robots, 2024.

[13] A. Patel and S. Song. GET-Zero: Graph Embodiment Transformer for Zero-shot Embodiment Generalization. In Proceedings of the International Conference on Robotics and Automation, pages 14262–14269, 2025.

[14] J. Song, C. Meng, and S. Ermon. Denoising Diffusion Implicit Models. arXiv preprint arXiv:2010.02502, 2020.

[15] O. M. Team, D. Ghosh, H. Walke, K. Pertsch, K. Black, O. Mees, S. Dasari, J. Hejna, T. Kreiman, C. Xu, et al. Octo: An Open-Source Generalist Robot Policy. arXiv preprint arXiv:2405.12213, 2024.

[16] T. Chen, A. Murali, and A. Gupta. Hardware Conditioned Policies for Multi-Robot Transfer Learning. 31:1–12, 2018.

[17] T. Wang, R. Liao, J. Ba, and S. Fidler. NerveNet: Learning Structured Policy with Graph Neural Networks. In Proceedings of the International Conference on Learning Representations, pages 1–26, 2018.

[18] W. Huang, I. Mordatch, and D. Pathak. One Policy to Control Them All: Shared Modular Policies for Agent-Agnostic Control. In Proceedings of the International Conference on Machine Learning, pages 4455–4464, 2020.

[19] Z. Yang, J. Mao, Y. Du, J. Wu, J. B. Tenenbaum, T. Lozano-Pérez, and L. P. Kaelbling. Compositional Diffusion-Based Continuous Constraint Solvers. In Proceedings of the Conference on Robot Learning, pages 3242–3265, 2023.

[20] Y. Luo, C. Sun, J. B. Tenenbaum, and Y. Du. Potential Based Diffusion Motion Planning. arXiv preprint arXiv:2407.06169, 2024.

[21] M. Du and S. Song. Dynaguide: Steering Diffusion Polices with Active Dynamic Guidance. Advances in Neural Information Processing Systems, 38:44192–44221, 2026.

[22] A. Graikos, N. Malkin, N. Jojic, and D. Samaras. Diffusion models as plug-and-play priors. 35: 14715–14728, 2022.

[23] H. Chung, J. Kim, M. T. Mccann, M. L. Klasky, and J. C. Ye. Diffusion Posterior Sampling for General Noisy Inverse Problems. arXiv preprint arXiv:2209.14687, 2022.

[24] A. Bansal, H.-M. Chu, A. Schwarzschild, R. Sengupta, M. Goldblum, J. Geiping, and T. Goldstein. Universal Guidance for Diffusion Models. In Proceedings of the International Conference on Learning Representations, pages 51304–51323, 2024.

[25] J. Ho, A. Jain, and P. Abbeel. Denoising Diffusion Probabilistic Models. Advances in neural information processing systems, 33:6840–6851, 2020.

[26] L. Pineda, T. Fan, M. Monge, S. Venkataraman, P. Sodhi, R. T. Chen, J. Ortiz, D. DeTone, A. Wang, S. Anderson, J. Dong, B. Amos, and M. Mukadam. Theseus: A Library for Differentiable Nonlinear Optimization. Advances in Neural Information Processing Systems, pages 3801–3818, 2022.

[27] R. Zurbrügg, A. Cramariuc, and M. Hutter. DexEvolve: Evolutionary Optimization for Robust and Diverse Dexterous Grasp Synthesis. arXiv preprint arXiv:2602.15201, 2026.

[28] Franka Robotics. Franka Panda robot arm. https://franka.de/, 2024. Accessed: 2026-05-26.

[29] Duatic AG. DynaArm: Ultra-lightweight robotic arm. https://www.duatic.com/ dynaarm, 2024. Accessed: 2026-05-26.

[30] B. Sundaralingam, S. K. S. Hari, A. Fishman, C. Garrett, K. Van Wyk, V. Blukis, A. Millane, H. Oleynikova, A. Handa, F. Ramos, et al. cuRoBo: Parallelized Collision-Free Minimum-Jerk Robot Motion Generation. arXiv preprint arXiv:2310.17274, 2023.

[31] R. Zurbrügg, A. Cramariuc, and M. Hutter. GraspQP: Differentiable Optimization of Force Closure for Diverse and Robust Dexterous Grasping. In Proceedings of the Conference on Robot Learning, pages 2583–2602, 2025.

[32] T. Engelbracht, R. Zurbrügg, M. Wohlrapp, M. Büchner, A. Valada, M. Pollefeys, H. Blum, and Z. Bauer. Hoi!–A Multimodal Dataset for Force-Grounded, Cross-View Articulated Manipulation. arXiv preprint arXiv:2512.04884, 2025.

[33] R. Zurbrugg, T. Portela, A. Bhardwaj, A. E. Vijayan, M. Wilder-Smith, and M. Hutter. VR-DAgger: Immersive VR for Dexterous Data Collection and Uncertainty-Guided On-Policy Correction. arXiv preprint arXiv:2605.27114, 2026.

[34] B. L. Wächter A. On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming. 106(1):25–57, 2006.

## 8 Supplementary Material

## Contents

8 Supplementary Material 13  
8.1 Denoising Diffusion Implicit Models 13  
8.1.1 Diffusion Models Implementation Details 13  
8.2 Bayesian interpretation of optimization-constrained denoising 14  
8.2.1 MAP Derivation 14  
8.2.2 Connection to DDIM 14  
8.3 Nonlinear Least-Squares Relaxation 15  
8.4 Visuomotor manipulation 15  
8.5 Optimization Modules 15  
8.5.0 Kinematic Feasibility Surrogate 16  
8.5.1 Closed-Loop Rollout Model 18  
8.5.2 Implementation Details 20  
8.5.3 Base-pose difficulty stratification 22  
8.6 Evaluation Metrics 23  
8.7 Collision-environment results 24  
8.7.1 Real-World Deployment 25

## 8.1 Denoising Diffusion Implicit Models

The following is based on the formulations of [14]. We consider a data distribution $q ( x _ { 0 } )$ and a model distribution $p _ { \theta } ( x _ { 0 } )$ , which approximates the data distribution and is easy to sample from. From $p _ { \theta } ( x )$ a sample $x _ { k - 1 }$ can be generated from the previous sample $x _ { k }$

$$
x _ {k - 1} = \sqrt {\alpha_ {k - 1}} \left(\frac {x _ {k} - \sqrt {1 - \alpha_ {k}} \hat {\varepsilon} _ {\theta} ^ {(k)} (x _ {k})}{\sqrt {\alpha_ {k}}}\right) + \sqrt {1 - \alpha_ {k - 1} - \sigma_ {k} ^ {2}} \hat {\varepsilon} _ {\theta} ^ {(k)} (x _ {k}) + \sigma_ {k} \varepsilon_ {k}, \quad \varepsilon_ {k} \sim \mathcal {N} (0, I)\tag{9}
$$

where $\alpha _ { k }$ encodes how much original signal remains at timestep $k ( \mathrm { i . e . }$ . is a noise-scheduler), and the parameters θ are then learned by minimizing the loss

$$
\mathcal {L} (\varepsilon_ {\theta}) = \sum_ {k = 1} ^ {K} \mathbb {E} _ {x _ {0} \sim q (x _ {0}), \varepsilon_ {k} \sim \mathcal {N} (0, I)} \left[ | | \varepsilon_ {\theta} ^ {(k)} (\sqrt {\alpha_ {k}} x _ {0} + \sqrt {1 - \alpha_ {k}} \varepsilon_ {k}) - \varepsilon_ {k} | | _ {2} ^ {2} \right],\tag{10}
$$

and

$$
\sigma_ {k} = \sqrt {\frac {1 - \alpha_ {k - 1}}{1 - \alpha_ {k}} \left(1 - \frac {\alpha_ {k}}{\alpha_ {k - 1}}\right)}.\tag{11}
$$

## 8.1.1 Diffusion Models Implementation Details

We use existing diffusion-model architectures for both tasks. For grasp synthesis, we use the Grasp Pose Diffusion model and training scheme from [27], omitting the keypoint embedding to reduce inference time. For visuomotor servoing, we use the Diffusion Transformer with a CNN image encoder from [33]. The model is conditioned on the two most recent RGB observations and hand poses, expressed in the current gripper frame, and predicts a chunk of six future wrist poses and finger states in the same gripper frame.

## 8.2 Bayesian interpretation of optimization-constrained denoising

This appendix shows that the cost terms in (4) correspond to the negative log-density of a posterior over guided trajectories, justifying the L2 penalty on $\delta _ { k }$ as a Gaussian prior inherited from the pretrained reverse process.

## 8.2.1 MAP Derivation

We view the guided denoising trajectory $\tau = ( x _ { K } , x _ { K - 1 } , \dots , x _ { 0 } )$ , with each state $\boldsymbol { x } _ { k } \in \mathbb { R } ^ { d }$ , as a latent variable and define a posterior over trajectories conditioned on feasibility:

$$
p (\tau \mid \text {feasible}) \propto p _ {\theta} (\tau) \prod_ {k = 1} ^ {K} \ell_ {k} (x _ {k}).\tag{12}
$$

Here, $p _ { \theta } ( \tau )$ denotes the trajectory distribution induced by the pretrained DDIM sampler. The factors $\ell _ { k } ( x _ { k } )$ act as feasibility pseudo-likelihoods and assign higher probability to lower-cost states:

$$
\ell_ {k} (x _ {k}) \propto \exp (- \beta_ {k} J (x _ {k})).\tag{13}
$$

The cost $J ( x _ { k } )$ measures violation of the desired feasibility criterion, and $\beta _ { k }$ controls the strength of the guidance at denoising step k. Thus, states with smaller feasibility cost are exponentially preferred.

The pretrained prior factorizes along the reverse chain as

$$
p _ {\theta} (\tau) = p (x _ {K}) \prod_ {k = 1} ^ {K} p _ {\theta} (x _ {k - 1} \mid x _ {k}).\tag{14}
$$

Taking the negative logarithm yields

$$
- \log p (\tau \mid \text {feasible}) = - \log p (x _ {K}) - \sum_ {k = 1} ^ {K} \log p _ {\theta} (x _ {k - 1} \mid x _ {k}) + \sum_ {k = 1} ^ {K} \beta_ {k} J (x _ {k}) + c,\tag{15}
$$

where, here and below, c collects all terms independent of the optimization variables (such as the posterior normalizer) and may take a different value in each equation. MAP inference corresponds to minimizing (15) subject to the problem constraints: the terminal tolerance on $x _ { 0 }$ and the initial-sample restriction on $x _ { K }$

## 8.2.2 Connection to DDIM

Under the stochastic DDIM update (3) with $\sigma _ { k } > 0$ , the reverse kernel is Gaussian,

$$
p _ {\theta} (x _ {k - 1} \mid x _ {k}) = \mathcal {N} \big (x _ {k - 1}; \mu_ {\theta} (x _ {k}, k), \sigma_ {k} ^ {2} I \big),\tag{16}
$$

so that

$$
- \log p _ {\theta} (x _ {k - 1} \mid x _ {k}) = \frac {1}{2 \sigma_ {k} ^ {2}} \| x _ {k - 1} - \mu_ {\theta} (x _ {k}, k) \| _ {2} ^ {2} + \frac {d}{2} \log (2 \pi \sigma_ {k} ^ {2}).\tag{17}
$$

We reparameterize each transition through its injected noise,

$$
x _ {k - 1} = \mu_ {\theta} (x _ {k}, k) + \sigma_ {k}   \delta_ {k}, \qquad \delta_ {k} \sim \mathcal {N} (0, I).\tag{18}
$$

Substituting $x _ { k - 1 } - \mu _ { \theta } ( x _ { k } , k ) = \sigma _ { k } \delta _ { k }$ cancels the $\sigma _ { k } ^ { 2 }$ and gives the identity

$$
- \log p _ {\theta} (x _ {k - 1} \mid x _ {k}) = \frac {1}{2} \| \delta_ {k} \| _ {2} ^ {2} + \frac {d}{2} \log (2 \pi \sigma_ {k} ^ {2}).\tag{19}
$$

The second term is independent of $\delta _ { k }$ . Summing over the chain and folding all such terms into $c ,$ (15) becomes the MAP objective

$$
- \log p (\tau \mid \text {feasible}) = - \log p (x _ {K}) + \sum_ {k = 1} ^ {K} \frac {1}{2} \| \delta_ {k} \| _ {2} ^ {2} + \sum_ {k = 1} ^ {K} \beta_ {k} J (x _ {k}) + c.\tag{20}
$$

This recovers the objective in $( 4 )$ , up to the prior term on $x _ { K }$ and the replacement of the likelihood term on $x _ { 0 }$ by the terminal-tolerance constraint used in our constrained formulation.

## 8.3 Nonlinear Least-Squares Relaxation

We give the full objective used by the differentiable solver of Section 4.2. The optimization variables are the denoising states and guidance perturbations, $\boldsymbol { z } = \left\{ x _ { K } , \dots , x _ { 0 } , \delta _ { K } , \dots , \delta _ { 1 } \right\}$ , and we solve

$$
\begin{array}{l l} \min _ {z} & \underbrace {\sum_ {k = 1} ^ {K} \beta_ {\delta} \| \delta_ {k} \| _ {2} ^ {2}} _ {\text {guidance regularizer}} + \underbrace {\beta_ {\text {init}}   d _ {\mathcal {X} _ {\text {init}}} (x _ {K}) ^ {2}} _ {\text {initial - sample regularizer}} \\ & + \underbrace {\sum_ {k = 1} ^ {K} \beta_ {\text {rev}} \left\| x _ {k - 1} - \mu_ {\theta} (x _ {k} , k) - \sigma_ {k} \delta_ {k} \right\| _ {2} ^ {2}} _ {\text {reverse - process residuals}} \\ & + \underbrace {\sum_ {k = 1} ^ {K} \beta_ {J , k} J (x _ {k}) ^ {2}} _ {\text {soft feasibility cost}} + \underbrace {\sum_ {k \in \mathcal {K} _ {J} \setminus \{0 \}} \beta_ {\text {path}} [ J (x _ {k}) - \varepsilon_ {\text {tol}} ] _ {+} ^ {2} + \beta_ {\text {term}} [ J (x _ {0}) - \varepsilon_ {\text {tol}} ] _ {+} ^ {2}} _ {\text {soft feasibility constraints}}, \end{array}\tag{21}
$$

with $[ \cdot ] _ { + } = \operatorname* { m a x } ( 0 , \cdot )$ . The initial-sample term $d _ { X _ { \mathrm { i n i t } } } ( x _ { K } )$ is replaced by a regularizer $\beta _ { \mathrm { i n i t } } \| x _ { K } \| _ { 2 } ^ { 2 }$ when no bounded ${ \mathcal { X } } _ { \mathrm { i n i t } }$ is used. The reverse-process residuals softly enforce consistency with the guided DDIM dynamics; the quadratic $\beta _ { J , k } J ( x _ { k } ) ^ { 2 }$ provides smooth gradients toward feasibility, while the hinge penalties concentrate weight on samples that exceed the tolerance. The path hinge is applied only to intermediate denoising states, and the separate terminal hinge $\beta _ { \mathrm { t e r m } }$ on $x _ { 0 }$ reflects that final-output feasibility is the primary requirement. The last reverse step uses its own weight $\beta _ { \mathrm { r e v } , 0 }$ to couple $x _ { 0 }$ more tightly to the preceding denoising state. The exact weights and full formulation are described in 8.5.2.

## 8.4 Visuomotor manipulation

We evaluate visuomotor manipulation on two image-conditioned tasks, shown in Fig. 4. In both tasks, the policy receives RGB observations and predicts short-horizon end-effector reference trajectories, which are executed on the target embodiment using a damped-least-squares impedance controller. The differentiable rollout model used to evaluate controller-level executability is described in Appendix 8.5.1; implementation details for the corresponding guidance methods are provided in Appendix 8.5.2.

We report two success metrics. The task success rate $\mathrm { S R } _ { \mathrm { t a s k } }$ measures the fraction of episodes in which the full task is completed: the drawer is fully opened for drawer manipulation, and the pan is correctly placed on the target burner for tabletop pick-and-place. The subtask success rate $\mathrm { S R } _ { \mathrm { s u b } }$ gives partial credit for meaningful intermediate progress. For drawer manipulation, the subtasks are grasping the handle and opening the drawer by at least a small distance; for pick-and-place, they are grasping the pan and lifting it from the table.

Table 3 reports task success, subtask success, executed end-effector distance, and solve time for both tasks and robot embodiments. Across the embodiment-aware methods, the optimization-based L-BFGS variant achieves the highest full task success in all settings. The improvement is particularly pronounced for tabletop pick-and-place, where successful execution often requires recovering from imperfect grasps or regrasping the pan before placement. L-BFGS also generally solves successful episodes faster than the baselines, despite the additional optimization step during replanning.

The executed-distance metric should be interpreted together with task success. Methods that solve more episodes also complete more of the reaching, recovery, and placement motions, which can naturally increase accumulated path length. Thus, the longer paths observed for L-BFGS reflect its ability to recover from near-failure cases, including through corrective motions such as regrasping, while still completing successful episodes faster overall.

## 8.5 Optimization Modules

The optimization-guided samplers require differentiable modules that translate robot-specific execution constraints into costs that can be optimized during denoising. We use two such modules. For grasp synthesis, a learned kinematic surrogate approximates the distance of a task-space wrist target to the reachable set. For visuomotor manipulation, a differentiable closed-loop rollout model approximates how the low-level controller tracks a predicted reference trajectory. This section describes both modules, including their inputs, training or rollout construction, and how their outputs are used as guidance costs.

![](images/1b966abf6bdee2a04238c11fb844c3ab49b95f523b875800fa024074874d8d77.jpg)

![](images/5cae5a2aa1e57c937adfb2f135e1d14c703be055d8f3e680e5abf8dbbbbeedd5.jpg)

![](images/884d1c780c51c28b49b4ba0987b323e4df8a990eb7b2057ef9e5e7f678a08d09.jpg)

![](images/bb73272e433eedb897d54dc5dc944c2a6fd7ee879469370d24792d410c1936a6.jpg)  
Figure 4: Example visuomotor manipulation tasks. We evaluate two image-conditioned manipulation tasks across two robotic manipulators. The first two rows show tabletop pick-and-place, where the robot must grasp a pan and place it on the target burner. The last two rows show drawer manipulation, where the robot must grasp the handle and open the drawer.

Table 3: VISUOMOTOR MANIPULATION RESULTS. Success rates, executed end-effector distance, and completion time for drawer opening and tabletop pick-and-place on the Dynaarm and Franka. We report full task success $\mathrm { S R } _ { \mathrm { t a s k } } .$ subtask success $\mathrm { S R } _ { \mathrm { s u b } }$ , accumulated end-effector distance, and solve time. DDIM<sup>†</sup> denotes the raw diffusion output deployed on a floating gripper without embodiment constraints and is shown as an upperreference point rather than as an embodiment-aware method. Among embodiment-aware methods, our L-BFGS guidance achieves the highest task success in all settings and generally reduces solve time. Best values are bold, second-best are underlined.

<table><tr><td rowspan="2">Task</td><td rowspan="2">Method</td><td colspan="4">Dynaarm</td><td colspan="4">Franka</td></tr><tr><td> $SR_{task [ \% ] \uparrow}$ </td><td> $SR_{sub [ \% ] \uparrow}$ </td><td>Dist [m]↓</td><td>Time [s]↓</td><td> $SR_{task [ \% ] \uparrow}$ </td><td> $SR_{sub [ \% ] \uparrow}$ </td><td>Dist [m]↓</td><td>Time [s]↓</td></tr><tr><td rowspan="5">Drawer</td><td> $\mathsf{DDIM^{\dagger}}$ </td><td>99.5±1.0</td><td>100.0±0.0</td><td>0.66±0.00</td><td>7.67±0.10</td><td>99.5±1.0</td><td>100.0±0.0</td><td>0.66±0.00</td><td>7.67±0.10</td></tr><tr><td>DDIM [27]</td><td>79.5±1.9</td><td>84.0±0.8</td><td>0.68±0.01</td><td>9.00±0.06</td><td>57.8±2.2</td><td>66.3±3.1</td><td>0.69±0.00</td><td>10.06±0.43</td></tr><tr><td>Gradient Guidance (σ)</td><td>79.8±1.7</td><td>85.0±1.6</td><td>0.69±0.01</td><td>9.03±0.18</td><td>57.8±7.8</td><td>66.8±6.8</td><td>0.70±0.00</td><td>10.29±0.42</td></tr><tr><td>Gradient Guidance ( $x_t$ -nudge)</td><td>79.8±3.3</td><td>83.8±3.0</td><td>0.68±0.01</td><td>9.09±0.04</td><td>58.0±2.4</td><td>67.0±2.8</td><td>0.70±0.01</td><td>10.21±0.43</td></tr><tr><td>Ours (L-BFGS)</td><td>81.8±2.9</td><td>87.3±2.9</td><td>0.73±0.01</td><td>8.12±0.19</td><td>60.0±3.4</td><td>70.0±1.6</td><td>0.78±0.02</td><td>9.42±0.44</td></tr><tr><td rowspan="5">Pick and Place</td><td> $\mathsf{DDIM^{\dagger}}$ </td><td>62.5±6.2</td><td>95.3±2.1</td><td>1.70±0.05</td><td>7.39±0.07</td><td>62.5±6.2</td><td>95.3±2.1</td><td>1.70±0.05</td><td>7.39±0.07</td></tr><tr><td>DDIM [27]</td><td>26.0±2.0</td><td>63.2±3.3</td><td>1.21±0.01</td><td>9.33±0.20</td><td>44.0±4.5</td><td>89.0±3.4</td><td>0.98±0.04</td><td>8.94±0.35</td></tr><tr><td>Gradient Guidance (σ)</td><td>29.3±4.3</td><td>69.3±4.6</td><td>1.21±0.01</td><td>10.08±0.73</td><td>41.0±1.4</td><td>92.3±1.3</td><td>1.00±0.00</td><td>8.99±0.32</td></tr><tr><td>Gradient Guidance ( $x_t$ -nudge)</td><td>26.3±3.4</td><td>63.2±5.0</td><td>1.19±0.01</td><td>9.56±0.57</td><td>43.3±1.7</td><td>90.3±3.8</td><td>0.97±0.01</td><td>9.22±0.33</td></tr><tr><td>Ours (L-BFGS)</td><td>41.8±3.8</td><td>66.0±3.6</td><td>1.46±0.03</td><td>8.40±0.49</td><td>67.0±3.4</td><td>98.3±1.3</td><td>1.21±0.01</td><td>8.02±0.41</td></tr></table>

## 8.5.0 Kinematic Feasibility Surrogate

For the grasping experiments in Section 5.1, we approximate the distance-to-reachability cost $J _ { \mathrm { I K } }$ with a differentiable surrogate $\hat { J } _ { \mathrm { I K } }$ . The surrogate replaces repeated calls to an IK solver during denoising with a cheap neural approximation, while still providing gradients with respect to the generated grasp pose. In addition to a scalar reachability estimate, the network predicts a corrective task-space offset and a joint configuration used for forward-kinematics consistency during training.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1: Kinematic feasibility surrogate $\hat{J}_{\text{IK}}$: architecture and training
Const: $H = 256$, $\text{dof} \in \{6, 7\}$, act = GELU; epochs = 5, batch = 4096, $\text{lr} = 3 \times 10^{-4}$;
$\lambda_{\text{err}} = 10$, $\lambda_{\text{pose}} = 1$, $\lambda_{\text{fk}} = 1$
Function IKMLPNet ($p$):
    // $p = [\text{pos}(3)$, $\text{rot6d}(6)]$; quat input converted
    $h \leftarrow \text{GELU}(\text{Linear}(9, 256)(p))$ // input embedding
    $h \leftarrow \text{GELU}(\text{Linear}(256, 256)(h))$ // backbone block 1
    $h \leftarrow \text{GELU}(\text{Linear}(256, 256)(h))$ // backbone block 2
    $\hat{q} \leftarrow \text{Linear}(128, \text{dof})(\text{GELU}(\text{Linear}(256, 128)(h)))$ // joint config (for FK loss)
    $\hat{J}_{\text{IK}} \leftarrow \text{softplus}(\text{Linear}(64, 1)(\text{GELU}(\text{Linear}(256, 64)(h))))$ // reachability $\geq 0$ $\Delta p \leftarrow \text{Linear}(128, 9)(\text{GELU}(\text{Linear}(256, 128)(h)))$ // pose correction
    return $\hat{q}$, $\hat{J}_{\text{IK}}$, $\Delta p$
// Data: $10^5$ targets labeled offline by reference IK solver
sample $p = [\text{pos}, \text{rot}] \sim \mathcal{U}([-2, 2]^3) \times SO(3)$ $(q^*, J^*, p^{\text{reach}}) \leftarrow \text{IKSolver}(p)$
// Pre-Compute Dataset
$\theta \leftarrow \text{init IKMLPNet}$; opt $\leftarrow \text{AdamW}$; sched $\leftarrow \text{CosineAnnealing}$
for epoch = 1 to 50 do
    for minibatch ($p$, $J^*, p^{\text{reach}}$) do
        $\hat{q}$, $\hat{J}_{\text{IK}}$, $\Delta p \leftarrow \text{IKMLPNet}(p)$ $\mathcal{L}_{\text{err}} \leftarrow ||\hat{J}_{\text{IK}} - J^*||^2$ $\mathcal{L}_{\text{pose}} \leftarrow \text{KeypointMSE}(p + \Delta p, p^{\text{reach}})$ $\mathcal{L}_{\text{fk}} \leftarrow \text{KeypointMSE}(FK(\hat{q}), p^{\text{reach}})$ $\mathcal{L} \leftarrow \lambda_{\text{err}}\mathcal{L}_{\text{err}} + \lambda_{\text{pose}}\mathcal{L}_{\text{pose}} + \lambda_{\text{fk}}\mathcal{L}_{\text{fk}}$
        opt.step($\nabla_\theta\mathcal{L}$)
    sched.step()
return $\theta$
</div>

Network architecture. The surrogate is a feed-forward MLP that maps a single task-space wrist target to a reachability estimate and a corrective offset. The target is parameterized as a 9-D vector $\displaystyle p = [ t \in \mathbb { R } ^ { 3 } , \ r _ { \mathrm { 6 D } } \in \mathbb { R } ^ { 6 } ]$ , where t is the wrist position in the robot base frame and $r _ { \mathrm { G D } }$ is the continuous 6-D rotation representation (the first two columns of the rotation matrix, with the full matrix recovered by Gram–Schmidt).

A quaternion input (t+ wxyz) is accepted and converted internally. The backbone is an input embedding Linear(9→256) followed by two hidden blocks of width 256; every block is Linear→ GELU with no normalization and no dropout. Three heads branch from the final 256-D feature:

• Reachability head $\hat { J } _ { \mathrm { I K } } \colon$ : Linear $( 2 5 6 \to 6 4 ) \to \mathrm { G E L U } \to \mathrm { L i n e a r } ( 6 4 \to 1 )$ followed by a softplus, which guarantees a non-negative scalar distance ${ \hat { J } } _ { \mathrm { I K } } ( \pmb { p } ) \geq 0$ rather than a feasibility logit. It regresses the residual pose error of the reference IK solver; reachability is decided by the threshold $\hat { J } _ { \mathrm { I K } } \leq \varepsilon _ { \mathrm { I K } }$

• Pose-correction head $\Delta \pmb { p } \in \mathbb { R } ^ { 9 } ;$ : an additive offset (in the same $[ t , r _ { \mathrm { 6 D } } ]$ space) that nudges an infeasible target toward the reachable set; it supplies the guidance gradient during denoising.

• Configuration head $\hat { \pmb q } ^ { \mathrm { ~ \alpha ~ } } \in \mathbb { R } ^ { \mathrm { d o f } }$ : predicts a joint solution, used for the forward-kinematics consistency loss and for visualization.

The network has $\approx 2 . 2 \times 1 0 ^ { 5 }$ parameters (≈ 218k for the 6-DoF arm, ≈ 219k for the 7-DoF arm). Linear layers use Kaiming-normal initialization.

Table 4: Kinematic feasibility surrogate accuracy. Held-out accuracy of $\hat { J } _ { \mathrm { I K } }$ on uniformly sampled task-space targets. RMSE reports the error relative to the ground-truth reachability cost $J _ { \mathrm { G T } }$ . Ranking AUC reports how well $- \hat { J } _ { \mathrm { I K } }$ predicts cuRobo IK solvability under a 5 mm/0.10 rad tolerance. Both arms achieve millimeter-level prediction error, and high AUC values show that lower surrogate costs reliably rank targets as more reachable.

<table><tr><td>Embodiment</td><td> $\hat{J}_{\text{IK}}$  RMSE [mm]</td><td>Ranking AUC</td></tr><tr><td>Franka (7 DoF)</td><td>2.9</td><td>0.993</td></tr><tr><td>Dynaarm (6 DoF)</td><td>5.2</td><td>0.930</td></tr></table>

Training-data generation. We generate training data separately for each embodiment, namely the Franka and Dynaarm. For each sample, we draw the target position uniformly from a $[ - 2 , 2 ] ^ { 3 }$ m box around the robot base and sample the target orientation from a normalized Gaussian quaternion, $\pmb q \sim \mathcal { N } ( 0 , I _ { 4 } )$ . Each target is labeled by solving an inverse-kinematics (IK) problem, which returns the closest reachable pose and the corresponding joint configuration. For each embodiment, we generate $1 0 ^ { 5 }$ training targets and $2 \times 1 0 ^ { 3 }$ validation targets.

Training objective and optimization. Training minimizes a weighted regression objective (no classification loss):

$$
\mathcal {L} = \lambda_ {\mathrm{err}} \left\| \hat {J} _ {\mathrm{IK}} - J ^ {\mathrm{GT}} \right\| _ {2} ^ {2} + \lambda_ {\mathrm{pose}} \mathcal {L} _ {\mathrm{pose}} + \lambda_ {\mathrm{fk}} \mathcal {L} _ {\mathrm{fk}} + \lambda_ {\boldsymbol {q}} \left\| \hat {\boldsymbol {q}} - \boldsymbol {q} ^ {\star} \right\| _ {2} ^ {2},\tag{22}
$$

where $\mathcal { L } _ { \mathrm { p o s e } }$ is a keypoint MSE between the corrected pose $\pmb { p } + \Delta \pmb { p }$ and the reachable reference pose, and ${ \mathcal L } _ { \mathrm { f k } }$ is a forward-kinematics consistency MSE between $\mathrm { F K } ( \hat { \pmb q } )$ and that reference pose. We use $\lambda _ { \mathrm { e r r } } = 1 0 , \lambda _ { \mathrm { p o s e } } = 1 , \lambda _ { \mathrm { f k } } = 1$ , and $\lambda _ { q } = 0$ (direct joint supervision is off; the configuration head is still trained implicitly through ${ \mathcal L } _ { \mathrm { f k } } )$ . Since $\lambda _ { q } = 0 .$ , the joint-supervision term vanishes and the loss reduces to the three-term objective used in Algorithm 1. Optimization uses AdamW with learning rate $3 \times 1 0 ^ { - 4 }$ , a cosine-annealing schedule, batch size 4096, 2 − 5 epochs, and fixed seed 42. The resulting held-out surrogate accuracy is reported in Table 4.

## 8.5.1 Closed-Loop Rollout Model

This section specifies the differentiable closed-loop rollout map $\phi ( x ; q _ { 0 } ; \kappa )$ used to evaluate the controller-level executability cost $J _ { \mathrm { d y n } }$ (8) in the dynamic-manipulation experiments (Section 5.2). The map predicts how a planned end-effector trajectory would actually be executed by the controller, so that $\boldsymbol { J } _ { \mathrm { { d y n } } }$ can score the executed poses against the planned references. Its inputs are

• the N predicted end-effector waypoints $x \ = \ [ x _ { 0 \mid k } , \ldots , x _ { N - 1 \mid k } ]$ , where each $\begin{array} { r l } { x _ { t \vert k } } & { { } = } \end{array}$ $( p _ { t } ^ { \mathrm { r e f } } , R _ { t } ^ { \mathrm { r e f } } ) \in \mathrm { S E } ( 3 )$ is expressed in the frame $E _ { 0 }$ of the end-effector at replanning time k;

• the measured arm configuration $q _ { 0 } \in \mathbb { R } ^ { n }$ (with $n = 6$ for the Dynaarm and $n = 7$ for the Franka); and

• the fixed Cartesian impedance controller parameters $\kappa ,$

and its output is the sequence of planned executed poses $\hat { x } _ { t } = \phi ( x ; q _ { 0 } ; \kappa ) _ { t }$

Per-step controller update. We approximate the closed-loop controller with a kinematic rollout. The model assumes that gravity compensation is handled by feed-forward torques, and simulates only the tracking behavior of a damped-least-squares inverse-kinematics controller. At each reference step t, the rollout starts from the current joint configuration $q _ { t } .$ , computes the end-effector pose $T _ { E } ^ { \bar { W } } ( q _ { t } ) = ( p _ { t } ^ { e e } , R _ { t } ^ { e e } )$ , and updates the joints through the following four steps.

1. Task-space error. We first compute the geometric pose error between the reference pose and the current end-effector pose:

$$
\Delta x _ {t} = \left[ \begin{array}{c} p _ {t} ^ {\mathrm{ref}} - p _ {t} ^ {e e} \\ \omega_ {t} ^ {\mathrm{err}} \end{array} \right] \in \mathbb {R} ^ {6}, \qquad \omega_ {t} ^ {\mathrm{err}} = \log \left(R _ {t} ^ {\mathrm{ref}} R _ {t} ^ {e e, \top}\right) ^ {\vee} \in \mathbb {R} ^ {3}.
$$

Here, $\mathrm { l o g ( \cdot ) ^ { v } : S O ( 3 ) \to \mathbb { R } ^ { 3 } }$ denotes the logarithmic map from rotations to axis-angle vectors. Optionally, this error can be weighted by a diagonal task-space stiffness $K _ { p } ^ { \mathrm { t a s k } }$ . In all reported experiments, we set $K _ { p } ^ { \mathrm { t a s k } } = I ,$ , so $\Delta { x } _ { t }$ is the unweighted geometric pose error.

2. Resolved-rate joint update. The task-space error is mapped to a joint-space increment with a damped-least-squares resolved-rate update:

$$
\delta q _ {t} = J (q _ {t}) ^ {\top} \big (J (q _ {t}) J (q _ {t}) ^ {\top} + \lambda^ {2} I _ {6} \big) ^ {- 1} \Delta x _ {t}, \quad \lambda = 0. 0 5,
$$

where $J ( q _ { t } )$ is the geometric end-effector Jacobian.

3. Authority limits. Before applying the update, we clip the joint increment to the motion that the robot can realize within one reference step. The per-joint bound is

$$
\bar {\delta q} = \min \left(\dot {q} _ {\mathrm{max}} \Delta t _ {\mathrm{ref}}, \frac {\tau_ {\mathrm{max}}}{k _ {p} ^ {\mathrm{joint}}}\right),
$$

where $\dot { q } _ { \mathrm { m a x } }$ and $\tau _ { \mathrm { m a x } }$ are the robot’s velocity and effort limits, and $k _ { p } ^ { \mathrm { j o i n t } }$ is the joint-space PD stiffness specified by the robot model. This bound captures both velocity limits and finite torque authority.

4. PD lag and integration. Finally, we account for the fact that the low-level PD controller closes only part of the commanded joint-space gap during one reference step. We model this with a first-order lag factor $\alpha _ { \mathrm { e f f } }$ and integrate the clipped increment:

$$
q _ {t + 1} = \mathrm{clip} \Big (q _ {t} + \alpha_ {\mathrm{eff}} \odot \mathrm{clip} (\delta q _ {t}, \pm \bar {\delta q}), \underline {{q}}, \overline {{q}} \Big),
$$

with

$$
\alpha_ {\mathrm{eff}, j} = 1 - \left(1 - \frac {k _ {p , j} ^ {\mathrm{joint}}}{k _ {p , j} ^ {\mathrm{joint}} + k _ {d , j} ^ {\mathrm{joint}} / \Delta t _ {\mathrm{in}}}\right) ^ {n _ {\mathrm{sub}}}.
$$

Here, $k _ { d } ^ { \mathrm { j o i n t } }$ denotes the joint-space damping gains from the robot model, $\Delta t _ { \mathrm { i n } }$ is the inner control timestep, and $n _ { \mathrm { s u b } }$ is the number of inner controller steps per reference interval. The final clipping enforces the joint limits $\underline { { q } }$ and ${ \overline { { q } } } .$

Executability cost. Given the executed poses $\hat { x } _ { t } = \phi ( x ; q _ { 0 } ; \kappa ) _ { i }$ , the executability cost (8) is the step-weighted tracking residual

$$
J _ {\mathrm{dyn}} = \sum_ {t = 0} ^ {N - 1} c _ {t} \left(\left\| \hat {p} _ {t} - p _ {t} ^ {\mathrm{ref}} \right\| ^ {2} + \left\| (\hat {R} _ {t} - R _ {t} ^ {\mathrm{ref}}) \mathcal {K} \right\| ^ {2}\right),\tag{23}
$$

where the orientation term is scored through three orthogonal keypoints $\kappa$ at distance 0.15 m. The position and orientation terms are equally weighted, and the step weights increase linearly as $c _ { t } = t { + } 1$

Controller offset. Using (23) directly as a guidance signal can produce an undesirable feedback effect. Since $\boldsymbol { J } _ { \mathrm { { d y n } } }$ penalizes tracking error, the easiest way to reduce this cost is not necessarily to make the motion more executable, but to make the reference itself less demanding. This occurs because the actions passed to the impedance controller also define the target reference used to evaluate tracking. As a result, the gradient can reduce the dynamic cost by suppressing fast or aggressive motions, rather than by improving their closed-loop execution.

For example, suppose the policy predicts a fast upward motion in the $+ z$ direction. If the controller cannot track this motion from the nominal command alone, directly penalizing the resulting tracking residual would encourage the policy to reduce the upward acceleration. This can be undesirable, since the acceleration may be essential for completing the task. Instead, if possible, we would like to preserve the intended reference if the controller could track it with the help of a bounded feed-forward command offset.

$$
J _ {\mathrm{dyn}} ^ {\star} = \min _ {o \in \mathcal {B}} J _ {\mathrm{dyn}} \big (\phi (x + o, q _ {0}; \kappa), x \big) + \lambda_ {\mathrm{reg}} \mathcal {R} (o).\tag{24}
$$

Here, $x + o$ denotes the offset command sent to the controller, while the cost $\boldsymbol { J } _ { \mathrm { { d y n } } }$ is still evaluated against the original, unshifted reference x. The regularizer $\mathcal { R } ( o )$ penalizes undesirable offsets, such as large or non-smooth corrections, and B bounds the admissible offset magnitude.

The resulting cost $J _ { \mathrm { d y n } } ^ { \star }$ measures the tracking error that remains after the best bounded feed-forward correction has been applied. Thus, the guidance signal captures the part of the reference that cannot be absorbed by the controller, instead of rewarding the policy for simply slowing down the motion.

## 8.5.2 Implementation Details

Kinematic feasibility model. For the grasping experiments, we approximate the distance-toreachability cost $J _ { \mathrm { I K } }$ with a differentiable surrogate $\hat { J } _ { \mathrm { I K } }$ , implemented as a small MLP trained on reachable and unreachable task-space targets for each robot embodiment. During optimization, $\hat { J } _ { \mathrm { I K } }$ is used in place of the ideal distance in Eq. (5), making the feasibility cost cheap to evaluate and differentiable with respect to the generated grasp representation. Network architecture, training-data generation, and held-out accuracy are reported in Appendix 8.5.0.

Constrained NLP solver. For the constrained formulation, we use IPOPT [34] with a maximum of 45 iterations. The optimization variables are the initial denoising state $x _ { K }$ and the correction variables $\{ \delta _ { k } \} _ { k = 1 } ^ { K }$ . The formulation is

$$
\begin{array}{l l l} \min _ {x _ {K}, \{\delta_ {k} \} _ {k = 1} ^ {K}} & L _ {\delta} + L _ {\mathrm{IK}} \\ \text {s.t.} & x _ {k - 1} = \mu_ {\theta} (x _ {k}, k) + \sigma_ {k} \delta_ {k}, & k = K, \ldots , 1, \\ & J _ {\mathrm{IK}} (x _ {0}) \leq \varepsilon_ {\mathrm{IK}}, \\ & - 1 \leq x _ {K} \leq 1, \\ & - 1 \leq \delta_ {k} \leq 1, & k = 1, \ldots , K, \end{array}\tag{25}
$$

with $\varepsilon _ { \mathrm { I K } } = 0 . 0 1$ . The objective terms are

$$
L _ {\delta} = \lambda_ {\delta} \sum_ {k = 1} ^ {K} \| \delta_ {k} \| _ {2} ^ {2},\tag{26}
$$

$$
L _ {\mathrm{IK}} = \lambda_ {\mathrm{IK}} \sum_ {k = 0} ^ {K} J _ {\mathrm{IK}} (x _ {k}),\tag{27}
$$

where $\lambda _ { \delta } ~ = ~ 1 . 0$ and $\lambda _ { \mathrm { I K } } ~ = ~ 1 . 0$ . When additional task constraints are present, further cost terms can be added to the objective. For example, we include an object-penetration penalty $\begin{array} { r } { L _ { \mathrm { o b j } } ~ = ~ \lambda _ { \mathrm { o b j } } \sum _ { k \in \mathcal { K } _ { \mathrm { o b i } } } P ( x _ { k } ) } \end{array}$ , with $\lambda _ { \mathrm { o b j } } ~ = ~ 1 . 0$ and ${ \cal K } _ { \mathrm { o b j } } = \{ 0 , 1 \}$ , where $P ( x _ { k } )$ discourages hand–object interpenetration during the final denoising steps.

Differentiable NLS relaxation. For the differentiable relaxation, we use Theseus with 15 Levenberg– Marquardt iterations. Instead of enforcing the reverse dynamics and terminal feasibility as hard constraints, we optimize the soft objective

$$
L _ {\mathrm{NLS}} = L _ {\mathrm{rev}} + L _ {\mathrm{rev,0}} + L _ {\delta} + L _ {\mathrm{clamp}} + L _ {\mathrm{IK,cost}} + L _ {\mathrm{IK,term}} + L _ {\mathrm{IK,path}},\tag{28}
$$

with

$$
L _ {\mathrm{rev}} = 1 0. 0 \sum_ {k = 1} ^ {K} \| x _ {k - 1} - \mu_ {\theta} (x _ {k}, k) - \sigma_ {k} \delta_ {k} \| _ {2} ^ {2},\tag{29}
$$

$$
L _ {\mathrm{rev,0}} = 5 0. 0 \left\| x _ {0} - \mu_ {\theta} (x _ {1}, 0) - \sigma_ {1} \delta_ {1} \right\| _ {2} ^ {2},\tag{30}
$$

$$
L _ {\delta} = 1. 0 \sum_ {k = 1} ^ {K} \| \delta_ {k} \| _ {2} ^ {2},\tag{31}
$$

$$
L _ {\mathrm{clamp}} = 1 0 0. 0 \sum_ {k = 1} ^ {K} \| \max (0, \delta_ {k} - 1) + \max (0, - 1 - \delta_ {k}) \| _ {2} ^ {2},\tag{32}
$$

$$
L _ {\mathrm{IK,cost}} = 1. 0 \sum_ {k = 1} ^ {K} J _ {\mathrm{IK}} (x _ {k}) ^ {2},\tag{33}
$$

$$
L _ {\mathrm{IK,term}} = 1 0. 0 \max (0, J _ {\mathrm{IK}} (x _ {0}) - 0. 0 0 5) ^ {2},\tag{34}
$$

$$
L _ {\mathrm{IK,path}} = 1 0. 0 \sum_ {k = 1} ^ {K} \max (0, J _ {\mathrm{IK}} (x _ {k}) - 0. 0 0 5) ^ {2}.\tag{35}
$$

This relaxation preserves the same denoising structure as the constrained NLP but replaces hard feasibility with weighted penalties.

Gradient-guidance baseline. For the gradient-guidance baseline, we apply a per-step guidance update with scale

$$
s _ {k} = g w (p _ {k}) \frac {\beta_ {k}}{\sqrt {1 - \alpha_ {k}}},\tag{36}
$$

where $g = 0 . 2 , p _ { k } = 1 - k / K$ , and $w ( p _ { k } ) = \cos ^ { 2 } ( \pi p _ { k } )$ . The factor $\beta _ { k } / \sqrt { 1 - \alpha _ { k } }$ accounts for the diffusion noise schedule. This baseline follows the same denoising schedule as the proposed method, but uses a scaled cost gradient instead of solving an optimization problem at each guided step.

Note that the gradient-guidance baseline uses a cosine schedule and the optimization-guided methods (IPOPT and Theseus) use constant schedulers. Both scheduler types were tested for all methods and the best performing one selected and reported. Similarly, various scales $g$ were tested, and $g = 0 . 2$ yielded the best results.

Solver settings. Unless stated otherwise, the grasping experiments use IPOPT with 45 iterations and Theseus with 15 Levenberg–Marquardt iterations. For the image-conditioned dynamic manipulation experiments, we use the lightweight L-BFGS backend during the final denoising steps to keep online replanning tractable, which is detailed below.

Visuomotor manipulation guidance. For the dynamic-manipulation experiments, we compare three ways of using the controller-level executability cost $\boldsymbol { J } _ { \mathrm { { d y n } } }$ from (8). All three methods use the same frozen diffusion model, replanning horizon, and differentiable rollout model from Appendix 8.5.1. The cost is evaluated on the final denoised action chunk $x _ { 0 }$ . The methods differ only in how the correction enters the sampling process.

The first baseline is an $x _ { t }$ -nudge guidance method. At each denoising step, we compute the guidance gradient $g$ from $J _ { \mathrm { d y n } }$ and update

$$
x _ {t} \leftarrow x _ {t} + \lambda \bar {\alpha} _ {t} g.\tag{37}
$$

The standard DDIM update is then applied to the nudged sample. Since the diffusion network receives the corrected $x _ { t } .$ , both the denoised estimate $\hat { x } _ { 0 } ( x _ { t } )$ and the predicted noise $\boldsymbol { \epsilon } _ { \boldsymbol { \theta } } ( x _ { t } , t , \boldsymbol { c } )$ are affected by the correction. The factor $\bar { \alpha } _ { t }$ concentrates the update near the final denoising steps. We use $\lambda = 1$ and $w _ { \mathrm { d y n } } = 5$

The second baseline is σ-guidance, which leaves the network input unchanged and injects the guidance gradient through the stochastic channel of the DDIM update. After evaluating the network at the original $x _ { t } ,$ , we compute the deterministic DDIM mean $\mu _ { t }$ and set

$$
x _ {t - 1} = \mu_ {t} + \sigma_ {t}   g, \qquad \sigma_ {t} = \eta \sqrt {\frac {1 - \bar {\alpha} _ {t - 1}}{1 - \bar {\alpha} _ {t}} \left(1 - \frac {\bar {\alpha} _ {t}}{\bar {\alpha} _ {t - 1}}\right)}.\tag{38}
$$

Its magnitude follows $\sigma _ { t }$ , which is largest at noisy steps and vanishes at the clean step. Therefore, this guidance is active only for $\eta > 0 ;$ we use $\eta = 1$

In both guidance schemes, the guidance gradient g depends on the controller offsets $^ { O , }$ which are themselves the solution of the inner optimization problem in (24). We therefore treat the correction as a two-stage (bilevel) optimization: at each denoising iteration we first compute the optimal offsets o with a few inner gradient steps, and then evaluate the guidance gradient with these o held fixed.

The optimization-based variant uses direct-shooting L-BFGS. Instead of modifying each denoising step online, it optimizes the full sampling process once per replanning step. The decision variable contains the initial latent sample $x _ { T }$ , the injected denoising noises $\left\{ \varepsilon _ { i } \right\}$ , and the feed-forward offset o:

$$
z = \left[ x _ {T} \mid \varepsilon \mid o \right].
$$

We optimize z with torch.optim.LBFGS using learning rate 0.1, at most 10 inner iterations, at most 5 closure evaluations, history size 10, strong-Wolfe line search, and one outer step. Each closure re-runs the full DDIM unroll with the frozen diffusion network, evaluates the differentiable rollout on the terminal chunk $x _ { 0 }$ , and back-propagates through both the sampler and rollout. Since the direct-shooting parameterization explicitly unrolls the diffusion dynamics, the generated trajectory remains consistent with the sampler by construction.

## 8.5.3 Base-pose difficulty stratification

We stratify the evaluation by base-pose difficulty: poses are classified as easy or hard based on the success rate of the DDIM approach, which hard-projects the un-guided diffusion output DDIM†. Base poses with lower success rates are considered more difficult. Figure 5 highlights the drop in success rate when projecting the diffusion output onto the inverse-kinematic feasible set, across both embodiments (Dynaarm and Franka).

![](images/de4cb19cf38e16255b71ebc251303d75674cadf7d612edca6241f5bcb5af4c79.jpg)  
Figure 5: Success rate by base-pose. Per-method SR on the Dynaarm and Franka for each base-pose on a smaller subset of objects, with poses stratified by the success rate of the DDIM approach.

Based on Figure 5, we consider poses nominal, npp and npp, npn as hard base pose configurations for the Dynaarm and Franka, respectively. The nominal base pose is given as $[ - 0 . 6 5 , 0 , 0 ]$ . The index $p$ refers to a shift in positive direction, n in negative direction by 0.25 from the nominal pose. Pose npp therefore refers to $[ - 0 . 6 5 - 0 . 2 5 , 0 . 2 5 , 0 . 2 5 ] = [ - 0 . 9 , 0 . 2 5 , 0 . 2 5 ]$

Figure 6 reports per-pose $( \mathrm { S R } ^ { ( 1 ) }$ , faded outer) and per-grasp $\mathrm { ( S R ^ { a l l } }$ , hatched inner) success rates across both arms. DDIM† serves as an upper bound on achievable success rate given the trained diffusion prior. The trend of decreased success rates $( S R ^ { ( 1 ) } )$ across all methods for the more challenging arm base poses is particularly evident for the Franka arm. Notably, both baselines (Gradient guidance and Projection guidance) suffer significant drops in success rate (from 70 to 23 and 57 to 9 respectively). The optimization-guided approaches (IPOPT and Theseus) are more robust to base pose changes with smaller drops in success rate (from 72 to 70 and 65 to 56 respectively). The Dynaarm shows similar trends, although less pronounced. Gradient guidance and Projection guidance drop from 59 to 58 and 28 to 17 while the optimization-guided IPOPT drops from 71 to 68. Interestingly, Theseus actually increases in success rate when changing from the easy to hard base pose configuration, going from 61 to 67. Additionally, while Gradient guidance outperforms Theseus, and almost approaches the success rate achieved by IPOPT, on the Franka arm in the easy base pose category, scores drop significantly on the hard category, where Theseus clearly outperforms Gradient guidance.

![](images/64d8acd394f0e8ddccd24a740de88648aeb9792cd8cde887f05a00cd9ebf7a64.jpg)  
Figure 6: Grasp success by base-pose difficulty. Per-method success rates on the Dynaarm and Franka, split into easy and hard base-pose bins. Each bar overlays the per-pose success rate $( \mathrm { S R } ^ { \mathrm { \bar { ( 1 ) } } }$ , faded outer) and the per-grasp success rate $( \mathrm { S } \hat { \mathrm { R } } ^ { \mathrm { a l l } }$ , hatched inner). Our optimization-constrained variants (IPOPT, Theseus) dominate every (arm, difficulty) bin and degrade the least as poses harden.

## 8.6 Evaluation Metrics

We collect the definitions of all metrics used in the main text and in the collision tables here.

## Grasp success and feasibility.

$\mathrm { S R } ^ { ( 1 ) }$ (one-axis grasp success): a grasp is successful if it remains stable in simulation under at least one pull axis. To evaluate a grasp, we apply 3 N pulls along the ±x, ±y, and ±z axes of the object frame, grouped into three axis-wise phases. A grasp passes an axis if the object’s center of mass remains within 5 cm of its initial position throughout that phase.

• SR<sup>all</sup> (all-axis grasp success): a grasp is successful under the same evaluation if it passes all three axis-wise pull phases. This is the stricter counterpart of SR<sup>(1)</sup>, requiring the same grasp to remain stable under pulls along all object-frame axes.

$\mathrm { S R ^ { I K } }$ (kinematic feasibility): the fraction of predictions that are reachable by the target arm. For the snap-to-feasible DDIM baseline, this metric is computed before projection (Footnote 2).

• CR (collision rate, collision tables only): the fraction of predicted grasps whose hand geometry penetrates an obstacle by more than 3 mm. We compute hand surface points from the predicted grasp pose and query their signed distances to the obstacle SDFs. A grasp is marked as colliding if the maximum penetration over all hand points and obstacles exceeds the tolerance. This is a purely geometric check and does not involve physics simulation.

$\mathrm { S R ^ { T o t } }$ (total success, collision tables only): the fraction of predictions that are both graspsuccessful and collision-free. We compute this by zeroing out the per-grasp success indicator for any grasp in collision, and then averaging over all grasps.

## Grasp quality.

$E _ { \mathrm { f c } }$ (force-closure violation): the GraspQP span-metric residual [31], capturing how close the grasp is to force closure (lower is better). In practice, we use the method of [27] to resample contact points and rely on the four-sided friction pyramid using $\mu = 0 . 2$

$Q _ { 1 }$ (grasp wrench quality, higher is better): the Ferrari-Canny ε-metric, given by the radius of the largest origin-centered ball inside the convex hull of the grasp wrench set.

## 8.7 Collision-environment results

For the collision-aware experiments, we use the same diffusion architecture as in the main grasping experiments, but train it on the full Dataset from [31] to improve robustness under the tighter scene constraints and to support sim-to-real transfer. In addition, the initial wrist poses $x _ { K }$ are sampled above the table plane to warm-start diffusion, providing a better initialization from which gradient-based methods can more easily find solutions.

Table 5 evaluates collision-aware grasp generation across four constrained environments: floor, walls, clutter, and tunnels. The walls setting places a wall perpendicular to the arm base, whereas the tunnel setting places walls on both sides of the arm, forming a funnel-like constraint. All methods are evaluated on 4 objects with 10 grasps per object in each environment. Since these scenes substantially restrict the set of valid grasps, the experiment stresses the ability of each method to balance grasp quality, kinematic reachability, and collision avoidance.

We report the single-grasp success rate $\mathrm { S R } ^ { ( 1 ) }$ , inverse-kinematics success rate $\mathrm { S R ^ { I K } }$ , collision rate CR, and total success rate $\mathrm { S R ^ { T o t } }$ , where total success denotes grasps that both succeed and remain collision-free. All guided methods use a proximity cost based on the signed distance field (SDF),

$$
J _ {\mathrm{coll}} (x) = \max (0, s (x) + d _ {\mathrm{safe}}),\tag{39}
$$

with safety margin $d _ { \mathrm { s a f e } } = 5 \mathrm { c m }$ . This formulation provides a non-zero optimization signal before actual obstacle penetration occurs. For IPOPT, we include $J _ { \mathrm { c o l l } }$ as a cost term rather than a hard constraint to avoid overly restrictive local optimization landscapes.

Table 5: Grasp Predictions with Collision Avoidance: Comparison of methods across evaluation metrics (averaged over 4 objects and 10 grasps per object) for 4 different environments, with the walls oriented perpendicular to the arm base.

<table><tr><td rowspan="3">Arm</td><td rowspan="3">Method</td><td colspan="17">Environment</td></tr><tr><td colspan="4">Floor</td><td colspan="4">Walls</td><td colspan="4">Clutter</td><td colspan="5">Tunnels</td></tr><tr><td> $SR^{Tot}\uparrow$ </td><td> $SR^{(1)}\uparrow$ </td><td> $SR^{IK}\uparrow$ </td><td> $CR\downarrow$ </td><td> $SR^{Tot}\uparrow$ </td><td> $SR^{(1)}\uparrow$ </td><td> $SR^{IK}\uparrow$ </td><td> $CR\downarrow$ </td><td> $SR^{Tot}\uparrow$ </td><td> $SR^{(1)}\uparrow$ </td><td> $SR^{IK}\uparrow$ </td><td> $CR\downarrow$ </td><td> $SR^{Tot}\uparrow$ </td><td> $SR^{(1)}\uparrow$ </td><td> $SR^{IK}\up \uparrow$ </td><td> $CR\downarrow$ </td><td></td></tr><tr><td rowspan="5">Dynaarm</td><td>DDIM [27]</td><td>5.0</td><td>16.25</td><td>32.5</td><td>60.0</td><td>2.5</td><td>16.25</td><td>32.5</td><td>62.5</td><td>4.37</td><td>19.99</td><td>32.5</td><td>67.5</td><td>0.0</td><td>16.25</td><td>32.5</td><td>80.0</td><td></td></tr><tr><td>Gradient Guidance</td><td>33.12</td><td>33.12</td><td>100</td><td>0.0</td><td>17.5</td><td>19.99</td><td>100</td><td>25.0</td><td>28.12</td><td>28.12</td><td>100</td><td>0.0</td><td>30.63</td><td>30.62</td><td>100</td><td>0.0</td><td></td></tr><tr><td>Projection Guidance</td><td>19.38</td><td>22.49</td><td>100</td><td>20.0</td><td>0.63</td><td>5.62</td><td>100</td><td>20.0</td><td>0.0</td><td>0.0</td><td>75.0</td><td>0.0</td><td>14.37</td><td>19.37</td><td>100</td><td>22.5</td><td></td></tr><tr><td>Theseus</td><td>36.88</td><td>36.87</td><td>77.5</td><td>0.0</td><td>35.63</td><td>44.36</td><td>77.5</td><td>17.5</td><td>23.13</td><td>28.37</td><td>82.5</td><td>37.5</td><td>24.37</td><td>46.24</td><td>80.0</td><td>32.5</td><td></td></tr><tr><td>IPOPT</td><td>46.88</td><td>51.24</td><td>75.0</td><td>10.0</td><td>22.5</td><td>43.74</td><td>75.0</td><td>32.5</td><td>31.25</td><td>39.99</td><td>65.0</td><td>20.0</td><td>23.75</td><td>26.24</td><td>52.5</td><td>32.5</td><td></td></tr><tr><td rowspan="5">Franka</td><td>DDIM [27]</td><td>11.87</td><td>68.73</td><td>100</td><td>82.5</td><td>10.0</td><td>68.73</td><td>100</td><td>85.0</td><td>6.25</td><td>60.61</td><td>100</td><td>90.0</td><td>7.5</td><td>68.73</td><td>100</td><td>87.5</td><td></td></tr><tr><td>Gradient Guidance</td><td>22.5</td><td>22.49</td><td>100</td><td>0.0</td><td>20.62</td><td>20.62</td><td>100</td><td>30.0</td><td>23.13</td><td>23.74</td><td>100</td><td>17.5</td><td>18.12</td><td>18.12</td><td>100</td><td>5.0</td><td></td></tr><tr><td>Projection Guidance</td><td>58.13</td><td>60.61</td><td>100</td><td>2.5</td><td>43.13</td><td>45.61</td><td>100</td><td>2.5</td><td>0.0</td><td>0.0</td><td>57.5</td><td>0.0</td><td>45.0</td><td>47.49</td><td>100</td><td>10.0</td><td></td></tr><tr><td>Theseus</td><td>58.75</td><td>58.74</td><td>100</td><td>0.0</td><td>50.62</td><td>55.61</td><td>100</td><td>17.5</td><td>26.87</td><td>38.12</td><td>97.5</td><td>27.5</td><td>38.75</td><td>59.36</td><td>100</td><td>30.0</td><td></td></tr><tr><td>IPOPT</td><td>65.62</td><td>69.98</td><td>100</td><td>12.5</td><td>60.62</td><td>76.86</td><td>100</td><td>20.0</td><td>55.63</td><td>70.61</td><td>100</td><td>27.5</td><td>61.88</td><td>68.11</td><td>100</td><td>17.5</td><td></td></tr></table>

Across both embodiments, the optimization-based methods obtain the best total success in most environments. On the Franka, IPOPT achieves the highest $\mathrm { S R ^ { T o t } }$ in all four environments, reaching 65.62% on Floor, 60.62% on Walls, 55.63% on Clutter, and 61.88% on Tunnels. On the Dynaarm, IPOPT and Theseus also improve over the baselines in several constrained settings, with IPOPT performing best on Floor and Theseus on Walls. The main exception is the Tunnels environment, where Gradient Guidance achieves the highest $\mathrm { S R ^ { T o t } }$ for the Dynaarm. In this case, the local gradient signal appears to provide a favorable correction direction, reducing collisions without substantially degrading grasp quality.

![](images/9599585d2f2284cefcca95b659c3cf2e6b4b2178cff04267c5908a831aa0587c.jpg)  
Floor

![](images/57f2cf32d8563f0bf3d55c08ff76b9e582fc9a0afce639762d7517cf70a96be6.jpg)  
Walls

![](images/8281122d22d2911fa7e074fe3ad51f0d2b3936c87ad14d4104dd08d3922d8c6e.jpg)  
Clutter

![](images/e5a35e98e1b6f812a9346bbabb04f43383b9ddd4e6e2142a2e7532d42624b894.jpg)  
Tunnel  
Figure 7: Hardware Setup. Illustration of the four environments considered during hardware deployment. A video of all hardware experiments is provided in the supplementary material.

However, this behavior is not consistent across environments or embodiments. The same gradientbased update often satisfies one objective at the expense of another. For example, Gradient Guidance achieves a 0.0% collision rate in several Dynaarm environments, but this frequently comes with reduced grasp success. Similarly, Projection Guidance reaches a 0.0% collision rate in the Clutter environment for both arms, but also collapses to 0.0% grasp success, which corresponds in simulation to grasps being moved into free space rather than onto the object. These cases indicate that local gradient or projection updates can work well when the correction direction is aligned with the learned grasp prior, but can also move samples away from successful grasp modes when the objectives conflict. In contrast, the optimization-based formulations provide a more robust mechanism for trading off grasp quality, reachability, and collision avoidance, leading to higher total success in most of the evaluated settings.

## 8.7.1 Real-World Deployment

The dexterous manipulation task was additionally deployed on hardware using a Franka Panda arm equipped with the XHand. We evaluated both the gradient-based guidance baseline and our IPOPT-based method across all four environment settings, Floor, Walls, Clutter, and Tunnels, shown in Fig. 7. While the simulation study in Table 5 considers four objects, we restrict the hardware evaluation to two representative objects, Mug and Spices. Both methods are deployed zero-shot, without retraining or fine-tuning of the diffusion model. Additional hardware rollouts are shown in the supplementary video.

For the hardware evaluation, we generated multiple grasp candidates per method, object, and environment. In the Floor environment, we generated three grasps for each method-object pair. For the more constrained Walls, Clutter, and Tunnels environments, we generated five grasps per method-object pair to account for the smaller set of valid solutions. We then deployed all candidates that were executable by the hardware pipeline.

We note that the arm approach motion was planned using cuRobo, which performs collision avoidance during execution. This component is part of the deployment pipeline, but not part of our proposed method or evaluation claim: our method only optimizes the final grasp configuration with respect to grasp success, kinematic reachability, and environment collisions. Consequently, some generated grasps that satisfied the final-state criteria could not be deployed because no valid collision-free approach motion was found by the controller. We therefore interpret the hardware experiments as initial deployment evidence rather than a complete end-to-end benchmark of motion planning and grasp execution.

Within this protocol, the hardware experiments support the qualitative trend observed in Table 5. In particular, gradient guidance can produce collision-free final grasps while moving the hand away from useful object-centered grasp modes, leading to missed or unstable grasps even in the comparatively simple Floor environment, as illustrated in Fig. 8. In contrast, the IPOPT-based method more reliably preserved grasp structure in the deployed trials while accounting for the environmental constraints, as also shown in Fig. 8. These results provide ini tial evidence that the same optimization-guided corrections used in simulation can transfer zeroshot to physical deployment.

Further hardware experiments for all four environments can be found in the supplementary video.

![](images/71fe3b5b6cc09f2ed4d1eab5b39ad45ea9d3a24094e7b321df60ca7bf357bb42.jpg)  
Gradient Guidance

![](images/efff844c9d1fddf8f4f3dcf9ccb04cf893b8d42dcf0179d32c99bb9e64b3f554.jpg)  
Figure 8: Out-of-distribution failure case. Example of gradient guidance and IPOPT in the Floor environment. Gradient guidance respects the collision-avoidance objective with respect to the floor, but fails to grasp the object.  
IPOPT