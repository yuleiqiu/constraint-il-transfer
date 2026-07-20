# Temporal Logic Guidance for Action-Only Diffusion Policies with World Models

Moritz Zoellner, Anastasios Manganaris, and Rohan Paleja

Department of Computer Science, Purdue University West Lafayette, IN, USA

{zoellner, amangana, rpaleja}@purdue.edu

Abstract—Diffusion policies enable multimodal robot behavior but offer limited ability to choose among behavior modes at inference time, even though such control is desirable in humanrobot settings. Prior solutions to this lack of control have utilized Signal Temporal Logic (STL) to express human intentions and provide corresponding guidance for diffusion policy inference. However, these approaches can only guide diffusion policies that jointly generate future actions and states, increasing both complexity and runtime. We propose a novel guidance method for action-only diffusion policies that uses a separate learned world model to enable differentiable evaluation of STL robustness, with its gradient then injected into the diffusion process. This steers behavior toward constraint satisfaction without retraining, improving constraint adherence while preserving task performance. On the Can Transport task from Robomimic, our method maintains 100% task success while reducing constraint violations from over 80% for baseline methods to 4%. We also discuss extensions toward improved robustness and more complex constraints.

Index Terms—Diffusion Policies, Inference-Time Guidance, Signal Temporal Logic, Human-Robot Interaction

## I. INTRODUCTION

For robots to operate effectively in human environments, including those involving non-expert users, they must be able to adapt their behavior to individual preferences [1], [2]. As robots become increasingly integrated into personal spaces, such preferences can vary significantly across users and situations. Consider a household robot loading a dishwasher: a cautious user might prioritize the safety of delicate glassware, instructing the robot to “space the dishes far apart so they don’t break,” while another may value efficiency, demanding it “stack them tightly to fit as much as possible.”

Since the same robot policy is deployed across different users, it must accommodate various preferences without relying on costly retraining. Modern robot learning methods, such as diffusion policies, learn from diverse demonstrations and can represent multiple valid ways of performing a task, taking in examples from a distribution of intended behaviors and outputting actions that imitate those examples [3], [4]. As a result, desired behaviors may already exist within the learned data distribution, but a fixed policy will not be able to reliably select them. This motivates inference-time guidance mechanisms that steer policy behavior without retraining.

Methods for inference-time guidance of diffusion models typically rely on a user’s preference expressed as a guidance objective against which sampled action trajectories are optimized. Prior work has shown guidance to be successful using guidance objectives based on classifier models [5], arbitrary auxiliary networks [6], visual observations from goalstates [7], and STL expressions [8]–[10]. STL [11] offers a structured and expressive way to specify complex tasks and safety constraints [12], [13]. Furthermore, it is comparatively well suited for human-robot interaction as it supports a natural language interface for users, with prior work demonstrating the feasibility of mapping language to STL specifications [14]– [18]. However, existing methods for STL-based diffusion guidance rely on predicted states during diffusion to evaluate constraints. Producing these states requires joint action-state generation, increasing complexity and training cost [19]. Many modern policies generate only action sequences, making such approaches not directly applicable.

![](images/873f3d491ac098f8d143d494589b13302851a8796d4b9bb93f64fd9e6a34ce6b.jpg)  
Fig. 1. Our method guides the behavior of diffusion policies at inferencetime to satisfy STL expressions. We use a world model to roll out the action chunk created by the policy and compute an STL robustness value based on the predicted future states. With all steps being differentiable, we can incorporate the gradient of this robustness value in the denoising process. Our approach allows users to define arbitrary STL constraints at runtime, enabling interactive steering of robot behavior without retraining.

We propose an approach, illustrated in Figure 1, for inference-time STL guidance of diffusion policies that only produce actions. We use a separately trained, differentiable world model to predict future states of proposed actions, enabling constraint evaluation without requiring state predictions from the policy. This allows our method to work with the majority of existing diffusion policies. It also allows the horizon over which STL satisfaction is evaluated to differ from the horizon over which the diffusion policy predicts actions.

## II. METHOD

We consider a diffusion policy $\pi ( a _ { t : t + H } \mid s _ { t } )$ that generates an action sequence of horizon H conditioned on the current state $s _ { t } .$ . Let $\phi$ denote a STL specification over state trajectories. Our objective is to guide the sampled action sequence such that the induced trajectory satisfies ϕ, without retraining the policy. Our approach builds on inference-time guidance of the diffusion denoising process [3], [6]. Starting from noisy actions $\mathbf { a } ^ { k } : = a _ { t : t + H } ^ { k }$ at denoising step k, the policy predicts a cleaner action sequence via

$$
\mathbf {a} ^ {k - 1} = \mu_ {\theta} (\mathbf {a} ^ {k}, s _ {t}, k) + \sigma^ {k} \epsilon .\tag{1}
$$

Here, $\mu _ { \theta }$ denotes the denoising model, $\sigma ^ { k }$ is the variance for the k-th denoising step, and $\epsilon \sim \mathcal { N } ( 0 , I )$ . Inference-time guidance modifies this update by adding the gradient of a guidance objective J with respect to the action sequence

$$
\mathbf {a} ^ {k - 1} = \mu_ {\theta} (\mathbf {a} ^ {k}, s _ {t}, k) + \lambda \nabla_ {\mathbf {a} ^ {k}} J (\phi , \mathbf {a} ^ {k}, s _ {t}) + \sigma^ {k} \epsilon .\tag{2}
$$

Here, λ controls the guidance strength. We additionally apply a small number of gradient ascent steps on $J$ to the final action sequence $\mathbf { a } ^ { 0 }$ , which we find improves constraint satisfaction.

The guidance objective $J ( \phi , a _ { t : t + H } , s _ { t } )$ is based on the robustness measure for an STL specification $\phi ,$ the proposed action sequence $\scriptstyle a _ { t : t + H } .$ , and the current state $s _ { t }$ . The robustness measure, denoted $\rho _ { \phi } ( \tau )$ , quantifies the degree to which a trajectory satisfies $\phi .$ The resulting objective is given by

$$
J (\phi , a _ {t: t + H}, s _ {t}) = \rho_ {\phi} (\hat {s} _ {t + 1: t + H}).\tag{3}
$$

Here, $\hat { s } _ { t + 1 : t + H }$ denotes a prediction of the future state trajectory induced by the action sequence. To obtain this trajectory, we employ a separately trained world model $\hat { s } _ { t + 1 } = f _ { \theta } ( s _ { t } , a _ { t } )$ learned from the same data used by the diffusion policy, that approximates the next state given the current state and action. By iteratively applying this model over the action sequence, we obtain $\hat { s } _ { t + 1 : t + H }$ . Since the STL robustness is differentiable with respect to a state trajectory, and the state trajectory is differentiable with respect to the input actions through the world model, we can compute the gradient of J with respect to the action sequence and inject it into the diffusion denoising process, thereby steering the policy toward constraint-compliant behavior at inference time.

## III. RESULTS

We evaluate our approach on the Can Transport task from Robomimic [20], where a robot must grasp a can and drop it into a box, using a diffusion policy trained on mixed human demonstrations to capture diverse behaviors. We constrain the policy to keep the can upright with the STL specification $\mathbf { G } ( R _ { z z } > \cos ( 5 ^ { \circ } ) )$ , where $R _ { z z }$ denotes the alignment between the can orientation and the world z-axis. Figure 2(a) illustrates that the learned policy exhibits multiple behaviors for the same initial state, including trajectories that violate the uprightness constraint, while the learned world model accurately predicts these rollouts over the action horizon. Figure 2(b) shows that our guidance method can steer undesirable policy rollouts toward trajectories that satisfy the constraint.

(a) Policy Rollouts and World Model Prediction  
![](images/16358b7fb907266a089425c0b690aa8c2783037ec236bb0d9cf20930672b6b72.jpg)

(b) Guided vs. Unguided Rollouts  
![](images/59f4c4a7375cf129da7fe21c0b6a117b6eea4e5d9a594c49c067dd08541552a3.jpg)  
Fig. 2. Top: The diffusion policy exhibits multiple behaviors for the same initial state. The learned world model accurately predicts multi-step rollouts over the action horizon. Bottom: When the policy selects an undesirable mode, our method can guide it toward trajectories that satisfy the constraint.

TABLE I  
CONSTRAINT SATISFACTION AND TASK SUCCESS.

<table><tr><td>Method</td><td>Avg. Tilt (°) ↓</td><td>Succ. (%) ↑</td><td>Viol. (%) ↓</td></tr><tr><td>Base Policy</td><td>8.51</td><td>100.0</td><td>84.0</td></tr><tr><td>Sample &amp; Rank</td><td>8.42</td><td>98.0</td><td>82.0</td></tr><tr><td>Guidance (Ours)</td><td>1.93</td><td>100.0</td><td>4.0</td></tr></table>

We evaluate the constraint satisfaction rate over 50 rollouts and compare against the base diffusion policy and a sampleand-rank baseline [21] using the same robustness objective. As shown in Table I, our method improves constraint satisfaction while maintaining task success, outperforming both baselines. This suggests that gradient-based guidance more effectively steers the policy toward constraint-compliant behaviors than sampling-based selection, potentially recovering underrepresented modes in the learned distribution.

## IV. FUTURE WORK

We plan to evaluate our approach across more tasks, environments, and STL specifications to better understand its generality. A promising direction is improving stability or efficiency of the guidance process using evolutionary or second-order optimization methods. The most significant direction for our approach is toward supporting long-horizon STL constraints, which is a limitation of most existing STLguidance approaches [8]–[10], [22]. A separate world model allows evaluating the consequences of chosen actions at the abstraction level provided by the STL specification’s automaton representation [23]. Instead of predicting the next environment state given the current environment state and proposed action, we can learn to predict the next automaton state given the current environment state, automaton state, and action mode chosen by the policy. We view our current approach as one step toward this idea to enable significantly longer term, specification-relevant prediction, which will be necessary for finally supporting diffusion policy guidance with arbitrarily complex task specifications.

## REFERENCES

[1] M. Natarajan, E. Seraj, B. Altundas, R. Paleja, S. Ye, L. Chen, R. Jensen, K. C. Chang, and M. Gombolay, “Human-robot teaming: grand challenges,” Current Robotics Reports, vol. 4, no. 3, pp. 81–100, 2023.

[2] R. Paleja, A. Silva, L. Chen, and M. Gombolay, “Interpretable and personalized apprenticeship scheduling: Learning interpretable scheduling policies from heterogeneous user demonstrations,” Advances in Neural Information Processing Systems, vol. 33, pp. 6417–6428, 2020.

[3] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in neural information processing systems, vol. 33, pp. 6840– 6851, 2020.

[4] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song, “Diffusion policy: Visuomotor policy learning via action diffusion,” The International Journal of Robotics Research, vol. 44, no. 10-11, pp. 1684–1704, 2025.

[5] P. Dhariwal and A. Q. Nichol, “Diffusion models beat GANs on image synthesis,” in Advances in Neural Information Processing Systems, A. Beygelzimer, Y. Dauphin, P. Liang, and J. W. Vaughan, Eds., 2021. [Online]. Available: https://openreview.net/forum?id=AAWuCvzaVt

[6] A. Bansal, H.-M. Chu, A. Schwarzschild, S. Sengupta, M. Goldblum, J. Geiping, and T. Goldstein, “Universal guidance for diffusion models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 843–852.

[7] M. Du and S. Song, “Dynaguide: Steering diffusion polices with active dynamic guidance,” arXiv preprint arXiv:2506.13922, 2025.

[8] Z. Zhong, D. Rempe, D. Xu, Y. Chen, S. Veer, T. Che, B. Ray, and M. Pavone, “Guided conditional diffusion for controllable traffic simulation,” in IEEE International Conference on Robotics and Automation (ICRA), 2023, pp. 3560–3566.

[9] Y. Meng and C. Fan, “Diverse controllable diffusion policy with signal temporal logic,” IEEE Robotics and Automation Letters, vol. 9, no. 10, pp. 8354–8361, 2024.

[10] Z. Feng, H. Luan, P. Goyal, and H. Soh, “Ltldog: Satisfying temporallyextended symbolic constraints for safe diffusion-based planning,” IEEE Robotics and Automation Letters, vol. 9, no. 10, pp. 8571–8578, 2024.

[11] O. Maler and D. Nickovic, “Monitoring temporal properties of continuous signals,” in Formal Techniques, Modelling and Analysis of Timed and Fault-Tolerant Systems (FTRTFT), ser. Lecture Notes in Computer Science, Y. Lakhnech and S. Yovine, Eds., vol. 3253. Berlin, Heidelberg: Springer, 2004, pp. 152–166.

[12] A. Manganaris, V. Giammarino, A. H. Qureshi, and S. Jagannathan, “Formal methods in robot policy learning and verification: A survey on current techniques and future directions,” arXiv preprint arXiv:2602.06971, 2026.

[13] P. Kapoor, K. Mizuta, E. Kang, and K. Leung, “Stlcg++: A masking approach for differentiable signal temporal logic specification,” IEEE Robotics and Automation Letters, 2025.

[14] J. X. Liu, Z. Yang, B. Schornstein, S. Liang, I. Idrees, S. Tellex, and A. Shah, “Lang2ltl: Translating natural language commands to temporal specification with large language models,” in Workshop on Language and Robotics at CoRL 2022, 2022.

[15] J. X. Liu, A. Shah, G. Konidaris, S. Tellex, and D. Paulius, “Lang2ltl-2: Grounding spatiotemporal navigation commands using large language and vision-language models,” in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 2325– 2332.

[16] J. He, E. Bartocci, D. Nickovi ˇ c, H. Isakovic, and R. Grosu, “Deepstl:´ from english requirements to signal temporal logic,” in Proceedings of the 44th International Conference on Software Engineering, 2022, pp. 610–622.

[17] S. Mohammadinejad, S. Paul, Y. Xia, V. Kudalkar, J. Thomason, and J. V. Deshmukh, “Systematic translation from natural language robot task descriptions to stl,” in International Conference on Bridging the Gap between AI and Reality. Springer, 2024, pp. 259–276.

[18] I. Hurley, R. Paleja, A. Suh, J. D. Pena, and H. C. Siu, “Stl: Still tricky˜ logic (for system validation, even when showing your work),” Advances in Neural Information Processing Systems, vol. 37, pp. 119 099–119 122, 2024.

[19] M. Janner, Y. Du, J. B. Tenenbaum, and S. Levine, “Planning with diffusion for flexible behavior synthesis,” arXiv preprint arXiv:2205.09991, 2022.

[20] A. Mandlekar, D. Xu, J. Wong, S. Nasiriany, C. Wang, R. Kulkarni, L. Fei-Fei, S. Savarese, Y. Zhu, and R. Mart´ın-Mart´ın, “What matters in learning from offline human demonstrations for robot manipulation,” in Proceedings of the 5th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, A. Faust, D. Hsu, and G. Neumann, Eds., vol. 164. PMLR, 08–11 Nov 2022, pp. 1678–1690. [Online]. Available: https://proceedings.mlr.press/v164/mandlekar22a.html

[21] H. Qi, H. Yin, Y. Du, and H. Yang, “Strengthening generative robot policies through predictive world modeling,” arXiv e-prints, pp. arXiv– 2502, 2025.

[22] P. Kapoor, A. Ganlath, M. Clifford, C. Liu, S. Scherer, and E. Kang, “Safedec: Constrained decoding for safe autoregressive generalist robot policies,” 2026. [Online]. Available: https://openreview.net/forum?id=dLO7MhVbbB

[23] C. Baier, J.-P. Katoen, and K. G. Larsen, Principles of Model Checking. MIT Press, 2008.