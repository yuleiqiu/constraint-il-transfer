# Language-Guided Object-Centric Diffusion Policy for Collision-Aware Robotic Manipulation

Hang Li<sup>∗,1,2</sup>, Qian Feng<sup>∗,1,2</sup>, Zhi Zheng<sup>1</sup>, Jianxiang Feng<sup>1,2</sup>, Alois Knoll<sup>1</sup> <sup>1</sup>Technical University of Munich, <sup>2</sup>Agile Robots Equal Contribution, {hang1.li, qian.feng}@tum.de

Abstract: Learning from demonstrations faces challenges in generalizing beyond the training data and is fragile even to slight visual variations. To tackle this problem, we introduce Lan-o3dp, a language guided object centric diffusion policy that takes 3d representation of task relevant objects as conditional input and can be guided by cost function for safety constraints at inference time. Lan-o3dp enables strong generalization in various aspects, such as background changes, visual ambiguity and can avoid novel obstacles that are unseen during the demonstration process. Specifically, We first train a diffusion policy conditioned on point clouds of target objects and then harness a large language model to decompose the user instruction into task related units consisting of target objects and obstacles, which can be used as visual observation for the policy network or converted to a cost function, guiding the generation of trajectory towards collision free region at test time. Our proposed method shows training efficiency and higher success rates compared with the baselines in simulation experiments. In real world experiments, our method exhibits strong generalization performance towards unseen instances, cluttered scenes, scenes of multiple similar objects and demonstrates training free capability of obstacle avoidance.

Keywords: Imitation learning, Object-centric representation, Guided diffusion, Large language model

![](images/b5e054e2a4bd62e0bcd9cfe59648dda1608d8b123df144a7180a7002bd26f18d.jpg)  
Figure 1: Language guided object centric diffusion policy(Lan-o3dp) enables generalization beyond training data and has the training free obstacle avoidance ability. Lan-o3dp uses a large language model to select a policy and the corresponding target objects. It can also locate the obstacles specified by language and convert the information of obstacles into cost function to guide the model to generate collision free trajectory. The shown task is pouring from the bowl into the pan.

## 1 Introduction

Recently, diffusion models have shown great potential in the field of robotic manipulation [1, 2, 3]. In the realm of imitation learning, diffusion based methods [4, 5] have demonstrated strong capabilities towards learning complex manipulation tasks. Compared to traditional imitation learning algorithms, diffusion models offer the advantages of stable training, high-dimensional output spaces, and the ability to capture the multi-modal distribution of actions [4].

However, their performance is limited when testing scenes are different from training in terms of scenes of similar objects but with different appearances, new backgrounds, camera view shift, unseen obstacles, and so on. Generalization across different scenarios and safe deployment are crucial for the widespread application of robotics. To address the challenges of generalization and safe obstacle avoidance, we propose Lan-o3dp, a language-guided, object-centric collision-aware diffusion policy. By leveraging large language model and vision language model, our approach identifies task-relevant objects in the scene based on language instructions, segments the relevant objects point clouds from the overall scene point cloud, and encodes this data into a compact 3D representation. The diffusion model is trained to predict robot end effector trajectory conditioned on these 3D representations of task-relevant objects by denoising random noise into a coherent action sequence. Moreover, at testing stage, it can also segment the point cloud of obstacles in addition to target objects based on the language instruction. The location and geometric information of obstacles are obtained from the segmented point cloud and can be used to construct a cost function. The gradient of the cost function is then integrated into guided sampling [6] to direct the trajectory prediction phase, allowing the robotic arm’s end effector to avoid obstacles. The training free obstacle avoidance can be effectively integrated into the proposed 3D representation pipeline, as it allows for the acquisition of the location and geometric information of obstacles from a calibrated camera on one hand, and on the other hand, the policy does not fail due to the changes in the scene caused by the addition of extra obstacles.

To evaluate our algorithm, we conduct experiments across seven simulated tasks in the RLBench [7] environment and three real-world tasks. We demonstrate the effectiveness of our proposed methods against state-of-the-art diffusion-based methods in simulation and further evaluate the generalization and obstacle avoidance capabilities of our method in the real world.

In summary, our contributions are three-fold:

1. We propose Lan-o3dp, an effective language-guided collision-aware visuomotor policy that generalizes across diverse aspects such as background changes, camera view shift, and even scenes of multiple similar objects.

2. We introduce a novel guidance mechanism to avoid obstacles by analyzing the limitations of the existing guided sampling approaches. We present theoretical explanations and validations on real robots. Given language instructions, Lan-o3dp identifies novel obstacles and avoids collision given training data of no obstacles, which further improves the generalization capability.

3. The proposed method is evaluated in both simulation and real world experiments and shows the effectiveness and universality compared to baselines.

## 2 Related Work

## 2.1 Diffusion Models in Robotics

Diffusion model is a type of probabilistic generative model that learns to generate new data by progressively applying a denoising process on a randomly sampled noise. Learning based robotic grasping [8, 9, 10] and manipulation skills [11, 12] are longstanding problems. Due to its advantage of stable training and impressive expressiveness, diffusion model has been applied in several robotic fields such as reinforcement learning [13, 1, 14], imitation learning [4, 5], grasp synthesis [15, 16] and motion planning [17, 18, 19]. In this work, we utilize object-centric 3D representation to maximize the generalization ability of the trained policy based on imitation learning via diffusion model and introduce a novel guided diffusion mechanism for obstacle avoidance.

## 2.2 Object-Centric Representation Learning

Object-centric representations have been widely studied to reason about visual observations modularly in the robotic field. In robotics, researchers commonly use 6D poses [20, 21, 22], bounding boxes [23, 24] or segmented masks [25] to represent objects in a scene. These representations are limited to known object categories or instances. Recent progress in open-world visual recognition has led to the development of substantial models across various domains, including object detection [26], object segmentation [27], video object segmentation [28]. Groot[11] trains a transformer policy using segmented 3D objects. However, Groot uses a Segmentation Correspondence Model to identify the target object and cannot handle scenes with multiple similar objects. We use open vocabulary segmentation, which allows for a more convenient specification of target objects through language.

## 2.3 Language models for robotics

Large language models(LLMs) possess powerful language comprehension abilities and a wealth of common knowledge. As a result, they can be effectively utilized to understand human instructions and to plan robotic tasks at a high level. Code as Policies[29] explored using a code writing LLM to generate robot policy code based on language commands. Voxposer[30] plans the task and generates code by a LLM to compose value maps for zero shot manipulation. SayCan[31] use a LLM to select skills from a library of pre-trained skills. Many works also explore using LLMs to write reward functions for training robotic skills [32, 33, 34]. In addition, With the help of pre-trained open vocabulary vision language model [26, 35, 36], the robot can ground the user’s instruction to the real world and accomplish various and complex tasks [30, 37, 38]. In this work, we use a language model to select the desired policy and extract objects and obstacles from the user’s instructions and obtain an object point cloud with the vision language model.

## 3 Method

## 3.1 Problem Formulation

In this work, we address the generalization problem of the diffusion policy and introduce collision awareness at inference time by adjusting the visual conditioning and sampling guidance.

Diffusion policy visual conditioning: Diffusion policy[4] uses DDPM to model the action sequence $P ( A _ { t } \mid O _ { t } )$ . Wherein, $A _ { t } = \{ { \mathrm { a t } } , \dots , { \mathrm { a t } } + { \mathrm { n } } \}$ is the predicted next n action steps, which is a sequence of end-effector poses. The prediction horizon n indicates that the diffusion policy predicts a trajectory over a shorter horizon instead of the entire trajectory. $O _ { t } = \{ V _ { t } , S _ { t } \}$ represents the visual observation $V _ { t }$ and robot states observation $S _ { t }$ . The observation features are fused to the policy network through Film [39]. While diffusion policy takes pure RGB images as visual observation, we employ the segmented point clouds of task-relevant objects from a calibrated camera for policy learning. By removing redundant visual information and retaining only task-relevant information, our model can minimize the negative effects of scene changes, thereby improving generalization performance.

Guided sampling formulation: The diffusion model is trained to predict the added noise $\epsilon ( O , A _ { k } , k )$ at each diffusion timestep k, and during the reverse diffusion process, it gradually denoises a Gaussian noise to a smooth noise-free trajectory. The reverse process step is $A _ { k - 1 } ~ =$ $\mu _ { k } + \sigma _ { k } z$ , where $\begin{array} { r } { \mu _ { k } = \frac { 1 } { \sqrt { \alpha _ { k } } } \left( A _ { k } - \frac { 1 - \alpha _ { k } } { \sqrt { 1 - \overline { { \alpha } } _ { k } } } \varepsilon \right) , z \sim \mathcal { N } ( 0 , I ) } \end{array}$ , α<sub>k</sub> ∈ <sup>R</sup> and $\begin{array} { r } { \overline { { \alpha } } _ { k } : = \prod _ { s = 1 } ^ { k } \alpha _ { s } } \end{array}$ <sub>s</sub> predefined scheduling parameters. The subscript time t of the trajectory index is dropped for ease of notation. Much prior work has explored guided sampling of the diffusion model. At the inference stage, guidance $g _ { k } = \nabla _ { A _ { k } } D$ as a gradient term of cost/distance D with respect to $A _ { k }$ is added to the model’s predicted mean such that each denoising step becomes:

$$
A _ {k - 1} = \mu_ {k} - \rho g _ {k} + \sigma_ {k} z\tag{1}
$$

![](images/0f5e6ca1c7d5bcaef469baa0a5164b6e8b4a005fcd23c3511ed728eeb89f22f7.jpg)  
Figure 2: An illustration of the proposed pipeline of Lan-o3dp. At the training stage, the visual observations in the demonstrations we collected only contained objects relevant to the task. During the deployment phase, we utilize a large language model to decompose users’ instructions into target objects and obstacles and select the corresponding policy. Target objects are used as visual observation for the model, while obstacles are transformed into a cost function to guide the model in generating collision-free trajectories.

, where $\rho$ is a scaling factor to control the effect of guidance. In this work, we model newly emerged obstacles in the scene as a cost function to guide the model in generating collision-free trajectories. The cost gradient and distance gradient have opposite signs for avoiding obstacles. Benefiting from the object-centric 3D representation of the pipeline, we can obtain the locations and basic geometric information of obstacles within the scene.

## 3.2 Approach

Training stage: To obtain task-relevant target point clouds, we leverage open vocabulary segmentation to acquire real-time masks of the target objects and map these masks onto the point clouds. Figure 2 shows our pipeline. As shown in the training stage, before starting to record the demonstrations, the task related objects are specified simply by words. A vision language model (VLM) is firstly called to detect corresponding objects within the scene to obtain the bounding boxes of the target objects, which are then passed as prompt to Segment Anything Model (SAM) [27] to obtain segmentation masks. Upon completion of segmentation, recording commences, and a video object segmentation model is employed to track the objects in real-time. The tracked masks are projected to point clouds, resulting in point cloud representations of the objects for each frame. The point clouds of objects are further downsampled by farthest point sampling.

Language guided deployment: During the deployment phase, our model is applicable to different scenarios. We use a large language model to decompose the user’s commands into policy, target objects, and obstacles. Similarly, open vocabulary segmentation is used to obtain the point cloud of the target objects and obstacles in each frame. The point cloud of target objects is subsequently inputted as an observation into the trained policy, while the point cloud of obstacles is processed and transformed into a cost function. The gradient of this cost function is then utilized to guide the trajectory generation towards collision free areas.

Cost guided generation: In the field of robotics, many guided sampling techniques rely on reward models [1, 14], which are, however, often difficult to obtain. We choose to use a flexibly constructed cost function instead. To convert obstacles information to cost function, we calculate the distance between every waypoint in the generated action sequence and the centers of obstacles $C _ { o b }$ . As previously mentioned, most guided sampling methods calculate the cost/distance $D ( A _ { k } , C _ { o b } )$ of each intermediate action $A _ { k }$ generated during the reverse diffusion process and compute the gradient $g _ { k } \ = \ \nabla _ { A _ { k } } D ( A _ { k } , C _ { o b } )$ . However, a cost function that is independent of the timestep k of the diffusion process becomes less meaningful because of the noisy trajectories, especially in the early stages of the denoising process. Consequently, the cost of noisy trajectories struggles to provide effective guidance. Unlike previous methods [19, 18], refer to FreeDoM [40], we calculate the cost at each step based on the estimated $A _ { 0 | k }$ , an estimated clean trajectory.

$$
A _ {0 | k} := \mathbb {E} [ A _ {0} | A _ {k} ] = \frac {A _ {k} - \sqrt {1 - \overline {{\alpha}} _ {k}} \epsilon_ {\theta} (A _ {k})}{\sqrt {\overline {{\alpha}} _ {k}}}\tag{[41]}
$$

(2)

We calculate the cost/distance of $A _ { 0 }$ estimated from $A _ { k }$ at each timestep and use this cost to compute the gradient with respect to $A _ { k } ,$ that is $\nabla _ { A _ { k } } D ( A _ { 0 | k } , C _ { o b } )$ . Therefore, the equation 1 becomes:

$$
A _ {k - 1} = \mu_ {k} - \rho \nabla_ {A _ {k}} D (A _ {0 | k}, C _ {o b}) + \sigma_ {k} z\tag{3}
$$

As discussed in [40], it is difficult to achieve effective guidance during the early stages of the diffusion process due to too chaotic sample. We choose to guide the generation during specific time periods. The detailed algorithm is shown in Algo 1

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 Cost guided diffusion sampling, given a diffusion model $\epsilon_{\theta}$, cost/distance measurement $D(x,y)$, and gradient scale $\rho_t$.
$A_T \leftarrow$ sample from $\mathcal{N}(0,I)$
for $k = T$ to 1 do
    $\mu_k \leftarrow \frac{1}{\sqrt{\alpha_k}} \left( A_k - \frac{1-\alpha_k}{\sqrt{1-\overline{\alpha_k}}} \epsilon_\theta \right)$ $A_{k-1} \leftarrow \mu_k + \sigma_k z$
    if $k \leq S$ then
        $A_{0|k} = \frac{A_k - \sqrt{1-\overline{\alpha_k}} \epsilon_\theta(A_k)}{\sqrt{\overline{\alpha_k}}}$ $A_{k-1} = A_{k-1} - \rho \nabla_{A_k} D(A_{0|k}, C_{ob})$
    end if
end for
Return $A_0$
</div>

## 4 Experiments

In our experiments, we show the following aspects: (1) Object-centric 3d diffusion policy achieves a higher average success rate in simulation experiments compared to the baselines; (2) Our method has strong generalization capabilities for scene changes; and (3) Cost guided generation can effectively avoid language specified obstacles.

## 4.1 Simulation Experiments

We conduct simulation experiments in RLBench to evaluate the success rate of proposed Lan-o3dp compared with two baselines namely diffusion policy [4] and 3D diffusion policy [5]. To keep consistent with real-world experiments, we only use the front camera and collect 40 demonstrations for each task across 7 tasks, which cover manipulation, pick-and-place, single object, and multiple objects. Examples are shown in figure 3, we extract the task related object point cloud. Each demonstration of every task has variation, such as position changes of the objects.

We use the convolutional network based diffusion policy. We train 500 epochs for each task, evaluate 20 episodes every 50 epochs, and then compute the average of the highest 5 success rates. The episodes for evaluation also have variations. As shown in table 1 achieves an overall 68.8 % success rate across 7 RLBench tasks.

Ablation Study. We conduct further ablation studies of design choices such as point cloud encoder, ”epsilon” or ”sample” prediction as a learning objective. As the success rates indicate, both learning to directly predict the initial trajectory ’sample’ and learning to predict the added noise at each timestep yield competitive results. Following the 3D diffusion policy, we study the DP3 Encoder, which is a simple three-layer MLP, and PointNet [42] encoder. As shown in table 2, DP3 Encoder with a residual connection can further improve the success rate by 4.7%.

![](images/cec7627ea6fd0de683c3c16b0dddff08671fb4a591fee1d0856a433408c1f38a.jpg)  
Figure 3: Visualization of some simulation tasks. We use a single front camera and extract the segmented point cloud of task-related objects to keep them consistent with the real world. The top row shows RGB images from the front camera, and the bottom row shows visualizations of the corresponding object point clouds.

Table 1: Simulation Results on RLBench

<table><tr><td>Tasks</td><td>open drawer</td><td>open wine bottle</td><td>sweep to dustpan</td><td>phone on base</td><td>put item in drawer</td><td>water plants</td><td>close microwave</td><td>Average Succ Rate</td></tr><tr><td>Diffusion Policy</td><td>70.0%</td><td>38.0%</td><td>57.0%</td><td>11.0%</td><td>32.0%</td><td>41.0%</td><td>95.0%</td><td>49.2%</td></tr><tr><td>3D Diffusion</td><td>94.0%</td><td>49.0%</td><td>66.0%</td><td>6.0%</td><td>5.0%</td><td>21.0%</td><td>94.0%</td><td>49.6%</td></tr><tr><td>Lan-o3dp</td><td>90.0%</td><td>77.0%</td><td>77.0%</td><td>57.0%</td><td>50.0%</td><td>37.5%</td><td>93.0%</td><td>68.8%</td></tr></table>

Table 2: Ablation study

<table><tr><td>Tasks</td><td>open drawer</td><td>open wine bottle</td><td>sweep to dustpan</td><td>phone on base</td><td>put item in drawer</td><td>water plants</td><td>close microwave</td><td>Average Succ Rate</td></tr><tr><td>Ours (Residual)</td><td>90.0%</td><td>77.0%</td><td>77.0%</td><td>57.0%</td><td>50.0%</td><td>37.5%</td><td>93.0%</td><td>68.8%</td></tr><tr><td>Ours (MLP)</td><td>93.0%</td><td>75.0%</td><td>64.0%</td><td>45.0%</td><td>40.0%</td><td>35.0%</td><td>97.0%</td><td>64.1%</td></tr><tr><td>Ours (PointNet)</td><td>0.0%</td><td>2.0%</td><td>23.0%</td><td>0.0%</td><td>0.0%</td><td>2.0%</td><td>77.0%</td><td>14.9%</td></tr><tr><td>Ours (Epsilon)</td><td>90.0%</td><td>90.0%</td><td>25.0%</td><td>53.0%</td><td>73%</td><td>34%</td><td>95.0%</td><td>65.7%</td></tr></table>

## 4.2 Real world Experiments

In the real-world experiments, we aim to verify the generalization in the following aspects: (1) instance changes, (2) multiple similar objects, (3) camera view shift, and (4) language informed obstacle avoidance.

## 4.2.1 Experiment Setup

System setup and task design: We conduct real-world experiments on 3 tasks with a Diana 7 robot arm. We use one RealSense D415 camera to capture the RGB image and point cloud. Our tasks are Bowl pour: grasp the bowl and pour the contents of the bowl into the pan; Bottle upright: stand the horizontal bottle upright. Bottle in drawer: put the bottle into the drawer. We use GPT-4 [43] at testing time to extract policy, target objects and obstacles from user instruction and generate code to run the policy.

Demonstrations collection: Demonstrations are collected by teleoperation with a space mouse and keyboard. We collect 40 demonstrations for each task and the training scenes and position variation are shown in the upper part of figure 4. In the bottle upright task and bottle in drawer task, the orientation of the bottle is not changed. Given task-related objects, we invoke open vocabulary detector GroundingDINO [26] to predict the bounding boxes, Segment Anything [27] to obtain the segmentation masks, and finally track the masks using video tracker Cutie [44]. We record observations consisting of objects point clouds and state observations, including the robot end effector poses and gripper state.

## 4.2.2 Generalization Evaluation

![](images/3c564c31a0107acf375870e31ad31500674ee5b3dfef17f376184f54c8c2d9ee.jpg)  
Figure 4: Training scenes (top row) and cluttered test scenes (bottom row). The red lines indicate the position variations in collected demonstrations. From left to right, the tasks are bottle upright, bowl pour, and bottle in drawer.

![](images/312a0c53c8f95b5e5aa3ff698703dbf8bb4cb90f1f26d8313fff3ab77fba7c4e.jpg)  
Figure 5: Scenes with obstacles (top row) and camera view shift (bottom row). The top row is the task bowl pour with novel obstacles. The bottom row is the setup of two cameras and the segmentation masks of current scene from two cameras.

Instance changes: We evaluate the generalization ability to objects with similar geometry through all three tasks. We have observed that our model can handle changes in the appearance of objects, but it tends to perform less effectively with objects that undergo larger changes in geometric shape.

Scenes of multiple similar objects: In scenes with multiple objects, language models are crucial due to the visual ambiguity caused by the presence of multiple similar objects. We use natural language to specify which object should be the target. The success rates of multiple similar objects in figure 6 are about manipulating the training objects.

Camera view shift: As shown in figure 5 bottom row, we use the camera in the red circle for demonstration collection and the camera in the

![](images/5d089c0dc8e1336ef2382c85c1562ed787facef0282f6c2bca6b54a7dc18a524.jpg)  
Figure 6: Generalization evaluation: we evaluate three tasks under different experiment conditions: camera view shift, cluttered with multiple similar objects scene, instance changes.

orange circle is only for testing. As shown in figure 6, there is no performance drop when the camera changes from red circle to orange circle.

## 4.2.3 Testing time obstacles avoidance

Scenes of obstacles: We test the cost guided obstacle avoidance in the bowl pouring task shown in figure 5. We construct the cost and let the generated trajectory change horizontally. The obstacles are modeled as cylinders, and the radius is the sum of the diameter of the bowl and the radius of the obstacle in the horizontal orientation. We notice that the robot can successfully avoid obstacles and finish the tasks. The gradient scale significantly influences the quality of the generated trajectory and the effect of obstacle avoidance.

Cost function constructed from obstacles: We calculate the distance to the center of obstacles $C _ { o b }$ for all waypoints $a _ { i }$ in the estimated trajectory $A _ { 0 \mid k }$ . If any distance $D ( a _ { i } , C _ { o b } )$ ) in $D ( A _ { 0 \vert k } , C _ { o b } )$ is shorter than a safety critical distance $Q ^ { * }$ , a non-zero gradient will be assigned to the corresponding waypoint. In real world experiments, we only consider the distance of x and y coordinates.

$$
G r a d i e n t = \left\{ \begin{array}{l l} \nabla D (a _ {i}, C _ {o b}), & \text {if} D (a _ {i}, C _ {o b}) \leq Q ^ {*} \\ 0, & \text {if} D (a _ {i}, C _ {o b}) > Q ^ {*} \end{array} \right.\tag{4}
$$

Visualization and explanation of proposed guidance mechanism: The generated trajectory of the denoising process is too chaotic especially at the early stage as illustrated in figure 7. The cost of trajectory with too much noise is barely meaningful. We calculate the cost of estimated clean trajectory $A _ { 0 \mid k }$ instead. The red circle in the following figures indicates the obstacle and the blue points are the waypoints of the generated action sequence. The visualization is only about x and y coordinations.

![](images/84f3881fd5cacbe09526901ecda4633e6aa11d96f116b5ce8e800be6ba0a2dee.jpg)  
k = 99

![](images/77a61ac7bf751cb613651deba93586b4b7f7bacaf34f9d85571096491b4b55cf.jpg)  
k = 50

![](images/a34acc8063f4141ccb35e31e5c1dad74676615bf0d50d24c30e33829f8feb663.jpg)  
k = 2

![](images/4d6e6d3d3362e45309294786fd76f775580f6732e33d686fa51d700d503ecc93.jpg)  
k = 0  
Figure 7: Visualization of intermediate trajectory $A _ { k }$ generated at different denoising timesteps.

The impact of different gradient scales $\rho$ on the generated final trajectory is shown in figure 8. A point within the red circle is a point has collision with the obstacle. In our experiments, a gradient scale greater than 0.0003 can effectively avoid obstacles.

![](images/2693fa94d335a0a887a53cf99e578fdf316a06aca18f9b197a91585aa973703f.jpg)  
No guidance

![](images/9b2d773460fa2f1063fc4e5a44a96814c48ba32362fcf5ee96d6d55314549956.jpg)  
ρ= 0.0001

![](images/64f4b42dbbecb44ebcb6b8af89e370803e2fdd96679f6ff155905f077bd5ffe9.jpg)  
ρ= 0.0002

![](images/0e5e7c817415fa9e063b9aef10c164d41d85ff218e253dcd84b8f849feff09e1.jpg)  
ρ= 0.0003  
Figure 8: Effect of different gradient scales.

Sample vs. epsilon prediction: Although the ”sample” prediction shows a bit higher success rate than ”epsilon” prediction in the simulation, we found that the guidance is almost ineffective if each step predicts the initial sample $A _ { 0 }$ directly. Using sample prediction can not successfully avoid obstacles. Since there is not much performance drop of our proposed method when using epsilon prediction, we use epsilon prediction in real-world.

## 5 Limitations and Conclusion

In this work, we use the point cloud of the target objects as the input for the diffusion policy model to enhance the model’s generalization performance. By filtering out objects that are irrelevant to the task, our model can perform well in changed scenes. Additionally, we have introduced training free cost-guided trajectory generation for obstacle avoidance, converting necessary obstacles into costs to achieve safer deployment. This work has some limitations. We utilize the point cloud of objects as the output of the model, assuming that the target object can be successfully detected by the model. The performance of the current VLM for detecting objects is limited, which restricts the performance of our model. Additionally, our modeling of obstacles uses a relatively simple distance measure, which is insufficient for complex obstacles. Future work could involve using more advanced visual-language models, incorporating a task planner for complex task planning, and modeling obstacles in more detail to avoid more complex obstacles.

## References

[1] M. Janner, Y. Du, J. B. Tenenbaum, and S. Levine. Planning with diffusion for flexible behavior synthesis. In Proceedings of International Conference on Machine Learning (ICML), 2022.

[2] X. Ma, S. Patidar, I. Haughton, and S. James. Hierarchical diffusion policy for kinematicsaware multi-task robotic manipulation. In Proceedings of IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR), 2024.

[3] M. Reuss, M. X. Li, X. Jia, and R. Lioutikov. Goal conditioned imitation learning using scorebased diffusion policies. In Proceedings of Robotics: Science and Systems (RSS), 2023.

[4] C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. Burchfiel, and S. Song. Diffusion policy: Visuomotor policy learning via action diffusion. In Proceedings of Robotics: Science and Systems (RSS), 2023.

[5] Y. Ze, G. Zhang, K. Zhang, C. Hu, M. Wang, and H. Xu. 3d diffusion policy: Generalizable visuomotor policy learning via simple 3d representations. In Proceedings of Robotics: Science and Systems (RSS), 2024.

[6] P. Dhariwal and A. Nichol. Diffusion models beat gans on image synthesis. In Proceedings of NeurIPS, 2022.

[7] S. James, Z. Ma, D. R. Arrojo, and A. J. Davison. Rlbench: The robot learning benchmark & learning environment, 2019.

[8] H. Liang, X. Ma, S. Li, M. Gorner, S. Tang, B. Fang, F. Sun, and J. Zhang. Pointnetgpd: Detecting grasp configurations from point sets. In 2019 International Conference on Robotics and Automation (ICRA). IEEE, May 2019. doi:10.1109/icra.2019.8794435. URL http: //dx.doi.org/10.1109/ICRA.2019.8794435.

[9] V. Mayer\*, Q. Feng\*, J. Deng, Y. Shi, Z. Chen, and A. Knoll. Ffhnet: Generating multi-fingered robotic grasps for unknown objects in real-time. 2022 International Conference on Robotics and Automation (ICRA), pages 762–769, 2022. URL https://api. semanticscholar.org/CorpusID:250508500.

[10] Y. Burkhardt\*, Q. Feng\*, J. Feng, K. Sharma, Z. Chen, and A. Knoll. Multi-fingered dynamic grasping for unknown objects, 2024.

[11] Y. Zhu, Z. Jiang, P. Stone, and Y. Zhu. Learning generalizable manipulation policies with object-centric 3d representations. In Proceedings of Conference on Robot Learning (CoRL), 2023.

[12] Z. Liang, Y. Mu, H. Ma, M. Tomizuka, M. Ding, and P. Luo. Skilldiffuser: Interpretable hierarchical planning via skill abstractions in diffusion-based task execution. In Proceedings of IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR), 2024.

[13] A. Ajay, Y. Du, A. Gupta, J. Tenenbaum, T. Jaakkola, and P. Agrawal. Is conditional generative modeling all you need for decision-making? In Proceedings of International Conference on Learning Representations (ICLR), 2023.

[14] Z. Liang, Y. Mu, M. Ding, F. Ni, M. Tomizuka, and P. Luo. Adaptdiffuser: Diffusion models as adaptive self-evolving planners. In Proceedings of International Conference on Machine Learning(ICML), 2023.

[15] K. R. Barad, A. Orsula, A. Richard, J. Dentler, M. Olivares-Mendez, and C. Martinez. Graspldm: Generative 6-dof grasp synthesis using latent diffusion models, 2023.

[16] Z. Weng, H. Lu, D. Kragic, and J. Lundell. Dexdiffuser: Generating dexterous grasps with diffusion models, 2024.

[17] J. Urain, N. Funk, J. Peters, and G. Chalvatzaki. Se(3)-diffusionfields: Learning smooth cost functions for joint grasp and motion optimization through diffusion. In 2023 International Conference on Robotics and Automation (ICRA). IEEE, 2023.

[18] J. Carvalho, A. T. Le, M. Baierl, D. Koert, and J. Peters. Motion planning diffusion: Learning and planning of robot motions with diffusion models. In Proceedings of International Conference on Intelligent Robots and Systems (IROS), 2023.

[19] K. Saha, V. Mandadi, J. Reddy, A. Srikanth1, A. Agarwal, B. Sen, A. Singh, and M. Krishna1. Ensemble-of-costs-guided diffusion for motion planning. In Proceedings of International Conference on Robotics and Automation(ICRA), 2024.

[20] J. Tremblay, T. To, B. Sundaralingam, Y. Xiang, D. Fox, and S. Birchfield. Deep object pose estimation for semantic robotic grasping of household objects. CoRR, abs/1809.10790, 2018. URL http://arxiv.org/abs/1809.10790.

[21] S. Tyree, J. Tremblay, T. To, J. Cheng, T. Mosier, J. Smith, and S. Birchfield. 6-dof pose estimation of household objects for robotic manipulation: An accessible dataset and benchmark, 2022.

[22] T. Migimatsu and J. Bohg. Object-centric task and motion planning in dynamic environments. CoRR, abs/1911.04679, 2019. URL http://arxiv.org/abs/1911.04679.

[23] D. Wang, C. Devin, Q.-Z. Cai, F. Yu, and T. Darrell. Deep object-centric policies for autonomous driving, 2019.

[24] C. Devin, P. Abbeel, T. Darrell, and S. Levine. Deep object-centric representations for generalizable robot learning, 2017.

[25] M. Danielczuk, M. Matl, S. Gupta, A. Li, A. Lee, J. Mahler, and K. Goldberg. Segmenting unknown 3d objects from real depth images using mask R-CNN trained on synthetic point clouds. CoRR, abs/1809.05825, 2018. URL http://arxiv.org/abs/1809.05825.

[26] S. Liu, Z. Zeng, T. Ren, F. Li, J. Y. Hao Zhang, C. Li, J. Yang, H. Su, J. Zhu, and L. Zhang. Grounding dino: Marrying dino with grounded pre-training for open-set object detection, 2023.

[27] A. Kirillov, E. Mintun, N. Ravi, H. Mao, L. G. C. Rolland, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, and et al. Segment anything. In Proceedings of International Conference on Computer Vision (ICCV), 2023.

[28] H. K. Cheng and A. G. Schwing. Xmem: Long-term video object segmentation with an atkinson-shiffrin memory model. In Proceedings of European Conference on Computer Vision (ECCV), 2022.

[29] J. Liang, W. Huang, F. Xia, P. Xu, K. Hausman, B. Ichter, P. Florence, A. Zeng, and et.al. Code as policies: Language model programs for embodied control. In IEEE International Conference on Robotics and Automation (ICRA), 2023.

[30] W. Huang, C. Wang, R. Zhang, Y. Li, J. Wu, and L. Fei-Fei. Voxposer: Composable 3d value maps for robotic manipulation with language models. In Proceedings of Conference on Robot Learning(CoRL), 2023.

[31] M. Ahn, A. Brohan, N. Brown, Y. Chebotar, O. Cortes, and et.al. Do as i can, not as i say: Grounding language in robotic affordances. In Proceedings of Conference on Robot Learning(CoRL), 2022.

[32] W. Yu, N. Gileadi, C. Fu, S. Kirmani, K.-H. Lee, and et.al. Language to rewards for robotic skill synthesis. In Proceedings of Conference on Robot Learning(CoRL), 2023.

[33] T. Xie, S. Zhao, C. H. Wu, Y. Liu, Q. Luo, V. Zhong, Y. Yang, and T. Yu. Text2reward: Reward shaping with language models for reinforcement learning. In Proceedings of International Conference on Learning Representations(ICLR), 2024.

[34] J. Ma, W. Liang, G. Wang, D.-A. Huang, O. Bastani, D. Jayaraman, Y. Zhu, L. J. Fan, and A. Anandkumar1. Eureka: Human-level reward design via coding large language models. In Proceedings of International Conference on Learning Representations(ICLR), 2024.

[35] M. Minderer, A. Gritsenko, A. Stone, M. Neumann, and et.al. Simple open-vocabulary object detection with vision transformers. In Proceedings of European Conference on Computer Vision (ECCV), 2022.

[36] T. Cheng, L. Song, Y. Ge, W. Liu, X. Wang, and Y. Shan. Yolo-world: Real-time openvocabulary object detection. In Proceedings of Computer Vision and Pattern Recognition Conference (CVPR), 2024.

[37] A. Stone, T. Xiao, Y. Lu, K. Gopalakrishnan, K.-H. Lee, Q. Vuong, P. Wohlhart, S. Kirmani, B. Zitkovich, F. Xia, C. Finn, and K. Hausman. Open-world object manipulation using pretrained vision-language models. In Proceedings of Conference on Robot Learning(CoRL), 2023.

[38] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, and et.al. Rt-2: Vision-languageaction models transfer web knowledge to robotic control. In Proceedings of International Conference on Computer Vision, 2023.

[39] E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville. Film: Visual reasoning with a general conditioning layer. In Proceedings of Conference on Artificial Intelligence(AAAI), 2018.

[40] J. Yu, Y. Wang, C. Zhao, B. Ghanem, and J. Zhang. Freedom: Training-free energy-guided conditional diffusion model. In Proceedings of International Conference on Computer Vision, 2023.

[41] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. In Proceedings of NeurIPS, 2020.

[42] C. R. Qi, H. Su, K. Mo, and L. J. Guibas. Pointnet: Deep learning on point sets for 3d classification and segmentation. In Proceedings of Computer Vision and Pattern Recognition Conference (CVPR), 2017.

[43] OpenAI, J. Achiam, S. Adler, S. Agarwal, L. Ahmad, and et.al. Gpt-4 technical report, 2023.

[44] H. K. Cheng, S. W. Oh, B. Price, J.-Y. Lee, and A. Schwing. Putting the object back into video object segmentation. In Proceedings of Computer Vision and Pattern Recognition Conference (CVPR), 2024.