# Analysis: 3D Diffusion Policy (DP3)

**Paper:** 3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations  
**Authors:** Yanjie Ze, Gu Zhang, Kangning Zhang, Chenyuan Hu, Muhan Wang, Huazhe Xu  
**Venue:** RSS 2024  
**Year:** 2024  
**arXiv:** 2403.03954  

---

## Problem Setting

DP3 addresses visual imitation learning for robotic manipulation with a focus on **sample efficiency** and **generalization**. The core hypothesis is that 3D point cloud representations, when paired with diffusion policies, enable:
- Learning complex skills from very few demonstrations (as few as 10 in simulation, 40 in real world)
- Strong generalization across space, viewpoint, appearance, and object instance
- Safer real-world deployment compared to 2D image-based policies

The tasks span 72 simulated tasks (7 domains: MetaWorld, Adroit, Bi-DexHands, DexArt, DexDeform, DexMV, HORA) and 4 real-world dexterous manipulation tasks.

---

## Method

### Perception: Point Cloud Encoding

**DP3 Encoder** — This is the critical component we need to replicate. The official PyTorch implementation is provided in Appendix A:

```python
class DP3Encoder(nn.Module):
    def __init__(self, channels=3):
        self.mlp = nn.Sequential(
            nn.Linear(channels, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 256), nn.LayerNorm(256), nn.ReLU())
        self.projection = nn.Sequential(nn.Linear(256, 64), nn.LayerNorm(64))

    def forward(self, x):
        # x: B, N, 3
        x = self.mlp(x)      # B, N, 256
        x = torch.max(x, 1)[0]  # B, 256  (max-pooling over points)
        x = self.projection(x)   # B, 64
        return x
```

**Architecture breakdown:**
| Stage | Operation | Input Dim | Output Dim | Notes |
|-------|-----------|-----------|------------|-------|
| Layer 1 | Linear + LayerNorm + ReLU | N × 3 | N × 64 | Per-point MLP |
| Layer 2 | Linear + LayerNorm + ReLU | N × 64 | N × 128 | Per-point MLP |
| Layer 3 | Linear + LayerNorm + ReLU | N × 128 | N × 256 | Per-point MLP |
| Pooling | Max over N points | N × 256 | 256 | Order-invariant |
| Projection | Linear + LayerNorm | 256 | 64 | Final compact feature |

**Key architectural facts:**
- **No residual connections** — The official code and paper do not mention skip/residual connections in the encoder. It is a plain sequential MLP.
- **Max pooling** (not average pooling) over the point dimension to achieve permutation invariance.
- **LayerNorm** (not BatchNorm) interleaved after every linear layer. Table VI shows BatchNorm is actually harmful for this architecture.
- **No T-Net**, no hierarchical grouping (unlike PointNet++), no attention (unlike Point Transformer).
- **Output dimension: 64**.

**Point cloud preprocessing:**
- **Input size:** 512 or 1024 points (paper says both are sufficient; they use one or the other consistently across tasks).
- **Source:** Single-view depth image (84×84) converted to point cloud using camera intrinsics/extrinsics.
- **Color:** Intentionally discarded for appearance generalization.
- **Cropping:** Points outside a workspace bounding box are removed (table, ground filtered).
- **Downsampling:** Farthest Point Sampling (FPS), not uniform random sampling. FPS helps cover 3D space more uniformly and reduces sampling randomness.
- **Coordinate frame:** The paper uses fixed cameras; point clouds are in the **camera/world frame** (converted via extrinsics). For view generalization experiments, they manually transform point clouds when camera pose changes.

**Robot state encoding:**
- Robot poses are encoded by a separate small MLP:
  ```
  Linear(DimRobo, 64) → ReLU → Linear(64, 64)
  ```
- The 64-dim point cloud feature and 64-dim robot state feature are **concatenated** into a 128-dim global conditioning vector.

### Decision: Diffusion Policy Backbone

- **Architecture:** Convolutional network-based diffusion policy (the original Diffusion Policy [Chi et al.], NOT the transformer variant).
- **Conditioning:** The 128-dim concatenated feature is fed as **global conditioning** (`global_cond`) to the ConditionalUnet1D.
- **NO FiLM conditioning in DP3.** DP3 simply concatenates visual + state features and passes them as a global condition vector. If Lan-o3dp mentions FiLM, that is likely an addition they made, not part of the original DP3 encoder.
- **Noise scheduler:** DDIM.
- **Prediction type:** **Sample prediction** (not epsilon prediction). The paper explicitly states this brings faster convergence and better high-dimensional action generation. Table VII shows "w/o sample pred" drops average success from 78.3 to 67.0.
- **Horizon:** Short horizon — H=4 (prediction horizon), N_obs=2 (observation timesteps), N_act=3 (actions executed). This is much shorter than the original Diffusion Policy's H=16.
- **Timesteps:** 100 training timesteps, 10 inference timesteps.
- **Training:** 1000 epochs for MetaWorld, 3000 for others. Batch size 128.
- **Normalization:** Actions and observations scaled independently to [-1, 1].

---

## Key Contributions

1. **Efficiency:** Handles most tasks with just 10 demonstrations; 24.2% relative improvement over 2D Diffusion Policy averaged over 72 tasks.
2. **Generalization:** Superior spatial, viewpoint, appearance, and instance generalization due to 3D representation.
3. **Safety:** Rarely violates safety constraints in real-world deployment (0% safety violation rate vs. 32.5% for image-based DP).
4. **Design insights:** A simple 3-layer MLP encoder with max pooling outperforms complex pre-trained encoders (PointNeXt, Point Transformer, PointNet++). Key ablation findings:
   - Removing T-Net and BatchNorm from PointNet is the main reason DP3-style encoder works (Table VI).
   - Cropping point clouds is essential (w/o cropping drops performance from 78.3 to 45.3).
   - The projection head to 64-dim slightly speeds up inference without hurting accuracy.

---

## Relevance to Our Work

### Does it address Δvis/Δgeo decomposition?
**No.** DP3 is a pure imitation learning method. It does not explicitly decompose visual ambiguity (Δvis) from geometric obstruction (Δgeo), nor does it use any inference-time guidance or cost functions. It learns end-to-end from demonstrations.

### Action-space or trajectory-space representation?
**Action-space.** DP3 predicts action sequences (H=4, N_act=3), not EEF trajectories. This is the same representation as our current Diffusion Policy.

### Inference-side or training-side approach?
**Training-side.** DP3 improves the observation encoder and trains end-to-end. There is no inference-time modification like our obstacle guidance.

### Oracle information or deployable perception?
**Deployable perception.** DP3 uses real single-view depth cameras (RealSense L515) and processes raw depth→point clouds. No oracle state is used at test time. However, note they use a **fixed camera** with known intrinsics/extrinsics, and they **crop point clouds with a predefined workspace bounding box**.

### Can it be reproduced in our robomimic/robosuite framework?
**Yes, with moderate effort.** The encoder itself is trivial (~20 lines of PyTorch). The main integration work is:
1. **Observation pipeline:** We need to generate point clouds from our simulator's depth camera, downsample with FPS to 512/1024 points, crop to workspace bounds, and feed them into the encoder.
2. **Observation encoder integration:** In robomimic, we currently use `ObservationGroupEncoder` with image encoders (ResNet). We need to add a point cloud encoder branch or replace the image encoder.
3. **Global conditioning:** Our existing `ConditionalUnet1D` already accepts `global_cond_dim`. We just need to concatenate the 64-dim point cloud feature with the robot state feature and pass that as `global_cond`.

### Relationship to Lan-o3dp, STAR-Gen, and other works
- **Lan-o3dp** likely uses this exact DP3 encoder architecture (the paper explicitly provides the code). If Lan-o3dp mentions FiLM or other modifications, those are additions on top of DP3's base encoder. The encoder itself — 3-layer MLP, LayerNorm, max pool, projection to 64 — is exactly what we should implement.
- **STAR-Gen** — Unknown without reading, but DP3 provides a strong baseline for 3D-conditioned diffusion policies that any follow-up work likely builds upon.

### Can our existing `depth_mask_to_world_pointcloud` generate compatible point clouds?
**Partially, with modifications.** Our existing function:
- ✅ Back-projects depth to world-frame point clouds
- ✅ Supports workspace bounding box cropping (`workspace_bounds`)
- ❌ Uses **voxel downsampling** + uniform `max_points` cap, not **Farthest Point Sampling (FPS)**
- ❌ Does not currently support removing color (though we can just ignore RGB)

**Gap:** We need to implement FPS downsampling to exactly match DP3. FPS is important because it provides more uniform spatial coverage and reduces randomness compared to uniform subsampling. However, for an initial implementation, voxel downsampling to ~1024 points may be a reasonable approximation.

### Coordinate frame
Our function outputs points in the **world frame** (via `camera_to_world` matrix). DP3 also converts depth to world/camera frame via extrinsics. As long as our point clouds are in a consistent frame relative to the robot, the MLP encoder should learn the mapping. The paper notes that "accurate transformation isn't necessary due to the robustness of our network."

---

## Limitations and Gaps

1. **Fixed camera assumption:** DP3 assumes a fixed camera with known calibration. Our setup already has this.
2. **Workspace cropping required:** Performance drops dramatically without cropping (78.3 → 45.3). We need to define appropriate workspace bounds for PickPlaceCan variants.
3. **No explicit geometric reasoning:** While DP3 generalizes well spatially, it does not explicitly model obstacle avoidance or collision constraints. It may still fail on Δgeo (physical obstruction) if the training distribution does not include similar obstacles.
4. **Short action horizon:** H=4 is very short. Our current setup may benefit from a longer horizon for smooth trajectories.
5. **No guidance integration:** DP3 is pure behavior cloning. If we want to combine DP3's 3D encoder with our obstacle guidance (Route B), that is a novel combination not explored in the paper.
6. **Single-view only:** DP3 deliberately uses a single camera. Multi-view point clouds might help but are not studied.

---

## Potentially Reusable Code or Ideas

### 1. DP3 Encoder (direct copy-paste ready)
The official code is only ~15 lines. We can drop this directly into `robomimic/models/obs_nets.py` or as a standalone module:
```python
class DP3Encoder(nn.Module):
    def __init__(self, input_dim=3, output_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 256), nn.LayerNorm(256), nn.ReLU(),
        )
        self.projection = nn.Sequential(
            nn.Linear(256, output_dim), nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        # x: [B, N, 3]
        x = self.mlp(x)           # [B, N, 256]
        x = torch.max(x, dim=1)[0]  # [B, 256]
        x = self.projection(x)     # [B, 64]
        return x
```

### 2. FPS Downsampling
We need a PyTorch or NumPy implementation of Farthest Point Sampling. PyTorch3D has one, or we can implement a simple version.

### 3. Observation Config Changes
In `robomimic/config/diffusion_policy_config.py` and observation utils, we need to:
- Add a new observation key (e.g., `"point_cloud"`) with shape `(N, 3)` where N is variable or fixed.
- Register `DP3Encoder` as an encoder option.
- Ensure the encoder output (64-dim) is concatenated with robot state before being passed to `noise_pred_net`.

### 4. Training Recipe to Adopt
- **Sample prediction** instead of epsilon prediction (we should verify our current robomimic config).
- **DDIM** with 100 train / 10 inference steps.
- **Normalize everything to [-1, 1]**.
- **Short horizon** H=4, N_act=3 — we should ablate whether this works for our pick-and-place tasks.
- **Batch size 128** and 1000-3000 epochs.

### 5. What NOT to copy from DP3
- **No FiLM:** DP3 does not use FiLM. If Lan-o3dp uses FiLM, that is their addition. Our initial DP3 implementation should just use global conditioning concatenation.
- **No residual connections in encoder:** The encoder is intentionally simple.
- **No color:** Drop RGB from point clouds for better generalization (relevant if we later test on unseen object colors).

---

## Summary for Implementation

| Component | Our Current Setup | DP3 Target | Effort |
|-----------|------------------|------------|--------|
| Visual obs | Masked RGB image (2D) | Point cloud (N×3) | Medium — need depth rendering + backprojection |
| Encoder | ResNet-18 | 3-layer MLP + max pool | Low — ~20 lines |
| Downsampling | Voxel + uniform cap | FPS to 512/1024 | Low — implement FPS |
| Conditioning | Image feature + state | PC feature (64) + state (64) | Low — concat global_cond |
| Prediction type | Epsilon? | Sample | Low — config change |
| Horizon | Likely longer | H=4, N_act=3 | Medium — may need ablation |
| Crop | Mask-based | Workspace bbox | Low — add bounds to our function |

**Conclusion:** The DP3 encoder is exactly as simple as Lan-o3dp claims: a 3-layer MLP with LayerNorm, max pooling, and a projection head. It is highly implementable in our robomimic framework. The main engineering effort is plumbing the point cloud observation pipeline (depth rendering → backprojection → FPS → cropping) and wiring the encoder into the existing `ObservationGroupEncoder` → `ConditionalUnet1D` pipeline.
