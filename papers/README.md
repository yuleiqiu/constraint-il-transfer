# Paper Library

Each paper has one directory under `papers/<short-name>/`:

```text
<short-name>/
├── paper.pdf       # source PDF, preferably downloaded from arXiv
├── paper.md        # optional extracted, searchable text
├── images/         # optional figures referenced by the notes
└── analysis.md     # project-focused reading notes
```

| Directory | Paper | arXiv | Project relevance |
|---|---|---:|---|
| `dynaguide` | *DynaGuide: Steering Diffusion Policies with Active Dynamic Guidance* | 2506.13922 | Dynamics-model guidance during denoising |
| `dp3` | *3D Diffusion Policy* | 2403.03954 | Point-cloud diffusion-policy baseline |
| `lan-o3dp` | *Language-Guided Object-Centric Diffusion Policy for Generalizable and Collision-Aware Robotic Manipulation* | 2407.00451 | Object-centric perception and clean-action cost guidance |
| `embodisteer` | *EmbodiSteer* | 2606.12965 | Joint-space, whole-body collision guidance |
| `feedback-world-model` | *Feedback World Model Enables Precise Guidance of Diffusion Policy* | 2605.15705 | Feedback-corrected world-model guidance |
| `temporal-logic-guidance` | *Temporal Logic Guidance for Action-Only Diffusion Policies with World Models* | 2606.22729 | STL constraints guided through a world model |
| `cape` | *CAPE: Context-Aware Diffusion Policy Via Proximal Mode Expansion for Collision Avoidance* | 2511.22773 | Collision-aware iterative guided refinement |
| `vls` | *VLS: Steering Pretrained Robot Policies via Vision-Language Models* | 2602.03973 | Vision-language-derived trajectory rewards |

Do not keep parser-specific intermediate artifacts (for example `*_model.json`,
`*_middle.json`, or layout PDFs) in the long-term library. Add an `analysis.md`
when a paper has been reviewed in the context of this project.
