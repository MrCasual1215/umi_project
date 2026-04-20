# Disable Wandb For Local Debug Design

## Goal

Allow local macOS debugging runs to disable Weights & Biases cleanly through an
environment variable, without changing the project's default training behavior.

## Scope

This change targets the training workspaces currently relevant to local image
policy training:

- `diffusion_policy/workspace/train_diffusion_unet_image_workspace.py`
- `diffusion_policy/workspace/train_diffusion_transformer_timm_workspace.py`

The design does not change model logic, dataset logic, or default logging
behavior in normal training runs.

## Desired Behavior

- By default, training continues to use `wandb` exactly as it does today.
- If the user explicitly sets `WANDB_DISABLED=true` before launching training,
  the workspace will not initialize `wandb` tracking.
- In disabled mode, `Accelerator` should be created with `log_with=None`.
- The code should skip `accelerator.init_trackers(...)` when wandb logging is
  disabled.

## Why This Approach

This is the smallest safe change for the user's current need:

- It keeps Linux / GPU / official training behavior unchanged.
- It avoids forcing `wandb` login during local macOS validation.
- It avoids introducing new config schema changes across many yaml files.

## Interface

Normal behavior remains unchanged:

```bash
python3 train.py --config-name=train_diffusion_unet_timm_umi_workspace task=picknplace
```

Local debug mode disables wandb:

```bash
WANDB_DISABLED=true python3 train.py --config-name=train_diffusion_unet_timm_umi_workspace task=picknplace training.debug=True
```

Accepted truthy values should include:

- `1`
- `true`
- `yes`
- `on`

Case-insensitive.

## Implementation Notes

- Add a small helper inside each affected workspace to parse the environment
  variable.
- Derive `log_with` from that result:
  - enabled: `'wandb'`
  - disabled: `None`
- Guard the tracker initialization block so it only runs when wandb logging is
  enabled.

## Risks

- If some downstream code assumes a tracker always exists, local debug mode
  could expose that assumption. This risk is low because `accelerator.log(...)`
  is designed to no-op when no tracker is configured.
- The change should be applied consistently to both relevant workspaces to avoid
  surprising behavior when switching training configs.

## Non-Goals

- No yaml-level logging config redesign
- No changes to checkpointing
- No changes to rollout behavior
- No changes to wandb version pinning
