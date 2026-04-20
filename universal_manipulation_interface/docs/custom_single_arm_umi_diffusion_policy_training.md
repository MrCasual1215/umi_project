# 基于本项目的单臂 UMI Diffusion Policy 二次开发训练文档

## 1. 文档目标

这份文档面向如下场景：

- 你已经有自己的 UMI 风格数据
- 当前要做的是单臂训练
- 可用模态至少包括：
  - RGB 图像
  - 夹爪宽度
  - 末端执行器位姿
- 你的位姿当前是 `xyz + RPY`
- 你希望在尽量少改动本项目主干代码的前提下，跑通 `diffusion policy` 训练

本文档采用“最小改动接入”路线：

- 不重写训练主干
- 不优先改 policy 结构
- 不把深度图和鱼眼图作为第一版训练主线
- 只增加一层离线数据转换，把你的原始数据转成项目已经支持的 `dataset.zarr.zip`

## 2. 先说结论

对你当前的数据，最稳妥的做法是：

1. 从你现有的原始数据目录中选定一只手臂，例如 `pika_l`
2. 先选一 路 RGB 相机作为训练视觉输入
3. 将 `xyz + RPY` 离线转换为项目主线使用的 `xyz + axis-angle`
4. 将所有 episode 转成项目已有 `UmiDataset` 能直接读取的 `dataset.zarr.zip`
5. 尽量复用现成的 `task/umi.yaml` 和 `train_diffusion_unet_timm_umi_workspace.yaml` 或 `train_diffusion_transformer_umi_workspace.yaml`
6. 先跑通 `RGB + gripper_width + pose` 的训练闭环，再考虑鱼眼、深度或多相机扩展

对你当前这份数据，最小改动版本完全可行。

## 3. 项目训练主线

### 3.1 训练入口

项目训练入口是 `train.py`：

- `train.py` 通过 Hydra 读取 `diffusion_policy/config/` 下的配置
- 根据配置中的 `_target_` 动态实例化 workspace
- 最终执行 `workspace.run()`

入口链路如下：

```text
train.py
  -> 读取 Hydra config
  -> 实例化 workspace
  -> workspace.run()
```

对应代码：

- `train.py`
- `diffusion_policy/workspace/train_diffusion_unet_image_workspace.py`
- `diffusion_policy/workspace/train_diffusion_transformer_timm_workspace.py`

### 3.2 workspace 做了什么

无论是 UNet 还是 Transformer 训练 workspace，主流程都类似：

1. 实例化 policy
2. 实例化 dataset
3. 构建 dataloader
4. 从 dataset 计算 normalizer
5. 进入训练循环
6. 定期保存 checkpoint

这意味着如果你想最小改动接入自己的数据，优先改的是：

- 数据转换脚本
- task 配置
- 必要时少量 dataset/config 适配

而不是：

- `train.py`
- 训练循环主体
- diffusion model 主体实现

## 4. 本项目真正需要的数据格式

### 4.1 训练直接读取的不是原始目录

项目 README 里的主线不是直接拿原始相机/pose/json 训练，而是先生成：

```text
dataset.zarr.zip
```

训练时 `UmiDataset` 直接通过 `zarr.ZipStore` 读取这个文件。

也就是说，你当前的原始目录：

```text
umidata/double/episode0/
  camera/
  gripper/
  localization/
```

不能直接喂给训练主线，必须先转换成 zarr replay buffer。

### 4.2 ReplayBuffer 基本结构

目标训练文件的结构是：

```text
dataset.zarr.zip
  meta/
    episode_ends
  data/
    camera0_rgb
    robot0_eef_pos
    robot0_eef_rot_axis_angle
    robot0_gripper_width
    robot0_demo_start_pose
    robot0_demo_end_pose
```

其中：

- `meta/episode_ends` 是一个一维整型数组，记录每个 episode 的结束下标
- `data/*` 下每个数组的第一维都是时间维 `T`
- 所有 key 的总长度都要与 `episode_ends[-1]` 对齐

### 4.3 单臂 UMI 主线最关键的字段

对于单臂 `task=umi`，核心字段可以理解成：

- `camera0_rgb`: 视觉观测
- `robot0_eef_pos`: 末端位置，形状 `(T, 3)`
- `robot0_eef_rot_axis_angle`: 末端旋转，形状 `(T, 3)`
- `robot0_gripper_width`: 夹爪宽度，形状 `(T, 1)`

此外，项目还会使用：

- `robot0_demo_start_pose`
- `robot0_demo_end_pose`

它们用于生成相对起点姿态特征。

### 4.4 动作 `action` 是否必须显式存储

不是必须。

当前项目的 `SequenceSampler` 在 replay buffer 中没有显式 `action` 时，会自动将以下字段拼成动作：

- `robot0_eef_pos`
- `robot0_eef_rot_axis_angle`
- `robot0_gripper_width`

也就是原始单臂动作在 zarr 内部可以是 7 维：

```text
3(pos) + 3(axis-angle rot) + 1(gripper width) = 7
```

之后 `UmiDataset` 在取样时会把旋转从 axis-angle 转成 `rotation_6d`，最终训练张量里的 action 维度变成：

```text
3 + 6 + 1 = 10
```

这也是为什么 `task/umi.yaml` 里 `action.shape` 写的是 `[10]`。

## 5. 你的原始数据如何映射到本项目

### 5.1 你当前原始数据的大致结构

从你提供的目录可以看出，单个 episode 内至少包含：

```text
episode0/
  camera/
    color/
      pikaDepthCamera_l/
      pikaDepthCamera_r/
      pikaFisheyeCamera_l/
      pikaFisheyeCamera_r/
    depth/
      pikaDepthCamera_l/
      pikaDepthCamera_r/
  gripper/
    encoder/
      pika_l/
      pika_r/
  localization/
    pose/
      pika_l/
      pika_r/
```

从 `statistic.txt` 来看，采样频率大致是：

- RGB 约 30Hz
- pose 约 100Hz
- gripper encoder 约 120Hz

你已经确认三类数据时间上可以对齐，这对接入非常有利。

### 5.2 单臂最小改动的选择建议

第一版不要同时吃双臂和所有相机。

建议先固定为：

- 机械臂：`pika_l`
- RGB 相机：任选一 路稳定的相机
  - 推荐先选普通彩色相机
  - 如果 `pikaDepthCamera_l` 的 color 流稳定，可以先用它
- 暂时忽略：
  - `pika_r`
  - 深度图
  - 鱼眼图

也就是说，第一版训练样本只保留一套单臂字段。

### 5.3 字段映射表

你当前原始数据到训练字段的映射建议如下：

| 你的原始数据                                                | 项目中的目标字段                    | 说明                         |
| ----------------------------------------------------- | --------------------------- | -------------------------- |
| `camera/color/<chosen_camera>/*.png`                  | `camera0_rgb`               | 单路 RGB 图像                  |
| `localization/pose/pika_l/*.json` 中的 `x,y,z`          | `robot0_eef_pos`            | 位置，形状 `(T, 3)`             |
| `localization/pose/pika_l/*.json` 中的 `roll,pitch,yaw` | `robot0_eef_rot_axis_angle` | 需要先从 RPY 转成 axis-angle     |
| `gripper/encoder/pika_l/*.json` 中的 `distance`         | `robot0_gripper_width`      | 建议整理为 `(T, 1)`             |
| 当前 episode 第一帧 pose                                   | `robot0_demo_start_pose`    | 形状 `(T, 6)`，每帧重复同一个起点 pose |
| 当前 episode 最后一帧 pose                                  | `robot0_demo_end_pose`      | 形状 `(T, 6)`，每帧重复同一个终点 pose |

## 6. 为什么一定建议把 RPY 转成 axis-angle

### 6.1 项目内部默认的旋转字段名

项目 UMI 主线低维旋转字段是：

```text
robot0_eef_rot_axis_angle
```

这不是名字问题，而是数据路径、采样逻辑和后续 `rotation_6d` 转换都围绕这个字段设计的。

### 6.2 内部训练用的是 rotation\_6d，但输入存储建议仍然是 axis-angle

项目的做法是：

1. zarr 中存 `axis-angle`
2. `UmiDataset` 取样时，将 pose 转成矩阵
3. 再转换为 `rotation_6d`
4. 最终训练 action 和相关 obs 使用 `3 + 6 + 1`

因此最兼容的做法是：

- 原始数据如果是 `RPY`
- 离线转换成 `axis-angle`
- 写入 `robot0_eef_rot_axis_angle`

### 6.3 你的 RPY 转换时需要注意什么

你当前 pose json 的格式类似：

```json
{
  "pitch": ...,
  "roll": ...,
  "x": ...,
  "y": ...,
  "yaw": ...,
  "z": ...
}
```

离线转换时，要先明确你的 `roll / pitch / yaw` 到底对应什么旋转顺序。

通常需要在代码里显式写清楚，例如：

```python
from scipy.spatial.transform import Rotation

r = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=False)
rotvec = r.as_rotvec()
```

但这里最重要的一点不是 `xyz` 这个字符串本身，而是：

- 你上游系统定义的 `RPY` 顺序
- 它对应的是内旋还是外旋
- 单位是弧度还是角度

如果这一步定义错了，训练不会直接报错，但策略学出来会是错的。

建议你先随机抽 10 帧，人工验证：

- `RPY -> rotvec -> rotation matrix`
- 与原始姿态语义是否一致

## 7. 推荐的数据转换方案

### 7.1 核心思路

新增一个离线转换脚本，例如：

```text
scripts_custom/convert_umidata_single_arm_to_umi_zarr.py
```

这个脚本的职责只有一件事：

- 把你的原始目录转换为项目当前 `UmiDataset` 能直接读取的 `dataset.zarr.zip`

不要把训练逻辑塞进这个脚本，也不要在训练时动态读原始 json/png。

### 7.2 转换脚本建议职责

建议脚本按以下步骤工作：

1. 遍历你的 episode 目录
2. 对每个 episode 选定单臂，例如 `pika_l`
3. 对每个 episode 选定一 路 RGB 相机
4. 读取该 episode 内已对齐的时间戳序列
5. 读取对应时刻的：
   - RGB
   - gripper width
   - pose
6. 把 `xyz + RPY` 转成：
   - `robot0_eef_pos`
   - `robot0_eef_rot_axis_angle`
7. 计算该 episode 的：
   - `demo_start_pose`
   - `demo_end_pose`
8. 按 ReplayBuffer 结构写入 zarr
9. 保存成 `dataset.zarr.zip`

### 7.3 每个 episode 应如何组织

你现在的 `episode0` 非常适合直接映射成 replay buffer 里的一个 episode。

也就是说：

- 一个原始 `episodeX/`
- 对应 replay buffer 里一次 `add_episode()`

这样最简单，也最贴合项目已有逻辑。

### 7.4 目标数组形状建议

假设某个 episode 最终对齐后有 `T` 个时刻，则建议写入：

```text
camera0_rgb                 (T, H, W, 3)   uint8
robot0_eef_pos             (T, 3)         float32
robot0_eef_rot_axis_angle  (T, 3)         float32
robot0_gripper_width       (T, 1)         float32
robot0_demo_start_pose     (T, 6)         float32
robot0_demo_end_pose       (T, 6)         float32
```

其中：

- `robot0_demo_start_pose[t]` 都等于该 episode 第一帧 pose
- `robot0_demo_end_pose[t]` 都等于该 episode 最后一帧 pose

### 7.5 图像处理建议

第一版转换图像时建议：

- 统一 resize 到 `224 x 224`
- 存 `uint8`
- 使用普通 RGB 图像

如果你原始分辨率更高，可以在离线转换时做缩放。

不建议第一版在训练时才动态解码原始目录图像，因为：

- 读取慢
- 逻辑复杂
- 不符合这个项目当前主线

## 8. 最小改动代码落点建议

### 8.1 必须新增的部分

你大概率只需要新增两类内容：

1. 一个数据转换脚本
2. 一个你自己的 task 配置

建议文件如下：

```text
scripts_custom/convert_umidata_single_arm_to_umi_zarr.py
diffusion_policy/config/task/my_umi_single_arm.yaml
```

### 8.2 尽量不要动的部分

第一版尽量不要改：

- `train.py`
- `diffusion_policy/workspace/*`
- `diffusion_policy/policy/*`
- `diffusion_policy/dataset/umi_dataset.py`

因为你当前数据和项目主线已经足够接近，重写这些层收益不大。

### 8.3 task 配置怎么写

你可以直接参考 `diffusion_policy/config/task/umi.yaml` 新建一个自己的任务配置。

建议最小版本保留这些字段：

```yaml
name: my_umi_single_arm

camera_obs_latency: 0.0
robot_obs_latency: 0.0
gripper_obs_latency: 0.0
dataset_frequeny: 30
obs_down_sample_steps: 1

low_dim_obs_horizon: 2
img_obs_horizon: 2
action_horizon: 16
ignore_proprioception: False
```

`shape_meta` 建议从 `umi.yaml` 复制后简化为：

- `camera0_rgb`
- `robot0_eef_pos`
- `robot0_eef_rot_axis_angle`
- `robot0_gripper_width`

如果你不想用相对起点特征，也可以先去掉 `robot0_eef_rot_axis_angle_wrt_start` 相关项，但要同时确认 `UmiDataset` 中不会再依赖这些字段。

为了最小改动，第一版更建议沿用现有 `umi.yaml` 的写法。

### 8.4 训练配置怎么选

对于你当前场景，建议优先尝试两个现成主线之一：

1. `train_diffusion_unet_timm_umi_workspace.yaml`
2. `train_diffusion_transformer_umi_workspace.yaml`

建议顺序：

- 先用 `train_diffusion_unet_timm_umi_workspace.yaml`
- 跑通后再试 `train_diffusion_transformer_umi_workspace.yaml`

原因：

- UNet 图像策略通常更容易先跑通
- 训练显存和调参难度通常更可控

## 9. 关于采样频率和重采样

### 9.1 你的原始频率并不一致

从你的 `statistic.txt` 看，大致是：

- RGB 约 30Hz
- pose 约 100Hz
- gripper 约 120Hz

虽然你说已经对齐，但从训练视角看，最终仍建议整理到一个统一序列频率。

### 9.2 建议的统一频率

第一版建议统一到相机频率，例如：

```text
30Hz
```

原因：

- 视觉是主观测
- 统一到图像频率最自然
- 可以直接让每个 RGB 帧对应一个训练时刻

### 9.3 重采样规则建议

假设以 RGB 帧时间戳作为主时钟：

- RGB：直接取该帧
- pose：按该时刻插值或取最近邻
- gripper width：按该时刻插值或取最近邻

既然你已经对齐，这一步可以简化为“按统一时序取同一 index”。

## 10. 训练时真正吃到的数据长什么样

### 10.1 zarr 里的原始字段

zarr 内部你只需要存储：

- RGB 原图
- `pos`
- `axis-angle`
- `gripper_width`

### 10.2 dataset 输出的训练张量

`UmiDataset` 取样后会输出：

- `obs['camera0_rgb']`: `(T_obs, 3, H, W)`
- `obs['robot0_eef_pos']`: `(T_obs, 3)`
- `obs['robot0_eef_rot_axis_angle']`: 实际已转换为 `rotation_6d` 后对应的 6 维特征
- `obs['robot0_gripper_width']`: `(T_obs, 1)`
- `action`: `(T_action, 10)`

这里最容易误解的一点是：

- zarr 中的旋转存 3 维 axis-angle
- 训练张量里的旋转是 6 维 rotation-6d

这是项目内部正常行为，不是数据错了。

## 11. 推荐训练步骤

### 11.1 第一步：先做数据转换

你的第一目标不是立刻开训练，而是先能稳定生成：

```text
your_dataset.zarr.zip
```

建议先只转换 3 到 5 个 episode 做小样本验证。

### 11.2 第二步：写一个自己的 task 配置

例如：

```text
diffusion_policy/config/task/my_umi_single_arm.yaml
```

其中主要改：

- `name∆`
- `dataset_path`
- `shape_meta`
- `dataset.dataset_path`

### 11.3 第三步：直接复用现有训练配置

例如先跑：

```bash
python train.py \
  --config-name=train_diffusion_unet_timm_umi_workspace \
  task=my_umi_single_arm \
  task.dataset_path=/absolute/path/to/your_dataset.zarr.zip
```

或者：

```bash
python train.py \
  --config-name=train_diffusion_transformer_umi_workspace \
  task=my_umi_single_arm \
  task.dataset_path=/absolute/path/to/your_dataset.zarr.zip
```

如果多卡，再把命令前面换成：

```bash
accelerate --num_processes <ngpus> train.py ...
```

## 12. 强烈建议先做的 sanity check

在正式长训练前，至少检查下面几项。

### 12.1 检查 zarr 内字段和 shape

确认：

- `episode_ends` 正常递增
- `camera0_rgb.shape[0]`
- `robot0_eef_pos.shape[0]`
- `robot0_eef_rot_axis_angle.shape[0]`
- `robot0_gripper_width.shape[0]`

它们的时间维完全一致。

### 12.2 检查图像通道顺序

确保 zarr 中存的是：

```text
(T, H, W, 3)
```

不是：

- `(T, 3, H, W)`
- BGR
- 浮点图像

### 12.3 检查姿态转换是否正确

至少做两件事：

1. 随机抽几帧，验证 `RPY -> axis-angle` 合理
2. 画出 position / rotation / gripper width 的时间曲线，看是否连续

### 12.4 检查 batch 输出

最靠谱的方式是在训练前单独实例化 dataset，并打印一条 sample 的 shape。

你要确认：

- `obs['camera0_rgb']` 的时间维和配置一致
- `action.shape[-1] == 10`
- 没有 NaN

## 13. 常见坑

### 13.1 把 RPY 直接塞进 `robot0_eef_rot_axis_angle`

这是最常见错误。

字段名叫 `axis_angle`，就必须真的是 axis-angle，而不是把 `roll,pitch,yaw` 改个名字直接塞进去。

### 13.2 把 `distance` 当成 angle

你当前 gripper encoder json 示例里有：

```json
{
  "angle": 1.55,
  "distance": 0.090542910969271362
}
```

如果训练目标是夹爪开合宽度，应优先使用 `distance` 作为 `robot0_gripper_width`。

### 13.3 一个 episode 内不同模态长度不一致

你原始流频率不同，即使已对齐，也很容易在导出时出现：

- 图像 1114 帧
- pose 1113 帧
- gripper 1114 帧

转换时必须在单个 episode 内统一成同一个 `T`。

### 13.4 一开始就上多相机、鱼眼、深度

这样会把问题叠在一起：

- 数据转换复杂
- shape\_meta 更复杂
- 模型 encoder 不一定直接支持
- 出错后很难定位

第一版先单臂单 RGB 跑通，成功率最高。

### 13.5 把双臂数据直接喂单臂配置

你当前原始目录是 `double`，但本文档讲的是单臂最小改动接入。

所以第一版一定要明确：

- 只选 `pika_l` 或只选 `pika_r`
- 其余手臂数据先忽略

## 14. 你当前数据的推荐第一版方案

基于你现有目录，我建议第一版按下面方案做：

### 14.1 只选左臂

使用：

- pose: `localization/pose/pika_l`
- gripper: `gripper/encoder/pika_l`

### 14.2 只选一 路 RGB

优先候选：

- `camera/color/pikaDepthCamera_l`

如果你后续发现鱼眼视角对任务更关键，再做第二版扩展。

### 14.3 统一到 30Hz

因为 RGB 约 30Hz，所以建议最终训练序列也按 30Hz 整理。

### 14.4 输出兼容 UMI 主线的 zarr

输出字段：

```text
camera0_rgb
robot0_eef_pos
robot0_eef_rot_axis_angle
robot0_gripper_width
robot0_demo_start_pose
robot0_demo_end_pose
episode_ends
```

### 14.5 直接复用现有 image diffusion 训练配置

首选：

```text
train_diffusion_unet_timm_umi_workspace.yaml
```

等第一版稳定后，再试：

```text
train_diffusion_transformer_umi_workspace.yaml
```

## 15. 后续扩展路线

当前文档主线不把这些作为第一版，但后续可以按下面路线扩展。

### 15.1 加鱼眼

推荐两种方式：

1. 离线去畸变后当普通 RGB 用
2. 新增第二个 `camera1_rgb` 并修改 `shape_meta`

第一种更适合先验证。

### 15.2 加深度

深度不是当前主线直接支持的模态。

如果要加，至少要改：

- `shape_meta`
- dataset 图像读取逻辑
- obs encoder 输入定义
- 深度归一化方式

因此不建议作为第一版接入目标。

### 15.3 升级到双臂

等单臂跑通后，再把：

- `robot1_eef_pos`
- `robot1_eef_rot_axis_angle`
- `robot1_gripper_width`
- `camera1_rgb`

等字段补进来，并迁移到 `umi_bimanual` 风格配置。

## 16. 最终建议

如果你的目标是尽快在本项目上训练出第一版能工作的策略，最优路线是：

1. 只做单臂
2. 只用一 路普通 RGB
3. 使用 `distance` 作为夹爪宽度
4. 将 `xyz + RPY` 离线转换成 `xyz + axis-angle`
5. 导出成兼容 `UmiDataset` 的 `dataset.zarr.zip`
6. 先复用现有 `umi` 训练配置跑通

这条路线与当前项目代码最一致，风险最低，后续也最容易逐步扩展到鱼眼、深度和双臂。
