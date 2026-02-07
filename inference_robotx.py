"""
RobotX 推理与评估脚本

用法:
    # 在测试集上评估 (计算 MSE 等指标):
    python inference_robotx.py \
        --config configs/robotx_aloha_agilex.json \
        --ckpt_path checkpoints/robotx/robotx-00050000.ckpt \
        --action_stats checkpoints/robotx/action_stats.pt \
        --eval_dir /path/to/test_episodes \
        --output_dir results/

    # 单次预测 (给定图像和关节状态):
    python inference_robotx.py \
        --config configs/robotx_aloha_agilex.json \
        --ckpt_path checkpoints/robotx/robotx-00050000.ckpt \
        --action_stats checkpoints/robotx/action_stats.pt \
        --image_path observation.png \
        --instruction "use the hammer to beat the block" \
        --joint_state "0,0,0,0,0,0,1,0,0,0,0,0,0,1"

    # 可视化预测轨迹:
    python inference_robotx.py \
        --config configs/robotx_aloha_agilex.json \
        --ckpt_path checkpoints/robotx/robotx-00050000.ckpt \
        --eval_dir /path/to/test_episodes \
        --visualize \
        --num_visualize 5
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_audio_tools.inference.sampling import sample, sample_discrete_euler
from stable_audio_tools.robotics.robot_model import create_robot_model_from_config
from stable_audio_tools.robotics.action_space import denormalize_actions


def load_model(config_path: str, ckpt_path: str, device: str = "cuda"):
    """
    加载 RobotX 模型

    支持三种检查点格式:
    - .ckpt: PyTorch Lightning 检查点
    - .pt: 普通 PyTorch 模型
    - .safetensors: Safetensors 格式
    """
    print(f"Loading config from {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)

    print(f"Creating model (action_dim={config['action_dim']}, chunk_size={config.get('action_chunk_size', 50)})")
    model = create_robot_model_from_config(config)

    print(f"Loading checkpoint from {ckpt_path}")

    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(ckpt_path)
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # PyTorch Lightning 检查点格式
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            # Lightning 会在 key 前加上 "diffusion." 前缀
            # 需要移除这个前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("diffusion."):
                    new_key = k[len("diffusion."):]
                    new_state_dict[new_key] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
        else:
            state_dict = ckpt

    # 加载权重
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)} (first 5: {missing[:5]})")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")

    model = model.to(device).eval()
    print(f"Model loaded successfully on {device}")

    return model, config


@torch.no_grad()
def predict_actions(
    model,
    config: dict,
    proprio: torch.Tensor,
    video: torch.Tensor = None,
    instruction: str = "",
    cfg_scale: float = 3.0,
    num_steps: int = 50,
    device: str = "cuda",
) -> torch.Tensor:
    """
    生成动作轨迹预测

    Args:
        model: RobotX 模型
        config: 配置字典
        proprio: 当前关节状态 (action_dim,)
        video: 相机图像 (num_cameras, C, H, W)
        instruction: 语言指令
        cfg_scale: CFG 引导强度
        num_steps: 扩散采样步数
        device: 计算设备

    Returns:
        actions: 预测的动作序列 (action_dim, chunk_size)
    """
    action_dim = config["action_dim"]
    chunk_size = config.get("action_chunk_size", 50)

    # 准备 metadata (兼容 AudioX 格式)
    metadata = [{
        "prompt": instruction,
        "proprio": proprio.to(device),
        "video": video.to(device) if video is not None else torch.zeros(1, 3, 224, 224).to(device),
    }]

    # 获取条件编码
    with torch.cuda.amp.autocast():
        conditioning = model.conditioner(metadata, device)
    cond_inputs = model.get_conditioning_inputs(conditioning)

    # 从噪声采样
    noise = torch.randn(1, action_dim, chunk_size).to(device)

    with torch.cuda.amp.autocast():
        if model.diffusion_objective == "v":
            actions = sample(
                model.model, noise, num_steps, 0,
                **cond_inputs, cfg_scale=cfg_scale, batch_cfg=True
            )
        else:
            actions = sample_discrete_euler(
                model.model, noise, num_steps,
                **cond_inputs, cfg_scale=cfg_scale, batch_cfg=True
            )

    return actions[0]  # (action_dim, chunk_size)


@torch.no_grad()
def evaluate_on_dataset(
    model,
    config: dict,
    eval_dir: str,
    action_stats: dict,
    output_dir: str,
    cfg_scale: float = 3.0,
    num_steps: int = 50,
    device: str = "cuda",
    max_samples: int = None,
):
    """
    在数据集上评估模型

    计算指标:
    - MSE: 预测动作与真实动作的均方误差
    - Per-joint MSE: 每个关节的 MSE
    - Action accuracy: 动作在阈值内的比例
    """
    from stable_audio_tools.robotics.robotwin_dataset import create_robotwin_dataloader

    os.makedirs(output_dir, exist_ok=True)

    # 创建数据加载器
    dataset_config = config.get("dataset", {})
    dataloader, dataset = create_robotwin_dataloader(
        data_dir=eval_dir,
        batch_size=1,
        action_chunk_size=config.get("action_chunk_size", 50),
        image_size=dataset_config.get("image_size", 224),
        camera_names=dataset_config.get("camera_names", ["front_camera", "head_camera"]),
        task_description=dataset_config.get("task_description", "complete the task"),
        normalize=True,
        action_stats=action_stats,
        shuffle=False,
        augment=False,
        num_workers=0,
    )

    print(f"\nEvaluating on {len(dataset)} samples...")

    all_mse = []
    all_per_joint_mse = []
    all_predictions = []
    all_ground_truth = []

    num_samples = min(len(dataloader), max_samples) if max_samples else len(dataloader)

    for batch_idx, (actions, metadata) in enumerate(tqdm(dataloader, total=num_samples)):
        if max_samples and batch_idx >= max_samples:
            break

        actions = actions.to(device)  # (1, action_dim, chunk_size)

        # 获取条件
        with torch.cuda.amp.autocast():
            conditioning = model.conditioner(metadata, device)
        cond_inputs = model.get_conditioning_inputs(conditioning)

        # 预测
        noise = torch.randn_like(actions)
        with torch.cuda.amp.autocast():
            if model.diffusion_objective == "v":
                preds = sample(
                    model.model, noise, num_steps, 0,
                    **cond_inputs, cfg_scale=cfg_scale, batch_cfg=True
                )
            else:
                preds = sample_discrete_euler(
                    model.model, noise, num_steps,
                    **cond_inputs, cfg_scale=cfg_scale, batch_cfg=True
                )

        # 计算 MSE
        mse = F.mse_loss(preds, actions).item()
        all_mse.append(mse)

        # Per-joint MSE
        per_joint_mse = F.mse_loss(preds, actions, reduction="none").mean(dim=2)[0].cpu().numpy()
        all_per_joint_mse.append(per_joint_mse)

        # 保存预测结果
        all_predictions.append(preds[0].cpu().numpy())
        all_ground_truth.append(actions[0].cpu().numpy())

    # 汇总结果
    all_mse = np.array(all_mse)
    all_per_joint_mse = np.stack(all_per_joint_mse, axis=0)

    results = {
        "num_samples": len(all_mse),
        "mse": {
            "mean": float(all_mse.mean()),
            "std": float(all_mse.std()),
            "median": float(np.median(all_mse)),
            "min": float(all_mse.min()),
            "max": float(all_mse.max()),
        },
        "per_joint_mse": {
            f"joint_{i}": float(all_per_joint_mse[:, i].mean())
            for i in range(all_per_joint_mse.shape[1])
        },
        "config": {
            "cfg_scale": cfg_scale,
            "num_steps": num_steps,
            "action_chunk_size": config.get("action_chunk_size", 50),
        }
    }

    # 保存结果
    results_path = os.path.join(output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # 保存预测轨迹 (可用于可视化)
    np.save(os.path.join(output_dir, "predictions.npy"), np.stack(all_predictions))
    np.save(os.path.join(output_dir, "ground_truth.npy"), np.stack(all_ground_truth))

    # 打印结果
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"  Samples: {results['num_samples']}")
    print(f"  MSE: {results['mse']['mean']:.6f} ± {results['mse']['std']:.6f}")
    print(f"  Median MSE: {results['mse']['median']:.6f}")
    print(f"  Per-joint MSE:")
    for i in range(min(14, len(results['per_joint_mse']))):
        print(f"    Joint {i}: {results['per_joint_mse'][f'joint_{i}']:.6f}")
    print(f"\n  Results saved to: {results_path}")

    return results


def visualize_predictions(
    output_dir: str,
    num_samples: int = 5,
):
    """可视化预测 vs 真实轨迹"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping visualization")
        return

    predictions = np.load(os.path.join(output_dir, "predictions.npy"))
    ground_truth = np.load(os.path.join(output_dir, "ground_truth.npy"))

    num_samples = min(num_samples, len(predictions))
    action_dim = predictions.shape[1]

    for i in range(num_samples):
        fig, axes = plt.subplots(min(7, action_dim), 2, figsize=(14, 12))
        fig.suptitle(f"Sample {i}: Predicted vs Ground Truth")

        pred = predictions[i]  # (action_dim, chunk_size)
        gt = ground_truth[i]

        for j in range(min(7, action_dim)):
            # Left arm
            ax = axes[j, 0] if action_dim > 1 else axes[0]
            ax.plot(gt[j], label="GT", alpha=0.7)
            ax.plot(pred[j], label="Pred", linestyle="--", alpha=0.7)
            ax.set_ylabel(f"Joint {j}")
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.3)

            # Right arm (if dual-arm)
            if j + 7 < action_dim:
                ax = axes[j, 1]
                ax.plot(gt[j + 7], label="GT", alpha=0.7)
                ax.plot(pred[j + 7], label="Pred", linestyle="--", alpha=0.7)
                ax.set_ylabel(f"Joint {j + 7}")
                ax.legend(fontsize=6)
                ax.grid(True, alpha=0.3)

        axes[-1, 0].set_xlabel("Timestep")
        axes[-1, 1].set_xlabel("Timestep")

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"trajectory_{i}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Saved visualization to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="RobotX 推理与评估")

    # 模型配置
    parser.add_argument("--config", type=str, required=True,
                        help="模型配置文件路径")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="检查点路径 (.ckpt, .pt, 或 .safetensors)")
    parser.add_argument("--action_stats", type=str, default=None,
                        help="动作归一化统计量路径")
    parser.add_argument("--device", type=str, default="cuda")

    # 采样参数
    parser.add_argument("--cfg_scale", type=float, default=3.0,
                        help="CFG 引导强度")
    parser.add_argument("--num_steps", type=int, default=50,
                        help="扩散采样步数")

    # 评估模式
    parser.add_argument("--eval_dir", type=str, default=None,
                        help="评估数据目录")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大评估样本数")
    parser.add_argument("--visualize", action="store_true",
                        help="生成可视化图表")
    parser.add_argument("--num_visualize", type=int, default=5,
                        help="可视化样本数量")

    # 单次预测模式
    parser.add_argument("--image_path", type=str, default=None,
                        help="输入图像路径")
    parser.add_argument("--instruction", type=str, default="",
                        help="语言指令")
    parser.add_argument("--joint_state", type=str, default=None,
                        help="当前关节状态 (逗号分隔)")

    args = parser.parse_args()

    # 加载模型
    model, config = load_model(args.config, args.ckpt_path, args.device)

    # 加载动作统计量
    action_stats = None
    if args.action_stats and os.path.exists(args.action_stats):
        action_stats = torch.load(args.action_stats, map_location="cpu")
        print(f"Loaded action stats from {args.action_stats}")

    if args.eval_dir:
        # 评估模式
        results = evaluate_on_dataset(
            model, config, args.eval_dir, action_stats,
            args.output_dir, args.cfg_scale, args.num_steps,
            args.device, args.max_samples,
        )

        if args.visualize:
            print("\nGenerating visualizations...")
            visualize_predictions(args.output_dir, args.num_visualize)

    else:
        # 单次预测模式
        action_dim = config["action_dim"]

        # 准备关节状态
        if args.joint_state:
            proprio = torch.tensor(
                [float(x) for x in args.joint_state.split(",")],
                dtype=torch.float32,
            )
        else:
            proprio = torch.zeros(action_dim)

        # 准备图像
        video = None
        if args.image_path:
            clip_mean = [0.48145466, 0.4578275, 0.40821073]
            clip_std = [0.26862954, 0.26130258, 0.27577711]
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=clip_mean, std=clip_std),
            ])
            img = Image.open(args.image_path).convert("RGB")
            video = transform(img).unsqueeze(0)  # (1, C, H, W)

        # 预测
        actions = predict_actions(
            model, config, proprio, video,
            args.instruction, args.cfg_scale, args.num_steps, args.device,
        )

        # 反归一化
        if action_stats:
            actions_t = actions.unsqueeze(0).permute(0, 2, 1)  # (1, chunk, dim)
            actions_denorm = denormalize_actions(actions_t, action_stats)
            actions_np = actions_denorm[0].cpu().numpy()
        else:
            actions_np = actions.permute(1, 0).cpu().numpy()

        print(f"\n预测轨迹: shape {actions_np.shape}")
        print(f"  (chunk_size={actions_np.shape[0]}, action_dim={actions_np.shape[1]})")
        print(f"\n前5步动作:")
        for t in range(min(5, actions_np.shape[0])):
            print(f"  t={t}: {actions_np[t].round(4)}")

        # 保存
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, "predicted_actions.npy")
        np.save(output_path, actions_np)
        print(f"\n保存到: {output_path}")


if __name__ == "__main__":
    main()
