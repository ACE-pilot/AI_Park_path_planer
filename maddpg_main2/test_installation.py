#!/usr/bin/env python3
"""
安装验证脚本 - 逐项检查 Rebuttal 实验所需的全部依赖
运行: python test_installation.py
"""
import sys
import os

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"

results = {"pass": 0, "fail": 0, "warn": 0}


def check(name, condition, msg_pass="", msg_fail="", critical=True):
    if condition:
        results["pass"] += 1
        print(f"  {PASS} {name}" + (f" - {msg_pass}" if msg_pass else ""))
        return True
    else:
        if critical:
            results["fail"] += 1
            print(f"  {FAIL} {name}" + (f" - {msg_fail}" if msg_fail else ""))
        else:
            results["warn"] += 1
            print(f"  {WARN} {name}" + (f" - {msg_fail}" if msg_fail else ""))
        return False


print("=" * 60)
print("Rebuttal 实验环境验证")
print("=" * 60)

# ========== 1. Python 版本 ==========
print("\n--- 1. Python 版本 ---")
py_ver = sys.version_info
check(
    f"Python {py_ver.major}.{py_ver.minor}.{py_ver.micro}",
    py_ver.major == 3 and py_ver.minor == 9,
    "Python 3.9 ✓",
    f"需要 Python 3.9，当前 {py_ver.major}.{py_ver.minor}",
)

# ========== 2. CUDA / GPU ==========
print("\n--- 2. CUDA / GPU ---")
try:
    import paddle

    check("PaddlePaddle 导入", True, f"版本 {paddle.__version__}")
    cuda_ok = paddle.device.is_compiled_with_cuda()
    check("CUDA 编译支持", cuda_ok, msg_fail="PaddlePaddle 未编译 CUDA 支持")
    if cuda_ok:
        gpu_count = paddle.device.cuda.device_count()
        check(f"GPU 可用 ({gpu_count} 块)", gpu_count > 0, msg_fail="未检测到 GPU")
        if gpu_count > 0:
            place = paddle.CUDAPlace(0)
            # 简单 GPU 计算测试
            x = paddle.randn([2, 3])
            y = paddle.matmul(x, x.T)
            check("GPU 张量计算", y.shape == [2, 2], "矩阵乘法正常")
except ImportError:
    check("PaddlePaddle 导入", False, msg_fail="未安装 paddlepaddle-gpu")

# ========== 3. PARL 框架 ==========
print("\n--- 3. PARL 框架 ---")
try:
    import parl

    check("PARL 导入", True, f"版本 {parl.__version__}")

    # 检查 MADDPG 算法
    from parl.algorithms.paddle.maddpg import MADDPG

    check("MADDPG 算法", True)

    # 检查是否为 editable 安装（源码目录）
    parl_dir = os.path.dirname(parl.__file__)
    project_root = os.path.dirname(os.path.abspath(__file__))
    is_editable = os.path.commonpath([parl_dir, project_root]) == project_root
    check(
        "PARL editable 安装",
        is_editable,
        f"源码: {parl_dir}",
        f"PARL 安装在 {parl_dir}，不在项目目录下。建议: pip install -e .",
        critical=False,
    )
except ImportError as e:
    check("PARL 导入", False, msg_fail=f"未安装: {e}")

# ========== 4. 核心依赖 ==========
print("\n--- 4. 核心依赖 ---")
deps = {
    "numpy": "numpy",
    "pandas": "pandas",
    "gym": "gym",
    "scipy": "scipy",
    "shapely": "shapely",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
}
for import_name, pip_name in deps.items():
    try:
        mod = __import__(import_name)
        ver = getattr(mod, "__version__", "?")
        check(f"{pip_name} ({ver})", True)
    except ImportError:
        check(f"{pip_name}", False, msg_fail=f"pip install {pip_name}")

# ========== 5. 环境代码 ==========
print("\n--- 5. 环境代码 ---")
project_root = os.path.dirname(os.path.abspath(__file__))

env_files = {
    "park_envE_ablation.py": os.path.join(
        project_root, "examples", "ParkEnvE_ablation_code_v2", "park_envE_ablation.py"
    ),
    "hyper_model.py (CNN-MLP)": os.path.join(
        project_root, "examples", "MADDPG_18_hyper_vision", "hyper_model.py"
    ),
    "mlp_model.py (Pure MLP)": os.path.join(
        project_root, "examples", "ParkEnvE_ablation_code_v2", "mlp_model.py"
    ),
    "train.py (主训练)": os.path.join(
        project_root, "examples", "MADDPG_18_hyper_vision", "train.py"
    ),
    "MADDPGWrapper.py": os.path.join(
        project_root, "examples", "MADDPG_18_hyper_vision", "MADDPGWrapper.py"
    ),
}
for name, path in env_files.items():
    check(name, os.path.exists(path), msg_fail=f"文件不存在: {path}")

# ========== 6. 环境可导入测试 ==========
print("\n--- 6. 环境可导入测试 ---")
try:
    # 添加必要路径
    examples_dir = os.path.join(project_root, "examples")
    sys.path.insert(0, os.path.join(examples_dir, "ParkEnvE_ablation_code_v2"))
    sys.path.insert(0, os.path.join(examples_dir, "MADDPG_18_hyper_vision"))
    sys.path.insert(0, project_root)

    from park_envE_ablation import ParkEnv

    check("ParkEnvE 环境导入", True)

    # 尝试创建环境实例
    from MADDPGWrapper import MADDPGWrapper
    env = ParkEnv(num_agents=3, render_mode=False)
    env = MADDPGWrapper(env)
    obs = env.reset()
    n_agents = env.n
    obs_dim = env.obs_shape_n[0][0]
    act_dim = env.act_shape_n[0]
    check(
        f"ParkEnvE 初始化 (agents={n_agents}, obs={obs_dim}, act={act_dim})",
        n_agents == 3 and obs_dim == 180 and act_dim == 4,
        msg_fail=f"维度异常: agents={n_agents}, obs={obs_dim}, act={act_dim}",
    )
except Exception as e:
    check("ParkEnvE 环境", False, msg_fail=str(e))

# ========== 7. 模型可导入测试 ==========
print("\n--- 7. 模型可导入测试 ---")
try:
    from hyper_model import MAModel as CNNMLPModel

    model = CNNMLPModel(180, 4, [(180,)] * 3, [4] * 3, True)
    check("CNN-MLP MAModel 创建", True)
except Exception as e:
    check("CNN-MLP MAModel", False, msg_fail=str(e))

try:
    from mlp_model import MAModel as MLPModel

    model = MLPModel(180, 4, [(180,)] * 3, [4] * 3, True)
    check("Pure MLP MAModel 创建", True)
except Exception as e:
    check("Pure MLP MAModel", False, msg_fail=str(e))

# ========== 8. 实验脚本存在性 ==========
print("\n--- 8. 实验脚本 ---")
exp_dir = os.path.join(project_root, "examples", "reviewer_experiments")
experiments = [
    ("exp_01", "code/mdp_definition.py"),
    ("exp_02", "code/eval_slope_ada.py"),
    ("exp_03", "code/analyze_reward.py"),
    ("exp_04", "code/analyze_degradation.py"),
    ("exp_05", "code/independent_ddpg_train.py"),
    ("exp_06", "code/visualize_cooperation.py"),
    ("exp_07", "code/train_mlp_ablation.py"),
    ("exp_08", "code/input_channel_analysis.py"),
    ("exp_09", "code/train_with_seed.py"),
    ("exp_10", "code/evaluate_generalization.py"),
    ("exp_11", "code/evaluate_zero_shot.py"),
    ("exp_12", "code/trajectory_postprocess.py"),
    ("exp_13", "code/explainability_analysis.py"),
]
for exp_name, script in experiments:
    # Find the full directory name
    matches = [
        d
        for d in os.listdir(exp_dir)
        if d.startswith(exp_name) and os.path.isdir(os.path.join(exp_dir, d))
    ]
    if matches:
        full_path = os.path.join(exp_dir, matches[0], script)
        check(f"{matches[0]}/{script}", os.path.exists(full_path), msg_fail="脚本缺失")
    else:
        check(f"{exp_name}", False, msg_fail="目录不存在")

# ========== 9. 任务文档 ==========
print("\n--- 9. 任务文档 ---")
task_doc = os.path.join(project_root, "docs", "rebuttal_tasks.md")
check("rebuttal_tasks.md", os.path.exists(task_doc))

# ========== 汇总 ==========
print("\n" + "=" * 60)
total = results["pass"] + results["fail"] + results["warn"]
print(f"总计: {total} 项检查")
print(f"  {PASS} 通过: {results['pass']}")
print(f"  {FAIL} 失败: {results['fail']}")
print(f"  {WARN} 警告: {results['warn']}")

if results["fail"] == 0:
    print(f"\n\033[92m✓ 环境配置完成，可以开始实验！\033[0m")
else:
    print(f"\n\033[91m✗ 有 {results['fail']} 项失败，请修复后重试。\033[0m")

sys.exit(results["fail"])
