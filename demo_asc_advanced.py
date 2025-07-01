"""
ASC高级提取系统演示脚本
Demo Script for Advanced ASC Extraction System

展示核心功能:
1. 完整ASC参数提取 {A, α, x, y, L, φ_bar}
2. 自适应迭代提取
3. 多散射类型识别
4. 可视化对比展示
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time
import warnings

warnings.filterwarnings("ignore")

# 导入ASC系统
try:
    from asc_extraction_advanced import ASCExtractionAdvanced
except ImportError:
    print("❌ 无法导入ASC高级系统，请确保asc_extraction_advanced.py在当前目录")
    exit(1)


class ASCAdvancedDemo:
    """ASC高级系统演示类"""

    def __init__(self):
        """初始化演示系统"""
        print("🎯 ASC高级提取系统演示")
        print("=" * 60)

        # 初始化ASC系统
        self.asc_system = ASCExtractionAdvanced(
            image_size=(128, 128),
            adaptive_threshold=0.08,  # 8% 自适应阈值
            max_iterations=30,
            min_scatterers=3,
            max_scatterers=20,
        )

        # 设置输出目录
        self.output_dir = "results/asc_demo"
        os.makedirs(self.output_dir, exist_ok=True)

    def create_synthetic_target(self) -> np.ndarray:
        """创建合成测试目标"""
        print("🎨 生成合成ASC测试目标...")

        complex_image = np.zeros((128, 128), dtype=complex)

        # 定义不同类型的散射中心
        scatterers = [
            # 三面角反射器 (α=1.0)
            {"x": 0.3, "y": 0.2, "amplitude": 1.0, "alpha": 1.0, "length": 0.1, "phi_bar": 0.0},
            # 平面反射器 (α=0.0)
            {"x": -0.2, "y": 0.4, "amplitude": 0.8, "alpha": 0.0, "length": 0.2, "phi_bar": np.pi / 4},
            # 边缘绕射 (α=-1.0)
            {"x": 0.1, "y": -0.3, "amplitude": 0.6, "alpha": -1.0, "length": 0.05, "phi_bar": np.pi / 2},
            # 复合散射 (α=0.5)
            {"x": -0.4, "y": -0.1, "amplitude": 0.7, "alpha": 0.5, "length": 0.15, "phi_bar": -np.pi / 3},
        ]

        # 频率参数
        fc = 1e10  # 中心频率
        B = 1e9  # 带宽
        omega = np.pi / 3  # 合成孔径角

        # 频率采样
        fx_range = np.linspace(-B / 2, B / 2, 128)
        fy_range = np.linspace(-fc * np.sin(omega / 2), fc * np.sin(omega / 2), 128)

        print(f"   生成 {len(scatterers)} 个不同类型的散射中心:")

        for i, scatterer in enumerate(scatterers):
            print(f"   - 散射中心 {i+1}: α={scatterer['alpha']}, L={scatterer['length']:.3f}")

            # 生成ASC原子
            atom = self.asc_system._generate_asc_atom(
                scatterer["x"],
                scatterer["y"],
                scatterer["alpha"],
                scatterer["length"],
                scatterer["phi_bar"],
                fx_range,
                fy_range,
            )

            # 添加到图像
            contribution = scatterer["amplitude"] * np.exp(1j * scatterer["phi_bar"]) * atom
            complex_image += contribution

        # 添加噪声
        noise_level = 0.05
        noise = noise_level * (np.random.randn(128, 128) + 1j * np.random.randn(128, 128))
        complex_image += noise

        print(f"   已添加噪声水平: {noise_level}")
        print(f"   合成图像能量: {np.linalg.norm(complex_image):.3f}")

        return complex_image, scatterers

    def load_real_data(self, filename: str) -> np.ndarray:
        """加载真实MSTAR数据"""
        data_path = f"datasets/SAR_ASC_Project/02_Data_Processed_raw/{filename}"

        if os.path.exists(data_path):
            print(f"📂 加载真实MSTAR数据: {filename}")
            magnitude, complex_image = self.asc_system.load_raw_data(data_path)
            return complex_image
        else:
            print(f"⚠️ 文件不存在，使用合成数据: {data_path}")
            complex_image, _ = self.create_synthetic_target()
            return complex_image

    def run_asc_extraction_demo(self, complex_image: np.ndarray, title: str = "ASC提取演示") -> dict:
        """运行ASC提取演示"""
        print(f"\n🎯 开始{title}...")
        print("-" * 50)

        start_time = time.time()

        # 1. 数据预处理
        print("⚙️ 数据预处理...")
        signal = self.asc_system.preprocess_data(complex_image)

        # 2. 构建ASC字典 (控制规模以便演示)
        print("📚 构建ASC字典...")
        dictionary, param_grid = self.asc_system.build_asc_dictionary(
            position_samples=6, azimuth_samples=3  # 较小规模用于演示
        )

        # 3. 自适应ASC提取
        print("🔍 自适应ASC提取...")
        scatterers = self.asc_system.adaptive_asc_extraction(signal, dictionary, param_grid)

        # 4. 参数精化
        if len(scatterers) > 0:
            print("🔧 参数精化...")
            refined_scatterers = self.asc_system.refine_parameters(scatterers[:5], signal)  # 只精化前5个
        else:
            refined_scatterers = scatterers

        # 5. 图像重构
        print("🔄 图像重构...")
        reconstructed = self.asc_system.reconstruct_asc_image(refined_scatterers)

        # 6. 结果分析
        analysis = self.asc_system.analyze_asc_results(refined_scatterers)

        processing_time = time.time() - start_time

        # 计算PSNR
        original_magnitude = np.abs(complex_image)
        reconstructed_magnitude = np.abs(reconstructed)
        mse = np.mean((original_magnitude - reconstructed_magnitude) ** 2)
        psnr = 20 * np.log10(np.max(original_magnitude) / np.sqrt(mse)) if mse > 0 else float("inf")

        results = {
            "scatterers": refined_scatterers,
            "reconstructed": reconstructed,
            "analysis": analysis,
            "processing_time": processing_time,
            "psnr": psnr,
            "dictionary_size": dictionary.shape[1],
        }

        print(f"✅ {title}完成!")
        print(f"   提取散射中心数: {len(refined_scatterers)}")
        print(f"   处理时间: {processing_time:.2f}秒")
        print(f"   重构PSNR: {psnr:.1f}dB")
        print(f"   字典规模: {dictionary.shape[1]} 个原子")

        return results

    def create_comprehensive_visualization(
        self, original_image: np.ndarray, results: dict, title: str, ground_truth: list = None
    ) -> None:
        """创建全面的可视化展示"""
        print("🎨 生成综合可视化...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"{title} - ASC高级提取系统结果", fontsize=16, fontweight="bold")

        # 1. 原始图像
        axes[0, 0].imshow(np.abs(original_image), cmap="hot", aspect="auto")
        axes[0, 0].set_title("原始SAR图像")
        axes[0, 0].set_xlabel("距离维")
        axes[0, 0].set_ylabel("方位维")

        # 2. 重构图像
        if results["reconstructed"] is not None:
            axes[0, 1].imshow(np.abs(results["reconstructed"]), cmap="hot", aspect="auto")
            axes[0, 1].set_title(f'ASC重构图像\nPSNR: {results["psnr"]:.1f}dB')
            axes[0, 1].set_xlabel("距离维")
            axes[0, 1].set_ylabel("方位维")

        # 3. 散射中心位置 (按α值着色)
        axes[0, 2].imshow(np.abs(original_image), cmap="gray", alpha=0.6, aspect="auto")

        # α值颜色映射
        alpha_colors = {-1.0: "blue", -0.5: "cyan", 0.0: "green", 0.5: "orange", 1.0: "red"}

        # 标记提取的散射中心
        for i, scatterer in enumerate(results["scatterers"]):
            x_pixel = int((scatterer["x"] + 1) * 64)
            y_pixel = int((scatterer["y"] + 1) * 64)
            alpha_val = scatterer["alpha"]
            color = alpha_colors.get(alpha_val, "purple")

            # 圆圈大小表示幅度
            radius = max(2, int(scatterer["estimated_amplitude"] * 5))
            circle = Circle((x_pixel, y_pixel), radius, color=color, fill=True, alpha=0.8)
            axes[0, 2].add_patch(circle)

            # 添加编号
            axes[0, 2].text(x_pixel + 3, y_pixel - 3, str(i + 1), color="white", fontsize=8, fontweight="bold")

        # 如果有真实值，用星号标记
        if ground_truth:
            for gt in ground_truth:
                x_pixel = int((gt["x"] + 1) * 64)
                y_pixel = int((gt["y"] + 1) * 64)
                axes[0, 2].scatter(
                    x_pixel, y_pixel, marker="*", s=100, c="yellow", edgecolors="black", linewidth=2, label="真实位置"
                )

        axes[0, 2].set_title("提取的散射中心\n(颜色=α值, 大小=幅度)")
        axes[0, 2].set_xlim(0, 128)
        axes[0, 2].set_ylim(0, 128)

        # 添加α值颜色图例
        from matplotlib.patches import Patch

        legend_elements = [Patch(facecolor=color, label=f"α={alpha}") for alpha, color in alpha_colors.items()]
        axes[0, 2].legend(handles=legend_elements, loc="upper right", fontsize=8)

        # 4. α值分布饼图
        if "analysis" in results and "alpha_distribution" in results["analysis"]:
            alpha_dist = results["analysis"]["alpha_distribution"]
            if alpha_dist:
                axes[1, 0].pie(
                    alpha_dist.values(),
                    labels=[f"α={k}\n({v}个)" for k, v in alpha_dist.items()],
                    autopct="%1.1f%%",
                    startangle=90,
                    colors=[alpha_colors.get(k, "gray") for k in alpha_dist.keys()],
                )
                axes[1, 0].set_title("散射类型分布\n(按α值)")
            else:
                axes[1, 0].text(0.5, 0.5, "无α分布数据", ha="center", va="center", transform=axes[1, 0].transAxes)

        # 5. 参数统计
        if results["scatterers"]:
            # 幅度分布
            amplitudes = [s["estimated_amplitude"] for s in results["scatterers"]]
            axes[1, 1].hist(amplitudes, bins=min(8, len(amplitudes)), alpha=0.7, color="green", edgecolor="black")
            axes[1, 1].set_xlabel("散射幅度")
            axes[1, 1].set_ylabel("频次")
            axes[1, 1].set_title("幅度分布")
            axes[1, 1].grid(True, alpha=0.3)

            # 长度参数分布
            lengths = [s["length"] for s in results["scatterers"]]
            axes[1, 2].hist(lengths, bins=min(8, len(lengths)), alpha=0.7, color="orange", edgecolor="black")
            axes[1, 2].set_xlabel("长度参数 L")
            axes[1, 2].set_ylabel("频次")
            axes[1, 2].set_title("长度参数分布")
            axes[1, 2].grid(True, alpha=0.3)

        # 移除空白子图
        for ax in axes.flat:
            if not ax.has_data():
                ax.axis("off")

        plt.tight_layout()

        # 保存图像
        filename = title.replace(" ", "_").replace("-", "_")
        output_path = os.path.join(self.output_dir, f"{filename}_results.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"   📊 可视化结果已保存: {output_path}")

    def print_detailed_results(self, results: dict) -> None:
        """打印详细结果"""
        print("\n📋 详细ASC提取结果:")
        print("-" * 60)

        if not results["scatterers"]:
            print("⚠️ 未提取到散射中心")
            return

        print(f"🎯 提取散射中心详情 (共{len(results['scatterers'])}个):")
        print("-" * 60)
        print(f"{'序号':<4} {'X位置':<8} {'Y位置':<8} {'α值':<6} {'长度L':<8} {'幅度A':<8} {'相位φ':<8} {'优化':<4}")
        print("-" * 60)

        for i, scatterer in enumerate(results["scatterers"], 1):
            optimized = "✓" if scatterer.get("optimization_success", False) else "✗"
            print(
                f"{i:<4} {scatterer['x']:<8.3f} {scatterer['y']:<8.3f} "
                f"{scatterer['alpha']:<6.1f} {scatterer['length']:<8.3f} "
                f"{scatterer['estimated_amplitude']:<8.3f} "
                f"{scatterer['estimated_phase']:<8.3f} {optimized:<4}"
            )

        if "analysis" in results:
            analysis = results["analysis"]
            print(f"\n📊 统计分析:")
            print(f"   α值分布: {analysis.get('alpha_distribution', {})}")
            print(
                f"   幅度统计: 均值={analysis.get('amplitude_stats', {}).get('mean', 0):.3f}, "
                f"标准差={analysis.get('amplitude_stats', {}).get('std', 0):.3f}"
            )
            print(f"   优化成功率: {analysis.get('optimization_success_rate', 0):.1%}")

    def run_demo(self, use_real_data: bool = False, filename: str = None):
        """运行完整演示"""
        print("🚀 启动ASC高级提取系统演示")
        print("=" * 70)

        # 准备测试数据
        if use_real_data and filename:
            complex_image = self.load_real_data(filename)
            ground_truth = None
            title = f"真实MSTAR数据 - {filename}"
        else:
            complex_image, ground_truth = self.create_synthetic_target()
            title = "合成测试目标"

        # 运行ASC提取
        results = self.run_asc_extraction_demo(complex_image, title)

        # 生成可视化
        self.create_comprehensive_visualization(complex_image, results, title, ground_truth)

        # 打印详细结果
        self.print_detailed_results(results)

        print(f"\n✅ ASC演示完成! 结果保存在: {self.output_dir}")

        return results


def main():
    """主演示函数"""
    print("🎯 ASC高级提取系统演示")
    print("=" * 70)

    # 初始化演示系统
    demo = ASCAdvancedDemo()

    # 运行合成数据演示
    print("\n📍 模式1: 合成目标演示")
    synthetic_results = demo.run_demo(use_real_data=False)

    # 检查是否有真实数据
    data_dir = "datasets/SAR_ASC_Project/02_Data_Processed_raw"
    if os.path.exists(data_dir):
        raw_files = [f for f in os.listdir(data_dir) if f.endswith(".raw")]
        if raw_files:
            print(f"\n📍 模式2: 真实MSTAR数据演示")
            print(f"   找到 {len(raw_files)} 个RAW文件，使用第一个进行演示...")
            real_results = demo.run_demo(use_real_data=True, filename=raw_files[0])
        else:
            print("\n⚠️ 未找到真实MSTAR数据，仅运行合成数据演示")
    else:
        print("\n⚠️ 数据目录不存在，仅运行合成数据演示")

    print("\n🎉 ASC高级提取系统演示完成!")
    print(f"   结果保存在: {demo.output_dir}")
    print("\n💡 关键改进:")
    print("   ✅ 自适应散射中心数量 (不再固定40个)")
    print("   ✅ 完整ASC参数提取 {A, α, x, y, L, φ_bar}")
    print("   ✅ 多散射类型识别 (α值区分)")
    print("   ✅ 精确位置估计 (非网格约束)")
    print("   ✅ 参数后处理优化")

    return demo


if __name__ == "__main__":
    demo_system = main()
