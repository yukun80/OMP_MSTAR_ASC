#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSTAR OMP散射中心提取 - 完整工作流程指南
MSTAR OMP Scattering Center Extraction - Complete Workflow Guide

指导用户完成从原始MSTAR数据到最终分析结果的完整流程
"""

import os
import sys
from datetime import datetime


class WorkflowGuide:
    """工作流程指导器"""

    def __init__(self):
        self.project_root = os.getcwd()
        self.data_dir = os.path.join(self.project_root, "datasets", "SAR_ASC_Project")

    def print_header(self, title: str):
        """打印标题"""
        print("\n" + "=" * 80)
        print(f"🎯 {title}")
        print("=" * 80)

    def print_step(self, step_num: int, title: str):
        """打印步骤"""
        print(f"\n📋 步骤 {step_num}: {title}")
        print("-" * 60)

    def check_prerequisites(self):
        """检查前置条件"""
        self.print_step(0, "检查前置条件")

        # 检查Python环境
        print(f"🐍 Python版本: {sys.version}")

        # 检查必要模块
        required_modules = ["numpy", "matplotlib", "sklearn", "scipy", "pandas"]

        missing_modules = []
        for module in required_modules:
            try:
                __import__(module)
                print(f"✅ {module}: 已安装")
            except ImportError:
                print(f"❌ {module}: 未安装")
                missing_modules.append(module)

        if missing_modules:
            print(f"\n⚠️  缺少依赖模块，请安装:")
            print(f"   pip install {' '.join(missing_modules)}")
            return False

        # 检查数据目录
        if os.path.exists(self.data_dir):
            print(f"✅ 数据目录存在: {self.data_dir}")
        else:
            print(f"❌ 数据目录不存在: {self.data_dir}")
            return False

        # 检查关键脚本
        key_scripts = ["omp_asc_final.py", "test_single_mstar.py", "process_mstar_data.py", "analyze_results.py"]

        for script in key_scripts:
            if os.path.exists(script):
                print(f"✅ {script}: 存在")
            else:
                print(f"❌ {script}: 不存在")
                return False

        print(f"\n🎉 所有前置条件满足！")
        return True

    def show_data_status(self):
        """显示数据状态"""
        self.print_step(1, "数据状态检查")

        subdirs = [
            ("00_Data_Raw", "原始MSTAR数据"),
            ("01_Data_Processed_mat", "MAT格式数据"),
            ("02_Data_Processed_raw", "RAW格式数据"),
            ("03_OMP_Results", "OMP处理结果"),
        ]

        for subdir, desc in subdirs:
            full_path = os.path.join(self.data_dir, subdir)
            if os.path.exists(full_path):
                files = []
                for root, dirs, filenames in os.walk(full_path):
                    files.extend(filenames)
                print(f"✅ {desc}: {len(files)} 个文件")
            else:
                print(f"❌ {desc}: 目录不存在")

        # 检查RAW文件（处理的关键输入）
        raw_dir = os.path.join(self.data_dir, "02_Data_Processed_raw", "SN_S7")
        if os.path.exists(raw_dir):
            raw_files = [f for f in os.listdir(raw_dir) if f.endswith(".raw")]
            print(f"\n📊 可用RAW文件: {len(raw_files)} 个")
            for i, file in enumerate(raw_files[:3], 1):
                print(f"   {i}. {file}")
            if len(raw_files) > 3:
                print(f"   ... 和另外 {len(raw_files)-3} 个文件")
        else:
            print(f"\n❌ RAW数据目录不存在: {raw_dir}")

    def show_workflow_overview(self):
        """显示工作流程概览"""
        self.print_header("完整工作流程概览")

        workflow_steps = [
            ("数据预处理", "MATLAB", "已完成 ✅", "step1_MSTAR2mat.m → step2_MSTAR_mat2raw.m"),
            ("算法验证", "Python", "准备执行", "test_single_mstar.py"),
            ("批量处理", "Python", "准备执行", "process_mstar_data.py"),
            ("结果分析", "Python", "准备执行", "analyze_results.py"),
            ("结果可视化", "自动生成", "自动完成", "PNG图像 + 统计报告"),
        ]

        print(f"{'阶段':<12} {'工具':<10} {'状态':<12} {'说明':<40}")
        print("-" * 80)
        for step, tool, status, desc in workflow_steps:
            print(f"{step:<12} {tool:<10} {status:<12} {desc:<40}")

    def show_execution_guide(self):
        """显示执行指南"""
        self.print_header("详细执行指南")

        # 步骤1: 快速测试
        self.print_step(1, "执行单文件测试（推荐）")
        print("目的: 验证算法是否正常工作")
        print("命令: python test_single_mstar.py")
        print("预期: 显示处理结果和性能评估")
        print("时间: 约30-60秒")

        # 步骤2: 批量处理
        self.print_step(2, "执行批量处理")
        print("目的: 处理所有MSTAR数据文件")
        print("命令: python process_mstar_data.py")
        print("预期: 生成所有文件的OMP处理结果")
        print("时间: 约5-10分钟（取决于文件数量）")
        print("输出: datasets/SAR_ASC_Project/03_OMP_Results/")

        # 步骤3: 结果分析
        self.print_step(3, "执行结果分析")
        print("目的: 分析处理结果，生成统计报告")
        print("命令: python analyze_results.py")
        print("预期: 生成综合分析报告和可视化图表")
        print("时间: 约30-60秒")
        print("输出: datasets/SAR_ASC_Project/03_OMP_Results/analysis/")

    def show_expected_outputs(self):
        """显示预期输出"""
        self.print_header("预期输出文件")

        outputs = [
            ("单文件测试", ["datasets/SAR_ASC_Project/test_results/single_test_*.png"]),
            (
                "批量处理结果",
                [
                    "datasets/SAR_ASC_Project/03_OMP_Results/*_scatterers.pkl",
                    "datasets/SAR_ASC_Project/03_OMP_Results/*_visualization.png",
                    "datasets/SAR_ASC_Project/03_OMP_Results/*_summary.txt",
                    "datasets/SAR_ASC_Project/03_OMP_Results/processing_summary.txt",
                ],
            ),
            (
                "结果分析",
                [
                    "datasets/SAR_ASC_Project/03_OMP_Results/analysis/comprehensive_analysis_report.txt",
                    "datasets/SAR_ASC_Project/03_OMP_Results/analysis/analysis_dashboard.png",
                    "datasets/SAR_ASC_Project/03_OMP_Results/analysis/all_scatterers_data.csv",
                    "datasets/SAR_ASC_Project/03_OMP_Results/analysis/file_statistics.csv",
                ],
            ),
        ]

        for category, files in outputs:
            print(f"\n📁 {category}:")
            for file in files:
                print(f"   • {file}")

    def show_performance_expectations(self):
        """显示性能预期"""
        self.print_header("性能预期指标")

        expectations = [
            ("处理时间", "每个128×128图像: 20-60秒"),
            ("重构质量", "PSNR > 20 dB"),
            ("散射中心数", "每个文件: 20-40个"),
            ("位置精度", "归一化坐标误差 < 0.1"),
            ("内存使用", "峰值 < 2GB"),
            ("成功率", "> 95%"),
        ]

        print(f"{'指标':<15} {'预期值':<30}")
        print("-" * 50)
        for metric, value in expectations:
            print(f"{metric:<15} {value:<30}")

    def show_troubleshooting(self):
        """显示故障排除"""
        self.print_header("常见问题与解决方案")

        issues = [
            ("内存不足", ["减少position_grid_size参数", "减少phase_levels参数", "逐个处理文件而非批量"]),
            ("处理时间过长", ["使用快速配置参数", "减少n_scatterers数量", "检查CPU使用率"]),
            ("PSNR过低", ["检查数据质量", "调整字典参数", "验证数据格式"]),
            ("找不到散射中心", ["检查信号幅度", "调整稀疏度参数", "验证算法配置"]),
        ]

        for issue, solutions in issues:
            print(f"\n❓ {issue}:")
            for solution in solutions:
                print(f"   • {solution}")

    def create_execution_script(self):
        """创建自动执行脚本"""
        script_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSTAR OMP处理 - 自动执行脚本
Auto-execution script for MSTAR OMP processing
"""

import os
import subprocess
import sys
from datetime import datetime

def run_command(cmd, description):
    """运行命令并处理结果"""
    print(f"\\n🚀 {description}")
    print(f"执行命令: {cmd}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {description} 成功完成")
            if result.stdout:
                print("输出:")
                print(result.stdout[-500:])  # 显示最后500字符
        else:
            print(f"❌ {description} 失败")
            print(f"错误: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ 执行失败: {str(e)}")
        return False
    
    return True

def main():
    """主执行函数"""
    print("🎯 MSTAR OMP处理 - 自动执行流程")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 步骤1: 单文件测试
    if not run_command("python test_single_mstar.py", "单文件测试"):
        print("❌ 单文件测试失败，停止执行")
        return False
    
    # 步骤2: 批量处理
    if not run_command("python process_mstar_data.py", "批量处理"):
        print("❌ 批量处理失败，停止执行")
        return False
    
    # 步骤3: 结果分析
    if not run_command("python analyze_results.py", "结果分析"):
        print("❌ 结果分析失败")
        return False
    
    print(f"\\n🎉 所有处理步骤完成！")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\\n📁 查看结果:")
    print(f"   • 处理结果: datasets/SAR_ASC_Project/03_OMP_Results/")
    print(f"   • 分析报告: datasets/SAR_ASC_Project/03_OMP_Results/analysis/")
    
    return True

if __name__ == "__main__":
    success = main()
'''

        script_path = "run_complete_workflow.py"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)

        print(f"\n💾 自动执行脚本已创建: {script_path}")
        print(f"使用方法: python {script_path}")

    def show_next_steps(self):
        """显示下一步操作"""
        self.print_header("下一步操作建议")

        print("🎯 推荐执行顺序:")
        print("1. 运行快速测试: python test_single_mstar.py")
        print("2. 如果测试通过，运行批量处理: python process_mstar_data.py")
        print("3. 处理完成后，运行结果分析: python analyze_results.py")
        print("4. 或者直接运行自动脚本: python run_complete_workflow.py")

        print("\n📋 处理完成后您将获得:")
        print("• 每个MSTAR文件的40个散射中心参数")
        print("• 高质量的可视化结果图像")
        print("• 详细的统计分析报告")
        print("• CSV格式的数据导出")
        print("• 综合性能评估报告")

        print("\n🚀 开始处理吧！")


def main():
    """主指南函数"""
    guide = WorkflowGuide()

    print("🎯 MSTAR OMP散射中心提取 - 完整工作流程指南")
    print("=" * 80)
    print("本指南将引导您完成从原始MSTAR数据到最终分析结果的完整流程")

    # 检查前置条件
    if not guide.check_prerequisites():
        print("\n❌ 前置条件不满足，请先解决相关问题")
        return False

    # 显示数据状态
    guide.show_data_status()

    # 显示工作流程概览
    guide.show_workflow_overview()

    # 显示执行指南
    guide.show_execution_guide()

    # 显示预期输出
    guide.show_expected_outputs()

    # 显示性能预期
    guide.show_performance_expectations()

    # 显示故障排除
    guide.show_troubleshooting()

    # 创建自动执行脚本
    guide.create_execution_script()

    # 显示下一步操作
    guide.show_next_steps()

    return True


if __name__ == "__main__":
    success = main()
