#!/usr/bin/env python3
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
    print(f"\n🚀 {description}")
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
    
    print(f"\n🎉 所有处理步骤完成！")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📁 查看结果:")
    print(f"   • 处理结果: datasets/SAR_ASC_Project/03_OMP_Results/")
    print(f"   • 分析报告: datasets/SAR_ASC_Project/03_OMP_Results/analysis/")
    
    return True

if __name__ == "__main__":
    success = main()
