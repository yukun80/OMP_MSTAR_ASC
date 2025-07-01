# MSTAR散射中心提取项目 - 使用指南

基于正交匹配追踪(OMP)和自适应属性散射中心(ASC)提取的完整解决方案

---

## 📋 项目概述

本项目提供了**三套不同的散射中心提取算法**和**四套测试演示程序**，满足从快速处理到科研级精度的不同需求。经过完整的算法重构，解决了网格约束、固定稀疏度、参数提取不完整等核心技术问题。

**核心特性**：

- ✅ **多算法架构**：3套算法覆盖不同精度需求
- ✅ **完整测试框架**：4套程序验证算法性能
- ✅ **真实数据支持**：支持MSTAR数据处理
- ✅ **用户友好**：清晰的运行顺序和使用指南

---

## 📂 项目文件结构

### **核心算法程序（3个）**

| 文件名                          | 功能         | 特点                       | 推荐场景     |
| ------------------------------- | ------------ | -------------------------- | ------------ |
| `omp_asc_final.py`            | 传统OMP算法  | 稳定快速，5-10秒处理       | 日常快速处理 |
| `asc_extraction_advanced.py`  | ASC高级算法  | 完整6维参数，散射类型识别  | 科研分析     |
| `asc_extraction_precision.py` | 精度优化算法 | 4级精度模式，3秒-5分钟可选 | 高精度需求   |

### **测试演示程序（4个）**

| 文件名                              | 功能           | 运行时间 | 推荐对象       |
| ----------------------------------- | -------------- | -------- | -------------- |
| `demo_asc_advanced.py`            | ASC功能演示    | 30秒     | 新用户了解功能 |
| `test_asc_advanced_comparison.py` | 三算法性能对比 | 2-3分钟  | 选择最佳算法   |
| `test_mstar_quick_precision.py`   | 快速精度测试   | 15秒     | 实用性验证     |
| `test_mstar_precision.py`         | 高精度测试     | 2-5分钟  | 科研级验证     |

---

## 🚀 推荐使用流程

### **第一步：环境准备**

```bash
# 安装依赖
pip install -r requirements.txt

# 确认数据目录存在
ls datasets/SAR_ASC_Project/02_Data_Processed_raw/SN_S7/
```

### **第二步：选择运行路径**

#### **路径A：新用户快速体验** ⭐ 推荐新手

```bash
# 1. 功能演示（了解ASC算法能力）
python demo_asc_advanced.py
# ⏱️ 30秒 | 📊 6图展示 | 🎯 理解算法原理

# 2. 算法对比（选择最适合的算法）
python test_asc_advanced_comparison.py  
# ⏱️ 2-3分钟 | 📊 性能对比 | 🎯 算法选择依据
```

#### **路径B：快速验证测试** ⭐ 推荐实用

```bash
# 快速精度测试（验证算法在你的数据上的效果）
python test_mstar_quick_precision.py
# ⏱️ 15秒 | 📊 balanced模式 | 🎯 实用性验证
```

#### **路径C：高精度科研** ⭐ 推荐科研

```bash
# 高精度测试（科研级最高精度验证）
python test_mstar_precision.py
# ⏱️ 2-5分钟 | 📊 ultra模式 | 🎯 科研级验证
```

### **第三步：选择核心算法进行实际处理**

根据第二步的测试结果，选择最适合你需求的核心算法：

#### **方案1：传统OMP算法**（快速稳定）

```python
from omp_asc_final import OMPASCExtractor

# 5-10秒快速处理
omp = OMPASCExtractor(n_scatterers=40)
results = omp.process_mstar_file("your_data.raw")
```

#### **方案2：ASC高级算法**（功能完整）

```python
from asc_extraction_advanced import ASCExtractorAdvanced

# 完整6维参数+散射类型识别
asc = ASCExtractorAdvanced(max_scatterers=30)
scatterers = asc.extract_asc_scatterers(complex_data)
```

#### **方案3：精度优化算法**（多级精度）

```python
from asc_extraction_precision import PrecisionASCExtractor

# 4级精度可选：fast/balanced/high/ultra
precision_asc = PrecisionASCExtractor()
results = precision_asc.extract_with_precision(data, mode="balanced")
```

---

## 📊 算法性能对比

| 算法               | 处理时间  | 散射中心数     | 参数维度 | 适用场景     |
| ------------------ | --------- | -------------- | -------- | ------------ |
| **传统OMP**  | 5-10秒    | 40个(固定)     | 4维基础  | 日常快速处理 |
| **ASC高级**  | 20-40秒   | 5-30个(自适应) | 6维完整  | 科研分析     |
| **精度优化** | 3秒-5分钟 | 5-30个(自适应) | 6维完整  | 灵活精度需求 |

### **精度模式详细说明**（精度优化算法）

| 模式               | 处理时间 | 位置采样   | 推荐用途    |
| ------------------ | -------- | ---------- | ----------- |
| **fast**     | 3-5秒    | 64个位置   | 快速预览    |
| **balanced** | 10-15秒  | 256个位置  | 日常处理 ⭐ |
| **high**     | 30-60秒  | 1024个位置 | 科研分析    |
| **ultra**    | 2-5分钟  | 4096个位置 | 极高精度    |

---

## 🛠️ 核心算法使用示例

### **快速开始示例**

```python
# 最简单的使用方式
from omp_asc_final import OMPASCExtractor

# 初始化算法
extractor = OMPASCExtractor(n_scatterers=40)

# 处理单个文件
magnitude, complex_data = extractor.load_raw_data("HB03333.017.128x128.raw")
results = extractor.extract_scatterers(extractor.preprocess_data(complex_data))

# 查看结果
print(f"提取到 {len(results['scatterers'])} 个散射中心")
for i, sc in enumerate(results['scatterers'][:5]):  # 显示前5个
    print(f"散射中心{i+1}: 位置({sc['x']:.3f}, {sc['y']:.3f}), 幅度{sc['estimated_amplitude']:.3f}")
```

### **高级功能示例**

```python
# ASC高级算法 - 完整参数提取
from asc_extraction_advanced import ASCExtractorAdvanced

extractor = ASCExtractorAdvanced(max_scatterers=30, adaptive_threshold=0.01)
scatterers = extractor.extract_asc_scatterers(complex_data)

# 查看完整ASC参数
for sc in scatterers[:3]:  # 显示前3个
    print(f"位置: ({sc['x']:.3f}, {sc['y']:.3f})")
    print(f"散射类型: {sc['scattering_type']} (α={sc['alpha']})")
    print(f"长度: {sc['length']:.3f}, 相位: {sc['phi_bar']:.3f}")
    print("---")
```

---

## 📁 输出结果说明

### **可视化输出**

- **原始SAR图像**：输入的MSTAR数据显示
- **重构图像**：基于提取散射中心的重构结果
- **散射中心分布**：散射中心在空间中的位置分布
- **参数统计**：幅度、相位、散射类型等统计信息
- **性能指标**：PSNR、处理时间、精度评估

### **数据输出**

- **JSON报告**：详细的性能指标和参数统计
- **可视化图片**：PNG格式的分析图表
- **控制台信息**：实时的处理进度和关键结果

---

## 🔧 故障排除

### **常见问题**

**Q1: 提示找不到数据文件**

```bash
解决：确认数据路径正确
ls datasets/SAR_ASC_Project/02_Data_Processed_raw/SN_S7/
```

**Q2: 处理时间过长**

```bash
解决：使用更快的模式
# 精度优化算法使用fast模式
python test_mstar_quick_precision.py  # 自动使用balanced模式
```

**Q3: 内存不足**

```bash
解决：降低精度模式或使用传统OMP
python demo_asc_advanced.py  # 使用内存较少的算法
```

### **性能优化建议**

1. **首次使用**：先运行demo了解功能，再选择合适算法
2. **日常处理**：推荐传统OMP算法，稳定快速
3. **科研分析**：推荐ASC高级算法，参数完整
4. **高精度需求**：推荐精度优化算法的balanced模式
5. **极限精度**：精度优化算法的ultra模式，需要充足时间和内存

---

## 📖 参考文档

- **技术详情**：`doc/project_problem.md` - 算法设计思路演进
- **开发历程**：`project-status.md` - 完整开发日志
- **数据格式**：`dataProcess/` - MATLAB预处理脚本

---

**项目状态**：✅ 生产就绪，七套程序完整，支持从快速处理到科研级精度的全方位需求

**快速开始建议**：新用户运行 `python demo_asc_advanced.py`，熟悉用户运行 `python test_asc_advanced_comparison.py`
