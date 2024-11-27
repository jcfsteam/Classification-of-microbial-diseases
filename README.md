
# 微生物疾病分类项目

本项目旨在开发一个基于机器学习的模型，用于对微生物疾病进行分类。通过分析微生物数据，模型能够准确地识别和分类不同类型的微生物疾病。

## 项目结构

- `catboost_info/`：存储CatBoost模型的训练信息和日志。
- `config/`：包含项目的配置文件。
- `data/`：用于存放原始数据和预处理后的数据集。
- `outputs/`：保存模型的输出结果，如预测结果和评估指标。
- `src/`：项目的源代码，包括数据处理、模型训练和评估等模块。
- `.gitattributes`：Git属性配置文件。
- `.gitignore`：指定需要Git忽略的文件和目录。
- `main.py`：项目的主程序入口，负责协调各模块的运行。
- `requirements.txt`：列出项目所需的Python依赖库。

## 安装指南

1. **克隆仓库**：

   ```bash
   git clone https://github.com/jcfsteam/Classification-of-microbial-diseases.git
   ```

2. **进入项目目录**：

   ```bash
   cd Classification-of-microbial-diseases
   ```

3. **创建虚拟环境（可选）**：

   ```bash
   python -m venv venv
   source venv/bin/activate  # 对于Windows用户，使用 `venv\Scripts\activate`
   ```

4. **安装依赖**：

   ```bash
   pip install -r requirements.txt
   ```

## 使用说明

1. **数据准备**：将您的数据集放置于`data/`目录下。确保数据格式符合项目要求。

2. **配置修改**：根据您的需求，修改`config/`目录下的配置文件，以适应不同的训练参数和模型设置。

3. **运行主程序**：

   ```bash
   python main.py
   ```

   主程序将按照配置文件中的设置，执行数据预处理、模型训练和评估等步骤。

## 贡献指南

欢迎对本项目进行贡献！如果您有任何建议或发现了问题，请提交Issue或Pull Request。

## 许可证

本项目采用MIT许可证。有关详细信息，请参阅LICENSE文件。

## 联系方式

如有任何疑问或需要进一步的信息，请联系项目维护者：[您的邮箱地址]。 