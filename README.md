# EasyRAG

## 1. Docker运行

```bash
chmod +x run.sh
./run.sh
```

## 2. 直接运行

```bash
pip install -r requirements.txt
cd src

# 运行测试题目
python3 main.py 

# 复制答案文件
cp submit_result.jsonl ../answer.jsonl
```

## 3. 代码结构

仅对复赛可能用到的代码进行讲解

```yaml
- src
	- custom
        - splitter.py # 自定义块分割
        - hierarchical.py # 分层分块
        - transformation.py # 文件路径和标题抽取
        - embeddings # 为GTE单独实现一个embedding类，方便自定义使用
        	- ...
        - retrievers.py # 实现基于qdrant的密集检索器，中文BM25检索器，实现了rrf和简单合并的融合检索器
		- rerankers.py # 单独为bge系列的reranker实现一些类，方便自定义使用
        - template.py # QA提示词模板
	- pipeline
		- ingestion.py # 数据处理流程：数据读入，元数据提取，文档分块，编码，元数据过滤器，向量数据库建立
		- pipeline.py # EasyRAG流程类，包含对各种数据和模型的初始化、具体的RAG流程定义
		- rag.py # rag的一些工具函数
        - qa.py # 读入问题文件，保存答案
	- utils # 适配hf的custom llm在国内使用，直接由对应模型的hf链接中的代码复制而来
		- ...
    - configs
		- easyrag.yaml # 配置文件
	- data
		- nltk_data # nltk中的停用词表和分词器数据
		- hit_stopwords.txt # hit中文停用词表
		- imgmap_filtered.json # 由get_ocr_data.py处理而来
		- question.jsonl # 复赛测试集
	- main.py # 主函数，入口文件
	- preprocess_zedx.py # zedx数据预处理
	- get_ocr_data.py # paddleocr+glm4v抽取图像内容
	- submit.py # 初赛提交结果
- requirements.txt # python依赖
```

## 4. 可能的问题及解答

### python:3.10.14-slim镜像如何获取？

如果遇到从Docker Hub拉取镜像失败，则可以采取更改docker config代理设置的方法重新拉取镜像或者使用本地安装的方式安装python:3.10.14-slim的linux/amd64版本镜像，其中的 `python3-10-14.tar` 文件在同目录下提供。

```bash
docker load -i python3-10-14.tar
```

### 使用的魔搭模型链接为什么是自己上传的，而不是官方的？

1. 我们前期一直使用hf下载到本地的版本，最后需要提交时考虑到国内评测不方便才转换到modelscope。我们**自己上传的模型没有经过任何微调**，**直接由hf最新版本下载而来，并上传到魔搭**，直接使用hf官方的模型效果是一致的。（注：buaadreamer是队长的魔搭用户名）
2. bge-reranker-v2-minicpm-layerwise官方并没有在魔搭上传模型
3. 综上，我们使用了自己上传的魔搭模型链接，用于复现时模型下载
