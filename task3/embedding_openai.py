import os
from openai import OpenAI
import pandas as pd

# 初始化 OpenAI 客户端
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API Key is missing. Please set it in your environment variables.")

client = OpenAI(api_key=api_key)

# 文件路径
file_path = "运动鞋店铺知识库.txt"

# 读取知识库文件
data = pd.read_csv(file_path, sep=":", header=None, names=["category", "content"])

# 调用 OpenAI Embedding API 的函数
def get_embeddings(texts, model="text-embedding-3-small"):
    try:
        response = client.embeddings.create(model=model, input=texts)
        
        # 检查返回值是否符合预期
        embeddings = []
        for item in response.data:
            if hasattr(item, "embedding"):
                embeddings.append(item.embedding)
            else:
                raise ValueError(f"Unexpected response item: {item}")
        return embeddings

    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return []

# 分批处理
batch_size = 10
embeddings = []

for i in range(0, len(data), batch_size):
    batch = data["content"].iloc[i:i+batch_size].tolist()
    batch_embeddings = get_embeddings(batch)
    if batch_embeddings:
        embeddings.extend(batch_embeddings)
    else:
        embeddings.extend([[0] * 1536] * len(batch))  # 填充空向量

# 检查长度是否匹配
if len(embeddings) != len(data):
    raise ValueError(
        f"Length mismatch: embeddings ({len(embeddings)}) != data ({len(data)})"
    )

# 将嵌入向量存储到 Pandas DataFrame
data["embedding"] = pd.Series(embeddings, index=data.index)

# 保存到 CSV 文件
output_file = "embedded_knowledge_base.csv"
data.to_csv(output_file, index=False)
print(f"嵌入向量已生成并保存到 {output_file}")
