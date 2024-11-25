import pandas as pd
import numpy as np
import faiss
import os
from openai import OpenAI
import pymysql

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# MySQL 数据库配置
db_config = {
    "host": "172.17.0.4",
    "port": 3306,
    "user": "root",
    "password": "1234asdf",
    "database": "my_database"
}


# 索引和元数据文件路径
index_file = "faiss_index.bin"
metadata_file = "metadata.csv"

# 加载 FAISS 索引
index = faiss.read_index(index_file)
print(f"已加载 FAISS 索引 {index_file}，向量维度：{index.d}")

# 加载元数据
metadata = pd.read_csv(metadata_file)

# 函数：调用 LLM 生成响应
def generate_response(query, context):
    prompt = f"用户查询: {query}\n上下文:\n{context}\n\n请生成详细的答案:"
    response = client.chat.completions.create(model="gpt-4-turbo",
    messages=[
        {"role": "system", "content": "你是一个智能销售客服，负责回答用户的问题。"},
        {"role": "user", "content": prompt}
    ])
    return response.choices[0].message.content


# 从 MySQL 加载元数据
def load_metadata_from_db():
    """从 MySQL 数据库加载元数据"""
    connection = pymysql.connect(**db_config)
    try:
        with connection.cursor(pymysql.cursors.DictCursor) as cursor:
            query = "SELECT id, title FROM ai_context"  # 查询 ID 和标题
            cursor.execute(query)
            metadata = cursor.fetchall()
            metadata_array = np.array(metadata)  # 转为数组，便于索引
            return metadata
    finally:
        connection.close()

# 获取 GPT 嵌入向量
def get_query_embedding(query, model="text-embedding-3-small"):
    """通过 OpenAI API 获取查询文本的嵌入向量"""
    try:
        response = client.embeddings.create(model=model, input=query)
        return np.array(response.data[0].embedding).astype("float32")
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

# 从 MySQL 检索上下文
def get_context_from_db(indices):
    """从数据库检索上下文记录"""
    connection = pymysql.connect(**db_config)
    try:
        with connection.cursor(pymysql.cursors.DictCursor) as cursor:  # 使用字典游标
            query = f"SELECT * FROM ai_context WHERE id IN ({','.join(map(str, indices))})"
            cursor.execute(query)
            results = cursor.fetchall()
            return results
    finally:
        connection.close()

# 查询函数
def search_by_title(title, top_k=5):
    """通过标题查询最相关的上下文"""
    # 获取标题的嵌入向量
    query_vector = get_query_embedding(title)
    if query_vector is None:
        print("无法生成查询嵌入向量，检索失败。")
        return None

     # 检索最近邻
    query_vector = query_vector.reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)
    indices = [i + 1 for i in indices[0]]  # 调整索引偏移以匹配数据库 ID
    print(f"检索到的索引（修正后）：{indices}")

    # 通过索引检索上下文
    results = get_context_from_db(indices)
    return results

# 示例查询
if __name__ == "__main__":
    title = input("请输入标题：")
    results = search_by_title(title, top_k=3)
    if results:
        print("\n检索结果：")
        for result in results:
            print(result)
    else:
        print("未找到相关记录。")
        
    # 4. 生成最终响应
    response = generate_response(title, results)
    print("\nAI 响应：")
    print(response)


