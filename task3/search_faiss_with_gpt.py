import pandas as pd
import numpy as np
import faiss
import os
from openai import OpenAI
import pymysql
from flask import Flask, request, jsonify
import json
from flask_cors import CORS
# 初始化 Flask 应用
app = Flask(__name__)
CORS(app)
app.config['JSON_AS_ASCII'] = False

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
def generate_response(query,history,context):
    context_text = "\n".join([f"{item['id']}: {item['text']}" for item in context])
    prompt = f"历史上下文 {history}你现在是一只奶牛猫，也是一个鞋子售后客服,请只回答用户提问相关的知识，用户查询: {query}\n数据库查找的内容:\n{context_text}\n\n请生成详细的答案"
    print(prompt)
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


# HTTP 接口定义
@app.route("/chat", methods=["POST"])
def chat():
    """
    处理聊天请求。
    接收参数：
    - message: 用户输入
    - historyMessages: 历史对话内容
    """
    data = request.json
    user_input = data.get("message", "")
    history = data.get("historyMessages", [])

    if not user_input:
        return jsonify({"error": "message 参数不能为空"}), 400

    # 检索上下文
    context = search_by_title(user_input, top_k=3)
    if not context:
        return jsonify({"error": "未找到相关上下文"}), 404

    # 生成回答
    response = generate_response(user_input, history, context)
    return jsonify({"response": response})

# 启动服务
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
