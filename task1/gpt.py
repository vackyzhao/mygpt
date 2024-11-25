import os
import openai

# 创建 OpenAI 客户端，使用环境变量中的 API 密钥
client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# 初始的对话消息，仅设置系统消息，不进行回复
messages = [
    {"role": "system", "content": "You are a helpful assistant. If the user wishes to end the conversation, respond with ENDCHAT."},
]

# 不进行初始回复，直接等待用户输入
print("System is ready. You can start the conversation.")

# 获取用户输入并继续对话
user_input = input("Your message: ")
while user_input.lower() != "exit":
    # 将用户消息添加到对话消息列表
    messages.append({"role": "user", "content": user_input})

    # 发送请求并启用流式响应
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True  # 启用流式响应
    )
    
    # 用于保存响应的完整内容
    full_response = ""

    # 获取并打印响应
    for chunk in response:
        chunk_message = chunk.choices[0].delta
        content = chunk_message.content if chunk_message.content else ""
        print(content, end="")
        
        # 拼接响应内容
        full_response += content

        # 检查是否收到 "ENDCHAT" 并结束对话
        if "ENDCHAT" in full_response:
            print("\nConversation ended.")
            user_input = "exit"  # 退出循环
            break

    print()  # 打印换行

    # 获取用户输入并继续对话
    if user_input != "exit":
        user_input = input("Your message: ")

print("Ending conversation.")
