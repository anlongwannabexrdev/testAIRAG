import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "moozi_llama-3.2-3B-Instruct",
    max_seq_length = 2048,
    dtype=None,
    device_map="auto",
)

def ask(question : str):
    prompt() = f"""
    You are an exeprt in answering questions about a moozi game based on the knowledge provided.

    Here are some relevant reviews: {reviews}

    Here is the question to answer: {question}
    """
    inputs = tokenizer(prompt, return_tensors = "pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=256)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Xin lỗi" in answer or not answer.strip():
        return "Không tìm thấy trong dữ liệu."
    return answer

if __name__ == "__main__":
    while True:
        question = input("Nhập prompt ('exit' để thoát): ")
        if question.lower() == "exit":
            break

        answer = ask(question)
        print("💡 Answer:", answer)