from src.retriever import retrieve
from src.generator import generate_answer

def chat():
    print("🤖 Chatbot đã sẵn sàng. Gõ 'exit' để thoát.\n")
    while True:
        query = input("Bạn: ")
        if query.lower() == "exit":
            break

        contexts = retrieve(query)
        answer = generate_answer(contexts, query)
        print(f"Bot: {answer}\n")

if __name__ == "__main__":
    chat()
