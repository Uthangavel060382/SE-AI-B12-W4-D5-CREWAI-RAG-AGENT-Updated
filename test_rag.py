
from rag_tool import CustomerSupportRAGTool

# Initialize custom CrewAI RAG tool
rag_tool = CustomerSupportRAGTool()

# Customer questions
questions = [
    "What is the return policy?",
    "How many days does a refund take?",
    "Does the warranty cover accidental damage?",
    "How long does express shipping take?"
]

# Test retrieval
for question in questions:
    print("\n" + "=" * 60)
    print("Customer Question:", question)

    result = rag_tool.run(query=question)

    print("\nRetrieved Knowledge:")
    print(result)
    print("=" * 60)
