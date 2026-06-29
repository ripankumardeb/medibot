from app import rag_chain
import traceback
try:
    print(rag_chain.invoke('Hello'))
except Exception as e:
    traceback.print_exc()
