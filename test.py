DATA_PATH = 'data-pdf\optimizers.pdf'
CHROMA_PATH = 'chroma'
# C:\Users\Shree123\rag-demo\data-pdf\
from load_and_split import load_content, split_content
raw_chunks = load_content(DATA_PATH)

chunks = split_content(raw_chunks)

from vector_store import chroma_vector_store
chroma_db = chroma_vector_store(CHROMA_PATH)

from uuid import uuid4
uuids = [str(uuid4()) for _ in range(len(chunks))]
print(f'Number of Chunks : {len(chunks)}')


chroma_db.add_documents(documents = chunks , ids = uuids)


results = chroma_db.similarity_search(
    "Gradient Descent",
    k=2,
    )
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")