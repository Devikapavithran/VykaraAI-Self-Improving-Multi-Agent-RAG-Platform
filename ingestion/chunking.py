from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=[
            "\nITEM 1A.", "\nItem 1A.", "\nRISK FACTORS",
            "\nITEM 1.", "\nItem 1.",
            "\nITEM 7.", "\nItem 7.",
            "\n\n"
        ]
    )
    return splitter.split_documents(docs)
