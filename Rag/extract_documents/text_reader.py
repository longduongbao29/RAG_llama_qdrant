from langchain_core.documents import Document
import fitz
from logs.loging import logger
import re
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate

MAX_CHUNK_SIZE = 1000

prompt = ChatPromptTemplate.from_template(
    """From input document, find the topic of document. Topic return needs to be specified, short.
                                          Document: 
                                          {document}
                                          Output (topic 1-5 word):
                                          """
)


class TextReader:
    doc_name: str = None
    file_path: str = None
    text: str = ""
    blocks: list = []

    def __init__(self, file_path, doc_name):
        self.doc_name = doc_name
        self.file_path = file_path

    def get_topics(self, llm):
        chain = prompt | llm | StrOutputParser()
        topic = chain.invoke({"document": self.text})
        return topic

    def readpdf(self):
        """
        Reads a PDF file and extracts text blocks from it.

        This function opens a PDF file specified by the file_path attribute,
        extracts text blocks from each page, and filters out blocks that contain
        URLs or web addresses. The filtered text blocks are then stored in the
        blocks attribute.

        Parameters:
        None

        Returns:
        None
        """
        doc = fitz.open(self.file_path)
        blocks = []
        for page in doc:
            output = page.get_text("blocks")
            for block in output:
                if block[6] == 0:
                    # We only take the text
                    text = block[4]
                    if (
                        "http://" not in text
                        and "www" not in text
                        and ".com" not in text
                    ):
                        blocks.append(block)
        self.blocks = blocks
        text_blocks = [block[4] for block in blocks]
        self.text = "\n".join(text_blocks)

    def create_documents(self):
        """Split and create documents from text"""
        documents = []
        chunks = []
        if "txt" in self.file_path:
            chunks = self.split_txt_by_paragraphs()
        else:
            chunks = self.split_pdf_by_paragraphs()
        for chunk in chunks:
            document = Document(
                page_content=chunk, metadata={"source_document": self.doc_name}
            )
            documents.append(document)
        return documents

    def split_txt_by_paragraphs(self):
        """Split text by paragraphs by .\n or \n\n

        Args:
            text (str): text to split

        Returns:
            list: chunks
        """
        paragraphs = re.split(r"\n", self.text)
        chunks = []
        chunk = ""
        for paragraph in paragraphs:
            if (
                len(chunk) + len(paragraph) > MAX_CHUNK_SIZE
            ):  # Define your max chunk size
                chunks.append(chunk)
                chunk = paragraph
            else:
                chunk += "\n\n" + paragraph
        if chunk:
            chunks.append(chunk)
        return chunks

    def split_pdf_by_paragraphs(self):
        """
        Splits the self. blocks extracted from a PDF into paragraphs.

        This function processes the text blocks stored in the `blocks` attribute,
        joining blocks that are part of the same paragraph and splitting them into
        chunks. It ensures that each chunk has a minimum number of tokens (words).

        Parameters:
        None

        Returns:
        list: A list of text chunks, each representing a paragraph or a joined set of blocks
              from the PDF, with a minimum number of tokens.
        """
        chunks = []
        prev_block = self.blocks[0]
        is_join = False
        for current_block in self.blocks[1:]:
            if is_join:
                prev_block = current_block
                is_join = False
                continue
            if prev_block[0] > current_block[0]:
                text = " ".join([prev_block[4], current_block[4]]).replace("\n", " ")
                chunks.append(text)
                is_join = True
            else:
                chunks.append(prev_block[4].replace("\n", " "))
            prev_block = current_block
        return_ = []
        for chunk in chunks:
            num_token = chunk.count(" ") + 1
            if num_token > 10:
                return_.append(chunk)
        return return_
