import os
import subprocess
from PyPDF2 import PdfWriter, PdfReader
from pypdf import PdfReader, PdfWriter
import random
import string


def optimize_pdfs(pdf_path: str):
    """
    Optimize a PDF file by compressing its content streams.

    Args:
        pdf_path (str): The path to the PDF file to be optimized.

    Returns:
        None

    Raises:
        Exception: If an error occurs during the PDF optimization process.
    """

    try:
        reader = PdfReader(pdf_path)
        writer = PdfWriter()
        for page in reader.pages:
            page.compress_content_streams()  # This compresses PDF content streams
            writer.add_page(page)
        with open(pdf_path, 'wb') as f:
            writer.write(f)
    except Exception as e:
        print(f"An error occurred during PDF optimization: {str(e)}")


def convert_words_to_pdfs(paths: list):
    """
    Convert Word documents to PDF format using LibreOffice.

    Args:
        paths (list): A list of file paths for the Word documents to be converted.

    Returns:
        list: A list of file paths for the generated PDF files, or None if an error occurs.

    Raises:
        FileNotFoundError: If an input file is not found.
        Exception: If the PDF conversion fails.
    """

    ouput_files_path = []
    for filename in paths:
        try:
            # Check if the input file exists
            if not os.path.exists(filename):
                raise FileNotFoundError(f"Input file not found: {filename}")
            
            # If output_path is not provided, create it based on the input_path
            output_path = os.path.splitext(filename)[0] + ".pdf"
            subprocess.call(['libreoffice', '--convert-to', 'pdf', filename, '--outdir', os.path.dirname(output_path)])
            
            # Verify the PDF was created successfully
            if not os.path.exists(output_path):
                raise Exception("PDF conversion failed")
            
            # Optimize the PDF (reduce file size)
            optimize_pdfs(output_path)
            print(f"Conversion successful. PDF saved at: {output_path}")
            ouput_files_path.append(output_path)
        except Exception as e:
            print(f"An error occurred during conversion: {str(e)}")
            return None
    return ouput_files_path

def add_encryption(files:list, length:int=16) -> str:
    """
    Add encryption to a list of PDF files using a randomly generated password.

    Args:
        files (list): A list of file paths for the PDF files to be encrypted.
        length (int, optional): The length of the randomly generated password. Defaults to 16.

    Returns:
        str: The randomly generated password used for encryption.
    """

    # Create possible pool of characters
    character_pool = ""
    character_pool += string.ascii_uppercase
    character_pool += string.ascii_lowercase
    character_pool += string.digits
    character_pool += string.punctuation

    # Generate the password
    password = ''.join(random.choice(character_pool) for _ in range(length))
    for file_path in files:
        reader = PdfReader(file_path)
        writer = PdfWriter()
        writer.append_pages_from_reader(reader)
        writer.encrypt(password)
        with open(file_path, "wb") as out_file:
            writer.write(out_file)
    return password

# Example usage
if __name__ == "__main__":
    filename = "test.docx"
    pdf_file = convert_words_to_pdfs([filename])
    # print(add_encryption(pdf_file, 8))
    # if pdf_file:
    #     print(f"Converted file: {pdf_file}")