from docx import Document

# read the .py file
with open("Lab1_basic.py", "r") as f:
    code = f.read()

# create a Word document
doc = Document()
doc.add_paragraph(code, style="Normal")  # keeps indentation
doc.save("Lab1_basic.docx")