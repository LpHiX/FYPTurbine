import pypdf
reader = pypdf.PdfReader(r'C:\Users\Martin\Zotero\storage\4AQPRHAV\Colclough - 1966 - Design of Turbine Blades Suitable for Supersonic Relative Inlet Velocities and the Investigation of.pdf')
text = ""
for i, page in enumerate(reader.pages):
    text += f"--- PAGE {i} ---\n" + page.extract_text() + "\n"
    if i > 25: break
with open(r'C:\Users\Martin\Active\FYPTurbine\temp_pdf_read.txt', 'w', encoding='utf-8') as f:
    f.write(text)
