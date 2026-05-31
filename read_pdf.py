import pypdf
reader = pypdf.PdfReader(r'C:\Users\Martin\Zotero\storage\99U6HX66\Wlodarski et al. - 1952 - Application of Supersonic Vortex-flow Theory to the Design of Supersonic Impulse Compressor- or Turb.pdf')
text = ""
for i, page in enumerate(reader.pages):
    text += f"--- PAGE {i} ---\n" + page.extract_text() + "\n"
    if i > 25: break
with open(r'C:\Users\Martin\Active\FYPTurbine\temp_pdf_read.txt', 'w', encoding='utf-8') as f:
    f.write(text)
