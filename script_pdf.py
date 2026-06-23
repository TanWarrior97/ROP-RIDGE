from markdown_pdf import MarkdownPdf, Section
import os
import base64
import re

def img_to_base64(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: Image {filepath} not found for PDF embed.")
        return ""
    with open(filepath, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode('utf-8')
    # Use image/png based on extension
    return f"data:image/png;base64,{encoded}"

def replace_images_with_base64(markdown_text):
    # Regex to find ![Alt Text](filename.png)
    # We will replace filename.png with its base64 equivalent
    pattern = r'!\[([^\]]*)\]\(([^)]+\.(?:png|jpg|jpeg))\)'
    
    def replacer(match):
        alt_text = match.group(1)
        filepath = match.group(2)
        b64_data = img_to_base64(filepath)
        if b64_data:
            return f'![{alt_text}]({b64_data})'
        return match.group(0) # Unchanged if not found
        
    return re.sub(pattern, replacer, markdown_text)

def create_pdf():
    pdf = MarkdownPdf(toc_level=2)
    pdf_path = "Final_Technical_Report.pdf"
    
    with open("Final_Technical_Report.md", "r", encoding="utf-8") as f:
        md_text = f.read()

    # Safely embed images into the markdown string
    md_text_embedded = replace_images_with_base64(md_text)

    pdf.add_section(Section(md_text_embedded, toc=False))
    pdf.meta["title"] = "AI ROP Screening Pipeline Architecture"
    pdf.meta["author"] = "Soham Gujar"
    
    pdf.save(pdf_path)
    print(f"Generated Base64-Embedded PDF at: {pdf_path}")

if __name__ == "__main__":
    create_pdf()
