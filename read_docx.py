import sys
import os
import zipfile
import re

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

def get_docx_text(path):
    try:
        with zipfile.ZipFile(path) as document:
            xml_content = document.read('word/document.xml')
            
        xml_str = xml_content.decode('utf-8')
        
        # Remove XML tags
        text = re.sub('<[^<]+?>', '', xml_str)
        return text
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    file_path = "c:/Users/user/Desktop/HMEAYC_Project/護理與幼保類群--朝陽科技大學--解碼教室裡的舞蹈：AI 如何看懂孩子的肢體學習語言 (1).docx"
    output_file = "paper.txt"
    if os.path.exists(file_path):
        text = get_docx_text(file_path)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Content written to {output_file}")
    else:
        print(f"File not found: {file_path}")
