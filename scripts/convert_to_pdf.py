#!/usr/bin/env python3
"""
Convert Markdown documentation to PDF with professional formatting
"""

import markdown2
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration
import sys
import os

def convert_md_to_pdf(md_file_path, pdf_file_path):
    """Convert a markdown file to a well-formatted PDF"""

    # Read the markdown file
    with open(md_file_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Convert markdown to HTML with extensions
    html_content = markdown2.markdown(
        md_content,
        extras=['fenced-code-blocks', 'tables', 'strike', 'task_list', 'header-ids']
    )

    # CSS for professional formatting
    css_content = """
    @page {
        size: A4;
        margin: 2cm;
        @bottom-center {
            content: counter(page) " / " counter(pages);
            font-size: 10pt;
            color: #666;
        }
    }

    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                     'Helvetica Neue', Arial, sans-serif;
        font-size: 11pt;
        line-height: 1.6;
        color: #333;
        max-width: 100%;
    }

    h1 {
        font-size: 24pt;
        font-weight: 700;
        color: #0078D4;
        border-bottom: 3px solid #0078D4;
        padding-bottom: 10px;
        margin-top: 0;
        margin-bottom: 20px;
        page-break-after: avoid;
    }

    h2 {
        font-size: 18pt;
        font-weight: 600;
        color: #0063B1;
        margin-top: 30px;
        margin-bottom: 15px;
        page-break-after: avoid;
    }

    h3 {
        font-size: 14pt;
        font-weight: 600;
        color: #333;
        margin-top: 20px;
        margin-bottom: 10px;
        page-break-after: avoid;
    }

    h4 {
        font-size: 12pt;
        font-weight: 600;
        color: #555;
        margin-top: 15px;
        margin-bottom: 8px;
    }

    p {
        margin-bottom: 10px;
        text-align: justify;
    }

    code {
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
        font-size: 9pt;
        background-color: #f4f4f4;
        padding: 2px 4px;
        border-radius: 3px;
        color: #d73a49;
    }

    pre {
        background-color: #f6f8fa;
        border: 1px solid #e1e4e8;
        border-radius: 6px;
        padding: 12px;
        overflow-x: auto;
        font-size: 9pt;
        line-height: 1.4;
        page-break-inside: avoid;
    }

    pre code {
        background-color: transparent;
        padding: 0;
        color: #24292e;
    }

    ul, ol {
        margin-left: 20px;
        margin-bottom: 10px;
    }

    li {
        margin-bottom: 5px;
    }

    strong {
        font-weight: 600;
        color: #222;
    }

    em {
        font-style: italic;
    }

    blockquote {
        border-left: 4px solid #0078D4;
        padding-left: 15px;
        margin-left: 0;
        color: #555;
        font-style: italic;
    }

    table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        page-break-inside: avoid;
    }

    table th {
        background-color: #0078D4;
        color: white;
        padding: 10px;
        text-align: left;
        font-weight: 600;
    }

    table td {
        padding: 8px;
        border-bottom: 1px solid #ddd;
    }

    table tr:nth-child(even) {
        background-color: #f9f9f9;
    }

    hr {
        border: none;
        border-top: 2px solid #e1e4e8;
        margin: 30px 0;
    }

    a {
        color: #0078D4;
        text-decoration: none;
    }

    a:hover {
        text-decoration: underline;
    }

    /* Special styling for Azure/Cloud elements */
    .azure-note {
        background-color: #e7f3ff;
        border-left: 4px solid #0078D4;
        padding: 10px;
        margin: 15px 0;
    }

    /* Emoji support */
    .emoji {
        font-size: 1.2em;
    }
    """

    # Wrap HTML with proper structure
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Azure Deployment Architecture</title>
        <style>
            {css_content}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    # Configure fonts
    font_config = FontConfiguration()

    # Convert HTML to PDF
    HTML(string=full_html).write_pdf(
        pdf_file_path,
        font_config=font_config
    )

    print(f"✅ PDF created successfully: {pdf_file_path}")

if __name__ == "__main__":
    # Define paths
    md_path = "docs/azure/AZURE_DEPLOYMENT_EXPLAINED.md"
    pdf_path = "docs/azure/AZURE_DEPLOYMENT_EXPLAINED.pdf"

    # Check if markdown file exists
    if not os.path.exists(md_path):
        print(f"❌ Error: Markdown file not found: {md_path}")
        sys.exit(1)

    # Create PDF
    try:
        convert_md_to_pdf(md_path, pdf_path)
        print(f"📄 PDF size: {os.path.getsize(pdf_path) / 1024:.1f} KB")
    except Exception as e:
        print(f"❌ Error creating PDF: {e}")
        sys.exit(1)