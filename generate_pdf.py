"""Generate a PDF containing all source code from the Smart Tomato Helper project."""
import os
from fpdf import FPDF

# Source code files to include (ordered logically)
SOURCE_FILES = [
    "app.py",
    "data_manager.py",
    "disease_hybrid.py",
    "disease_model.py",
    "yield_hybrid.py",
    "yield_model.py",
    "soil_hybrid.py",
    "fertilizer_hybrid.py",
    "fertilizer_logic.py",
    "model_trainer.py",
    "train_all.py",
    "ocr_utils.py",
    "weather.py",
    "requirements.txt",
    "Dockerfile",
    "docker-compose.yml",
]

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PDF = os.path.join(PROJECT_DIR, "smart_tomato_helper_source_code.pdf")


class SourceCodePDF(FPDF):
    def header(self):
        self.set_font("Courier", "B", 10)
        self.set_text_color(40, 40, 40)
        self.cell(0, 8, "Smart Tomato Helper - Source Code", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(0, 128, 0)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Courier", "I", 8)
        self.set_text_color(128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def add_file_section(self, filename, content):
        # File title
        self.set_font("Courier", "B", 11)
        self.set_fill_color(230, 245, 230)
        self.set_text_color(0, 80, 0)
        self.cell(0, 8, f"  {filename}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

        # Separator
        self.set_draw_color(180, 210, 180)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

        # Code content
        self.set_font("Courier", "", 7)
        self.set_text_color(30, 30, 30)

        lines = content.split("\n")
        line_num_width = len(str(len(lines)))

        for i, line in enumerate(lines, 1):
            # Line number
            line_num = str(i).rjust(line_num_width)

            # Handle long lines by truncating at page width
            # Replace tabs with spaces
            line = line.replace("\t", "    ")

            # Encode safely for PDF
            safe_line = line.encode("latin-1", errors="replace").decode("latin-1")

            text = f"{line_num} | {safe_line}"

            # Check if we need a new page
            if self.get_y() > 270:
                self.add_page()

            self.cell(0, 3.5, text, new_x="LMARGIN", new_y="NEXT")

        self.ln(6)


def main():
    pdf = SourceCodePDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.alias_nb_pages()

    # Title page
    pdf.add_page()
    pdf.ln(60)
    pdf.set_font("Courier", "B", 22)
    pdf.set_text_color(0, 100, 0)
    pdf.cell(0, 12, "Smart Tomato Helper", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Courier", "", 14)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 10, "Complete Source Code", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("Courier", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, "AI-Powered Tomato Crop Management System", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)

    # Table of Contents
    pdf.set_font("Courier", "B", 12)
    pdf.set_text_color(0, 80, 0)
    pdf.cell(0, 8, "Files Included:", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Courier", "", 9)
    pdf.set_text_color(60, 60, 60)
    for idx, f in enumerate(SOURCE_FILES, 1):
        pdf.cell(0, 5, f"  {idx:2d}. {f}", align="C", new_x="LMARGIN", new_y="NEXT")

    # Add each source file
    for filename in SOURCE_FILES:
        filepath = os.path.join(PROJECT_DIR, filename)
        if not os.path.exists(filepath):
            print(f"  [SKIP] {filename} not found")
            continue

        print(f"  [ADD]  {filename}")
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        pdf.add_page()
        pdf.add_file_section(filename, content)

    pdf.output(OUTPUT_PDF)
    print(f"\nPDF generated: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
