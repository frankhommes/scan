import os
import argparse
import subprocess
from pathlib import Path
from PIL import Image, ImageEnhance
from pypdf import PdfWriter, PdfReader
import openai
from llama_cpp import Llama

def generate_summary_local(text):
    """Generiert eine kurze Zusammenfassung aus OCR-Text mit einem lokalen LLM"""
    if not text.strip():
        return "Keine Zusammenfassung verfügbar (leerer Text)."

    # Pfad zum Llama-Modell (z. B. "ggml-model-q4_0.bin")
    model_path = "/path/to/your/model.gguf"

    llm = Llama(model_path=model_path)
    prompt = f"Erstelle eine professionelle Zusammenfassung des folgenden gescannten Textes:\n{text}\nZusammenfassung:"
    
    response = llm(prompt, max_tokens=100, stop=["\n"])
    return response["choices"][0]["text"].strip()


def generate_summary_cloud(text):
    """Generiert eine Zusammenfassung über die OpenAI API"""
    if not text.strip():
        return "Keine Zusammenfassung verfügbar (leerer Text)."

    openai.api_key = "DEIN_OPENAI_API_KEY"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "Erstelle eine professionelle Zusammenfassung des folgenden Textes."},
                  {"role": "user", "content": text}],
        max_tokens=100
    )

    return response["choices"][0]["message"]["content"].strip()


class DocumentConverter:
    def __init__(self, input_files, enhance=False, use_cloud_llm=False, title="Gescanntes Dokument", author="Automatische PDF-Konvertierung", subject="Optimiertes PDF mit OCR", producer="Python Script mit pypdf"):
        """Initialisiert den Dokumentenkonverter mit den Eingabedateien und optionalen Metadaten"""
        self.input_files = [Path(f) for f in input_files]
        self.enhance = enhance
        self.use_cloud_llm = use_cloud_llm
        self.temp_dir = Path("temp_pdfs")
        self.temp_dir.mkdir(exist_ok=True)
        self.pdf_files = []
        self.merged_pdf = self.temp_dir / "merged.pdf"
        self.compressed_pdf = self.temp_dir / "compressed.pdf"
        self.output_pdf = "output.pdf"

        # 🆕 LLM wird automatisch ausgewählt
        self.generate_summary = generate_summary_cloud if use_cloud_llm else generate_summary_local

        # Standard-Metadaten
        self.metadata = {
            "/Title": title,
            "/Author": author,
            "/Subject": subject,
            "/Producer": producer
        }

    def apply_ocr(self):
        """Führt OCR mit `ocrmypdf` durch und extrahiert danach den Text für die automatische Zusammenfassung"""
        print("🔠 Führe OCR durch...")
        ocr_command = [
            "ocrmypdf", "-l", "deu+eng", "--optimize", "1",
            "--output-type", "pdfa", "--jbig2-lossy",
            str(self.compressed_pdf), self.output_pdf
        ]
        subprocess.run(ocr_command, check=True)

        # 🆕 OCR-Text extrahieren
        extracted_text = extract_text_from_pdf(self.output_pdf)

        # 🆕 LLM-basierte Zusammenfassung generieren
        self.metadata["/Subject"] = self.generate_summary(extracted_text)
        print(f"📄 Automatische Zusammenfassung erstellt: {self.metadata['/Subject']}")


    def enhance_image(self, image):
        """Verbessert die Bildqualität: Kontrast, Helligkeit, Schärfe, Gamma"""
        print("🔧 Wende Bildverbesserung an...")
        gamma = 0.8
        image = image.point(lambda x: 255 * (x / 255) ** gamma)
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)

        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.1)

        sharpness = ImageEnhance.Sharpness(image)
        image = sharpness.enhance(2.0)

        return image

    def convert_tiff_to_pdf(self):
        """Konvertiert TIFF-Dateien in PDFs mit optionaler Bildverbesserung"""
        for tiff_path in self.input_files:
            pdf_path = self.temp_dir / f"{tiff_path.stem}.pdf"
            print(f"📄 Konvertiere {tiff_path} → {pdf_path}")

            with Image.open(tiff_path) as img:
                img = img.convert("RGB")
                if self.enhance:
                    img = self.enhance_image(img)
                img.save(pdf_path, "PDF", resolution=300, quality=85, optimize=True)

            self.pdf_files.append(str(pdf_path))

    def merge_pdfs(self):
            """Fügt alle erzeugten PDFs zu einer Datei zusammen und setzt Metadaten"""
            print("📑 Füge PDFs zusammen und setze Metadaten...")
            pdf_writer = PdfWriter()

            for pdf in self.pdf_files:
                with open(pdf, "rb") as f:
                    reader = PdfReader(f)
                    for page in reader.pages:
                        pdf_writer.add_page(page)

            # 🆕 **Setzt die Metadaten**
            pdf_writer.add_metadata(self.metadata)

            with open(self.merged_pdf, "wb") as f:
                pdf_writer.write(f)

    def compress_pdf(self):
        """Komprimiert die PDF mit Ghostscript für beste Qualität & Dateigröße"""
        print("📉 Komprimiere PDF mit Ghostscript...")
        gs_command = [
            "gs", "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.4",
            "-dPDFSETTINGS=/printer", "-dNOPAUSE", "-dQUIET", "-dBATCH",
            f"-sOutputFile={self.compressed_pdf}", str(self.merged_pdf)
        ]
        subprocess.run(gs_command, check=True)

    def apply_ocr(self):
        """Führt OCR mit `ocrmypdf` durch"""
        print("🔠 Führe OCR durch...")
        ocr_command = [
            "ocrmypdf", "-l", "deu+eng", "--optimize", "1",
            "--output-type", "pdfa", "--jbig2-lossy",
            str(self.compressed_pdf), self.output_pdf
        ]
        subprocess.run(ocr_command, check=True)

    def cleanup(self):
        """Löscht temporäre Dateien nach Fertigstellung"""
        print("🧹 Lösche temporäre Dateien...")
        for f in self.temp_dir.iterdir():
            f.unlink()
        self.temp_dir.rmdir()

    def run(self):
        """Hauptprozess: Konvertieren, Zusammenfügen, Komprimieren, OCR"""
        self.convert_tiff_to_pdf()
        self.merge_pdfs()
        self.compress_pdf()
        self.apply_ocr()
        self.cleanup()
        print(f"✅ Fertige Datei: {self.output_pdf}")


def main():
    """CLI-Parser für das Skript mit PDF-Metadaten-Optionen"""
    parser = argparse.ArgumentParser(description="Konvertiert TIFFs zu einem optimierten PDF mit optionaler Bildverbesserung und Metadaten.")

    # TIFF-Dateien
    parser.add_argument("files", nargs="+", help="Liste von TIFF-Dateien")
    
    # Optionale Bildverbesserung
    parser.add_argument("--enhance", action="store_true", help="Verbessert Helligkeit/Kontrast für Graustufenbilder")

    # **Metadaten-Optionen**
    parser.add_argument("--title", type=str, default="Gescanntes Dokument", help="Titel des PDFs")
    parser.add_argument("--author", type=str, default="Automatische PDF-Konvertierung", help="Autor des PDFs")
    parser.add_argument("--subject", type=str, default="Optimiertes PDF mit OCR", help="Thema des PDFs")
    parser.add_argument("--producer", type=str, default="Python Script mit pypdf", help="Producer des PDFs")

    args = parser.parse_args()

    converter = DocumentConverter(
        args.files, 
        enhance=args.enhance, 
        title=args.title,
        author=args.author,
        subject=args.subject,
        producer=args.producer
    )
    converter.run()



if __name__ == "__main__":
    main()
