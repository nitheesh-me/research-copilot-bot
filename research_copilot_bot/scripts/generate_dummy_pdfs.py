from reportlab.pdfgen import canvas
import os

def create_dummy_pdf(filename, title, content):
    c = canvas.Canvas(filename)
    c.drawString(100, 800, title)
    c.drawString(100, 780, "Authors: Dummy Author")
    c.drawString(100, 760, "Abstract: This is a dummy abstract.")

    y = 740
    for line in content.split('\n'):
        c.drawString(100, y, line)
        y -= 20
        if y < 50:
            c.showPage()
            y = 800

    c.save()

def main():
    os.makedirs("pdfs", exist_ok=True)

    papers = [
        ("paper1.pdf", "Deep Learning in SE", "Deep learning has shown promise in code generation.\nWe propose a new model."),
        ("paper2.pdf", "Testing with AI", "AI can automate test generation.\nHowever, oracle problem remains."),
        ("paper3.pdf", "Survey of LLMs", "LLMs are transforming software engineering.\nWe survey 100 papers."),
        ("paper4.pdf", "Automated Debugging", "We use agents to fix bugs automatically.\nResults show 50% fix rate."),
        ("paper5.pdf", "Requirements Engineering", "NLP helps in extracting requirements.\nAmbiguity is a challenge.")
    ]

    for filename, title, content in papers:
        create_dummy_pdf(os.path.join("pdfs", filename), title, content)
        print(f"Created {filename}")

if __name__ == "__main__":
    main()
