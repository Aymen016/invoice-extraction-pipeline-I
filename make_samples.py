#!/usr/bin/env python3
"""Generate sample invoice PDFs so you can test without hunting for real ones.

Deliberately varied: different layouts, currencies, one with a deliberate
arithmetic error, one scanned-style image. Uniform samples give you a
falsely flattering accuracy number.

    python make_samples.py
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path("samples")
styles = getSampleStyleSheet()


def build(filename: str, header: str, meta: list[tuple[str, str]],
          parties: list[tuple[str, str]], items: list[list], totals: list[tuple[str, str]]):
    OUT.mkdir(exist_ok=True)
    doc = SimpleDocTemplate(str(OUT / filename), pagesize=A4,
                            topMargin=20 * mm, bottomMargin=20 * mm)
    flow = [Paragraph(f"<b><font size=16>{header}</font></b>", styles["Normal"]),
            Spacer(1, 8 * mm)]

    meta_tbl = Table([[k, v] for k, v in meta], colWidths=[40 * mm, 60 * mm])
    meta_tbl.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 9),
                                  ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    flow += [meta_tbl, Spacer(1, 6 * mm)]

    for label, body in parties:
        flow.append(Paragraph(f"<b>{label}</b><br/>{body}", styles["Normal"]))
        flow.append(Spacer(1, 4 * mm))

    flow.append(Spacer(1, 4 * mm))
    tbl = Table([["Description", "Qty", "Unit Price", "Amount"]] + items,
                colWidths=[85 * mm, 20 * mm, 30 * mm, 30 * mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
    ]))
    flow += [tbl, Spacer(1, 6 * mm)]

    tot = Table([[k, v] for k, v in totals], colWidths=[115 * mm, 50 * mm])
    tot.setStyle(TableStyle([("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                             ("FONTSIZE", (0, 0), (-1, -1), 9),
                             ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold")]))
    flow.append(tot)
    doc.build(flow)
    print(f"  wrote samples/{filename}")


def main():
    # 1. Clean Pakistani invoice, PKR, tax adds up correctly.
    build(
        "invoice_clean_pkr.pdf", "TAX INVOICE",
        [("Invoice No:", "INV-2026-0412"), ("Invoice Date:", "14/03/2026"),
         ("Due Date:", "13/04/2026"), ("Currency:", "PKR")],
        [("From:", "Zenith Textiles (Pvt) Ltd<br/>34-C Gulberg III, Lahore<br/>NTN: 4210398-7"),
         ("Bill To:", "Meridian Retail Group<br/>Plot 12, Korangi, Karachi")],
        [["Cotton fabric, 60s combed", "120", "850.00", "102,000.00"],
         ["Polyester lining", "80", "310.00", "24,800.00"],
         ["Packaging and labelling", "1", "6,500.00", "6,500.00"]],
        [("Subtotal:", "PKR 133,300.00"), ("Sales Tax (18%):", "PKR 23,994.00"),
         ("Grand Total:", "PKR 157,294.00")],
    )

    # 2. USD, different labels, no tax line.
    build(
        "invoice_usd_notax.pdf", "INVOICE",
        [("Invoice #:", "SW-88213"), ("Date:", "2026-02-02"), ("Terms:", "Net 15")],
        [("Vendor:", "Softwave Analytics LLC<br/>500 Market St, San Francisco, CA<br/>VAT: US-99-4410221"),
         ("Bill To:", "Aurora Logistics Inc<br/>221 Dock Road, Seattle, WA")],
        [["Data platform licence, Q1", "1", "4,200.00", "4,200.00"],
         ["Onboarding and migration", "18", "150.00", "2,700.00"]],
        [("Subtotal:", "$6,900.00"), ("Total Due:", "$6,900.00")],
    )

    # 3. Deliberately broken arithmetic — validation should catch this.
    build(
        "invoice_bad_math.pdf", "PURCHASE ORDER",
        [("PO Number:", "PO-77401"), ("Date:", "05/01/2026")],
        [("Supplier:", "Highland Components Ltd<br/>Unit 4, Sialkot Industrial Estate<br/>NTN: 3390221-4"),
         ("Deliver To:", "Falcon Assembly Works, Faisalabad")],
        [["Bearing assembly SKF-6204", "200", "420.00", "84,000.00"],
         ["Hex bolts M8 (box of 100)", "15", "1,100.00", "16,500.00"]],
        # Subtotal below is wrong on purpose: real sum is 100,500.
        [("Subtotal:", "PKR 100,500.00"), ("GST (18%):", "PKR 18,090.00"),
         ("Total:", "PKR 121,000.00")],  # should be 118,590
    )

    # 4. Ambiguous date format (03/04/2026 — March or April?).
    build(
        "invoice_ambiguous_date.pdf", "INVOICE",
        [("Invoice Number:", "AB/2026/0087"), ("Invoice Date:", "03/04/2026"),
         ("Payment Due:", "03/05/2026")],
        [("From:", "Karachi Print House<br/>Shahrah-e-Faisal, Karachi<br/>NTN: 1122334-5"),
         ("To:", "Bright Futures School System")],
        [["Annual report printing, 500 copies", "500", "185.00", "92,500.00"]],
        [("Subtotal:", "Rs 92,500.00"), ("Sales Tax:", "Rs 16,650.00"),
         ("Amount Payable:", "Rs 109,150.00")],
    )

    print("\nAlso making a scanned-style image to exercise the OCR path...")
    make_scan()


def make_scan():
    """Render one PDF page to a JPEG so it has no text layer at all."""
    import pdfplumber

    src = OUT / "invoice_clean_pkr.pdf"
    with pdfplumber.open(src) as pdf:
        img = pdf.pages[0].to_image(resolution=150).original
    img = img.convert("L")  # greyscale, like a real scan
    out = OUT / "invoice_scanned.jpg"
    img.save(out, quality=70)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
