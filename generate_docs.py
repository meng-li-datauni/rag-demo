"""Generate PDF files of fictional, non-public knowledge for the RAG demo.

Every company, person, number and date here is invented, so the LLM cannot
answer questions about them from its training data - only from these files.
"""

from pathlib import Path

from fpdf import FPDF

DOCS_DIR = Path(__file__).parent / "docs"

DOCUMENTS = {
    "employee_handbook.pdf": (
        "Halvorsen Robotics - Employee Handbook (2026 edition)",
        [
            ("Paid time off",
             "Full-time employees at Halvorsen Robotics receive 27 days of paid time off per "
             "calendar year. Up to 6 unused days may be carried over into the next year, but "
             "carried-over days expire on 31 March. Employees who have been with the company "
             "for more than five years receive 3 additional days."),
            ("Hybrid work policy",
             "Halvorsen Robotics follows a hybrid model. Every employee must be in the Trondheim "
             "office on Tuesdays and Thursdays, which are the designated collaboration days. "
             "The remaining days may be worked remotely with manager approval."),
            ("Equipment stipend",
             "New hires receive a one-time home office equipment stipend of EUR 740. The stipend "
             "is refreshed every three years. Receipts must be submitted through the internal "
             "tool called Ledgerbird within 60 days of purchase."),
            ("On-call compensation",
             "Engineers on the on-call rotation receive EUR 95 per weekday on call and EUR 160 "
             "per weekend day on call. A rotation lasts one week and is handed over every "
             "Monday at 10:00 local time."),
        ],
    ),
    "project_kestrel_spec.pdf": (
        "Project Kestrel - Internal Technical Specification v2.3",
        [
            ("Overview",
             "Project Kestrel is the internal codename for Halvorsen Robotics' next-generation "
             "warehouse inspection drone. The lead engineer is Ines Varga, and the project is "
             "sponsored by the Autonomy division."),
            ("Hardware",
             "The Kestrel airframe carries a 41.5 Wh lithium-ion battery that gives a typical "
             "flight time of 34 minutes. Maximum payload is 3.2 kg. The drone uses six rotors "
             "and weighs 2.7 kg without payload."),
            ("Software",
             "Kestrel runs the in-house flight firmware Larkspur, currently at version 4.1. "
             "Obstacle avoidance relies on a fused stereo-camera and ultra-wideband positioning "
             "system called MeshSight."),
            ("Timeline",
             "Pilot deployments start in November 2026 at two customer sites. General "
             "availability is planned for 14 March 2027. The target list price is EUR 12,900 "
             "per unit."),
        ],
    ),
    "q3_2026_sales_report.pdf": (
        "Halvorsen Robotics - Q3 2026 Internal Sales Report (Confidential)",
        [
            ("Headline numbers",
             "Total revenue in Q3 2026 was EUR 18.4 million, up 22 percent from Q2 2026. Gross "
             "margin was 41 percent. Customer churn for the quarter was 3.1 percent."),
            ("Regions",
             "The Nordics region was the top-performing region with EUR 7.9 million in revenue, "
             "followed by DACH with EUR 5.2 million and Benelux with EUR 3.0 million. The "
             "remaining revenue came from other European markets."),
            ("Key customers",
             "The largest customer by volume was Bellmark Logistics, which ordered 212 units of "
             "the Osprey ground robot. The second largest was Fjordline Cold Storage with 96 "
             "units."),
            ("Outlook",
             "Sales leadership expects Q4 2026 revenue between EUR 19 and 21 million, driven by "
             "the Kestrel pilot programme and a renewal of the Bellmark contract."),
        ],
    ),
    "incident_postmortem_inc0917.pdf": (
        "Postmortem INC-0917: Telemetry Outage",
        [
            ("Summary",
             "On 22 August 2026 the fleet telemetry pipeline was unavailable for 47 minutes, "
             "from 09:12 to 09:59 CEST. During the outage, robots kept operating but no "
             "telemetry reached the customer dashboard."),
            ("Root cause",
             "The TLS certificate on the telemetry gateway tg-east-2 expired. The certificate "
             "had been issued manually 13 months earlier and was not covered by automatic "
             "renewal."),
            ("Resolution",
             "The on-call engineer, Tomas Reinholt, issued a new certificate and restarted the "
             "gateway. As a permanent fix, all gateway certificates were moved to automatic "
             "renewal with cert-manager, and an alert now fires 21 days before any certificate "
             "expires."),
            ("Customer impact",
             "Fourteen customers were affected. Each affected customer received a service "
             "credit of 5 percent of their monthly subscription fee."),
        ],
    ),
}


def write_pdf(path: Path, title: str, sections: list[tuple[str, str]]) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    for heading, body in sections:
        pdf.set_font("Helvetica", "B", 13)
        pdf.multi_cell(0, 8, heading, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, body, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)
    pdf.output(str(path))


def main() -> None:
    DOCS_DIR.mkdir(exist_ok=True)
    for filename, (title, sections) in DOCUMENTS.items():
        write_pdf(DOCS_DIR / filename, title, sections)
        print(f"wrote {DOCS_DIR / filename}")


if __name__ == "__main__":
    main()
