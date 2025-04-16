from fpdf import FPDF

# Create instance of FPDF class
pdf = FPDF()

# Add a page to the PDF
pdf.add_page()

# Set font: using Arial, regular, size 12
pdf.set_font("Arial", size=12)

# Your text content
text = """CLINICAL ADVANCES IN HYPERTENSION MANAGEMENT
Volume 28, No. 3 - March 2023

Abstract
This paper examines novel approaches to hypertension management in patients with comorbidities. We analyze treatment outcomes for 50 patients across multiple hospitals, focusing on medication efficacy and quality of life improvements.

Introduction
Hypertension affects approximately 30% of adults worldwide and is a major risk factor for cardiovascular disease. When combined with other chronic conditions, management becomes more complex and requires personalized approaches.

Case Study: Central Hospital Protocol
At Central Hospital, cardiologist Emily Wong developed a progressive treatment protocol for hypertensive patients. The protocol emphasizes careful beta blocker titration, beginning with lower doses of atenolol (25-50mg) for patients with multiple risk factors.

Diabetes Considerations
Hypertension often co-occurs with Type 2 Diabetes, complicating treatment. Patient R. Chen (67, male) exhibited improved control when metformin was administered at least 2 hours apart from atenolol, maintaining a consistent medication schedule.

Migraine Comorbidity
For patients suffering from both hypertension and migraine, neurologist James Miller (University Medical Center) recommends avoiding trigger foods and considering preventative medications. Patients often report confusing migraine symptoms with hypertension symptoms, leading to improper self-medication with over-the-counter NSAIDs like ibuprofen.

Conclusion
Successful hypertension management in complex patients requires interdisciplinary coordination. Electronic medical records should flag potential medication interactions, particularly when patients receive care across multiple specialties.

Keywords: hypertension, comorbidity, atenolol, diabetes, medication interaction
"""

# Split text into lines and add each line to the PDF
for line in text.splitlines():
    # Add a cell with the line; ln=True moves to the next line
    pdf.cell(0, 10, line, ln=True)

# Output the PDF file
pdf.output("research_paper.pdf")
