from pathlib import Path


source_path = Path(__file__).with_name("create_ieee_paper.py")
source = source_path.read_text(encoding="utf-8")
replacements = {
    "p.paragraph_format.first_line_indent = Inches(0.16) unless lead else None":
        "p.paragraph_format.first_line_indent = None if lead else Inches(0.16)",
    "Affiliation withheld for review | Correspondence: [email]":
        "Affiliation and contact details withheld for blind review",
}
for old, new in replacements.items():
    source = source.replace(old, new)
exec(compile(source, str(source_path), "exec"), {"__name__": "__main__", "__file__": str(source_path)})
