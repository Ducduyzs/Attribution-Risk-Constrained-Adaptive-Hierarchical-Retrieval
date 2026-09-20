from pathlib import Path


source_path = Path(__file__).with_name("create_ieee_paper.py")
source = source_path.read_text(encoding="utf-8")
source = source.replace(
    "p.paragraph_format.first_line_indent = Inches(0.16) unless lead else None",
    "p.paragraph_format.first_line_indent = None if lead else Inches(0.16)",
)
exec(compile(source, str(source_path), "exec"), {"__name__": "__main__", "__file__": str(source_path)})
