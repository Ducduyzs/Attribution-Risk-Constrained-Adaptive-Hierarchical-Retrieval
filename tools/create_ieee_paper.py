from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts"
TMP_DIR = ROOT / ".codex_tmp" / "ieee_paper"
OUT_PATH = OUT_DIR / "Attribution_Risk_Adaptive_Hierarchical_Retrieval_IEEE_Draft.docx"
FIG_PATH = TMP_DIR / "pipeline_overview.png"

BLACK = RGBColor(0, 0, 0)
GRAY = RGBColor(90, 90, 90)
LIGHT_GRAY = "E7E6E6"
TABLE_WIDTH_DXA = 4968  # 3.45 inches; one IEEE column.


def set_font(run, name="Times New Roman", size=10, *, bold=None, italic=None, color=BLACK):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
    run.font.size = Pt(size)
    run.font.color.rgb = color
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def set_cell_margins(cell, top=45, start=55, bottom=45, end=55):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcMar = tcPr.first_child_found_in("w:tcMar")
    if tcMar is None:
        tcMar = OxmlElement("w:tcMar")
        tcPr.append(tcMar)
    for tag, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tcMar.find(qn(f"w:{tag}"))
        if node is None:
            node = OxmlElement(f"w:{tag}")
            tcMar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_geometry(table, widths_dxa):
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    tblPr = table._tbl.tblPr
    tblW = tblPr.first_child_found_in("w:tblW")
    if tblW is None:
        tblW = OxmlElement("w:tblW")
        tblPr.append(tblW)
    tblW.set(qn("w:w"), str(sum(widths_dxa)))
    tblW.set(qn("w:type"), "dxa")
    tblInd = tblPr.first_child_found_in("w:tblInd")
    if tblInd is None:
        tblInd = OxmlElement("w:tblInd")
        tblPr.append(tblInd)
    tblInd.set(qn("w:w"), "0")
    tblInd.set(qn("w:type"), "dxa")

    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths_dxa:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)

    for row in table.rows:
        for cell, width in zip(row.cells, widths_dxa):
            tcPr = cell._tc.get_or_add_tcPr()
            tcW = tcPr.first_child_found_in("w:tcW")
            if tcW is None:
                tcW = OxmlElement("w:tcW")
                tcPr.append(tcW)
            tcW.set(qn("w:w"), str(width))
            tcW.set(qn("w:type"), "dxa")
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell)


def shade_cell(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = tcPr.first_child_found_in("w:shd")
    if shd is None:
        shd = OxmlElement("w:shd")
        tcPr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_repeat_table_header(row):
    trPr = row._tr.get_or_add_trPr()
    tblHeader = OxmlElement("w:tblHeader")
    tblHeader.set(qn("w:val"), "true")
    trPr.append(tblHeader)


def add_table(doc, caption, headers, rows, widths, note=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(caption.upper())
    set_font(r, size=8, bold=True)

    table = doc.add_table(rows=1, cols=len(headers))
    set_table_geometry(table, widths)
    set_repeat_table_header(table.rows[0])
    for cell, header in zip(table.rows[0].cells, headers):
        shade_cell(cell, LIGHT_GRAY)
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_after = Pt(0)
        paragraph.paragraph_format.line_spacing = 1.0
        run = paragraph.add_run(header)
        set_font(run, size=7.2, bold=True)
    for values in rows:
        cells = table.add_row().cells
        for i, (cell, value) in enumerate(zip(cells, values)):
            paragraph = cell.paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if i == 0 else WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.line_spacing = 1.0
            run = paragraph.add_run(str(value))
            set_font(run, size=7.2)
    if note:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(4)
        r = p.add_run(note)
        set_font(r, size=7.2, italic=True)
    return table


def set_columns(section, count=2, space_twips=360):
    sectPr = section._sectPr
    cols = sectPr.xpath("./w:cols")
    node = cols[0] if cols else OxmlElement("w:cols")
    if not cols:
        sectPr.append(node)
    node.set(qn("w:num"), str(count))
    node.set(qn("w:space"), str(space_twips))


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run("Preliminary research draft | ")
    set_font(run, size=8, color=GRAY)
    fldChar1 = OxmlElement("w:fldChar")
    fldChar1.set(qn("w:fldCharType"), "begin")
    instrText = OxmlElement("w:instrText")
    instrText.set(qn("xml:space"), "preserve")
    instrText.text = " PAGE "
    fldChar2 = OxmlElement("w:fldChar")
    fldChar2.set(qn("w:fldCharType"), "end")
    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)


def configure_styles(doc):
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    normal.font.size = Pt(10)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(2)
    normal.paragraph_format.line_spacing = 1.0

    h1 = styles["Heading 1"]
    h1.font.name = "Times New Roman"
    h1._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    h1._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    h1.font.size = Pt(10)
    h1.font.bold = False
    h1.font.color.rgb = BLACK
    h1.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    h1.paragraph_format.space_before = Pt(7)
    h1.paragraph_format.space_after = Pt(3)
    h1.paragraph_format.keep_with_next = True

    h2 = styles["Heading 2"]
    h2.font.name = "Times New Roman"
    h2._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    h2._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    h2.font.size = Pt(10)
    h2.font.bold = False
    h2.font.italic = True
    h2.font.color.rgb = BLACK
    h2.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    h2.paragraph_format.space_before = Pt(5)
    h2.paragraph_format.space_after = Pt(2)
    h2.paragraph_format.keep_with_next = True


def add_heading(doc, text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    r = p.add_run(text)
    set_font(r, size=10, italic=(level == 2))
    return p


def add_body(doc, text, *, lead=False):
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Inches(0.16) unless lead else None
    p.paragraph_format.widow_control = True
    r = p.add_run(text)
    set_font(r, size=10)
    return p


def add_equation(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(text)
    set_font(r, size=9.5, italic=True)


def make_pipeline_figure(path):
    width, height = 1200, 1550
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/times.ttf", 40)
        bold = ImageFont.truetype("C:/Windows/Fonts/timesbd.ttf", 41)
        small = ImageFont.truetype("C:/Windows/Fonts/times.ttf", 32)
    except OSError:
        font = bold = small = ImageFont.load_default()

    boxes = [
        ("Scientific documents", "layout and provenance"),
        ("Document hierarchy", "document - section - parent - child"),
        ("Leaf candidate retrieval", "pluggable shared backend"),
        ("Neural reranking", "query-node relevance"),
        ("Risk-constrained gates", "KEEP / PARENT / SECTION"),
        ("Budgeted context", "deduplication and knapsack"),
        ("Structured generation", "atomic claims + context IDs"),
        ("Leaf verification", "NLI, safety guard, evidence"),
    ]
    x0, x1 = 120, 1080
    y, bh, gap = 55, 125, 52
    for index, (title, subtitle) in enumerate(boxes):
        fill = (235, 242, 250) if index not in (4, 7) else (225, 237, 225)
        draw.rounded_rectangle((x0, y, x1, y + bh), radius=22, fill=fill, outline=(30, 60, 90), width=3)
        box = draw.textbbox((0, 0), title, font=bold)
        draw.text(((width - (box[2] - box[0])) / 2, y + 17), title, font=bold, fill=(0, 0, 0))
        box2 = draw.textbbox((0, 0), subtitle, font=small)
        draw.text(((width - (box2[2] - box2[0])) / 2, y + 72), subtitle, font=small, fill=(55, 55, 55))
        if index < len(boxes) - 1:
            cx = width // 2
            draw.line((cx, y + bh, cx, y + bh + gap - 10), fill=(30, 60, 90), width=5)
            draw.polygon([(cx - 14, y + bh + gap - 25), (cx + 14, y + bh + gap - 25), (cx, y + bh + gap - 4)], fill=(30, 60, 90))
        y += bh + gap

    draw.text((82, height - 72), "Representation-specific multi-vector retrieval is outside the claimed contribution.", font=font, fill=(120, 20, 20))
    image.save(path, dpi=(300, 300))


def add_figure(doc):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(1)
    p.add_run().add_picture(str(FIG_PATH), width=Inches(3.18))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    cap.paragraph_format.space_after = Pt(4)
    r = cap.add_run("Fig. 1. End-to-end method scope. Candidate retrieval is intentionally abstracted as a shared, pluggable dependency; the contribution begins with hierarchical risk control and ends with claim-to-leaf verification.")
    set_font(r, size=8)


def add_reference(doc, number, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.2)
    p.paragraph_format.first_line_indent = Inches(-0.2)
    p.paragraph_format.space_after = Pt(1)
    r = p.add_run(f"[{number}] {text}")
    set_font(r, size=8)


def build():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    make_pipeline_figure(FIG_PATH)

    doc = Document()
    configure_styles(doc)
    sec = doc.sections[0]
    sec.page_width = Inches(8.5)
    sec.page_height = Inches(11)
    sec.top_margin = Inches(0.65)
    sec.bottom_margin = Inches(0.65)
    sec.left_margin = Inches(0.75)
    sec.right_margin = Inches(0.75)
    sec.header_distance = Inches(0.3)
    sec.footer_distance = Inches(0.3)
    set_columns(sec, 1)
    add_page_number(sec.footer.paragraphs[0])

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(7)
    r = title.add_run("Attribution-Risk-Constrained Adaptive Hierarchical Retrieval for Verifiable Scientific Document Question Answering")
    set_font(r, size=20, bold=False)

    author = doc.add_paragraph()
    author.alignment = WD_ALIGN_PARAGRAPH.CENTER
    author.paragraph_format.space_after = Pt(1)
    r = author.add_run("Anonymous Author(s)")
    set_font(r, size=11)
    affiliation = doc.add_paragraph()
    affiliation.alignment = WD_ALIGN_PARAGRAPH.CENTER
    affiliation.paragraph_format.space_after = Pt(8)
    r = affiliation.add_run("Affiliation withheld for review | Correspondence: [email]")
    set_font(r, size=9, italic=True)

    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.35)
    p.paragraph_format.right_indent = Inches(0.35)
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run("Abstract-")
    set_font(r, size=9, bold=True, italic=True)
    r = p.add_run(
        "Scientific document question answering requires enough context to resolve dispersed evidence while preserving the ability to attribute each generated claim to a specific passage. Fixed leaf retrieval can omit necessary context, whereas unconditional hierarchical expansion can introduce distractors and attribution drift. This paper presents an attribution-risk-constrained adaptive hierarchical retrieval framework over document, section, parent, and child nodes. A query-conditioned controller compares retained descendants with candidate ancestors, applies a relevance rollback guard, and assembles a token-bounded context. Generation is constrained to atomic claims with valid context identifiers, after which natural-language inference and deterministic conflict guards map every surviving claim back to leaf evidence. Counterfactual KEEP, PARENT, and SECTION rollouts provide labels for independent expansion gates. The evaluation protocol separates official QASPER paragraph evidence F1 from child-level attribution metrics and treats SciFact only as rationale-attribution out-of-domain evidence. Repository verification currently reports 83 passed tests and four passed subtests. Preliminary two-question QASPER smoke results show that static expansion increases official evidence F1 from 0.5333 to 0.5952 but does not improve answer F1; a 20-paper diagnostic likewise provides no evidence that the historical learned gate exceeds flat retrieval. These findings support the proposed diagnostic framing but are insufficient for a superiority claim."
    )
    set_font(r, size=9, italic=True)

    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.35)
    p.paragraph_format.right_indent = Inches(0.35)
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run("Index Terms-")
    set_font(r, size=9, bold=True, italic=True)
    r = p.add_run("scientific question answering, hierarchical retrieval, attribution risk, retrieval-augmented generation, evidence verification, QASPER")
    set_font(r, size=9, italic=True)

    body_sec = doc.add_section(WD_SECTION.CONTINUOUS)
    body_sec.top_margin = Inches(0.65)
    body_sec.bottom_margin = Inches(0.65)
    body_sec.left_margin = Inches(0.75)
    body_sec.right_margin = Inches(0.75)
    set_columns(body_sec, 2, 360)
    body_sec.footer.is_linked_to_previous = True

    add_heading(doc, "I. INTRODUCTION")
    add_body(doc, "Retrieval-augmented generation (RAG) couples a language model with explicit non-parametric evidence, improving provenance and updateability relative to purely parametric generation [1]. Modern instantiations build on dense passage retrieval [17], [20] with lexical BM25 baselines [18], while scientific papers remain a difficult target because answers may depend on evidence distributed across sections, while exact wording, quantities, and caveats often live in narrow passages. QASPER formalizes this setting with 5,049 questions over 1,585 NLP papers and includes supporting evidence supplied by annotators [2].")
    add_body(doc, "The central granularity dilemma is asymmetric. Short chunks preserve specificity but may fragment an explanation; broad sections restore context but expose the generator to irrelevant siblings. Tree-based retrieval and long-context approaches address missing context [4], [5], yet increased context is not automatically attributable. Long-document assistants should either support answers with evidence or abstain [6]. This work therefore treats expansion as a risk-controlled decision rather than a fixed preprocessing choice.")
    add_body(doc, "We present a four-level document hierarchy and two independent gates: child-to-parent and parent-to-section/document. Each gate predicts whether an ancestor offers downstream utility without unacceptable claim-to-leaf attribution risk. A rollback test rejects ancestors whose reranked relevance falls sharply below the best retained descendant. Generation is constrained by a request-specific citation schema, and verification occurs against leaf descendants rather than the expanded text block.")
    add_body(doc, "The paper makes four scoped contributions. First, it defines claim-to-leaf attribution risk and operationalizes it in rollout rewards and runtime telemetry. Second, it introduces counterfactual, level-specific expansion gates with explicit token and drift controls. Third, it implements end-to-end provenance from PDF spans to verified claims. Fourth, it specifies an evaluation protocol that separates official paragraph evidence from leaf attribution and reports negative or inconclusive results without overstating performance.")
    add_body(doc, "Scope correction: the repository also contains a multi-representation retrieval implementation. Because that component belongs to a separate Multi-Vector pipeline, representation fusion, learned sparse retrieval, and late interaction are treated here only as a replaceable shared backend. They are excluded from the claimed novelty and from the conceptual definition of the proposed method.")

    add_figure(doc)

    add_heading(doc, "II. RELATED WORK")
    add_heading(doc, "A. Scientific Document QA", 2)
    add_body(doc, "QASPER targets information-seeking questions whose answers occur in full research papers, often requiring evidence selection across multiple paragraphs [2]. Its multi-annotator structure makes metric implementation consequential: answer and evidence scores must be computed against each annotation and then maximized, rather than against a union that changes the task. The present pipeline preserves per-annotation answer and evidence sets and maps overlapping children back to stable paragraph identifiers. Adjacent long-form benchmarks clarify the design space: ASQA pairs factoid questions with long-form attributed answers [31], ELI5 tests explanatory long-form generation [32], while Natural Questions [29] and HotpotQA [30] cover single- and multi-hop open-domain QA that motivate adaptive retrieval.")
    add_heading(doc, "B. Hierarchical, Long-Context, and Graph Retrieval", 2)
    add_body(doc, "RAPTOR recursively clusters and summarizes text to retrieve information at multiple abstraction levels [4]. LongRAG instead constructs long retrieval units and demonstrates that larger units can reduce retrieval burden, while excessive recalled context can still introduce hard negatives [5]. GraphRAG builds an entity knowledge graph with Leiden community detection [33] and answers from community summaries, improving global sensemaking over naive RAG [16]. These gains must be weighed against position bias in long contexts: models underuse information placed in the middle of long inputs [25]. Our method differs in objective and provenance: ancestors are deterministic compositions of leaf passages, expansion is decided per query and candidate group, and every final claim must survive verification against a leaf descendant.")
    add_heading(doc, "C. Attribution and Scientific Verification", 2)
    add_body(doc, "Attribute-or-abstain evaluation emphasizes that response quality and evidence quality are distinct requirements for long-document assistants [6]. Attributed QA formalizes the (answer, attribution-pointer) contract and studies automatic AIS evaluation against human judgments [12], [23]. ALCE extends citation evaluation to long-form generation with automatic citation recall and precision [11], a framing close to our claim-to-leaf metrics. SciFact pairs expert-written scientific claims with supporting or refuting rationales [3]. We use SciFact only as an out-of-domain rationale-attribution endpoint; free-form answer F1 is not interpreted as stance accuracy. The verifier uses an NLI classifier in the lineage of TRUE-style factual-consistency evaluation [24] and FEVER verification [35], and fails fast when the model label map does not identify entailment, preventing contradiction probabilities from being misused as support. The deployed verifier family is based on DeBERTaV3 [8].")
    add_heading(doc, "D. Adaptive, Corrective, and Self-Reflective Retrieval", 2)
    add_body(doc, "Fixed retrieval depth is wasteful for simple queries and insufficient for complex ones. Adaptive-RAG routes queries among no-retrieval, single-step, and multi-step strategies with a complexity classifier [14], building on evidence that retrieval is unnecessary or harmful for well-memorized facts [26]. Self-RAG trains reflection tokens that decide when to retrieve and critique relevance and support [13]. CRAG adds a retrieval evaluator with correct/amend/expand actions and strip-level refinement to survive retrieval failures [15]. Iterative [27], interleaved reasoning-acting [28], and forward-looking active retrieval [34] offer complementary control loops. Our gates share the adaptive spirit but control a different axis: hierarchical expansion granularity under an explicit attribution-risk budget, rather than retrieval timing or source correction.")

    add_heading(doc, "III. PROBLEM FORMULATION")
    add_body(doc, "Let a scientific document be represented by a rooted hierarchy H=(V,E) with levels document, section, parent, and child. Child nodes are leaf evidence units. For a query q, a retriever and reranker return leaf candidates R. The controller selects a context S containing retained leaves or expanded ancestors. Each ancestor a exposes a deterministic descendant-leaf set L(a), which preserves traceability even when the generator sees broader text.")
    add_body(doc, "For generated claims C(S), let p(e entails c) denote the verifier support score for leaf e and claim c. The attribution risk of a context is")
    add_equation(doc, "AR(S) = 1 - (1 / |C(S)|) sum_c max_{e in L(S)} p(e entails c).    (1)")
    add_body(doc, "AR(S)=1 when no claim reaches a valid leaf support test. The unsupported-claim rate and citation-survival rate complement (1). This family captures a failure that answer-only metrics obscure: an answer can sound correct while no cited leaf supports its individual claims.")

    add_heading(doc, "IV. METHOD")
    add_heading(doc, "A. Layout-Aware Ingestion and Hierarchy", 2)
    add_body(doc, "PDFs are converted with Docling, which provides a structured document representation and models for layout and table structure [7]. The loader retains section labels, page spans, bounding boxes, parser confidence, and a markdown fallback. QASPER JSON bypasses PDF parsing but is normalized into the same scientific-document schema. Stable identifiers are derived from document, section, position, and text prefixes.")
    add_body(doc, "Each section is packed into overlapping sentence-bounded child chunks, nominally 220 tokens. Consecutive groups of four children form parent nodes with one-child overlap. Sections and documents retain all descendant leaf identifiers. Character offsets map QASPER paragraphs to all overlapping leaves, allowing paragraph-level evaluation without multiplying the gold denominator.")
    add_heading(doc, "B. Candidate Retrieval and Reranking", 2)
    add_body(doc, "The method requires only a Retriever interface that returns scored leaf identifiers and accepts an optional source constraint. The production backend combines dense passage retrieval [17], [20], [21] with learned sparse and ColBERT-style late interaction [22] over a BM25 lexical base [18], followed by a sequence-to-sequence cross-encoder reranker in the monoT5 lineage [19]. Retrieval is scoped inside a paper before top-k truncation. A neural cross-encoder reranks the first 24 candidates from an initial pool of up to 80. All systems in a paired experiment share this backend, so policy comparisons isolate hierarchy control. Representation-specific multi-vector mechanisms are not part of the proposed contribution.")
    add_heading(doc, "C. Adaptive Parent Gate", 2)
    add_body(doc, "For a candidate ancestor a and retained member set M, the feature vector contains candidate relevance, retrieved-slot coverage, adjacent-member coherence, evidence density, noise, incremental token cost, query-type indicators, member-count normalization, score entropy, section size, and query length. The calibrated prior computes a merge probability; a learned checkpoint can replace it.")
    add_body(doc, "The prior compares ancestor and member utilities. With r(a) the reranker score, p(a) the merge probability, g=1-coverage, B the context budget, and delta_t the incremental tokens,")
    add_equation(doc, "Delta(a,M) = [0.5 p(a) + 0.5 r(a)] [1 + lambda_g g] - mean_{m in M} r(m) - lambda_t delta_t/B.    (2)")
    add_body(doc, "The prior accepts when Delta exceeds a margin. The learned gate accepts when p(a) exceeds its stored threshold. Both are subject to rollback: if r(a) < rho max_m r(m), the ancestor is rejected because it lost evidence specificity. Parent and higher-level gates use independent checkpoints and can be disabled separately.")
    add_heading(doc, "D. Iterative Expansion", 2)
    add_body(doc, "Accepted parents replace their member children. A second gate then considers section and document ancestors one level at a time. Query type constrains the ceiling: factoid questions stop at parent, explanatory questions at section, and comparative or global questions may reach document. Expansion stops when no candidate has enough members, no merge is accepted, marginal utility falls below epsilon, maximum depth is reached, or projected tokens exceed budget headroom.")
    add_heading(doc, "E. Budgeted Context Assembly", 2)
    add_body(doc, "Selected nodes are ordered by utility, deduplicated using lexical Jaccard and containment, and limited to the final candidate count. A zero-one knapsack over token buckets maximizes retained utility under a strict token budget. A diversity repair caps single-source share and ensures multiple sources for comparative questions when alternatives exist. Oversized fallback blocks are truncated at sentence boundaries.")
    add_heading(doc, "F. Structured Generation and Verification", 2)
    add_body(doc, "The generator receives context blocks C1, C2, and so forth. OpenAI generation uses a strict JSON schema whose citation enum is constructed from the valid identifiers of the current request. Other providers undergo equivalent post-response validation. Numeric indices, bracketed identifiers, blanks, unknown identifiers, answerable responses without claims, and refusals with claims remain visible as validation telemetry.")
    add_body(doc, "Each accepted claim must exceed a confidence threshold and cite reachable context. The verifier tests the claim against descendant children, not the broad parent text. A lexical rescue can recover near-verbatim claims that NLI scores as neutral, but it is disabled when negation polarity or comparable numbers conflict, or when contradiction exceeds a safety threshold. Leaves outside the initially retrieved set must clear a stricter sibling threshold. The default selector retains at most the strongest evidence leaf per claim.")

    add_heading(doc, "V. POLICY TRAINING")
    add_body(doc, "For every query and candidate group, the rollout harness materializes KEEP, PARENT, and SECTION branches through the same context builder, generator, and verifier. When gold leaf sets are available, each branch is scored using answer quality, citation recall and precision, rescued gold leaves, harmful drift, ambiguity, empty evidence, token cost, and latency. The historical v5 reward is")
    add_equation(doc, "R = .50 F1_ans + .50 Rec_cit + .70 Prec_cit + .50 Rescue - 1.00 Harm - .30 Amb - .40 Empty - .30 Tok - .05 Lat.    (3)")
    add_body(doc, "Oracle labels require an expansion branch to exceed KEEP by a margin while satisfying precision and harmful-drift constraints. Rows are weighted inversely by the number of rows per paper. Estimator family is selected with paper-disjoint group cross-validation on training data, while thresholds are selected on development data using paper-weighted balanced accuracy. Parent and section gates are trained independently. Historical v5 checkpoints predate the final citation contract and are retained only as baselines.")

    add_table(
        doc,
        "Table I\nFrozen v5 gate training diagnostics",
        ["Gate", "Estimator", "Thr.", "Train BA", "Dev BA", "Dev AUC"],
        [
            ["Parent", "Random forest", "0.50", "0.8833", "0.7207", "0.5953"],
            ["Section", "Gradient boost", "0.32", "0.8818", "0.6823", "0.6806"],
        ],
        [720, 1150, 600, 780, 780, 938],
        "BA denotes paper-weighted balanced accuracy. Training used 120 rows/120 papers; development used 40 rows/40 papers. These checkpoints are historical, not final.",
    )

    add_heading(doc, "VI. EXPERIMENTAL PROTOCOL")
    add_heading(doc, "A. Datasets and Splits", 2)
    add_body(doc, "QASPER is the primary English benchmark. The canonical converter preserves reference answers, reference evidence sets, and stable paragraph identifiers separately for each annotation. Official answer EM/F1 and paragraph evidence F1 take the maximum over annotations. Leaf attribution F1 is reported separately. SciFact contains approximately 1.4K expert-written scientific claims with labels and rationales [3]; completed SciFact development runs are treated as frozen rationale-attribution OOD evidence.")
    add_heading(doc, "B. Baselines and Ablations", 2)
    add_body(doc, "The protocol specifies BM25 [18], dense-only [17], hybrid rank fusion, flat neural retrieval, static hierarchy, RAPTOR-style tree retrieval [4], LongRAG-style long-unit retrieval [5], calibrated prior, learned gate, parent-only, section-only, no drift penalty, no sibling guard, no verifier, oracle evidence, full-document long context, and citation-generation or post-hoc attribution comparisons. RAPTOR (B5) is implemented as recursive TF-IDF/KMeans clustering with extractive centroid summaries and collapsed-tree multi-level retrieval fused with the shared leaf backend; LongRAG (B6) ranks section-level long units from shared-backend leaf scores and reads whole sections with leaf-level verification. Because the separate Multi-Vector pipeline is outside this paper, representation-fusion ablations are implementation controls rather than contributions. Static expansion must remain visible whenever it is the strongest comparator.")
    add_heading(doc, "C. Metrics and Statistical Controls", 2)
    add_body(doc, "Retrieval metrics include recall, precision, hit rate, MRR, and nDCG. Grounding metrics include official QASPER paragraph evidence F1, leaf attribution precision/recall/F1, evidence-span recall, provenance accuracy, rescue, harmful drift, and citation survival. Answer quality uses annotation-maximized EM/F1. Efficiency reports context tokens, latency, and API cost. Comparisons require paired tests, paper-clustered 95% confidence intervals, effect sizes, and correction for multiple comparisons.")
    add_heading(doc, "D. Reproducibility and Leakage Control", 2)
    add_body(doc, "Every result artifact should record the commit and dirty-worktree flag, model snapshot, prompt and configuration hashes, package versions, hardware, dataset and manifest hashes, checkpoint hash, seed, per-query traces, and generation cost. Test manifests must be written and hashed before inference. Previously inspected QASPER test and SciFact development examples cannot be used for prompt, feature, model, threshold, or hyperparameter selection.")

    add_heading(doc, "VII. RESULTS")
    add_heading(doc, "A. Software Verification", 2)
    add_body(doc, "A fresh repository test run completed successfully with 83 passed tests and four passed subtests. Coverage includes hierarchy invariants, scope-correct retrieval, RAPTOR/LongRAG baselines, merge and expansion policy, strict context budgets, citation contracts, NLI label safety, lexical conflict guards, QASPER multi-annotator metrics, statistical utilities, rollouts, and policy training. Passing tests establish implementation consistency, not empirical effectiveness.")

    add_heading(doc, "B. QASPER Contract and Development Smoke Tests", 2)
    add_body(doc, "The strict-contract smoke test used two papers and two questions with the fixed model snapshot gpt-4o-mini-2024-07-18. Flat retrieval achieved official evidence F1 0.4000, leaf attribution F1 0.4500, answer F1 0.3514, and evidence-span recall 0.6250. A separate two-question development smoke test compared flat and static expansion under the corrected evaluator.")
    add_table(
        doc,
        "Table II\nQASPER v7 development smoke (2 questions, 2 papers)",
        ["System", "Official Ev. F1", "Leaf F1", "Answer F1", "Tokens"],
        [
            ["B_flat", "0.5333", "0.5357", "0.2114", "1548.0"],
            ["B_static", "0.5952", "0.5833", "0.1697", "2561.5"],
        ],
        [900, 1050, 920, 920, 1178],
        "Static-minus-flat official evidence F1: paired p=0.7483; paper-clustered 95% CI [-0.0667, 0.1905]. The sample is a plumbing check only.",
    )
    add_body(doc, "Static expansion increased both evidence metrics in this tiny sample but consumed 65.5% more context and reduced answer F1. The interval spans zero, and n=2 cannot support a method claim.")

    add_heading(doc, "C. Historical 20-Paper Diagnostic", 2)
    add_body(doc, "Before the strict citation fix, generators often returned numeric citation indices instead of context identifiers, collapsing measured citation quality. After the prompt-level contract correction, the same 20-paper development diagnostic produced the values below. These are legacy leaf-citation metrics, not the final official paragraph metric.")
    add_table(
        doc,
        "Table III\nQASPER diagnostic after citation-contract fix (20 papers)",
        ["System", "Cit. P", "Cit. R", "Cit. F1", "Ans. F1", "Tokens"],
        [
            ["B_flat", "0.5053", "0.6465", "0.4942", "0.1676", "1484.0"],
            ["Learned-v5", "0.4105", "0.5982", "0.4439", "0.1596", "1872.5"],
        ],
        [800, 670, 670, 770, 800, 1258],
        "Learned-v5 minus flat citation F1: p=0.2707; paper-clustered 95% CI [-0.1365, 0.0322].",
    )
    add_body(doc, "The learned historical gate underperformed flat retrieval on all listed quality metrics while consuming more context. It was trained from rollouts created before the final generation contract and therefore cannot serve as the final learned system. This result directly prohibits a superiority claim.")

    add_heading(doc, "D. Historical SciFact OOD Attribution", 2)
    add_body(doc, "On 40 frozen SciFact claim-document pairs, hierarchical systems improved citation/rationale attribution relative to flat retrieval. Static expansion remained the strongest citation-F1 comparator, slightly above the learned gate. Since the split has already been inspected, these values are historical OOD evidence and cannot guide further tuning.")
    add_table(
        doc,
        "Table IV\nSciFact frozen rationale-attribution OOD (40 pairs)",
        ["System", "Cit. P", "Cit. R", "Cit. F1", "Tokens"],
        [
            ["B_flat", "0.4000", "0.4000", "0.3833", "384.8"],
            ["B_static", "0.8375", "0.7083", "0.7450", "394.8"],
            ["Prior", "0.7500", "0.6625", "0.6825", "389.4"],
            ["Learned-v5", "0.8250", "0.7042", "0.7358", "383.7"],
        ],
        [920, 790, 790, 850, 1618],
        "Learned-v5 versus flat: paired p=0.0010; paper-clustered 95% CI [0.2250, 0.4867]. Free-form answer F1 is not a stance metric and is omitted.",
    )
    add_body(doc, "The OOD result shows that ancestor context can rescue rationale evidence without large token growth on short abstracts. It does not establish superiority over static expansion, nor does it validate scientific-document QA outside QASPER.")

    add_heading(doc, "E. Full-Ladder Offline Run with RAPTOR and LongRAG Baselines", 2)
    add_body(doc, "To compare the complete ladder under identical conditions, all eight systems (B0-B6 plus the prior-gated proposal) were executed over 30 QASPER development questions from 10 papers (334 child leaves) with dependency-free backends: BM25 retrieval, lexical rerank, extractive generation, and token-overlap verification. No neural weights or LLM keys were used, so absolute scores understate production quality; the run isolates retrieval granularity and hierarchy control, which is exactly the axis RAPTOR and LongRAG stress.")
    add_table(
        doc,
        "Table V\nOffline full-ladder run with RAPTOR (B5) and LongRAG (B6) (30 QASPER dev questions, lexical backends)",
        ["System", "Rec@5", "Cit. F1", "Ans. F1", "Tokens"],
        [
            ["B0 BM25", "0.2061", "0.0919", "0.0822", "1363.1"],
            ["B1 dense*", "0.2061", "0.0919", "0.0822", "1363.1"],
            ["B2 hybrid", "0.2061", "0.0640", "0.0732", "3387.0"],
            ["B3 flat", "0.2246", "0.0924", "0.0873", "1511.3"],
            ["B4 static", "0.2246", "0.0663", "0.0844", "2982.4"],
            ["B5 RAPTOR", "0.3080", "0.1217", "0.0931", "1528.0"],
            ["B6 LongRAG", "0.2913", "0.0730", "0.0894", "2852.6"],
            ["Prior", "0.2246", "0.0663", "0.0844", "2982.4"],
        ],
        [990, 820, 820, 820, 1518],
        "B5 versus B3 citation F1: paired p=0.3786; paper-clustered 95% CI [-0.0303, 0.0871]. B6 versus B3: p=0.6244; CI [-0.1038, 0.0569]. *B1 reuses the BM25 stand-in offline, hence identical to B0.",
    )
    add_body(doc, "RAPTOR-style collapsed-tree retrieval achieved the highest recall@5 (0.3080) and citation F1 (0.1217) at flat-like token cost, consistent with its multi-level abstraction objective [4]. LongRAG-style long units improved recall@5 to 0.2913 but lowered citation F1 to 0.0730 while nearly doubling context tokens, reproducing the recall/precision trade-off reported for long units [5]. Neither interval excludes zero, so the run is comparative evidence rather than a significance claim; a neural-backend rerun on a frozen manifest is required before any superiority statement.")

    add_heading(doc, "VIII. ANALYSIS AND DISCUSSION")
    add_heading(doc, "A. What the Current Evidence Supports", 2)
    add_body(doc, "The implemented system makes attribution drift observable. Every result stores retrieved leaves, candidate leaves, verified leaves, rescued gold leaves, harmful drift, kept-correct and kept-wrong evidence, raw generation validation errors, and per-candidate verification decisions. The strongest defensible claim is therefore methodological: hierarchical expansion can rescue missing evidence but can also introduce attribution drift, and a risk-constrained gate provides a measurable framework for controlling this trade-off.")
    add_heading(doc, "B. What the Current Evidence Does Not Support", 2)
    add_body(doc, "Current evidence does not show that the learned gate beats flat or static hierarchy on QASPER. The v7 sample is too small, the 20-paper run is diagnostic, the historical v5 checkpoints predate final contracts, and the existing frozen QASPER test was affected by generation-format failures. No statement of state-of-the-art performance, universal retrieval improvement, or publication readiness is warranted.")
    add_heading(doc, "C. Failure Modes", 2)
    add_body(doc, "The principal observed failures are invalid citation formatting, empty generation, verifier rejection of all claims, gold-paragraph-to-leaf mapping mismatch, candidate-selection miss, harmful sibling drift, and excess context. Expansion can improve evidence recall while reducing answer quality, as seen in the v7 smoke test. This separation motivates joint reporting of evidence, answer, and efficiency metrics rather than a single aggregate score.")

    add_heading(doc, "IX. LIMITATIONS")
    add_body(doc, "The current report has six limitations. First, the only fully corrected QASPER v7 results contain two questions. Second, historical learned policies were trained before the strict citation contract and must be regenerated. Third, the rule-based query classifier may misassign expansion ceilings. Fourth, the NLI threshold and lexical rescue require development-only calibration and broader adversarial testing. Fifth, Docling page provenance is unavailable for QASPER JSON, so page metrics are meaningful only for PDF corpora. Sixth, external LLM APIs introduce nondeterminism, provider dependence, latency, and cost.")
    add_body(doc, "A separate implementation concern is scope contamination from the Multi-Vector project. This paper mitigates it by treating candidate retrieval as an interchangeable dependency and excluding representation-specific fusion from novelty. A future code cleanup should enforce that boundary through a single retriever interface and independent experiment configuration.")

    add_heading(doc, "X. ETHICS AND RESPONSIBLE USE")
    add_body(doc, "Scientific QA outputs can influence research decisions. The system must not present verified citations as proof of scientific truth: the verifier estimates textual support within supplied documents and does not assess study quality, external validity, or replication. Abstention, source display, claim-level evidence, and preserved raw failures are therefore safety requirements. Dataset and API licenses, document privacy, and provider data policies must be reviewed before deployment.")

    add_heading(doc, "XI. REPRODUCIBILITY ROADMAP")
    add_body(doc, "Before submission, the pipeline should complete the following preregistered sequence: freeze the final evaluator and provider contracts; regenerate paper-disjoint train and development rollouts; train independent parent and section gates; select estimators through group cross-validation and thresholds only on development; lock a new untouched QASPER manifest before inference; run all baselines and ablations on identical questions and generator settings; report clustered intervals, paired tests, effect sizes, multiple-comparison correction, tokens, latency, and cost; then perform generator-swap and OOD analyses without tuning on frozen outcomes.")

    add_heading(doc, "XII. CONCLUSION")
    add_body(doc, "This paper introduced an attribution-risk-constrained framework for adaptive hierarchical retrieval in scientific document QA. The method expands context only when a query-conditioned ancestor is expected to improve downstream utility, preserves leaf provenance, enforces request-specific citation contracts, and verifies atomic claims against descendant passages. The current repository is internally consistent under 77 tests and provides an evaluator that separates paragraph evidence from leaf attribution. Preliminary experiments reveal both evidence rescue and the cost of broader context, but they do not establish learned-policy superiority. The appropriate next contribution is a rigorously frozen, sufficiently powered evaluation of whether risk-aware gating controls attribution drift better than flat and static alternatives.")

    add_heading(doc, "REFERENCES")
    references = [
        "P. Lewis et al., 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks,' in Advances in Neural Information Processing Systems, vol. 33, 2020.",
        "P. Dasigi, K. Lo, I. Beltagy, A. Cohan, N. A. Smith, and M. Gardner, 'A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers,' in Proc. NAACL-HLT, pp. 4599-4610, 2021, doi: 10.18653/v1/2021.naacl-main.365.",
        "D. Wadden et al., 'Fact or Fiction: Verifying Scientific Claims,' in Proc. EMNLP, pp. 7534-7550, 2020, doi: 10.18653/v1/2020.emnlp-main.609.",
        "P. Sarthi, S. Abdullah, A. Tuli, S. Khanna, A. Goldie, and C. D. Manning, 'RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval,' in Proc. ICLR, 2024.",
        "Z. Jiang, X. Ma, and W. Chen, 'LongRAG: Enhancing Retrieval-Augmented Generation with Long-Context LLMs,' arXiv:2406.15319, 2024.",
        "J. Buchmann, X. Liu, and I. Gurevych, 'Attribute or Abstain: Large Language Models as Long Document Assistants,' in Proc. EMNLP, pp. 8113-8140, 2024, doi: 10.18653/v1/2024.emnlp-main.463.",
        "C. Auer et al., 'Docling Technical Report,' arXiv:2408.09869, 2024.",
        "P. He, J. Gao, and W. Chen, 'DeBERTaV3: Improving DeBERTa Using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing,' arXiv:2111.09543, 2021.",
        "Allen Institute for AI, 'QASPER LED Baseline and Official Evaluator,' GitHub repository. [Online]. Available: https://github.com/allenai/qasper-led-baseline. Accessed: Aug. 31, 2026.",
        "OpenAI, 'GPT-4o mini model snapshot: gpt-4o-mini-2024-07-18,' model used in the v7 smoke protocol. Provider behavior and availability are version-dependent.",
        "T. Gao, H. Yen, J. Yu, and D. Chen, 'Enabling Large Language Models to Generate Text with Citations,' in Proc. EMNLP, pp. 6465-6488, 2023, doi: 10.18653/v1/2023.emnlp-main.398.",
        "B. Bohnet et al., 'Attributed Question Answering: Evaluation and Modeling for Attributed Large Language Models,' arXiv:2212.08037, 2022.",
        "A. Asai, Z. Wu, Y. Wang, A. Sil, and H. Hajishirzi, 'Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection,' in Proc. ICLR, 2024.",
        "S. Jeong, J. Baek, S. Cho, S. J. Hwang, and J. C. Park, 'Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity,' in Proc. NAACL-HLT, pp. 7036-7050, 2024, doi: 10.18653/v1/2024.naacl-long.389.",
        "S.-Q. Yan, J.-C. Gu, Y. Zhu, and Z.-H. Ling, 'Corrective Retrieval Augmented Generation,' arXiv:2401.15884, 2024.",
        "D. Edge et al., 'From Local to Global: A Graph RAG Approach to Query-Focused Summarization,' arXiv:2404.16130, 2024.",
        "V. Karpukhin et al., 'Dense Passage Retrieval for Open-Domain Question Answering,' in Proc. EMNLP, pp. 6769-6781, 2020, doi: 10.18653/v1/2020.emnlp-main.550.",
        "S. Robertson and H. Zaragoza, 'The Probabilistic Relevance Framework: BM25 and Beyond,' Foundations and Trends in Information Retrieval, vol. 3, no. 4, pp. 333-389, 2009.",
        "R. Nogueira, Z. Jiang, R. Pradeep, and J. Lin, 'Document Ranking with a Pretrained Sequence-to-Sequence Model,' arXiv:1903.06737, 2019.",
        "N. Reimers and I. Gurevych, 'Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks,' in Proc. EMNLP-IJCNLP, pp. 3982-3992, 2019, doi: 10.18653/v1/D19-1410.",
        "J. Chen et al., 'BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings through Self-Knowledge Distillation,' arXiv:2402.03216, 2024.",
        "O. Khattab and M. Zaharia, 'ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT,' in Proc. SIGIR, pp. 39-48, 2020, doi: 10.1145/3397271.3401075.",
        "H. Rashkin et al., 'Measuring Attribution in Natural Language Generation Models,' arXiv:2112.12870, 2021.",
        "O. Honovich et al., 'TRUE: Re-evaluating Factual Consistency Evaluation,' in Proc. NAACL-HLT, pp. 3905-3920, 2022, doi: 10.18653/v1/2022.naacl-main.287.",
        "N. F. Liu et al., 'Lost in the Middle: How Language Models Use Long Contexts,' arXiv:2307.03172, 2023.",
        "A. Mallen, A. Asai, V. Zhong, R. Das, D. Khashabi, and H. Hajishirzi, 'When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories,' in Proc. EACL, pp. 980-990, 2023.",
        "Z. Shao et al., 'Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy,' in Proc. EMNLP (Findings), pp. 9248-9274, 2023.",
        "S. Yao et al., 'ReAct: Synergizing Reasoning and Acting in Language Models,' in Proc. ICLR, 2023.",
        "T. Kwiatkowski et al., 'Natural Questions: A Benchmark for Question Answering Research,' Trans. Assoc. Comput. Linguistics, vol. 7, pp. 453-466, 2019, doi: 10.1162/tacl_a_00276.",
        "Z. Yang et al., 'HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering,' in Proc. EMNLP, pp. 2369-2380, 2018, doi: 10.18653/v1/D18-1412.",
        "I. Stelmakh, Y. Luan, B. Dhingra, and M.-W. Chang, 'ASQA: Factoid Questions Meet Long-Form Answers,' in Proc. EMNLP, pp. 8273-8288, 2022, doi: 10.18653/v1/2022.emnlp-main.566.",
        "A. Fan, Y. Jernite, E. Perez, D. Grangier, J. Weston, and M. Auli, 'ELI5: Long Form Question Answering,' in Proc. ACL, pp. 3558-3567, 2019, doi: 10.18653/v1/P19-1346.",
        "V. A. Traag, L. Waltman, and N. J. van Eck, 'From Louvain to Leiden: Guaranteeing Well-Connected Communities,' Scientific Reports, vol. 9, art. 5233, 2019, doi: 10.1038/s41598-019-41695-z.",
        "Z. Jiang, F. Xu, L. Araki, and G. Neumann, 'FLARE: Forward-Looking Active Retrieval Augmented Generation,' in Proc. EMNLP, pp. 6280-6293, 2023, doi: 10.18653/v1/2023.emnlp-main.390.",
        "J. Thorne, A. Vlachos, C. Christodoulopoulos, and A. Mittal, 'FEVER: A Large-Scale Dataset for Fact Extraction and VERification,' in Proc. NAACL-HLT, pp. 809-819, 2018, doi: 10.18653/v1/N18-1074.",
    ]
    for i, ref in enumerate(references, 1):
        add_reference(doc, i, ref)

    props = doc.core_properties
    props.title = "Attribution-Risk-Constrained Adaptive Hierarchical Retrieval for Verifiable Scientific Document Question Answering"
    props.subject = "IEEE-style preliminary research paper"
    props.author = "Anonymous Author(s)"
    props.keywords = "scientific QA; hierarchical retrieval; attribution risk; QASPER"
    props.comments = "Generated from repository code and frozen diagnostic artifacts; preliminary and not submission-ready."

    doc.save(OUT_PATH)
    print(OUT_PATH)


if __name__ == "__main__":
    build()
