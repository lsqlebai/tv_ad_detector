#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

from openpyxl import Workbook
from openpyxl.drawing.image import Image
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Font, PatternFill, Side, Border
from openpyxl.worksheet.datavalidation import DataValidation


HEADERS = [
    "delete",
    "file",
    "start",
    "end",
    "start_seconds",
    "end_seconds",
    "score",
    "kind",
    "review_required",
    "sources",
    "start_before_frame",
    "start_frame",
    "middle_frame",
    "end_frame",
    "end_after_frame",
]

FRAME_COLUMNS = [
    ("start_before_frame", "K"),
    ("start_frame", "L"),
    ("middle_frame", "M"),
    ("end_frame", "N"),
    ("end_after_frame", "O"),
]


def format_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining = seconds - hours * 3600 - minutes * 60
    if hours:
        return f"{hours}:{minutes:02d}:{remaining:06.3f}"
    return f"{minutes}:{remaining:06.3f}"


def display_time(row: dict, name: str) -> str:
    seconds = row.get(f"{name}_seconds")
    if seconds not in (None, ""):
        return format_time(float(seconds))
    return row.get(name, "")


def read_candidates(output_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for csv_path in sorted(output_dir.glob("*.ads.csv")):
        video_stem = csv_path.name[: -len(".ads.csv")]
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for index, row in enumerate(reader, start=1):
                row["file"] = f"{video_stem}.mp4"
                row["candidate_id"] = f"{video_stem}:{index:03d}"
                row["delete"] = "NO" if row.get("review_required") == "yes" else "YES"
                rows.append(row)
    return rows


def resolve_image_path(workbook_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return workbook_dir / path


def add_image_or_label(ws, workbook_dir: Path, value: str, cell: str, width: int, height: int) -> None:
    if value == "__VIDEO_END__":
        target = ws[cell]
        target.value = "视频结束"
        target.alignment = Alignment(horizontal="center", vertical="center")
        target.font = Font(color="666666")
        return
    image_path = resolve_image_path(workbook_dir, value)
    if not image_path.is_file():
        return
    image = Image(str(image_path))
    image.width = width
    image.height = height
    ws.add_image(image, cell)


def build_workbook(rows: list[dict], output_dir: Path, xlsx_path: Path) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "Review"

    title_fill = PatternFill("solid", fgColor="1F4E78")
    header_fill = PatternFill("solid", fgColor="D9EAF7")
    trusted_fill = PatternFill("solid", fgColor="E2F0D9")
    review_fill = PatternFill("solid", fgColor="FFF2CC")
    border = Border(bottom=Side(style="thin", color="B7B7B7"))

    ws["A1"] = "Ad Candidate Review"
    ws["A1"].font = Font(bold=True, color="FFFFFF", size=14)
    ws["A1"].fill = title_fill
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(HEADERS))
    ws["A2"] = "Set delete=YES for ranges to remove. Rows with review_required=YES are heuristic candidates and need manual confirmation."
    ws["A2"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=len(HEADERS))

    for column, header in enumerate(HEADERS, start=1):
        cell = ws.cell(row=4, column=column, value=header)
        cell.font = Font(bold=True)
        cell.fill = header_fill
        cell.border = border
        cell.alignment = Alignment(horizontal="center", vertical="center")

    dv = DataValidation(type="list", formula1='"YES,NO"', allow_blank=False)
    ws.add_data_validation(dv)

    for row_index, row in enumerate(rows, start=5):
        values = [
            row.get("delete", "NO"),
            row.get("file", ""),
            display_time(row, "start"),
            display_time(row, "end"),
            float(row["start_seconds"]) if row.get("start_seconds") else None,
            float(row["end_seconds"]) if row.get("end_seconds") else None,
            float(row["score"]) if row.get("score") else None,
            row.get("kind", ""),
            row.get("review_required", ""),
            row.get("sources", ""),
            "",
            "",
            "",
            "",
            "",
        ]
        for column, value in enumerate(values, start=1):
            cell = ws.cell(row=row_index, column=column, value=value)
            cell.alignment = Alignment(wrap_text=True, vertical="center")
            cell.border = border
        dv.add(ws.cell(row=row_index, column=1))

        for frame_name, column_letter in FRAME_COLUMNS:
            add_image_or_label(ws, output_dir.parent, row.get(frame_name, ""), f"{column_letter}{row_index}", 180, 101)
        ws.row_dimensions[row_index].height = 84

    last_row = max(5, len(rows) + 4)
    ws.auto_filter.ref = f"A4:O{last_row}"
    ws.freeze_panes = "A5"
    ws.column_dimensions["A"].width = 10
    ws.column_dimensions["B"].width = 26
    ws.column_dimensions["C"].width = 10
    ws.column_dimensions["D"].width = 10
    ws.column_dimensions["E"].width = 10
    ws.column_dimensions["F"].width = 10
    ws.column_dimensions["G"].width = 10
    ws.column_dimensions["H"].width = 20
    ws.column_dimensions["I"].width = 16
    ws.column_dimensions["J"].width = 42
    ws.column_dimensions["K"].width = 26
    ws.column_dimensions["L"].width = 26
    ws.column_dimensions["M"].width = 26
    ws.column_dimensions["N"].width = 26
    ws.column_dimensions["O"].width = 26
    ws.column_dimensions["E"].hidden = True
    ws.column_dimensions["F"].hidden = True

    ws.conditional_formatting.add(
        f"A5:O{last_row}",
        FormulaRule(formula=["$A5=\"YES\""], fill=trusted_fill),
    )
    ws.conditional_formatting.add(
        f"A5:O{last_row}",
        FormulaRule(formula=["$I5=\"yes\""], fill=review_fill),
    )

    summary = wb.create_sheet("Summary")
    summary["A1"] = "Metric"
    summary["B1"] = "Value"
    summary["A2"] = "Candidate rows"
    summary["B2"] = len(rows)
    summary["A3"] = "Default delete=YES"
    summary["B3"] = sum(1 for row in rows if row.get("delete") == "YES")
    summary["A4"] = "Needs review"
    summary["B4"] = sum(1 for row in rows if row.get("review_required") == "yes")
    for cell in summary[1]:
        cell.font = Font(bold=True)
        cell.fill = header_fill
    summary.column_dimensions["A"].width = 24
    summary.column_dimensions["B"].width = 16

    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = xlsx_path.with_suffix(".tmp.xlsx")
    wb.save(tmp_path)
    tmp_path.replace(xlsx_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an Excel workbook for reviewing ad candidates.")
    parser.add_argument("--output-dir", type=Path, default=Path("output/detect"))
    parser.add_argument("--xlsx", type=Path, default=Path("output/ad_review.xlsx"))
    args = parser.parse_args()

    rows = read_candidates(args.output_dir)
    if not rows:
        raise SystemExit(f"No *.ads.csv files found in {args.output_dir}")
    build_workbook(rows, args.output_dir, args.xlsx)
    print(f"Wrote review workbook: {args.xlsx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
