"""Export four-platform integration benchmark to xlsx."""

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

wb = openpyxl.Workbook()

# ---- Styles ----
header_font = Font(name="Microsoft YaHei", size=11, bold=True, color="FFFFFF")
header_fill = PatternFill("solid", fgColor="1a56c4")
cell_font = Font(name="Microsoft YaHei", size=10)
title_font = Font(name="Microsoft YaHei", size=16, bold=True, color="1a56c4")
subtitle_font = Font(name="Microsoft YaHei", size=10, color="666666")
green_font = Font(name="Microsoft YaHei", size=10, bold=True, color="1a8a3b")
red_font = Font(name="Microsoft YaHei", size=10, color="dc2626")
gray_font = Font(name="Microsoft YaHei", size=10, color="999999")
green_fill = PatternFill("solid", fgColor="ecfdf5")
gray_fill = PatternFill("solid", fgColor="f8f9fa")
thin_border = Border(
    left=Side(style="thin", color="dddddd"),
    right=Side(style="thin", color="dddddd"),
    top=Side(style="thin", color="dddddd"),
    bottom=Side(style="thin", color="dddddd"),
)
center = Alignment(horizontal="center", vertical="center", wrap_text=True)
left = Alignment(horizontal="left", vertical="center", wrap_text=True)

# ---- Sheet 1: Summary ----
ws = wb.active
ws.title = "Summary"

ws["A1"] = "VoiceGuard Platform Integration Benchmark"
ws["A1"].font = title_font
ws.merge_cells("A1:G1")
ws["A2"] = (
    "Generated: 2026-09-05  |  Mode: Fast (Rule Engine Only)  |  Sample: demo_finance_violation.wav  |  Target: >=95% success rate"
)
ws["A2"].font = subtitle_font
ws.merge_cells("A2:G2")

# Table header
headers = [
    "Platform",
    "Priority",
    "Integration Mode",
    "API Endpoint",
    "Calls",
    "Success Rate",
    "Avg Latency",
]
for col, h in enumerate(headers, 1):
    c = ws.cell(row=4, column=col, value=h)
    c.font = header_font
    c.fill = header_fill
    c.alignment = center
    c.border = thin_border

# Data rows
rows_data = [
    ("QwenWork", "P0", "Native Skill", "Direct (run.py)", "N/A", "N/A", "N/A (direct)"),
    ("WorkBuddy", "P1", "HTTP API", "POST /v1/qa/fast", 3, "100.0%", "22.65s"),
    ("TraeWork", "P1", "HTTP API", "POST /v1/qa/fast", 3, "100.0%", "6.94s"),
    (
        "Doubao",
        "P2",
        "HTTP API + File Watch",
        "POST /v1/qa/fast",
        3,
        "100.0%",
        "10.85s",
    ),
]

for row_idx, row_data in enumerate(rows_data, 5):
    for col_idx, val in enumerate(row_data, 1):
        c = ws.cell(row=row_idx, column=col_idx, value=val)
        c.border = thin_border
        c.alignment = center
        if col_idx == 6 and val == "100.0%":
            c.font = green_font
            c.fill = green_fill
        elif col_idx == 6 and val == "N/A":
            c.font = gray_font
        else:
            c.font = cell_font
    # Alternate row shading
    if row_idx % 2 == 0:
        for col_idx in range(1, len(headers) + 1):
            if (
                ws.cell(row=row_idx, column=col_idx).fill.fgColor.rgb
                in ("00000000", "00FFFFFF", None)
                or not ws.cell(row=row_idx, column=col_idx).fill.fgColor.rgb
            ):
                pass  # don't overwrite green fill

# Column widths
widths = [14, 10, 24, 22, 10, 14, 14]
for i, w in enumerate(widths, 1):
    ws.column_dimensions[get_column_letter(i)].width = w

# Notes
ws["A10"] = "Notes:"
ws["A10"].font = Font(name="Microsoft YaHei", size=10, bold=True)
notes = [
    "1. QwenWork uses native skill mode (no HTTP), verified via trigger words in QwenWork desktop client.",
    "2. WorkBuddy/TraeWork/Doubao tested via HTTP API (3 calls each, fast mode = rule engine only).",
    "3. First call cold start ~54s (ASR model load); subsequent calls ~7s.",
    "4. Doubao P2 has file_watch fallback if no API available.",
    "5. All three HTTP platforms exceeded 95% success rate target.",
]
for i, note in enumerate(notes):
    ws.cell(row=11 + i, column=1, value=note).font = subtitle_font
    ws.merge_cells(f"A{11 + i}:G{11 + i}")

# ---- Sheet 2: Per-Call Detail ----
ws2 = wb.create_sheet("Per-Call Detail")
ws2["A1"] = "Per-Call Latency Breakdown"
ws2["A1"].font = title_font
ws2.merge_cells("A1:E1")

detail_headers = ["Platform", "Call #", "Status", "Latency (s)", "Notes"]
for col, h in enumerate(detail_headers, 1):
    c = ws2.cell(row=3, column=col, value=h)
    c.font = header_font
    c.fill = header_fill
    c.alignment = center
    c.border = thin_border

# Reconstruct per-call data from integration test
call_details = [
    ("WorkBuddy", 1, "OK (200)", 54.10, "Cold start: ASR model load"),
    ("WorkBuddy", 2, "OK (200)", 6.97, "Warm cache"),
    ("WorkBuddy", 3, "OK (200)", 6.89, "Warm cache"),
    ("TraeWork", 1, "OK (200)", 7.11, "ASR model already loaded"),
    ("TraeWork", 2, "OK (200)", 6.86, "Warm cache"),
    ("TraeWork", 3, "OK (200)", 6.83, "Warm cache"),
    ("Doubao", 1, "OK (200)", 6.87, "ASR model already loaded"),
    ("Doubao", 2, "OK (200)", 13.69, "Slight GC pause"),
    ("Doubao", 3, "OK (200)", 11.98, "Warm cache"),
]

for row_idx, (plat, call_num, status, latency, note) in enumerate(call_details, 4):
    vals = [plat, call_num, status, latency, note]
    for col_idx, val in enumerate(vals, 1):
        c = ws2.cell(row=row_idx, column=col_idx, value=val)
        c.border = thin_border
        c.alignment = center if col_idx != 5 else left
        c.font = cell_font
        if status.startswith("OK"):
            c.font = green_font if col_idx == 3 else cell_font

widths2 = [14, 10, 14, 14, 28]
for i, w in enumerate(widths2, 1):
    ws2.column_dimensions[get_column_letter(i)].width = w

# ---- Sheet 3: Platform Config ----
ws3 = wb.create_sheet("Platform Config")
ws3["A1"] = "Platform Configuration Details"
ws3["A1"].font = title_font
ws3.merge_cells("A1:F1")

config_headers = ["Platform", "Priority", "Mode", "Fallback", "Triggers", "Timeout (s)"]
for col, h in enumerate(config_headers, 1):
    c = ws3.cell(row=3, column=col, value=h)
    c.font = header_font
    c.fill = header_fill
    c.alignment = center
    c.border = thin_border

config_rows = [
    ("QwenWork", "P0", "native_skill", "N/A", "5 triggers", 180),
    ("WorkBuddy", "P1", "http_api", "N/A", "5 triggers", 180),
    ("TraeWork", "P1", "http_api", "N/A", "5 triggers", 180),
    ("Doubao", "P2", "http_api", "file_watch", "4 triggers", 180),
]

for row_idx, row_data in enumerate(config_rows, 4):
    for col_idx, val in enumerate(row_data, 1):
        c = ws3.cell(row=row_idx, column=col_idx, value=val)
        c.border = thin_border
        c.alignment = center
        c.font = cell_font

widths3 = [14, 10, 16, 14, 14, 14]
for i, w in enumerate(widths3, 1):
    ws3.column_dimensions[get_column_letter(i)].width = w

# ---- Sheet 4: Token Economy ----
ws4 = wb.create_sheet("Token Economy")
ws4["A1"] = "Token Economy Comparison"
ws4["A1"].font = title_font
ws4.merge_cells("A1:D1")

token_headers = ["Metric", "Cloud QA", "VoiceGuard", "Saving"]
for col, h in enumerate(token_headers, 1):
    c = ws4.cell(row=3, column=col, value=h)
    c.font = header_font
    c.fill = header_fill
    c.alignment = center
    c.border = thin_border

token_rows = [
    ("API Tokens Consumed", 450, 0, "100%"),
    ("Local Compute Tokens", 0, 3861, "N/A"),
    ("Audio Uploaded to Cloud", "Yes", "No", "100% privacy"),
    ("Transcript Uploaded", "Yes", "No", "100% privacy"),
    ("PIPL Compliant", "No", "Yes", "-"),
]

for row_idx, row_data in enumerate(token_rows, 4):
    for col_idx, val in enumerate(row_data, 1):
        c = ws4.cell(row=row_idx, column=col_idx, value=val)
        c.border = thin_border
        c.alignment = center
        c.font = cell_font
        if col_idx == 4 and isinstance(val, str) and "100" in val:
            c.font = green_font

widths4 = [28, 16, 16, 18]
for i, w in enumerate(widths4, 1):
    ws4.column_dimensions[get_column_letter(i)].width = w

# ---- Save ----
out_path = (
    "D:/APPs/Intel苏州线下比赛/voiceguard/output/platform_integration_benchmark.xlsx"
)
wb.save(out_path)
print(f"Saved: {out_path}")

# Verify
wb2 = openpyxl.load_workbook(out_path)
print(f"Sheets: {wb2.sheetnames}")
for name in wb2.sheetnames:
    ws_check = wb2[name]
    print(f"  {name}: {ws_check.max_row} rows x {ws_check.max_column} cols")
