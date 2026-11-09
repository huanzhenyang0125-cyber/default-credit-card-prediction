from pathlib import Path
import re
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether

root = Path(__file__).resolve().parents[1]
out = root / 'reports/Credit_Risk_Research_Report.pdf'
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='ReportTitle', parent=styles['Title'], fontName='Helvetica-Bold', fontSize=21, leading=25, textColor=colors.HexColor('#122c3b'), spaceAfter=12))
styles.add(ParagraphStyle(name='ReportBody', parent=styles['BodyText'], fontSize=9, leading=13.3, spaceAfter=7))
styles.add(ParagraphStyle(name='ReportCaption', parent=styles['BodyText'], fontSize=8, leading=11, textColor=colors.HexColor('#435c69'), spaceAfter=9))
styles['Heading1'].fontSize=14; styles['Heading1'].leading=18; styles['Heading1'].spaceBefore=14
styles['Heading2'].fontSize=11; styles['Heading2'].leading=14; styles['Heading2'].spaceBefore=10

def footer(canvas, doc):
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.HexColor('#627786'))
    canvas.drawString(20*mm, 13*mm, 'Credit Risk Decision Lab | Research portfolio')
    canvas.drawRightString(190*mm, 13*mm, str(doc.page))

def img(name, caption, width=165*mm):
    path = root / 'figures' / name
    image = Image(str(path)); image.drawHeight = image.imageHeight / image.imageWidth * width; image.drawWidth = width
    return KeepTogether([image, Paragraph(escape(caption), styles['ReportCaption'])])

def comparison():
    data = pd.read_csv(root / 'reports/model_comparison.csv')
    rows = [['Validation model', 'ROC-AUC', 'Average precision', 'Brier']]
    for _, r in data.iterrows(): rows.append([r['model'], f"{r.roc_auc:.4f}", f"{r.average_precision:.4f}", f"{r.brier:.4f}"])
    table = Table(rows, colWidths=[70*mm, 30*mm, 38*mm, 27*mm], repeatRows=1)
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eaf1f3')),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#9cafb8'))]))
    return KeepTogether([table, Spacer(1,4*mm), Paragraph('Validation-set candidate comparison. The calibrated forest did not improve Brier score in this experiment.', styles['ReportCaption'])])

text = (root / 'reports/credit_risk_research_report.md').read_text(encoding='utf-8')
story=[]
for block in re.split(r'\n\s*\n', text):
    block=block.strip()
    if not block: continue
    block = block.replace('–','-').replace('—','-').replace('−','-').replace('’',"'").replace('“','"').replace('”','"')
    lines=block.splitlines()
    if block.startswith('# '): story.append(Paragraph(escape(lines[0][2:]),styles['ReportTitle'])); continue
    if block.startswith('## '):
        title=lines[0][3:]
        story.append(Paragraph(escape(title),styles['Heading1']))
        if title == '4. Validation results': story.append(comparison())
        if title == '5. Held-out evaluation and decision analysis':
            story.append(img('report_discrimination.png','Figure 1. Final fixed forest: held-out ROC and precision-recall curves. Same-dataset test; prior full-sample exploration limits independence.'))
            story.append(img('report_capacity.png','Figure 2. Cumulative review coverage. The full curve is an exploratory post-evaluation diagnostic.'))
        continue
    if block.startswith('### '): story.append(Paragraph(escape(lines[0][4:]),styles['Heading2'])); continue
    if block.startswith('|'): continue
    clean=block.replace('**','').replace('`','').replace('*','').replace('\n',' ')
    story.append(Paragraph(escape(clean),styles['ReportBody']))

doc=SimpleDocTemplate(str(out), pagesize=(210*mm,297*mm), leftMargin=22*mm, rightMargin=22*mm, topMargin=20*mm, bottomMargin=20*mm, title='Credit Risk Decision Lab')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(f'Created: {out}')
