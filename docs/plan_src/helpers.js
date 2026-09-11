const d = require("docx");
const {
  Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType,
  ShadingType, AlignmentType, BorderStyle, LevelFormat, PositionalTab,
  PositionalTabAlignment, PositionalTabLeader,
} = d;

const INK = "1A1A1A";
const MUTED = "5A5A5A";
const ACCENT = "1F4E79";
const NEW = "8A4B08";
const HEADER_BG = "EEF2F6";
const NEW_BG = "FDF3E7";

const PAGE_W = 12240, PAGE_H = 15840, MARGIN = 1080;
const CONTENT_W = PAGE_W - 2 * MARGIN;

function p(text, opts = {}) {
  const runs = Array.isArray(text) ? text : [new TextRun({ text, ...(opts.run || {}) })];
  return new Paragraph({
    children: runs,
    spacing: { after: opts.after ?? 140, line: opts.line ?? 276 },
    alignment: opts.alignment,
    ...(opts.style ? { style: opts.style } : {}),
    ...(opts.border ? { border: opts.border } : {}),
  });
}

function t(text, o = {}) {
  return new TextRun({ text, color: o.color || INK, bold: o.bold, italics: o.italics,
    size: o.size || 21, font: o.font || "Calibri", break: o.break });
}
function mono(text, o = {}) {
  return new TextRun({ text, font: "Consolas", size: o.size || 18, color: o.color || INK, bold: o.bold });
}

function h(text, level, o = {}) {
  const sizes = { 1: 32, 2: 26, 3: 22 };
  return new Paragraph({
    heading: level === 1 ? HeadingLevel.HEADING_1 : level === 2 ? HeadingLevel.HEADING_2 : HeadingLevel.HEADING_3,
    spacing: { before: level === 1 ? 360 : 280, after: level === 1 ? 160 : 120 },
    children: [new TextRun({ text, bold: true, size: sizes[level], color: o.color || ACCENT, font: "Calibri" })],
    ...(level === 1 ? { border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "C9D6E4", space: 6 } } } : {}),
  });
}

function bullet(text, o = {}) {
  const runs = Array.isArray(text) ? text : [t(text, o)];
  return new Paragraph({ children: runs, numbering: { reference: "bullets", level: o.level || 0 },
    spacing: { after: 80, line: 276 } });
}
function numbered(text, o = {}) {
  const runs = Array.isArray(text) ? text : [t(text, o)];
  return new Paragraph({ children: runs, numbering: { reference: "numbers", level: 0 },
    spacing: { after: 80, line: 276 } });
}

function cell(children, o = {}) {
  const kids = typeof children === "string"
    ? [new Paragraph({ children: [t(children, { bold: o.bold, size: o.size || 19, color: o.color })],
        spacing: { after: 40, line: 250 }, alignment: o.alignment })]
    : children;
  return new TableCell({
    children: kids,
    width: { size: o.width, type: WidthType.DXA },
    shading: o.bg ? { type: ShadingType.CLEAR, fill: o.bg, color: "auto" } : undefined,
    margins: { top: 70, bottom: 70, left: 110, right: 110 },
  });
}

function table(widths, rows, o = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const scaled = widths.map((w) => Math.round((w / total) * CONTENT_W));
  return new Table({
    columnWidths: scaled,
    width: { size: CONTENT_W, type: WidthType.DXA },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4, color: "C9D6E4" },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: "C9D6E4" },
      left: { style: BorderStyle.SINGLE, size: 4, color: "C9D6E4" },
      right: { style: BorderStyle.SINGLE, size: 4, color: "C9D6E4" },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: "DDE5EE" },
      insideVertical: { style: BorderStyle.SINGLE, size: 2, color: "DDE5EE" },
    },
    rows: rows.map((r, i) =>
      new TableRow({
        tableHeader: i === 0 && o.header !== false,
        children: r.map((c, j) => {
          const isConf = typeof c === "object" && !Array.isArray(c) &&
            (c.text !== undefined || c.children !== undefined);
          const conf = isConf ? c : { text: c };
          return cell(conf.children || conf.text, {
            width: scaled[j],
            bold: i === 0 && o.header !== false ? true : conf.bold,
            bg: i === 0 && o.header !== false ? HEADER_BG : conf.bg,
            color: conf.color, alignment: conf.alignment, size: conf.size,
          });
        }),
      })
    ),
  });
}

function callout(title, lines, o = {}) {
  const kids = [
    new Paragraph({ children: [t(title, { bold: true, color: o.color || NEW, size: 20 })],
      spacing: { after: 60, line: 260 } }),
    ...lines.map((l) => new Paragraph({
      children: Array.isArray(l) ? l : [t(l, { size: 20 })],
      spacing: { after: 60, line: 260 } })),
  ];
  return table([100], [[{ children: kids, bg: o.bg || NEW_BG }]], { header: false });
}

const numbering = {
  config: [
    { reference: "bullets", levels: [
      { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 400, hanging: 200 } } } },
      { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 760, hanging: 200 } } } },
    ]},
    { reference: "numbers", levels: [
      { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 400, hanging: 220 } } } },
    ]},
  ],
};

module.exports = { d, p, t, mono, h, bullet, numbered, table, cell, callout, numbering,
  INK, MUTED, ACCENT, NEW, HEADER_BG, NEW_BG, PAGE_W, PAGE_H, MARGIN, CONTENT_W };
