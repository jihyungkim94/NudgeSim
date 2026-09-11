const fs = require("fs");
const H = require("./helpers.js");
const d = H.d;

const children = [
  ...require("./part1.js"),
  ...require("./part2.js"),
  ...require("./part3.js").blocks,
  ...require("./part4.js"),
  ...require("./part5.js"),
];

const doc = new d.Document({
  numbering: H.numbering,
  styles: {
    default: {
      document: { run: { font: "Calibri", size: 21, color: H.INK } },
    },
  },
  sections: [{
    properties: {
      page: {
        size: { width: H.PAGE_W, height: H.PAGE_H },
        margin: { top: H.MARGIN, right: H.MARGIN, bottom: H.MARGIN, left: H.MARGIN },
      },
    },
    footers: {
      default: new d.Footer({
        children: [new d.Paragraph({
          alignment: d.AlignmentType.CENTER,
          children: [new d.TextRun({
            children: ["NudgeSim Project Plan · v2.4 · September 2026 · page ", d.PageNumber.CURRENT, " of ", d.PageNumber.TOTAL_PAGES],
            size: 16, color: H.MUTED,
          })],
        })],
      }),
    },
    children,
  }],
});

d.Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(process.argv[2] || "Project_Plan_NudgeSim_v2.4.docx", buf);
  console.log("written:", process.argv[2], buf.length, "bytes");
});
