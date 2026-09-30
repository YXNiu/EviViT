// Rebuild vector assets from the released Table 1 summary and final-paper PNG.
// Run from any directory: node docs/assets/build-page-assets.mjs
// The numerical figure uses displayed paper values, not reconstructed test runs.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const assetDir = path.dirname(fileURLToPath(import.meta.url));
const repoDir = path.resolve(assetDir, '../..');
const summary = JSON.parse(fs.readFileSync(path.join(repoDir, 'results/paper_table1_local_pair_averages.json'), 'utf8'));
const rows = summary.paired_results;
if (rows.length !== 9 || new Set(rows.map(r => r.host)).size !== 9) throw new Error('Expected all nine host pairs.');
for (const row of rows) {
  if (Math.abs(row.evivit_average - row.base_average - row.gain) > 0.011) throw new Error(`Gain mismatch: ${row.host}`);
}
const esc = s => String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
const text = (x, y, value, size = 18, color = '#26384a', extra = '') => `<text x="${x}" y="${y}" font-size="${size}" fill="${color}" ${extra}>${esc(value)}</text>`;
const svg = (width, height, body, title, description) => `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img"><title>${esc(title)}</title><desc>${esc(description)}</desc><g font-family="Arial, Helvetica, sans-serif">${body}</g></svg>`;

let chart = '<rect width="1100" height="680" rx="18" fill="#ffffff"/>';
chart += text(30, 43, 'Fine-grained gains across all nine hosts', 27, '#18354a', 'font-weight="600"');
chart += text(30, 76, 'Matched local pairs · seven-benchmark mean · Table 1', 17, '#637386');
chart += text(30, 120, 'Host', 16, '#637386');
chart += text(310, 120, 'Mean accuracy (%)', 16, '#637386');
chart += text(565, 120, 'Paired gain (percentage points)', 16, '#637386');
const x0 = 565, span = 435, maxGain = 8, y0 = 160, step = 49;
for (let k = 0; k <= 8; k += 2) {
  const x = x0 + k / maxGain * span;
  chart += `<line x1="${x}" y1="139" x2="${x}" y2="583" stroke="#e5ebee" stroke-width="1"/>`;
  chart += text(x, 610, k, 15, '#637386', 'text-anchor="middle"');
}
rows.forEach((row, i) => {
  const y = y0 + i * step;
  if (i % 2 === 0) chart += `<rect x="18" y="${y - 22}" width="1064" height="43" rx="5" fill="#f7f9fb"/>`;
  chart += text(30, y + 6, row.host, 19);
  chart += text(310, y + 6, row.base_average.toFixed(2), 19, '#637386');
  chart += text(380, y + 6, '→', 19, '#637386');
  chart += text(420, y + 6, row.evivit_average.toFixed(2), 19, '#315c9b', 'font-weight="600"');
  chart += `<rect x="${x0}" y="${y - 13}" width="${row.gain / maxGain * span}" height="27" rx="4" fill="#72a77b"/>`;
  chart += text(x0 + row.gain / maxGain * span + 9, y + 6, `+${row.gain.toFixed(2)}`, 18, '#3e7950', 'font-weight="600"');
});
chart += text(30, 650, 'VP-E / VP-M / VP-H · V*Bench · HR-4K / HR-8K · full-image ZoomBench', 16, '#637386');
fs.writeFileSync(path.join(assetDir, 'paired-gains.svg'), svg(1100, 680, chart, 'EviViT paired mean gains', summary.average_definition));

let data = '<rect width="1200" height="400" rx="20" fill="#f6f9fc"/>';
data += text(34, 48, 'Human Search Traces', 30, '#18354a', 'font-weight="600"');
data += text(34, 82, '1,144 raw training sessions · question-conditioned evidence supervision', 19, '#637386');
const panels = [
  {x:34, color:'#315c9b', title:'Training question', lines:['VisualProbe source sample', 'Question + image identifier']},
  {x:434, color:'#825fb2', title:'Human interaction', lines:['Pointer / hover / zoom', 'Evidence-box operations']},
  {x:834, color:'#4b8857', title:'Released session', lines:['Ordered raw events', 'Final boxes + relative time']}
];
for (const p of panels) {
  data += `<rect x="${p.x}" y="124" width="332" height="155" rx="14" fill="#ffffff" stroke="#dde6ee"/>`;
  data += `<rect x="${p.x}" y="124" width="332" height="6" rx="3" fill="${p.color}"/>`;
  data += text(p.x+22, 170, p.title, 23, p.color, 'font-weight="600"');
  p.lines.forEach((line, i) => {data += text(p.x+22, 213+i*31, line, 18);});
}
data += '<path d="M378 200 H414 M405 192 L415 200 L405 208 M778 200 H814 M805 192 L815 200 L805 208" fill="none" stroke="#90a2b5" stroke-width="3"/>';
data += text(34, 324, 'De-identified annotations · no original images · no generated process text', 20, '#315c9b');
data += text(34, 363, 'Schematic workflow; event types may repeat or be absent.', 16, '#637386');
fs.writeFileSync(path.join(assetDir, 'dataset-overview.svg'), svg(1200, 400, data, 'Human Search Traces dataset overview', 'VisualProbe training questions are annotated through mouse interactions. The release contains raw session events and final boxes, not source images or generated text.'));

const teaser = fs.readFileSync(path.join(assetDir, 'teaser.png')).toString('base64');
let social = '<rect width="1280" height="640" fill="#ffffff"/><rect width="1280" height="9" fill="#315c9b"/>';
social += text(42, 77, 'EviViT', 54, '#18354a', 'font-weight="700"');
social += text(1238, 65, 'arXiv:2609.37123', 19, '#637386', 'text-anchor="end"');
social += text(42, 119, 'Evidence-Adaptive Vision Transformers', 27, '#26384a');
social += text(42, 155, 'for Fine-Grained Perception', 27, '#26384a');
social += `<image xmlns:xlink="http://www.w3.org/1999/xlink" x="32" y="185" width="1216" height="390" preserveAspectRatio="xMidYMid meet" xlink:href="data:image/png;base64,${teaser}"/>`;
social += text(42, 612, 'github.com/YXNiu/EviViT', 18, '#315c9b');
social += text(1238, 612, 'Human-search supervision · Detail in context', 18, '#4b8857', 'text-anchor="end"');
const socialOutput = process.argv[2];
if (socialOutput) fs.writeFileSync(socialOutput, svg(1280, 640, social, 'EviViT public project preview', 'EviViT arXiv preprint, with native-detail evidence allocation and sparse context fusion.'));
console.log('Generated paired-gains.svg and dataset-overview.svg; pass an optional output path for the social-preview SVG.');
