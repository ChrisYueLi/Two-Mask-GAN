const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

const ROOT = __dirname;
const OUTPUT_CSV = String.raw`F:\ProjectRelated\surfdrive1\dataset\Interruption\output_with_wer.csv`;
const GENDER_CSV = String.raw`F:\ProjectRelated\surfdrive1\Ego_speech_filter\ASR_RESULTnew.csv`;
const TOPIC_XLSX = String.raw`F:\ProjectRelated\surfdrive1\Ego_speech_filter\debateable_topics.xlsx`;
const OUT_TEX = path.join(ROOT, 'set3_appendix_tables.tex');

function parseCSV(text) {
  if (text.charCodeAt(0) === 0xfeff) {
    text = text.slice(1);
  }
  const rows = [];
  let row = [];
  let field = '';
  let inQuotes = false;

  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    const next = text[i + 1];
    if (inQuotes) {
      if (ch === '"' && next === '"') {
        field += '"';
        i += 1;
      } else if (ch === '"') {
        inQuotes = false;
      } else {
        field += ch;
      }
      continue;
    }

    if (ch === '"') {
      inQuotes = true;
    } else if (ch === ',') {
      row.push(field);
      field = '';
    } else if (ch === '\n') {
      row.push(field);
      rows.push(row);
      row = [];
      field = '';
    } else if (ch !== '\r') {
      field += ch;
    }
  }

  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }

  if (!rows.length) return [];
  const header = rows[0];
  return rows.slice(1).filter((r) => r.some((v) => v !== '')).map((r) => {
    const obj = {};
    header.forEach((key, idx) => {
      obj[key] = r[idx] ?? '';
    });
    return obj;
  });
}

function normId(value) {
  return String(value || '')
    .trim()
    .replace(/\.wav(?:\.wav)?$/i, '')
    .replace(/_filter$/i, '')
    .replace(/\.flac$/i, '');
}

function cleanText(value) {
  return String(value || '')
    .replace(/\s+/g, ' ')
    .replace(/^"+|"+$/g, '')
    .trim();
}

function escapeLatex(value) {
  return String(value || '')
    .replace(/\\/g, '\\textbackslash{}')
    .replace(/([#$%&_{}])/g, '\\$1')
    .replace(/~/g, '\\textasciitilde{}')
    .replace(/\^/g, '\\textasciicircum{}');
}

function tokenize(text) {
  return cleanText(text).split(/\s+/).filter(Boolean);
}

function normToken(token) {
  return token.toLowerCase().replace(/[“”"'.;,!?]+/g, '');
}

function trimSnippet(tokens, start, end, context = 2, maxTokens = 16) {
  if (start >= end) return '---';
  let from = Math.max(0, start - context);
  let to = Math.min(tokens.length, end + context);
  while (to - from > maxTokens) {
    if (start - from > to - end) {
      from += 1;
    } else {
      to -= 1;
    }
  }
  let out = tokens.slice(from, to).join(' ');
  if (from > 0) out = `... ${out}`;
  if (to < tokens.length) out = `${out} ...`;
  return out || '---';
}

function diffPair(baseText, modelText) {
  const a = tokenize(baseText);
  const b = tokenize(modelText);
  if (!a.length && !b.length) return { base: '---', model: '---' };

  let prefix = 0;
  while (
    prefix < a.length &&
    prefix < b.length &&
    normToken(a[prefix]) === normToken(b[prefix])
  ) {
    prefix += 1;
  }

  let suffix = 0;
  while (
    suffix < a.length - prefix &&
    suffix < b.length - prefix &&
    normToken(a[a.length - 1 - suffix]) === normToken(b[b.length - 1 - suffix])
  ) {
    suffix += 1;
  }

  const aEnd = a.length - suffix;
  const bEnd = b.length - suffix;
  const base = trimSnippet(a, prefix, aEnd);
  const model = trimSnippet(b, prefix, bEnd);
  if (base === '---' && model === '---') {
    return { base: '---', model: '---' };
  }
  return { base, model };
}

function formatWer(value) {
  const num = Number.parseFloat(value);
  if (!Number.isFinite(num)) return '--';
  return (num * 100).toFixed(2);
}

function wrapWords(text, wordsPerLine = 6, maxLines = 3) {
  const words = cleanText(text).split(/\s+/).filter(Boolean);
  if (!words.length) return ['---'];
  const lines = [];
  for (let i = 0; i < words.length; i += wordsPerLine) {
    lines.push(words.slice(i, i + wordsPerLine).join(' '));
    if (lines.length === maxLines) {
      if (i + wordsPerLine < words.length && !lines[lines.length - 1].endsWith('...')) {
        lines[lines.length - 1] += ' ...';
      }
      break;
    }
  }
  return lines;
}

function stackCell(text) {
  if (text === '---') return '---';
  const lines = wrapWords(text).map((line) => escapeLatex(line));
  return '\\shortstack[l]{``' + lines.join(' \\\\ ') + "''}";
}

function readXmlFromXlsx(entry) {
  return execFileSync('tar', ['-xOf', TOPIC_XLSX, entry], { encoding: 'utf8' });
}

function decodeXml(text) {
  return String(text || '')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'");
}

function parseTopics() {
  const sharedXml = readXmlFromXlsx('xl/sharedStrings.xml');
  const sheetXml = readXmlFromXlsx('xl/worksheets/sheet1.xml');

  const sharedStrings = [];
  const sharedRe = /<si>([\s\S]*?)<\/si>/g;
  let m;
  while ((m = sharedRe.exec(sharedXml))) {
    const segment = m[1];
    const chunks = [...segment.matchAll(/<t[^>]*>([\s\S]*?)<\/t>/g)].map((x) => decodeXml(x[1]));
    sharedStrings.push(chunks.join(''));
  }

  const topics = [];
  const rowRe = /<row[^>]*r="(\d+)"[^>]*>([\s\S]*?)<\/row>/g;
  while ((m = rowRe.exec(sheetXml))) {
    const rowNum = Number(m[1]);
    if (rowNum < 2) continue;
    const rowXml = m[2];
    const cellMatch = rowXml.match(/<c[^>]*r="A\d+"[^>]*t="s"[^>]*>[\s\S]*?<v>(\d+)<\/v>[\s\S]*?<\/c>/);
    if (!cellMatch) continue;
    const topic = sharedStrings[Number(cellMatch[1])] || '';
    topics.push({ topicIdx: rowNum - 2, topic });
  }
  return topics;
}

function genderLabel(raw) {
  const value = String(raw || '').trim().toLowerCase();
  if (value === '0') return 'Female';
  if (value === '1') return 'Male';
  if (value.startsWith('f')) return 'Female';
  if (value.startsWith('m')) return 'Male';
  return 'Unknown';
}

function buildGenderMap(rows) {
  const map = new Map();
  rows.forEach((row) => {
    const id = normId(row.filename);
    if (id) map.set(id, genderLabel(row.Gender));
  });
  return map;
}

function speakerStem(filename) {
  const id = normId(filename);
  const parts = id.split('_');
  return parts.slice(0, -1).join('_');
}

function topicIdxFromFilename(filename) {
  const id = normId(filename);
  const parts = id.split('_');
  return parts[parts.length - 1];
}

function buildAnonMaps(rows, genderMap) {
  const speakerOrder = { Female: new Map(), Male: new Map(), Unknown: new Map() };
  const counters = { Female: 0, Male: 0, Unknown: 0 };
  const segmentMap = new Map();

  rows.forEach((row) => {
    const id = normId(row.filename);
    if (!id || segmentMap.has(id)) return;
    const gender = genderMap.get(id) || 'Unknown';
    const stem = speakerStem(id);
    const topicIdx = topicIdxFromFilename(id);
    const group = speakerOrder[gender] || speakerOrder.Unknown;
    if (!group.has(stem)) {
      counters[gender] = (counters[gender] || 0) + 1;
      group.set(stem, counters[gender]);
    }
    const participantNum = group.get(stem);
    segmentMap.set(id, `${gender}${participantNum}_${topicIdx}`);
  });

  return segmentMap;
}

function makeTopicTable(topics) {
  const body = topics
    .map(({ topicIdx, topic }) => `${topicIdx} & ${escapeLatex(topic)} \\\\`)
    .join('\n');
  return [
    '\\begin{table}[t]',
    '\\caption{Debate topic index used in anonymized segment identifiers.}',
    '\\label{tab:topic_index}',
    '\\centering',
    '\\small',
    '\\setlength{\\tabcolsep}{6pt}',
    '\\begin{tabular}{@{}cl@{}}',
    '\\toprule',
    '\\textbf{topic\\_idx} & \\textbf{Topic} \\\\',
    '\\midrule',
    body,
    '\\bottomrule',
    '\\end{tabular}',
    '\\end{table}',
    '',
  ].join('\n');
}

function makeLongTable(rows, title, modelKey, modelLabel, werKey) {
  const lines = [];
  lines.push('\\begingroup');
  lines.push('\\footnotesize');
  lines.push('\\setlength{\\tabcolsep}{2.5pt}');
  lines.push('\\renewcommand{\\arraystretch}{1.08}');
  lines.push(`\\begin{longtable}{@{}p{0.86in}p{1.86in}p{1.86in}p{0.62in}p{0.70in}@{}}`);
  lines.push(`\\caption{Full version of Table~\\ref{tab:set3_examples} for ${title}. Snippets show pairwise differences relative to the RESF + ASR baseline; unchanged context is omitted.}\\\\`);
  lines.push('\\toprule');
  lines.push(`\\textbf{Segment ID} & \\textbf{RESF + ASR baseline} & \\textbf{${modelLabel}} & \\textbf{Base WER} & \\textbf{Model WER} \\\\`);
  lines.push('\\midrule');
  lines.push('\\endfirsthead');
  lines.push('\\toprule');
  lines.push(`\\textbf{Segment ID} & \\textbf{RESF + ASR baseline} & \\textbf{${modelLabel}} & \\textbf{Base WER} & \\textbf{Model WER} \\\\`);
  lines.push('\\midrule');
  lines.push('\\endhead');
  lines.push('\\midrule');
  lines.push('\\multicolumn{5}{r}{\\emph{Continued on next page}} \\\\');
  lines.push('\\midrule');
  lines.push('\\endfoot');
  lines.push('\\bottomrule');
  lines.push('\\endlastfoot');

  rows.forEach((row) => {
    const diff = diffPair(row.filter, row[modelKey]);
    const baseCell = stackCell(diff.base);
    const modelCell = stackCell(diff.model);
    lines.push(
      `${escapeLatex(row.segmentId)} & ` +
      `${baseCell} & ` +
      `${modelCell} & ` +
      `${formatWer(row.wer_filter)} & ${formatWer(row[werKey])} \\\\`
    );
  });

  lines.push('\\end{longtable}');
  lines.push('\\endgroup');
  lines.push('');
  return lines.join('\n');
}

function main() {
  const outputRows = parseCSV(fs.readFileSync(OUTPUT_CSV, 'utf8')).filter((row) => normId(row.filename));
  const genderRows = parseCSV(fs.readFileSync(GENDER_CSV, 'utf8')).filter((row) => normId(row.filename));
  const genderMap = buildGenderMap(genderRows);
  const anonMap = buildAnonMaps(outputRows, genderMap);
  const topics = parseTopics();

  const rows = outputRows.map((row) => ({
    ...row,
    filename: normId(row.filename),
    segmentId: anonMap.get(normId(row.filename)) || normId(row.filename),
  }));

  const doc = [
    '% Auto-generated by generate_set3_appendix.js',
    '\\section{Appendix: Full Version of Table~\\ref{tab:set3_examples}}',
    'This appendix provides the full version of Table~\\ref{tab:set3_examples} for the exported Evaluation Set~\\RomanNumeralCaps{3} segment list. Segment identifiers follow \\texttt{Gender\\_num\\_\\{topic\\_idx\\}}, where \\texttt{num} is the anonymized participant index within gender and \\texttt{topic\\_idx} refers to the debate topic index listed in Table~\\ref{tab:topic_index}. Each table reports pairwise transcript differences relative to the RESF + ASR baseline while retaining the baseline WER and comparator WER for each segment.',
    '',
    makeTopicTable(topics),
    makeLongTable(
      rows,
      'RESF + ASR baseline vs. Proposed Two-Mask CMGAN',
      'pred_two_mask',
      'Proposed',
      'wer_pred_two_mask'
    ),
    makeLongTable(
      rows,
      'RESF + ASR baseline vs. Retrained DCCRN',
      'pred_dccrn',
      'Retrained DCCRN',
      'wer_pred_dccrn'
    ),
    makeLongTable(
      rows,
      'RESF + ASR baseline vs. Original CMGAN',
      'pred_orig_cmgan',
      'Original CMGAN',
      'wer_pred_orig_cmgan'
    ),
  ].join('\n');

  fs.writeFileSync(OUT_TEX, doc, 'utf8');
  console.log(`Wrote ${OUT_TEX}`);
}

main();
