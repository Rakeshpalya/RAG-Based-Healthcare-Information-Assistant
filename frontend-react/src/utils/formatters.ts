/**
 * Utility formatting functions for clinical timestamps, file sizes, and citation titles.
 */

export function formatDate(dateString?: string | null): string {
  if (!dateString) return 'Just now';
  try {
    const d = new Date(dateString);
    if (isNaN(d.getTime())) return dateString;
    return d.toLocaleDateString(undefined, {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return dateString;
  }
}

export function formatFileSize(bytes?: number | null): string {
  if (bytes === null || bytes === undefined || bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

export function cleanDocumentName(filename?: string): string {
  if (!filename) return 'Clinical Reference Document';
  // Strip UUID prefixes or path prefixes
  const base = filename.split(/[/\\]/).pop() || filename;
  return base.replace(/^[0-9a-fA-F-]+_/, '').replace(/\.pdf$/i, '');
}

export function formatRelevanceScore(score?: number): string {
  if (score === undefined || score === null) return '';
  const pct = Math.round(Math.min(Math.max(score, 0), 1) * 100);
  return `${pct}% relevance`;
}

export function formatAssistantText(text?: string | null): string {
  if (!text) return '';
  let cleaned = text;

  // 1. Spacing around citations: normalize [Source1] -> [Source 1], ensure spaces around brackets
  cleaned = cleaned.replace(/\[Source(\d+)\]/g, '[Source $1]');
  cleaned = cleaned.replace(/\bSource(\d+)\b/g, 'Source $1');
  cleaned = cleaned.replace(/([A-Za-z0-9])(\[(?:Source\s+\d+|[Ss]ources?\b))/g, '$1 $2');
  cleaned = cleaned.replace(/(\[(?:Source\s+\d+|[Ss]ources?\b[^\]]*)\])([.,;:!?])?([A-Za-z0-9])/g, '$1$2 $3');

  // 2. Conjoined words and punctuation fixes
  cleaned = cleaned.replace(/([a-z0-9\]\)])([.!?])([A-Z])/g, '$1$2 $3');
  cleaned = cleaned.replace(/([a-z0-9\]\)])([,;:])([A-Za-z])/g, '$1$2 $3');
  cleaned = cleaned.replace(/\bwithouta\b/g, 'without a');
  cleaned = cleaned.replace(/\bItexplicitly\b/g, 'It explicitly');
  cleaned = cleaned.replace(/\bitexplicitly\b/g, 'it explicitly');
  cleaned = cleaned.replace(/\bspecifichealth\b/g, 'specific health');
  cleaned = cleaned.replace(/\bpossiblecomplications\b/g, 'possible complications');
  cleaned = cleaned.replace(/\bairwaycondition\b/g, 'airway condition');
  cleaned = cleaned.replace(/\bandthat\b/g, 'and that');
  cleaned = cleaned.replace(/\bExcessbody\b/g, 'Excess body');
  cleaned = cleaned.replace(/\bExcessweight\b/g, 'Excess weight');

  // 3. Ensure bullet points and numbered list items start on newlines (preserving bold **)
  cleaned = cleaned.replace(/([.:;!?\n])\s*([*\-•])(?!\*)\s*/g, '$1\n$2 ');
  cleaned = cleaned.replace(/([.:;!?\n])\s*(\d+[\.\)])\s*/g, '$1\n$2 ');
  cleaned = cleaned.replace(/([^\n\s])\s*([•])\s*/g, '$1\n$2 ');
  cleaned = cleaned.replace(/([^\n\s])\s+\*(?!\*)\s*([A-Za-z0-9])/g, '$1\n* $2');
  cleaned = cleaned.replace(/([^\n\s])\s+-\s+([A-Za-z0-9])/g, '$1\n- $2');
  cleaned = cleaned.replace(/([a-z0-9])\s*-\s*([A-Z][a-zA-Z]+)/g, '$1\n- $2');
  cleaned = cleaned.replace(/(\b[a-zA-Z]+)\s*\n\s*-\s*([a-z0-9]+\b)/g, '$1-$2');

  // 4. Clean empty bullet lines
  cleaned = cleaned.replace(/^\s*[-*•](?:\s+[-*•])*\s*$/gm, '');
  cleaned = cleaned.replace(/^\s*[-*•\s]+$/gm, '');

  return cleaned.trim();
}

export const fixConjoinedWords = formatAssistantText;
