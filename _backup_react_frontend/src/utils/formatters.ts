export function formatTimestamp(isoString: string): string {
  try {
    const d = new Date(isoString);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch {
    return '';
  }
}

export function formatSimilarityScore(score?: number): string {
  if (score === undefined || score === null) return 'N/A';
  const percentage = Math.round(score * 100);
  return `${percentage}% relevance`;
}

export function cleanDocumentName(rawName?: string): string {
  if (!rawName) return 'Clinical Document';
  // Strip UUID prefixes or extension noise if present
  return rawName.replace(/^[a-f0-9-]{36}_/, '');
}
