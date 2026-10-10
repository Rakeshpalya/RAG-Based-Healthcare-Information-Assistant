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
