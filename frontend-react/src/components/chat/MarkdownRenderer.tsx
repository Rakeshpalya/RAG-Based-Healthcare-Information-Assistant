import React, { useState } from 'react';
import { Copy, Check } from 'lucide-react';
import { fixConjoinedWords } from '../../utils/formatters';

interface MarkdownRendererProps {
  content: string;
  className?: string;
}

export const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content, className = '' }) => {
  // Pre-clean content to fix conjoined words and normalize line breaks
  const cleanedContent = fixConjoinedWords(content || '');

  // Split content by code blocks: ```lang\ncode\n```
  const codeBlockRegex = /```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g;
  const sections: { type: 'text' | 'code'; content: string; language?: string }[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = codeBlockRegex.exec(cleanedContent)) !== null) {
    if (match.index > lastIndex) {
      sections.push({
        type: 'text',
        content: cleanedContent.slice(lastIndex, match.index),
      });
    }
    sections.push({
      type: 'code',
      language: match[1] || 'text',
      content: match[2].trim(),
    });
    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < cleanedContent.length) {
    sections.push({
      type: 'text',
      content: cleanedContent.slice(lastIndex),
    });
  }

  return (
    <div className={`markdown-body space-y-3 leading-relaxed text-slate-800 ${className}`}>
      {sections.map((section, sIdx) => {
        if (section.type === 'code') {
          return (
            <CodeBlock
              key={sIdx}
              language={section.language || 'text'}
              code={section.content}
            />
          );
        }
        return <TextSection key={sIdx} text={section.content} />;
      })}
    </div>
  );
};

const CodeBlock: React.FC<{ language: string; code: string }> = ({ language, code }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="my-3 rounded-xl overflow-hidden border border-slate-700 bg-slate-900 text-slate-100 text-xs shadow-md">
      <div className="flex items-center justify-between px-3.5 py-1.5 bg-slate-800 border-b border-slate-700 text-slate-300 font-mono text-[11px]">
        <span className="uppercase tracking-wider font-semibold">{language}</span>
        <button
          onClick={handleCopy}
          className="inline-flex items-center gap-1 text-slate-400 hover:text-slate-200 transition-colors px-1.5 py-0.5 rounded hover:bg-slate-700"
          title="Copy code"
        >
          {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
          <span>{copied ? 'Copied' : 'Copy'}</span>
        </button>
      </div>
      <pre className="p-4 overflow-x-auto font-mono text-xs leading-relaxed text-emerald-300">
        <code>{code}</code>
      </pre>
    </div>
  );
};

const TextSection: React.FC<{ text: string }> = ({ text }) => {
  const lines = text.split('\n');
  const elements: React.ReactNode[] = [];
  let currentList: { type: 'ul' | 'ol'; items: string[] } | null = null;

  const flushList = () => {
    if (!currentList) return;
    if (currentList.type === 'ul') {
      elements.push(
        <ul key={`ul-${elements.length}`} className="my-2 space-y-1.5 pl-5 list-disc text-slate-700">
          {currentList.items.map((it, idx) => (
            <li key={idx} className="leading-relaxed">
              {renderInlineMarkdown(it)}
            </li>
          ))}
        </ul>
      );
    } else {
      elements.push(
        <ol key={`ol-${elements.length}`} className="my-2 space-y-1.5 pl-5 list-decimal text-slate-700">
          {currentList.items.map((it, idx) => (
            <li key={idx} className="leading-relaxed">
              {renderInlineMarkdown(it)}
            </li>
          ))}
        </ol>
      );
    }
    currentList = null;
  };

  for (let i = 0; i < lines.length; i++) {
    const rawLine = lines[i];
    const line = rawLine.trim();

    if (!line) {
      flushList();
      continue;
    }

    // Headings
    if (line.startsWith('#### ')) {
      flushList();
      elements.push(
        <h4 key={`h4-${elements.length}`} className="text-sm font-bold text-slate-900 mt-3 mb-1">
          {renderInlineMarkdown(line.slice(5))}
        </h4>
      );
    } else if (line.startsWith('### ')) {
      flushList();
      elements.push(
        <h3 key={`h3-${elements.length}`} className="text-base font-bold text-slate-900 mt-4 mb-1.5">
          {renderInlineMarkdown(line.slice(4))}
        </h3>
      );
    } else if (line.startsWith('## ')) {
      flushList();
      elements.push(
        <h2 key={`h2-${elements.length}`} className="text-lg font-bold text-slate-900 mt-5 mb-2 pb-1 border-b border-slate-200">
          {renderInlineMarkdown(line.slice(3))}
        </h2>
      );
    } else if (line.startsWith('# ')) {
      flushList();
      elements.push(
        <h1 key={`h1-${elements.length}`} className="text-xl font-bold text-slate-900 mt-6 mb-2 pb-1.5 border-b border-slate-300">
          {renderInlineMarkdown(line.slice(2))}
        </h1>
      );
    } else if (line.startsWith('> ')) {
      flushList();
      elements.push(
        <blockquote key={`bq-${elements.length}`} className="border-l-4 border-sky-500 bg-sky-50/50 pl-3.5 py-1.5 my-2 text-slate-700 italic rounded-r-md">
          {renderInlineMarkdown(line.slice(2))}
        </blockquote>
      );
    } else if (/^[-*•]\s+/.test(line)) {
      const itemContent = line.replace(/^[-*•]\s+/, '');
      if (!currentList || currentList.type !== 'ul') {
        flushList();
        currentList = { type: 'ul', items: [itemContent] };
      } else {
        currentList.items.push(itemContent);
      }
    } else if (/^\d+\.\s+/.test(line)) {
      const itemContent = line.replace(/^\d+\.\s+/, '');
      if (!currentList || currentList.type !== 'ol') {
        flushList();
        currentList = { type: 'ol', items: [itemContent] };
      } else {
        currentList.items.push(itemContent);
      }
    } else {
      flushList();
      elements.push(
        <p key={`p-${elements.length}`} className="my-1.5 leading-relaxed text-slate-800">
          {renderInlineMarkdown(rawLine)}
        </p>
      );
    }
  }

  flushList();

  return <>{elements}</>;
};

function renderInlineMarkdown(text: string): React.ReactNode {
  if (!text) return null;

  // Split tokens by bold (**...**), inline code (`...`), citations ([Source N]), and links
  const tokenRegex = /(\*\*.*?\*\*|`.*?`|\[(?:Source|SOURCE)\s+\d+\]|\[.*?\]\(.*?\))/g;
  const parts = text.split(tokenRegex);

  return parts.map((part, index) => {
    if (!part) return null;

    // Bold: **text**
    if (part.startsWith('**') && part.endsWith('**') && part.length >= 4) {
      return (
        <strong key={index} className="font-semibold text-slate-900">
          {part.slice(2, -2)}
        </strong>
      );
    }

    // Inline code: `code`
    if (part.startsWith('`') && part.endsWith('`') && part.length >= 2) {
      return (
        <code key={index} className="px-1.5 py-0.5 rounded bg-slate-100 text-slate-800 font-mono text-[12px] border border-slate-200">
          {part.slice(1, -1)}
        </code>
      );
    }

    // Citation badge: [Source N] or [SOURCE N]
    const citationMatch = part.match(/^\[(?:Source|SOURCE)\s+(\d+)\]$/i);
    if (citationMatch) {
      const num = citationMatch[1];
      return (
        <span
          key={index}
          className="inline-flex items-center px-1.5 py-0.5 mx-0.5 rounded-full text-[11px] font-semibold bg-sky-100 text-sky-800 border border-sky-300 align-baseline tracking-tight hover:bg-sky-200 transition-colors"
          title={`Verified Source Citation ${num}`}
        >
          [Source {num}]
        </span>
      );
    }

    // Links: [text](url)
    const linkMatch = part.match(/^\[(.*?)\]\((.*?)\)$/);
    if (linkMatch) {
      return (
        <a
          key={index}
          href={linkMatch[2]}
          target="_blank"
          rel="noopener noreferrer"
          className="text-sky-600 hover:text-sky-700 underline underline-offset-2"
        >
          {linkMatch[1]}
        </a>
      );
    }

    return <React.Fragment key={index}>{part}</React.Fragment>;
  });
}
