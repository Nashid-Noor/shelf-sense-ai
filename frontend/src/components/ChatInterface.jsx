/**
 * Chat interface component with streaming support and citations.
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import {
  Send,
  Loader2,
  BookOpen,
  AlertCircle,
  RefreshCw,
  Sparkles,
  User,
  Bot,
  Copy,
  Check,
  ChevronDown,
} from 'lucide-react';

function CitationLink({ citation }) {
  return (
    <Link
      to={`/library/${citation.book_id}`}
      className="inline-flex items-center gap-1 px-2 py-0.5 bg-indigo-50 text-indigo-700 rounded text-sm hover:bg-indigo-100 transition-colors"
    >
      <BookOpen size={12} />
      <span className="max-w-[200px] truncate">{citation.title}</span>
    </Link>
  );
}

function MessageContent({ content, citations = [] }) {
  // Parse content for citation markers like [Title by Author]
  // In production, this would be more sophisticated
  return (
    <div className="prose prose-sm max-w-none">
      <p className="whitespace-pre-wrap">{content}</p>

      {citations.length > 0 && (
        <div className="mt-3 pt-3 border-t border-gray-100">
          <p className="text-xs text-gray-500 mb-2">Referenced books:</p>
          <div className="flex flex-wrap gap-2">
            {citations.map((citation, index) => (
              <CitationLink key={index} citation={citation} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function ChatMessage({ message, isStreaming = false }) {
  const [copied, setCopied] = useState(false);
  const isUser = message.role === 'user';

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
      {/* Avatar */}
      <div
        className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${isUser
          ? 'bg-indigo-100 text-indigo-600'
          : 'bg-gradient-to-br from-purple-500 to-indigo-600 text-white'
          }`}
      >
        {isUser ? <User size={18} /> : <Bot size={18} />}
      </div>

      {/* Message */}
      <div
        className={`flex-1 max-w-[80%] ${isUser ? 'text-right' : ''}`}
      >
        <div
          className={`inline-block p-4 rounded-2xl ${isUser
            ? 'bg-indigo-600 text-white rounded-br-sm'
            : 'bg-white border border-gray-100 shadow-sm rounded-bl-sm'
            }`}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <MessageContent content={message.content} citations={message.citations || []} />
          )}

          {isStreaming && (
            <span className="inline-block w-2 h-4 ml-1 bg-current animate-pulse" />
          )}
        </div>

        {/* Actions */}
        {!isUser && !isStreaming && (
          <div className="flex items-center gap-2 mt-1 text-gray-400">
            <button
              onClick={handleCopy}
              className="p-1 hover:text-gray-600 hover:bg-gray-100 rounded transition-colors"
              title="Copy message"
            >
              {copied ? <Check size={14} /> : <Copy size={14} />}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

function SuggestionChips({ suggestions, onSelect }) {
  return (
    <div className="flex flex-wrap gap-2">
      {suggestions.map((suggestion, index) => (
        <button
          key={index}
          onClick={() => onSelect(suggestion)}
          className="px-3 py-1.5 bg-white border border-gray-200 rounded-full text-sm text-gray-600 hover:border-indigo-300 hover:text-indigo-600 hover:bg-indigo-50 transition-colors"
        >
          {suggestion}
        </button>
      ))}
    </div>
  );
}

const DEFAULT_SUGGESTIONS = [
  'What should I read next?',
  'Summarize my library',
  'Find similar books to...',
  'What genres do I read most?',
];

export default function ChatInterface({
  conversationId,
  onNewConversation,
  initialMessages = [],
  streamMessage,
}) {
  const [messages, setMessages] = useState(initialMessages);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState(null);
  const [currentConversationId, setCurrentConversationId] = useState(conversationId);

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const abortControllerRef = useRef(null);

  // Auto-scroll to bottom
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Sync messages from props
  useEffect(() => {
    setMessages(initialMessages);
  }, [initialMessages]);

  // Sync conversation ID from props
  useEffect(() => {
    if (conversationId) {
      setCurrentConversationId(conversationId);
    }
  }, [conversationId]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = { role: 'user', content: input.trim() };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setError(null);
    setIsLoading(true);

    // Create assistant message placeholder
    const assistantMessage = {
      role: 'assistant',
      content: '',
      citations: [],
    };
    setMessages((prev) => [...prev, assistantMessage]);
    setIsStreaming(true);

    try {
      // Use streaming API
      await new Promise((resolve, reject) => {
        streamMessage(
          userMessage.content,
          currentConversationId,
          // On chunk
          (chunk) => {
            if (chunk.type === 'content') {
              setMessages((prev) => {
                const updated = [...prev];
                const lastIndex = updated.length - 1;
                const lastMsg = { ...updated[lastIndex] }; // Clone to avoid mutation

                if (lastMsg.role === 'assistant') {
                  lastMsg.content += chunk.content;
                }

                updated[lastIndex] = lastMsg;
                return updated;
              });
            } else if (chunk.type === 'citation') {
              setMessages((prev) => {
                const updated = [...prev];
                const lastIndex = updated.length - 1;
                const lastMsg = { ...updated[lastIndex] }; // Clone to avoid mutation

                if (lastMsg.role === 'assistant') {
                  lastMsg.citations = [...(lastMsg.citations || []), chunk.citation];
                }

                updated[lastIndex] = lastMsg;
                return updated;
              });
            } else if (chunk.conversation_id) {
              setCurrentConversationId(chunk.conversation_id);
            }
          },
          // On done
          () => {
            setIsStreaming(false);
            setIsLoading(false);
            resolve();
          },
          // On error
          (err) => {
            reject(err);
          }
        );
      });
    } catch (err) {
      setError(err.message || 'Failed to send message. Please try again.');
      // Remove the empty assistant message
      setMessages((prev) => prev.slice(0, -1));
      setIsStreaming(false);
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSuggestionSelect = (suggestion) => {
    setInput(suggestion);
    inputRef.current?.focus();
  };

  const handleRetry = () => {
    setError(null);
    if (messages.length > 0) {
      const lastUserMessage = [...messages].reverse().find((m) => m.role === 'user');
      if (lastUserMessage) {
        setInput(lastUserMessage.content);
        setMessages((prev) => prev.slice(0, -1)); // Remove last message
      }
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center px-4">
            <div className="w-16 h-16 bg-gradient-to-br from-purple-500 to-indigo-600 rounded-2xl flex items-center justify-center mb-4 shadow-lg">
              <Sparkles className="w-8 h-8 text-white" />
            </div>
            <h2 className="text-xl font-semibold text-gray-900 mb-2">
              Ask me about your library
            </h2>
            <p className="text-gray-500 mb-6 max-w-md">
              I can help you find books, get recommendations, analyze your reading habits, and more.
            </p>
            <SuggestionChips
              suggestions={DEFAULT_SUGGESTIONS}
              onSelect={handleSuggestionSelect}
            />
          </div>
        ) : (
          <>
            {messages.map((message, index) => (
              <ChatMessage
                key={index}
                message={message}
                isStreaming={isStreaming && index === messages.length - 1}
              />
            ))}
          </>
        )}

        {error && (
          <div className="flex items-center gap-3 p-4 bg-red-50 border border-red-200 rounded-lg">
            <AlertCircle className="text-red-500 flex-shrink-0" />
            <div className="flex-1">
              <p className="text-red-700">{error}</p>
            </div>
            <button
              onClick={handleRetry}
              className="flex items-center gap-1 px-3 py-1.5 text-red-700 hover:bg-red-100 rounded-lg transition-colors"
            >
              <RefreshCw size={16} />
              <span>Retry</span>
            </button>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Scroll to bottom button */}
      {messages.length > 3 && (
        <button
          onClick={scrollToBottom}
          className="absolute bottom-24 right-6 p-2 bg-white shadow-lg rounded-full border border-gray-200 hover:bg-gray-50 transition-colors"
        >
          <ChevronDown size={20} className="text-gray-600" />
        </button>
      )}

      {/* Input */}
      <div className="border-t border-gray-200 bg-white p-4">
        <div className="flex items-end gap-3">
          <div className="flex-1 relative">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about your library..."
              rows={1}
              disabled={isLoading}
              className="w-full px-4 py-3 pr-12 bg-gray-50 border border-gray-200 rounded-xl resize-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent disabled:opacity-50 max-h-32"
              style={{
                height: 'auto',
                minHeight: '48px',
              }}
              onInput={(e) => {
                e.target.style.height = 'auto';
                e.target.style.height = Math.min(e.target.scrollHeight, 128) + 'px';
              }}
            />
          </div>

          <button
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
            className="p-3 bg-indigo-600 text-white rounded-xl hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isLoading ? (
              <Loader2 size={20} className="animate-spin" />
            ) : (
              <Send size={20} />
            )}
          </button>
        </div>

        <p className="text-xs text-gray-400 mt-2 text-center">
          Responses are based on your library. Press Enter to send, Shift+Enter for new line.
        </p>
      </div>
    </div>
  );
}
