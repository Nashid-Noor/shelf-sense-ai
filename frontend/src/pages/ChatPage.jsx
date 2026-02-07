/**
 * Chat page with AI-powered library assistant.
 */

import React, { useState, useEffect, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  Plus,
  MessageCircle,
  Trash2,
  ChevronLeft,
  Loader2,
  Clock,
  MoreVertical,
} from 'lucide-react';
import { chatApi } from '../services/api';
import ChatInterface from '../components/ChatInterface';

function ConversationListItem({ conversation, isActive, onClick, onDelete }) {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <button
      onClick={onClick}
      className={`w-full text-left p-3 rounded-lg transition-colors group ${isActive
        ? 'bg-indigo-50 border border-indigo-200'
        : 'hover:bg-gray-50 border border-transparent'
        }`}
    >
      <div className="flex items-start gap-3">
        <MessageCircle
          size={18}
          className={isActive ? 'text-indigo-600' : 'text-gray-400'}
        />
        <div className="flex-1 min-w-0">
          <p
            className={`font-medium truncate ${isActive ? 'text-indigo-900' : 'text-gray-900'
              }`}
          >
            {conversation.title || 'New conversation'}
          </p>
          <p className="text-xs text-gray-500 flex items-center gap-1 mt-0.5">
            <Clock size={12} />
            {new Date(conversation.updated_at).toLocaleDateString()}
          </p>
        </div>

        <div className="relative opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            onClick={(e) => {
              e.stopPropagation();
              setMenuOpen(!menuOpen);
            }}
            className="p-1 text-gray-400 hover:text-gray-600 rounded"
          >
            <MoreVertical size={16} />
          </button>

          {menuOpen && (
            <>
              <div
                className="fixed inset-0 z-10"
                onClick={(e) => {
                  e.stopPropagation();
                  setMenuOpen(false);
                }}
              />
              <div className="absolute right-0 top-full mt-1 z-20 bg-white rounded-lg shadow-lg border border-gray-200 py-1">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDelete(conversation);
                    setMenuOpen(false);
                  }}
                  className="w-full px-4 py-2 text-left text-sm text-red-600 hover:bg-red-50 flex items-center gap-2"
                >
                  <Trash2 size={14} />
                  Delete
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </button>
  );
}

function ConversationSidebar({
  conversations,
  activeId,
  onSelect,
  onNew,
  onDelete,
  loading,
}) {
  return (
    <div className="w-72 bg-white border-r border-gray-200 flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <button
          onClick={onNew}
          className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
        >
          <Plus size={18} />
          <span>New Chat</span>
        </button>
      </div>

      {/* Conversation list */}
      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {loading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="w-6 h-6 text-gray-400 animate-spin" />
          </div>
        ) : conversations.length === 0 ? (
          <div className="text-center py-8 text-gray-500 text-sm">
            No conversations yet
          </div>
        ) : (
          conversations.map((conv) => (
            <ConversationListItem
              key={conv.id}
              conversation={conv}
              isActive={conv.id === activeId}
              onClick={() => onSelect(conv)}
              onDelete={onDelete}
            />
          ))
        )}
      </div>
    </div>
  );
}

export default function ChatPage() {
  const [searchParams, setSearchParams] = useSearchParams();

  const [conversations, setConversations] = useState([]);
  const [activeConversation, setActiveConversation] = useState(null);
  const [messages, setMessages] = useState([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [loading, setLoading] = useState(true);
  const [loadingMessages, setLoadingMessages] = useState(false);

  // Get conversation ID from URL
  const conversationIdFromUrl = searchParams.get('id');

  // Fetch conversations on mount
  useEffect(() => {
    const fetchConversations = async () => {
      try {
        setLoading(true);
        const response = await chatApi.listConversations({ limit: 50 });
        setConversations(response.conversations || []);
      } catch (err) {
        console.error('Failed to fetch conversations:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchConversations();
  }, []);

  // Load conversation from URL
  useEffect(() => {
    if (conversationIdFromUrl && conversations.length > 0) {
      const conv = conversations.find((c) => c.id === conversationIdFromUrl);
      if (conv) {
        handleSelectConversation(conv);
      }
    }
  }, [conversationIdFromUrl, conversations]);



  const handleSelectConversation = async (conversation) => {
    setActiveConversation(conversation);
    setSearchParams({ id: conversation.id });

    // Fetch messages
    try {
      setLoadingMessages(true);
      const response = await chatApi.getConversation(conversation.id);
      setMessages(response.messages || []);
    } catch (err) {
      console.error('Failed to fetch messages:', err);
      setMessages([]);
    } finally {
      setLoadingMessages(false);
    }
  };

  const handleNewConversation = () => {
    setActiveConversation(null);
    setMessages([]);
    setSearchParams({});
  };

  const handleDeleteConversation = async (conversation) => {
    try {
      await chatApi.deleteConversation(conversation.id);
      setConversations((prev) => prev.filter((c) => c.id !== conversation.id));

      if (activeConversation?.id === conversation.id) {
        handleNewConversation();
      }
    } catch (err) {
      console.error('Failed to delete conversation:', err);
    }
  };

  const streamingIdRef = React.useRef(null);

  const handleStreamMessage = useCallback((message, conversationId, onChunk, onDone, onError) => {
    // Reset tracking ref for new conversations
    if (!conversationId) {
      streamingIdRef.current = null;
    }

    chatApi.streamMessage(
      message,
      conversationId,
      (chunk) => {
        // Handle new conversation ID
        // Only update if we haven't already processed this ID for this stream
        if (chunk.conversation_id && !activeConversation && streamingIdRef.current !== chunk.conversation_id) {
          streamingIdRef.current = chunk.conversation_id;

          // Add to conversations list immediately so it shows in sidebar
          setConversations((prev) => [
            {
              id: chunk.conversation_id,
              title: message.slice(0, 50) + (message.length > 50 ? '...' : ''),
              updated_at: new Date().toISOString(),
            },
            ...prev,
          ]);
        }
        onChunk(chunk);
      },
      () => {
        // When stream is done, update the URL to persist the conversation context
        // This will trigger a re-fetch, but that's fine since the stream is finished
        if (streamingIdRef.current && !activeConversation) {
          setSearchParams({ id: streamingIdRef.current });
          setActiveConversation({
            id: streamingIdRef.current,
            title: message.slice(0, 50) + (message.length > 50 ? '...' : ''),
            updated_at: new Date().toISOString(),
          });
        }
        onDone();
      },
      onError
    );
  }, [activeConversation, setSearchParams]);

  // Handle URL prompt parameter (e.g., ?prompt=recommend)
  useEffect(() => {
    const prompt = searchParams.get('prompt');
    if (prompt === 'recommend' && !activeConversation) {
      const initialMessage = 'What should I read next based on my library?';

      // Set initial UI state
      setMessages([
        {
          role: 'user',
          content: initialMessage,
        },
        {
          role: 'assistant',
          content: '',
          citations: [],
        }
      ]);

      // Trigger streaming immediately
      handleStreamMessage(
        initialMessage,
        null, // No conversation ID yet
        (chunk) => {
          setMessages((prev) => {
            const updated = [...prev];
            const lastIndex = updated.length - 1;
            const lastMsg = { ...updated[lastIndex] };

            if (lastMsg.role === 'assistant') {
              if (chunk.type === 'content') {
                lastMsg.content += chunk.content;
              } else if (chunk.type === 'citation') {
                lastMsg.citations = [...(lastMsg.citations || []), chunk.citation];
              }
            }

            updated[lastIndex] = lastMsg;
            return updated;
          });
        },
        () => {
          // On done
        },
        (err) => {
          console.error("Failed to auto-send prompt", err);
        }
      );

      // Clear param to prevent double trigger
      setSearchParams({});
    }
  }, [searchParams, activeConversation, handleStreamMessage, setSearchParams]);

  return (
    <div className="h-[calc(100vh-8rem)] flex bg-gray-50 rounded-xl overflow-hidden border border-gray-200">
      {/* Sidebar - desktop */}
      <div className={`hidden lg:flex ${sidebarOpen ? '' : 'hidden'}`}>
        <ConversationSidebar
          conversations={conversations}
          activeId={activeConversation?.id}
          onSelect={handleSelectConversation}
          onNew={handleNewConversation}
          onDelete={handleDeleteConversation}
          loading={loading}
        />
      </div>

      {/* Mobile sidebar toggle */}
      <div className="lg:hidden absolute top-4 left-4 z-10">
        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className="p-2 bg-white rounded-lg shadow-sm border border-gray-200"
        >
          {sidebarOpen ? <ChevronLeft size={20} /> : <MessageCircle size={20} />}
        </button>
      </div>

      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div className="lg:hidden fixed inset-0 z-40">
          <div
            className="absolute inset-0 bg-black/50"
            onClick={() => setSidebarOpen(false)}
          />
          <div className="absolute left-0 top-0 bottom-0 w-72">
            <ConversationSidebar
              conversations={conversations}
              activeId={activeConversation?.id}
              onSelect={(conv) => {
                handleSelectConversation(conv);
                setSidebarOpen(false);
              }}
              onNew={() => {
                handleNewConversation();
                setSidebarOpen(false);
              }}
              onDelete={handleDeleteConversation}
              loading={loading}
            />
          </div>
        </div>
      )}

      {/* Main chat area */}
      <div className="flex-1 flex flex-col bg-gray-50">
        {loadingMessages ? (
          <div className="flex-1 flex items-center justify-center">
            <Loader2 className="w-8 h-8 text-indigo-500 animate-spin" />
          </div>
        ) : (
          <ChatInterface
            conversationId={activeConversation?.id}
            initialMessages={messages}
            streamMessage={handleStreamMessage}
            onNewConversation={handleNewConversation}
          />
        )}
      </div>
    </div>
  );
}
