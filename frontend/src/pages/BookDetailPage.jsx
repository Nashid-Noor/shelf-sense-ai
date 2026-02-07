/**
 * Book detail page with full metadata, reading status, and actions.
 */

import React, { useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import {
  ArrowLeft,
  BookOpen,
  Calendar,
  Tag,
  ExternalLink,
  Edit2,
  Trash2,
  Clock,
  Check,
  Star,
  Loader2,
  MessageCircle,
  Share2,
} from 'lucide-react';
import { booksApi, analyticsApi } from '../services/api';
import { StarRating } from '../components/BookCard';
import BookCard from '../components/BookCard';

const READ_STATUS_OPTIONS = [
  { value: 'unread', label: 'To Read', icon: Clock, color: 'gray' },
  { value: 'reading', label: 'Reading', icon: BookOpen, color: 'blue' },
  { value: 'read', label: 'Read', icon: Check, color: 'green' },
];

function StatusButton({ status, currentStatus, onChange }) {
  const Icon = status.icon;
  const isActive = currentStatus === status.value;
  
  const colorClasses = {
    gray: isActive ? 'bg-gray-100 text-gray-700 border-gray-300' : 'hover:bg-gray-50',
    blue: isActive ? 'bg-blue-100 text-blue-700 border-blue-300' : 'hover:bg-blue-50',
    green: isActive ? 'bg-green-100 text-green-700 border-green-300' : 'hover:bg-green-50',
  };

  return (
    <button
      onClick={() => onChange(status.value)}
      className={`flex items-center gap-2 px-4 py-2 border rounded-lg transition-colors ${
        isActive ? colorClasses[status.color] : 'border-gray-200 text-gray-600 ' + colorClasses[status.color]
      }`}
    >
      <Icon size={18} />
      <span>{status.label}</span>
    </button>
  );
}

function DeleteConfirmModal({ book, onConfirm, onCancel }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/50" onClick={onCancel} />
      <div className="relative bg-white rounded-xl p-6 max-w-md w-full">
        <h3 className="text-lg font-semibold text-gray-900 mb-2">Delete Book?</h3>
        <p className="text-gray-600 mb-6">
          Are you sure you want to remove "{book.title}" from your library? This action cannot be undone.
        </p>
        <div className="flex gap-3 justify-end">
          <button
            onClick={onCancel}
            className="px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-lg"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
          >
            Delete
          </button>
        </div>
      </div>
    </div>
  );
}

export default function BookDetailPage() {
  const { bookId } = useParams();
  const navigate = useNavigate();
  
  const [book, setBook] = useState(null);
  const [similarBooks, setSimilarBooks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [updating, setUpdating] = useState(false);

  useEffect(() => {
    const fetchBook = async () => {
      try {
        setLoading(true);
        const bookData = await booksApi.get(bookId);
        setBook(bookData);

        // Fetch similar books
        try {
          const recs = await analyticsApi.getRecommendations({
            count: 4,
            basedOnBookId: bookId,
          });
          setSimilarBooks(recs.recommendations || []);
        } catch (err) {
          console.error('Failed to fetch similar books:', err);
        }
      } catch (err) {
        console.error('Failed to fetch book:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    if (bookId) {
      fetchBook();
    }
  }, [bookId]);

  const handleStatusChange = async (status) => {
    try {
      setUpdating(true);
      await booksApi.updateReadStatus(bookId, status);
      setBook((prev) => ({ ...prev, read_status: status }));
    } catch (err) {
      console.error('Failed to update status:', err);
    } finally {
      setUpdating(false);
    }
  };

  const handleRatingChange = async (rating) => {
    try {
      setUpdating(true);
      await booksApi.rate(bookId, rating);
      setBook((prev) => ({ ...prev, rating }));
    } catch (err) {
      console.error('Failed to update rating:', err);
    } finally {
      setUpdating(false);
    }
  };

  const handleDelete = async () => {
    try {
      await booksApi.delete(bookId);
      navigate('/library');
    } catch (err) {
      console.error('Failed to delete book:', err);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 text-indigo-500 animate-spin" />
      </div>
    );
  }

  if (error || !book) {
    return (
      <div className="text-center py-16">
        <BookOpen className="w-16 h-16 text-gray-300 mx-auto mb-4" />
        <h2 className="text-xl font-semibold text-gray-900 mb-2">Book not found</h2>
        <p className="text-gray-500 mb-6">{error || 'This book may have been removed from your library.'}</p>
        <Link
          to="/library"
          className="inline-flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg"
        >
          <ArrowLeft size={18} />
          Back to Library
        </Link>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Back button */}
      <button
        onClick={() => navigate(-1)}
        className="inline-flex items-center gap-2 text-gray-600 hover:text-gray-900"
      >
        <ArrowLeft size={20} />
        <span>Back</span>
      </button>

      {/* Main content */}
      <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
        <div className="md:flex">
          {/* Cover */}
          <div className="md:w-1/3 bg-gradient-to-br from-gray-100 to-gray-200 p-6 flex items-center justify-center">
            {book.cover_url ? (
              <img
                src={book.cover_url}
                alt={book.title}
                className="max-h-80 w-auto rounded-lg shadow-lg"
              />
            ) : (
              <div className="w-48 h-64 bg-gray-200 rounded-lg flex items-center justify-center">
                <BookOpen className="w-16 h-16 text-gray-300" />
              </div>
            )}
          </div>

          {/* Details */}
          <div className="md:w-2/3 p-6 md:p-8">
            {/* Title and author */}
            <h1 className="text-2xl md:text-3xl font-bold text-gray-900 mb-2">
              {book.title}
            </h1>
            <p className="text-lg text-gray-600 mb-4">{book.author}</p>

            {/* Rating */}
            <div className="flex items-center gap-4 mb-6">
              <StarRating
                rating={book.rating || 0}
                onChange={handleRatingChange}
                size="lg"
              />
              {book.rating && (
                <span className="text-gray-500">
                  {book.rating} out of 5
                </span>
              )}
            </div>

            {/* Status buttons */}
            <div className="mb-6">
              <label className="text-sm font-medium text-gray-700 mb-2 block">
                Reading Status
              </label>
              <div className="flex flex-wrap gap-2">
                {READ_STATUS_OPTIONS.map((status) => (
                  <StatusButton
                    key={status.value}
                    status={status}
                    currentStatus={book.read_status}
                    onChange={handleStatusChange}
                  />
                ))}
              </div>
            </div>

            {/* Metadata */}
            <div className="grid grid-cols-2 gap-4 mb-6">
              {book.publication_year && (
                <div className="flex items-center gap-2 text-gray-600">
                  <Calendar size={18} />
                  <span>{book.publication_year}</span>
                </div>
              )}
              {book.isbn && (
                <div className="flex items-center gap-2 text-gray-600">
                  <Tag size={18} />
                  <span className="font-mono text-sm">{book.isbn}</span>
                </div>
              )}
            </div>

            {/* Genres */}
            {book.genres?.length > 0 && (
              <div className="mb-6">
                <label className="text-sm font-medium text-gray-700 mb-2 block">
                  Genres
                </label>
                <div className="flex flex-wrap gap-2">
                  {book.genres.map((genre) => (
                    <Link
                      key={genre}
                      to={`/library?genre=${encodeURIComponent(genre)}`}
                      className="px-3 py-1.5 bg-gray-100 text-gray-700 rounded-full text-sm hover:bg-indigo-100 hover:text-indigo-700 transition-colors"
                    >
                      {genre}
                    </Link>
                  ))}
                </div>
              </div>
            )}

            {/* Actions */}
            <div className="flex flex-wrap gap-3 pt-4 border-t border-gray-100">
              <Link
                to={`/chat?prompt=Tell me about ${encodeURIComponent(book.title)}`}
                className="inline-flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700"
              >
                <MessageCircle size={18} />
                Ask AI about this book
              </Link>
              
              {book.isbn && (
                <a
                  href={`https://openlibrary.org/isbn/${book.isbn}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-4 py-2 border border-gray-200 text-gray-700 rounded-lg hover:bg-gray-50"
                >
                  <ExternalLink size={18} />
                  OpenLibrary
                </a>
              )}
              
              <button
                onClick={() => {
                  navigator.share?.({
                    title: book.title,
                    text: `Check out "${book.title}" by ${book.author}`,
                    url: window.location.href,
                  });
                }}
                className="inline-flex items-center gap-2 px-4 py-2 border border-gray-200 text-gray-700 rounded-lg hover:bg-gray-50"
              >
                <Share2 size={18} />
                Share
              </button>
              
              <button
                onClick={() => setShowDeleteModal(true)}
                className="inline-flex items-center gap-2 px-4 py-2 text-red-600 hover:bg-red-50 rounded-lg"
              >
                <Trash2 size={18} />
                Delete
              </button>
            </div>
          </div>
        </div>

        {/* Description */}
        {book.description && (
          <div className="p-6 md:p-8 border-t border-gray-100">
            <h2 className="text-lg font-semibold text-gray-900 mb-3">Description</h2>
            <p className="text-gray-700 whitespace-pre-line">{book.description}</p>
          </div>
        )}
      </div>

      {/* Similar Books */}
      {similarBooks.length > 0 && (
        <div>
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Similar Books</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {similarBooks.map((rec, i) => (
              <BookCard
                key={i}
                book={rec.book || rec}
                variant="grid"
                showActions={false}
              />
            ))}
          </div>
        </div>
      )}

      {/* Delete confirmation modal */}
      {showDeleteModal && (
        <DeleteConfirmModal
          book={book}
          onConfirm={handleDelete}
          onCancel={() => setShowDeleteModal(false)}
        />
      )}

      {/* Updating indicator */}
      {updating && (
        <div className="fixed bottom-4 right-4 bg-white rounded-lg shadow-lg px-4 py-2 flex items-center gap-2">
          <Loader2 className="w-4 h-4 animate-spin text-indigo-500" />
          <span className="text-sm text-gray-600">Saving...</span>
        </div>
      )}
    </div>
  );
}
