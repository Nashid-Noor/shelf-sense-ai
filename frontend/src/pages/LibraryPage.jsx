/**
 * Library page with book grid, filtering, and search.
 */

import React, { useState, useEffect, useCallback } from 'react';
import { useSearchParams, Link } from 'react-router-dom';
import {
  Search,
  Filter,
  Grid,
  List,
  SortAsc,
  SortDesc,
  X,
  Loader2,
  BookOpen,
  Plus,
  ChevronDown,
} from 'lucide-react';
import { booksApi } from '../services/api';
import BookCard from '../components/BookCard';

const SORT_OPTIONS = [
  { value: 'title', label: 'Title' },
  { value: 'author', label: 'Author' },
  { value: 'publication_year', label: 'Year Published' },
  { value: 'created_at', label: 'Date Added' },
  { value: 'rating', label: 'Rating' },
];

const READ_STATUS_OPTIONS = [
  { value: '', label: 'All Status' },
  { value: 'unread', label: 'To Read' },
  { value: 'reading', label: 'Reading' },
  { value: 'read', label: 'Read' },
];

function FilterDropdown({ label, value, options, onChange }) {
  const [open, setOpen] = useState(false);
  const selectedOption = options.find((opt) => opt.value === value);

  return (
    <div className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 px-3 py-2 bg-white border border-gray-200 rounded-lg hover:border-gray-300 transition-colors"
      >
        <span className="text-sm text-gray-700">
          {label}: <span className="font-medium">{selectedOption?.label || 'All'}</span>
        </span>
        <ChevronDown size={16} className="text-gray-400" />
      </button>

      {open && (
        <>
          <div className="fixed inset-0 z-10" onClick={() => setOpen(false)} />
          <div className="absolute top-full left-0 mt-1 z-20 w-48 bg-white rounded-lg shadow-lg border border-gray-200 py-1">
            {options.map((option) => (
              <button
                key={option.value}
                onClick={() => {
                  onChange(option.value);
                  setOpen(false);
                }}
                className={`w-full px-4 py-2 text-left text-sm hover:bg-gray-50 ${value === option.value ? 'text-indigo-600 font-medium' : 'text-gray-700'
                  }`}
              >
                {option.label}
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

function GenreFilter({ genres, selected, onChange }) {
  const [expanded, setExpanded] = useState(false);
  const displayGenres = expanded ? genres : genres.slice(0, 8);

  return (
    <div className="space-y-2">
      <h3 className="text-sm font-medium text-gray-700">Genres</h3>
      <div className="flex flex-wrap gap-2">
        {displayGenres.map((genre) => (
          <button
            key={genre}
            onClick={() => onChange(selected === genre ? '' : genre)}
            className={`px-3 py-1.5 rounded-full text-sm transition-colors ${selected === genre
              ? 'bg-indigo-600 text-white'
              : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
          >
            {genre}
          </button>
        ))}
        {genres.length > 8 && (
          <button
            onClick={() => setExpanded(!expanded)}
            className="px-3 py-1.5 text-sm text-indigo-600 hover:text-indigo-700"
          >
            {expanded ? 'Show less' : `+${genres.length - 8} more`}
          </button>
        )}
      </div>
    </div>
  );
}

function EmptyState({ hasFilters, onClearFilters }) {
  return (
    <div className="text-center py-16">
      <BookOpen className="w-16 h-16 text-gray-300 mx-auto mb-4" />
      <h3 className="text-xl font-semibold text-gray-900 mb-2">
        {hasFilters ? 'No books match your filters' : 'Your library is empty'}
      </h3>
      <p className="text-gray-500 mb-6 max-w-md mx-auto">
        {hasFilters
          ? 'Try adjusting your filters or search query'
          : 'Start by scanning a shelf photo or manually adding books'}
      </p>
      {hasFilters ? (
        <button
          onClick={onClearFilters}
          className="inline-flex items-center gap-2 px-4 py-2 text-indigo-600 hover:bg-indigo-50 rounded-lg"
        >
          <X size={18} />
          Clear filters
        </button>
      ) : (
        <a
          href="/scan"
          className="inline-flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700"
        >
          <Plus size={18} />
          Add your first books
        </a>
      )}
    </div>
  );
}

export default function LibraryPage() {
  const [searchParams, setSearchParams] = useSearchParams();

  const [books, setBooks] = useState([]);
  const [totalBooks, setTotalBooks] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [availableGenres, setAvailableGenres] = useState([]);

  // View state
  const [viewMode, setViewMode] = useState('grid');

  // Filter state from URL
  const searchQuery = searchParams.get('q') || '';
  const selectedGenre = searchParams.get('genre') || '';
  const selectedStatus = searchParams.get('status') || '';
  const sortBy = searchParams.get('sort') || 'created_at';
  const sortOrder = searchParams.get('order') || 'desc';
  const page = parseInt(searchParams.get('page') || '1', 10);
  const limit = 20;

  // Update URL params
  const updateParams = useCallback((updates) => {
    const newParams = new URLSearchParams(searchParams);
    Object.entries(updates).forEach(([key, value]) => {
      if (value) {
        newParams.set(key, value);
      } else {
        newParams.delete(key);
      }
    });
    // Reset to page 1 when filters change
    if (!updates.page) {
      newParams.delete('page');
    }
    setSearchParams(newParams);
  }, [searchParams, setSearchParams]);

  // Fetch books
  useEffect(() => {
    const fetchBooks = async () => {
      try {
        setLoading(true);

        let response;
        if (searchQuery) {
          // Use search endpoint
          response = await booksApi.search({
            query: searchQuery,
            filters: {
              genre: selectedGenre || undefined,
              read_status: selectedStatus || undefined,
            },
            limit,
          });
        } else {
          // Use list endpoint
          response = await booksApi.list({
            page,
            limit,
            genre: selectedGenre || undefined,
            readStatus: selectedStatus || undefined,
            sortBy,
            sortOrder,
          });
        }

        setBooks(response.books || response.results || []);
        setTotalBooks(response.total || 0);

        // Extract unique genres for filter
        if (response.genres) {
          setAvailableGenres(response.genres);
        }
      } catch (err) {
        console.error('Failed to fetch books:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchBooks();
  }, [searchQuery, selectedGenre, selectedStatus, sortBy, sortOrder, page]);

  // Fetch genres on mount
  useEffect(() => {
    const fetchGenres = async () => {
      try {
        const stats = await analyticsApi.getStats();
        if (stats.genre_distribution) {
          setAvailableGenres(stats.genre_distribution.map((g) => g.genre));
        }
      } catch (err) {
        console.error('Failed to fetch genres:', err);
      }
    };

    if (availableGenres.length === 0) {
      fetchGenres();
    }
  }, [availableGenres.length]);

  const hasFilters = searchQuery || selectedGenre || selectedStatus;

  const clearFilters = () => {
    setSearchParams(new URLSearchParams());
  };

  const handleDeleteBook = async (book) => {
    if (!window.confirm(`Are you sure you want to delete "${book.title}"?`)) {
      return;
    }

    try {
      await booksApi.delete(book.id);
      // Remove from local state
      setBooks((prev) => prev.filter((b) => b.id !== book.id));
      setTotalBooks((prev) => prev - 1);
    } catch (err) {
      console.error('Failed to delete book:', err);
      // Could show toast error here
      alert('Failed to delete book');
    }
  };

  const totalPages = Math.ceil(totalBooks / limit);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Library</h1>
          <p className="text-gray-500 mt-1">
            {totalBooks} book{totalBooks !== 1 ? 's' : ''} in your collection
          </p>
        </div>

        <Link
          to="/scan"
          className="inline-flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700"
        >
          <Plus size={18} />
          <span>Add Book</span>
        </Link>
      </div>

      {/* Search and Filters */}
      <div className="space-y-4">
        {/* Search bar */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={20} />
          <input
            type="text"
            placeholder="Search by title, author, or ISBN..."
            value={searchQuery}
            onChange={(e) => updateParams({ q: e.target.value })}
            className="w-full pl-10 pr-4 py-3 bg-white border border-gray-200 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
          />
          {searchQuery && (
            <button
              onClick={() => updateParams({ q: '' })}
              className="absolute right-3 top-1/2 -translate-y-1/2 p-1 text-gray-400 hover:text-gray-600"
            >
              <X size={18} />
            </button>
          )}
        </div>

        {/* Filter bar */}
        <div className="flex flex-wrap items-center gap-3">
          <FilterDropdown
            label="Status"
            value={selectedStatus}
            options={READ_STATUS_OPTIONS}
            onChange={(value) => updateParams({ status: value })}
          />

          <FilterDropdown
            label="Sort"
            value={sortBy}
            options={SORT_OPTIONS}
            onChange={(value) => updateParams({ sort: value })}
          />

          <button
            onClick={() => updateParams({ order: sortOrder === 'asc' ? 'desc' : 'asc' })}
            className="p-2 bg-white border border-gray-200 rounded-lg hover:border-gray-300"
            title={sortOrder === 'asc' ? 'Ascending' : 'Descending'}
          >
            {sortOrder === 'asc' ? <SortAsc size={20} /> : <SortDesc size={20} />}
          </button>

          <div className="flex-1" />

          {/* View toggle */}
          <div className="flex bg-white border border-gray-200 rounded-lg overflow-hidden">
            <button
              onClick={() => setViewMode('grid')}
              className={`p-2 ${viewMode === 'grid' ? 'bg-indigo-50 text-indigo-600' : 'text-gray-600 hover:bg-gray-50'}`}
            >
              <Grid size={20} />
            </button>
            <button
              onClick={() => setViewMode('list')}
              className={`p-2 ${viewMode === 'list' ? 'bg-indigo-50 text-indigo-600' : 'text-gray-600 hover:bg-gray-50'}`}
            >
              <List size={20} />
            </button>
          </div>
        </div>

        {/* Genre filter */}
        {availableGenres.length > 0 && (
          <GenreFilter
            genres={availableGenres}
            selected={selectedGenre}
            onChange={(value) => updateParams({ genre: value })}
          />
        )}

        {/* Active filters */}
        {hasFilters && (
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-500">Active filters:</span>
            {searchQuery && (
              <span className="inline-flex items-center gap-1 px-2 py-1 bg-indigo-100 text-indigo-700 rounded-full text-sm">
                Search: {searchQuery}
                <button onClick={() => updateParams({ q: '' })}>
                  <X size={14} />
                </button>
              </span>
            )}
            {selectedGenre && (
              <span className="inline-flex items-center gap-1 px-2 py-1 bg-indigo-100 text-indigo-700 rounded-full text-sm">
                Genre: {selectedGenre}
                <button onClick={() => updateParams({ genre: '' })}>
                  <X size={14} />
                </button>
              </span>
            )}
            {selectedStatus && (
              <span className="inline-flex items-center gap-1 px-2 py-1 bg-indigo-100 text-indigo-700 rounded-full text-sm">
                Status: {selectedStatus}
                <button onClick={() => updateParams({ status: '' })}>
                  <X size={14} />
                </button>
              </span>
            )}
            <button
              onClick={clearFilters}
              className="text-sm text-gray-500 hover:text-gray-700"
            >
              Clear all
            </button>
          </div>
        )}
      </div>

      {/* Loading state */}
      {loading && (
        <div className="flex items-center justify-center py-16">
          <Loader2 className="w-8 h-8 text-indigo-500 animate-spin" />
        </div>
      )}

      {/* Error state */}
      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
          {error}
        </div>
      )}

      {/* Books grid/list */}
      {!loading && !error && (
        <>
          {books.length > 0 ? (
            <div
              className={
                viewMode === 'grid'
                  ? 'grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4'
                  : 'space-y-3'
              }
            >
              {books.map((book) => (
                <BookCard
                  key={book.id}
                  book={book}
                  variant={viewMode}
                  onDelete={handleDeleteBook}
                />
              ))}
            </div>
          ) : (
            <EmptyState hasFilters={hasFilters} onClearFilters={clearFilters} />
          )}

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-center gap-2 pt-6">
              <button
                onClick={() => updateParams({ page: String(page - 1) })}
                disabled={page <= 1}
                className="px-4 py-2 border border-gray-200 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
              >
                Previous
              </button>
              <span className="px-4 py-2 text-gray-600">
                Page {page} of {totalPages}
              </span>
              <button
                onClick={() => updateParams({ page: String(page + 1) })}
                disabled={page >= totalPages}
                className="px-4 py-2 border border-gray-200 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
              >
                Next
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}

// Import for genre fetch
import { analyticsApi } from '../services/api';
