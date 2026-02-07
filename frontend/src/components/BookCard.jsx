/**
 * Book card component with cover image, metadata, and actions.
 */

import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import {
  BookOpen,
  Star,
  Check,
  Clock,
  MoreVertical,
  Edit,
  Trash2,
  ExternalLink,
} from 'lucide-react';

const READ_STATUS_CONFIG = {
  unread: { label: 'To Read', color: 'bg-gray-100 text-gray-700', icon: Clock },
  reading: { label: 'Reading', color: 'bg-blue-100 text-blue-700', icon: BookOpen },
  read: { label: 'Read', color: 'bg-green-100 text-green-700', icon: Check },
};

function StarRating({ rating, onChange, size = 'md', readonly = false }) {
  const [hovered, setHovered] = useState(0);
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-5 h-5',
    lg: 'w-6 h-6',
  };

  return (
    <div className="flex gap-0.5">
      {[1, 2, 3, 4, 5].map((star) => (
        <button
          key={star}
          type="button"
          onClick={() => !readonly && onChange?.(star)}
          onMouseEnter={() => !readonly && setHovered(star)}
          onMouseLeave={() => !readonly && setHovered(0)}
          disabled={readonly}
          className={`${readonly ? '' : 'hover:scale-110'} transition-transform`}
        >
          <Star
            className={`${sizeClasses[size]} ${star <= (hovered || rating)
                ? 'text-amber-400 fill-amber-400'
                : 'text-gray-300'
              }`}
          />
        </button>
      ))}
    </div>
  );
}

function DropdownMenu({ items, onClose }) {
  return (
    <>
      <div className="fixed inset-0 z-40" onClick={onClose} />
      <div className="absolute right-0 top-full mt-1 z-50 w-48 bg-white rounded-lg shadow-lg border border-gray-200 py-1">
        {items.map((item, index) => (
          <button
            key={index}
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              item.onClick();
              onClose();
            }}
            className={`w-full px-4 py-2 text-left flex items-center gap-3 hover:bg-gray-50 ${item.danger ? 'text-red-600 hover:bg-red-50' : 'text-gray-700'
              }`}
          >
            {item.icon && <item.icon size={16} />}
            <span>{item.label}</span>
          </button>
        ))}
      </div>
    </>
  );
}

export default function BookCard({
  book,
  variant = 'grid', // 'grid' | 'list' | 'compact'
  onStatusChange,
  onRatingChange,
  onEdit,
  onDelete,
  showActions = true,
}) {
  const [menuOpen, setMenuOpen] = useState(false);

  const {
    id,
    title,
    author,
    cover_url,
    genres = [],
    publication_year,
    read_status = 'unread',
    rating,
    isbn,
  } = book;

  const statusConfig = READ_STATUS_CONFIG[read_status] || READ_STATUS_CONFIG.unread;
  const StatusIcon = statusConfig.icon;

  const menuItems = [
    { label: 'Edit', icon: Edit, onClick: () => onEdit?.(book) },
    isbn && {
      label: 'View on OpenLibrary',
      icon: ExternalLink,
      onClick: () => window.open(`https://openlibrary.org/isbn/${isbn}`, '_blank'),
    },
    { label: 'Delete', icon: Trash2, onClick: () => onDelete?.(book), danger: true },
  ].filter(Boolean);

  // Grid layout (default)
  if (variant === 'grid') {
    return (
      <Link
        to={`/library/${id}`}
        className="group block bg-white rounded-xl overflow-hidden shadow-sm border border-gray-100 hover:shadow-md transition-all duration-200"
      >
        {/* Cover Image */}
        <div className="aspect-[2/3] relative bg-gradient-to-br from-gray-100 to-gray-200 overflow-hidden">
          {cover_url ? (
            <img
              src={cover_url}
              alt={title}
              className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center">
              <BookOpen className="w-16 h-16 text-gray-300" />
            </div>
          )}

          {/* Status badge */}
          <div className="absolute top-2 left-2">
            <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${statusConfig.color}`}>
              <StatusIcon size={12} />
              {statusConfig.label}
            </span>
          </div>

          {/* Actions menu */}
          {showActions && (
            <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
              <div className="relative">
                <button
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    setMenuOpen(!menuOpen);
                  }}
                  className="p-1.5 bg-white/90 rounded-lg hover:bg-white shadow-sm"
                >
                  <MoreVertical size={16} className="text-gray-600" />
                </button>
                {menuOpen && (
                  <DropdownMenu items={menuItems} onClose={() => setMenuOpen(false)} />
                )}
              </div>
            </div>
          )}
        </div>

        {/* Info */}
        <div className="p-3 space-y-1">
          <h3 className="font-medium text-gray-900 line-clamp-2 leading-tight group-hover:text-indigo-600 transition-colors">
            {title}
          </h3>
          <p className="text-sm text-gray-500 truncate">{author}</p>

          <div className="flex items-center justify-between pt-1">
            {rating ? (
              <StarRating rating={rating} size="sm" readonly />
            ) : (
              <span className="text-xs text-gray-400">Not rated</span>
            )}
            {publication_year && (
              <span className="text-xs text-gray-400">{publication_year}</span>
            )}
          </div>
        </div>
      </Link>
    );
  }

  // List layout
  if (variant === 'list') {
    return (
      <Link
        to={`/library/${id}`}
        className="group flex gap-4 p-4 bg-white rounded-xl border border-gray-100 hover:shadow-md transition-all duration-200"
      >
        {/* Cover */}
        <div className="w-20 h-28 flex-shrink-0 bg-gradient-to-br from-gray-100 to-gray-200 rounded-lg overflow-hidden">
          {cover_url ? (
            <img
              src={cover_url}
              alt={title}
              className="w-full h-full object-cover"
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center">
              <BookOpen className="w-8 h-8 text-gray-300" />
            </div>
          )}
        </div>

        {/* Info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2">
            <div className="min-w-0">
              <h3 className="font-medium text-gray-900 group-hover:text-indigo-600 transition-colors truncate">
                {title}
              </h3>
              <p className="text-sm text-gray-500">{author}</p>
            </div>

            <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${statusConfig.color}`}>
              <StatusIcon size={12} />
              {statusConfig.label}
            </span>
          </div>

          {genres.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-2">
              {genres.slice(0, 3).map((genre) => (
                <span
                  key={genre}
                  className="px-2 py-0.5 bg-gray-100 text-gray-600 text-xs rounded-full"
                >
                  {genre}
                </span>
              ))}
              {genres.length > 3 && (
                <span className="px-2 py-0.5 text-gray-400 text-xs">
                  +{genres.length - 3}
                </span>
              )}
            </div>
          )}

          <div className="flex items-center gap-4 mt-2">
            {rating ? (
              <StarRating rating={rating} size="sm" readonly />
            ) : (
              <span className="text-xs text-gray-400">Not rated</span>
            )}
            {publication_year && (
              <span className="text-xs text-gray-400">{publication_year}</span>
            )}
          </div>
        </div>

        {/* Actions */}
        {showActions && (
          <div className="relative flex-shrink-0">
            <button
              onClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                setMenuOpen(!menuOpen);
              }}
              className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg"
            >
              <MoreVertical size={20} />
            </button>
            {menuOpen && (
              <DropdownMenu items={menuItems} onClose={() => setMenuOpen(false)} />
            )}
          </div>
        )}
      </Link>
    );
  }

  // Compact layout (for detection results)
  if (variant === 'compact') {
    return (
      <div className="flex gap-3 p-3 bg-white rounded-lg border border-gray-100">
        {/* Cover */}
        <div className="w-12 h-16 flex-shrink-0 bg-gray-100 rounded overflow-hidden">
          {cover_url ? (
            <img
              src={cover_url}
              alt={title}
              className="w-full h-full object-cover"
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center">
              <BookOpen className="w-5 h-5 text-gray-300" />
            </div>
          )}
        </div>

        {/* Info */}
        <div className="flex-1 min-w-0">
          <h4 className="font-medium text-gray-900 text-sm truncate">{title}</h4>
          <p className="text-xs text-gray-500 truncate">{author}</p>
          {publication_year && (
            <p className="text-xs text-gray-400 mt-0.5">{publication_year}</p>
          )}
        </div>
      </div>
    );
  }

  return null;
}

export { StarRating };
