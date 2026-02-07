/**
 * Detection results component showing identified books from shelf image.
 */

import React, { useState, useMemo } from 'react';
import {
  Check,
  X,
  Plus,
  AlertTriangle,
  BookOpen,
  Eye,
  ChevronDown,
  ChevronUp,
  Loader2,
  Edit2,
} from 'lucide-react';
import BookCard from './BookCard';

const CONFIDENCE_THRESHOLDS = {
  high: 0.8,
  medium: 0.5,
};

function ConfidenceBadge({ confidence }) {
  const percent = Math.round(confidence * 100);
  
  let color, label;
  if (confidence >= CONFIDENCE_THRESHOLDS.high) {
    color = 'bg-green-100 text-green-700';
    label = 'High';
  } else if (confidence >= CONFIDENCE_THRESHOLDS.medium) {
    color = 'bg-yellow-100 text-yellow-700';
    label = 'Medium';
  } else {
    color = 'bg-red-100 text-red-700';
    label = 'Low';
  }

  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${color}`}>
      {label} ({percent}%)
    </span>
  );
}

function DetectedBookItem({
  detection,
  selected,
  onToggleSelect,
  onAddToLibrary,
  onEdit,
  onViewDetails,
  isAdding,
}) {
  const [expanded, setExpanded] = useState(false);
  const book = detection.identified_book;
  const isLowConfidence = detection.confidence < CONFIDENCE_THRESHOLDS.medium;

  return (
    <div
      className={`border rounded-xl overflow-hidden transition-all duration-200 ${
        selected
          ? 'border-indigo-500 bg-indigo-50/50'
          : 'border-gray-200 bg-white hover:border-gray-300'
      }`}
    >
      {/* Main content */}
      <div className="p-4">
        <div className="flex gap-4">
          {/* Selection checkbox */}
          <button
            onClick={() => onToggleSelect(detection.id)}
            className={`w-6 h-6 rounded-lg border-2 flex items-center justify-center flex-shrink-0 transition-colors ${
              selected
                ? 'bg-indigo-600 border-indigo-600'
                : 'border-gray-300 hover:border-indigo-400'
            }`}
          >
            {selected && <Check size={14} className="text-white" />}
          </button>

          {/* Book cover */}
          <div className="w-16 h-24 flex-shrink-0 bg-gray-100 rounded-lg overflow-hidden">
            {book?.cover_url ? (
              <img
                src={book.cover_url}
                alt={book.title}
                className="w-full h-full object-cover"
              />
            ) : (
              <div className="w-full h-full flex items-center justify-center">
                <BookOpen className="w-8 h-8 text-gray-300" />
              </div>
            )}
          </div>

          {/* Book info */}
          <div className="flex-1 min-w-0">
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0">
                {book ? (
                  <>
                    <h4 className="font-medium text-gray-900 truncate">
                      {book.title}
                    </h4>
                    <p className="text-sm text-gray-500">{book.author}</p>
                    {book.publication_year && (
                      <p className="text-xs text-gray-400 mt-0.5">
                        {book.publication_year}
                      </p>
                    )}
                  </>
                ) : (
                  <div>
                    <p className="font-medium text-gray-900">Unknown Book</p>
                    <p className="text-sm text-gray-500 mt-1">
                      OCR: "{detection.ocr_text}"
                    </p>
                  </div>
                )}
              </div>
              
              <ConfidenceBadge confidence={detection.confidence} />
            </div>

            {/* Warning for low confidence */}
            {isLowConfidence && (
              <div className="flex items-center gap-2 mt-2 p-2 bg-amber-50 border border-amber-200 rounded-lg">
                <AlertTriangle size={16} className="text-amber-500 flex-shrink-0" />
                <span className="text-xs text-amber-700">
                  Low confidence match. Please verify before adding.
                </span>
              </div>
            )}

            {/* Action buttons */}
            <div className="flex items-center gap-2 mt-3">
              {book && (
                <>
                  <button
                    onClick={() => onAddToLibrary(detection)}
                    disabled={isAdding}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition-colors"
                  >
                    {isAdding ? (
                      <Loader2 size={14} className="animate-spin" />
                    ) : (
                      <Plus size={14} />
                    )}
                    <span>Add to Library</span>
                  </button>
                  
                  <button
                    onClick={() => onViewDetails(detection)}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
                  >
                    <Eye size={14} />
                    <span>Details</span>
                  </button>
                </>
              )}
              
              <button
                onClick={() => onEdit(detection)}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <Edit2 size={14} />
                <span>Edit</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Expandable details */}
      {book && (
        <>
          <button
            onClick={() => setExpanded(!expanded)}
            className="w-full px-4 py-2 flex items-center justify-center gap-1 text-sm text-gray-500 hover:bg-gray-50 border-t border-gray-100"
          >
            {expanded ? (
              <>
                <span>Hide details</span>
                <ChevronUp size={16} />
              </>
            ) : (
              <>
                <span>Show more</span>
                <ChevronDown size={16} />
              </>
            )}
          </button>
          
          {expanded && (
            <div className="px-4 pb-4 space-y-3 border-t border-gray-100">
              {book.description && (
                <div>
                  <h5 className="text-xs font-medium text-gray-500 uppercase mb-1">
                    Description
                  </h5>
                  <p className="text-sm text-gray-700 line-clamp-3">
                    {book.description}
                  </p>
                </div>
              )}
              
              {book.genres?.length > 0 && (
                <div>
                  <h5 className="text-xs font-medium text-gray-500 uppercase mb-1">
                    Genres
                  </h5>
                  <div className="flex flex-wrap gap-1">
                    {book.genres.map((genre) => (
                      <span
                        key={genre}
                        className="px-2 py-0.5 bg-gray-100 text-gray-600 text-xs rounded-full"
                      >
                        {genre}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              
              {book.isbn && (
                <div>
                  <h5 className="text-xs font-medium text-gray-500 uppercase mb-1">
                    ISBN
                  </h5>
                  <p className="text-sm text-gray-700 font-mono">{book.isbn}</p>
                </div>
              )}
              
              <div>
                <h5 className="text-xs font-medium text-gray-500 uppercase mb-1">
                  Detection Info
                </h5>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div>
                    <span className="text-gray-500">OCR Text:</span>
                    <span className="ml-1 text-gray-700">
                      {detection.ocr_text || 'N/A'}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-500">Position:</span>
                    <span className="ml-1 text-gray-700">
                      ({Math.round(detection.bbox.x)}, {Math.round(detection.bbox.y)})
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default function DetectionResults({
  detections = [],
  imageUrl,
  onAddSelected,
  onAddAll,
  onEdit,
  isAddingAll,
}) {
  const [selectedIds, setSelectedIds] = useState(new Set());
  const [addingIds, setAddingIds] = useState(new Set());
  const [filter, setFilter] = useState('all'); // 'all', 'high', 'low', 'unknown'

  const toggleSelect = (id) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const selectAll = () => {
    setSelectedIds(new Set(filteredDetections.map((d) => d.id)));
  };

  const deselectAll = () => {
    setSelectedIds(new Set());
  };

  const filteredDetections = useMemo(() => {
    return detections.filter((d) => {
      if (filter === 'all') return true;
      if (filter === 'high') return d.confidence >= CONFIDENCE_THRESHOLDS.high;
      if (filter === 'low') return d.confidence < CONFIDENCE_THRESHOLDS.medium;
      if (filter === 'unknown') return !d.identified_book;
      return true;
    });
  }, [detections, filter]);

  const stats = useMemo(() => ({
    total: detections.length,
    high: detections.filter((d) => d.confidence >= CONFIDENCE_THRESHOLDS.high).length,
    low: detections.filter((d) => d.confidence < CONFIDENCE_THRESHOLDS.medium).length,
    unknown: detections.filter((d) => !d.identified_book).length,
  }), [detections]);

  const handleAddSingle = async (detection) => {
    setAddingIds((prev) => new Set(prev).add(detection.id));
    try {
      await onAddSelected([detection]);
    } finally {
      setAddingIds((prev) => {
        const next = new Set(prev);
        next.delete(detection.id);
        return next;
      });
    }
  };

  const handleAddSelected = async () => {
    const selected = detections.filter((d) => selectedIds.has(d.id));
    await onAddSelected(selected);
    setSelectedIds(new Set());
  };

  if (detections.length === 0) {
    return (
      <div className="text-center py-12">
        <BookOpen className="w-12 h-12 text-gray-300 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-gray-900 mb-1">No books detected</h3>
        <p className="text-gray-500">
          Try uploading a clearer image of your bookshelf
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Summary stats */}
      <div className="flex flex-wrap items-center gap-4 p-4 bg-gray-50 rounded-xl">
        <div className="flex-1">
          <h3 className="font-semibold text-gray-900">
            {stats.total} book{stats.total !== 1 ? 's' : ''} detected
          </h3>
          <p className="text-sm text-gray-500">
            {stats.high} high confidence • {stats.low} low confidence • {stats.unknown} unidentified
          </p>
        </div>
        
        <div className="flex items-center gap-2">
          <button
            onClick={onAddAll}
            disabled={isAddingAll || stats.high === 0}
            className="inline-flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition-colors"
          >
            {isAddingAll ? (
              <Loader2 size={18} className="animate-spin" />
            ) : (
              <Plus size={18} />
            )}
            <span>Add All High Confidence</span>
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm text-gray-500">Filter:</span>
        {[
          { key: 'all', label: 'All' },
          { key: 'high', label: 'High Confidence' },
          { key: 'low', label: 'Low Confidence' },
          { key: 'unknown', label: 'Unidentified' },
        ].map((option) => (
          <button
            key={option.key}
            onClick={() => setFilter(option.key)}
            className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
              filter === option.key
                ? 'bg-indigo-100 text-indigo-700'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            {option.label}
          </button>
        ))}
        
        <div className="flex-1" />
        
        {/* Selection controls */}
        {selectedIds.size > 0 ? (
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-500">
              {selectedIds.size} selected
            </span>
            <button
              onClick={handleAddSelected}
              className="inline-flex items-center gap-1 px-3 py-1.5 bg-indigo-600 text-white text-sm rounded-lg hover:bg-indigo-700"
            >
              <Plus size={14} />
              Add Selected
            </button>
            <button
              onClick={deselectAll}
              className="p-1.5 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded"
            >
              <X size={18} />
            </button>
          </div>
        ) : (
          <button
            onClick={selectAll}
            className="text-sm text-indigo-600 hover:text-indigo-700"
          >
            Select All
          </button>
        )}
      </div>

      {/* Detection list */}
      <div className="space-y-3">
        {filteredDetections.map((detection) => (
          <DetectedBookItem
            key={detection.id}
            detection={detection}
            selected={selectedIds.has(detection.id)}
            onToggleSelect={toggleSelect}
            onAddToLibrary={handleAddSingle}
            onEdit={onEdit}
            onViewDetails={(d) => console.log('View details', d)}
            isAdding={addingIds.has(detection.id)}
          />
        ))}
      </div>

      {filteredDetections.length === 0 && (
        <div className="text-center py-8 text-gray-500">
          No books match the current filter
        </div>
      )}
    </div>
  );
}
