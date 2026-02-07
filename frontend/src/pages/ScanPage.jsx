/**
 * Scan page for detecting books from shelf images.
 */

import React, { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Camera,
  Upload,
  Loader2,
  CheckCircle,
  AlertCircle,
  ArrowRight,
  BookOpen,
  Sparkles,
} from 'lucide-react';
import { detectionApi, booksApi } from '../services/api';
import ImageUploader from '../components/ImageUploader';
import DetectionResults from '../components/DetectionResults';

const STEPS = {
  UPLOAD: 'upload',
  PROCESSING: 'processing',
  RESULTS: 'results',
  COMPLETE: 'complete',
};

function StepIndicator({ currentStep }) {
  const steps = [
    { key: STEPS.UPLOAD, label: 'Upload Image' },
    { key: STEPS.PROCESSING, label: 'Detect Books' },
    { key: STEPS.RESULTS, label: 'Review & Add' },
  ];

  const currentIndex = steps.findIndex((s) => s.key === currentStep);

  return (
    <div className="flex items-center justify-center gap-4 mb-8">
      {steps.map((step, index) => {
        const isComplete = index < currentIndex;
        const isCurrent = index === currentIndex;

        return (
          <React.Fragment key={step.key}>
            <div className="flex items-center gap-2">
              <div
                className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium transition-colors ${isComplete
                  ? 'bg-green-500 text-white'
                  : isCurrent
                    ? 'bg-indigo-600 text-white'
                    : 'bg-gray-200 text-gray-500'
                  }`}
              >
                {isComplete ? <CheckCircle size={18} /> : index + 1}
              </div>
              <span
                className={`text-sm font-medium ${isCurrent ? 'text-gray-900' : 'text-gray-500'
                  }`}
              >
                {step.label}
              </span>
            </div>
            {index < steps.length - 1 && (
              <div
                className={`w-12 h-0.5 ${index < currentIndex ? 'bg-green-500' : 'bg-gray-200'
                  }`}
              />
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
}

function ProcessingState({ progress }) {
  const stages = [
    { label: 'Analyzing image...', threshold: 20 },
    { label: 'Detecting book spines...', threshold: 40 },
    { label: 'Extracting text with OCR...', threshold: 60 },
    { label: 'Identifying books...', threshold: 80 },
    { label: 'Fetching metadata...', threshold: 100 },
  ];

  const currentStage = stages.find((s) => progress < s.threshold) || stages[stages.length - 1];

  return (
    <div className="max-w-md mx-auto text-center py-12">
      <div className="w-20 h-20 mx-auto mb-6 relative">
        <div className="absolute inset-0 bg-indigo-100 rounded-full animate-ping opacity-50" />
        <div className="relative w-full h-full bg-gradient-to-br from-indigo-500 to-purple-600 rounded-full flex items-center justify-center">
          <Sparkles className="w-10 h-10 text-white animate-pulse" />
        </div>
      </div>

      <h2 className="text-xl font-semibold text-gray-900 mb-2">
        {currentStage.label}
      </h2>
      <p className="text-gray-500 mb-6">
        This may take a few moments depending on the image size
      </p>

      <div className="w-full bg-gray-200 rounded-full h-2 mb-2">
        <div
          className="bg-gradient-to-r from-indigo-500 to-purple-600 h-2 rounded-full transition-all duration-500"
          style={{ width: `${progress}%` }}
        />
      </div>
      <p className="text-sm text-gray-400">{Math.round(progress)}% complete</p>
    </div>
  );
}

function CompletionState({ added, failed, onScanAnother, onViewLibrary }) {
  return (
    <div className="max-w-md mx-auto text-center py-12">
      <div className="w-20 h-20 mx-auto mb-6 bg-green-100 rounded-full flex items-center justify-center">
        <CheckCircle className="w-10 h-10 text-green-600" />
      </div>

      <h2 className="text-xl font-semibold text-gray-900 mb-2">
        Books added to your library!
      </h2>
      <p className="text-gray-500 mb-6">
        Successfully added {added} book{added !== 1 ? 's' : ''}
        {failed > 0 && ` (${failed} failed)`}
      </p>

      <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
        <button
          onClick={onScanAnother}
          className="inline-flex items-center gap-2 px-4 py-2 border border-gray-200 rounded-lg hover:bg-gray-50"
        >
          <Camera size={18} />
          Scan Another
        </button>
        <button
          onClick={onViewLibrary}
          className="inline-flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700"
        >
          View Library
          <ArrowRight size={18} />
        </button>
      </div>
    </div>
  );
}

export default function ScanPage() {
  const navigate = useNavigate();

  const [step, setStep] = useState(STEPS.UPLOAD);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [uploadedImageUrl, setUploadedImageUrl] = useState(null);
  const [detections, setDetections] = useState([]);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const [addingAll, setAddingAll] = useState(false);
  const [completionStats, setCompletionStats] = useState({ added: 0, failed: 0 });

  const handleUpload = useCallback(async (file) => {
    setUploadedFile(file);
    setUploadedImageUrl(URL.createObjectURL(file));
    setError(null);
    setStep(STEPS.PROCESSING);
    setProgress(0);

    try {
      // Simulate progress updates (in production, use SSE or polling)
      const progressInterval = setInterval(() => {
        setProgress((prev) => Math.min(prev + Math.random() * 15, 90));
      }, 500);

      // Call detection API
      const response = await detectionApi.detect(file, {
        mode: 'full',
        autoIdentify: true,
      });

      clearInterval(progressInterval);
      setProgress(100);

      // Process detections
      const processedDetections = (response.detected_books || []).map((det, index) => ({
        id: `det-${index}`,
        confidence: det.detection_confidence,
        bbox: det.bbox,
        ocr_text: det.ocr_text,
        identified_book: det.identified ? {
          title: det.title,
          author: det.author,
          isbn: det.book_id, // Using book_id as pseudo-ISBN/ID for now
          // Additional fields if available in backend response
          confidence: det.identification_confidence,
          // Map backend fields
          genres: det.genres || [],
          description: det.description || "",
          cover_url: det.cover_url || null,
          publication_year: det.publication_year || null,
          publisher: det.publisher || null,
          isbn_13: det.isbn_13 || null,
        } : null,
      }));

      setDetections(processedDetections);
      setStep(STEPS.RESULTS);
    } catch (err) {
      console.error('Detection failed:', err);
      setError(err.message || 'Failed to process image. Please try again.');
      setStep(STEPS.UPLOAD);
    }
  }, []);

  const handleAddSelected = useCallback(async (selectedDetections) => {
    const results = { added: 0, failed: 0 };

    for (const detection of selectedDetections) {
      if (!detection.identified_book) continue;

      try {
        await booksApi.create({
          title: detection.identified_book.title,
          author: detection.identified_book.author,
          isbn: detection.identified_book.isbn,
          cover_url: detection.identified_book.cover_url,
          publication_year: detection.identified_book.publication_year,
          genres: detection.identified_book.genres,
          description: detection.identified_book.description,
        });
        results.added++;
      } catch (err) {
        console.error('Failed to add book:', err);
        results.failed++;
      }
    }

    // Remove added detections from list
    const addedIds = new Set(selectedDetections.map((d) => d.id));
    setDetections((prev) => prev.filter((d) => !addedIds.has(d.id)));

    return results;
  }, []);

  const handleAddAll = useCallback(async () => {
    setAddingAll(true);

    const highConfidenceDetections = detections.filter(
      (d) => d.identified_book && d.confidence >= 0.8
    );

    const results = await handleAddSelected(highConfidenceDetections);

    setAddingAll(false);

    if (detections.length - highConfidenceDetections.length === 0) {
      setCompletionStats(results);
      setStep(STEPS.COMPLETE);
    }
  }, [detections, handleAddSelected]);

  const handleEdit = useCallback((detection) => {
    // TODO: Open edit modal
    console.log('Edit detection:', detection);
  }, []);

  const handleScanAnother = useCallback(() => {
    setStep(STEPS.UPLOAD);
    setUploadedFile(null);
    setUploadedImageUrl(null);
    setDetections([]);
    setProgress(0);
    setError(null);
    setCompletionStats({ added: 0, failed: 0 });
  }, []);

  return (
    <div className="max-w-4xl mx-auto">
      {/* Header */}
      <div className="text-center mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Scan Your Bookshelf</h1>
        <p className="text-gray-500 mt-1">
          Upload a photo and let AI detect and identify your books
        </p>
      </div>

      {/* Step indicator */}
      {step !== STEPS.COMPLETE && <StepIndicator currentStep={step} />}

      {/* Main content */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        {step === STEPS.UPLOAD && (
          <div className="space-y-6">
            <ImageUploader
              onUpload={handleUpload}
              uploading={false}
              error={error}
            />

            {/* Tips */}
            <div className="bg-gray-50 rounded-lg p-4">
              <h3 className="font-medium text-gray-900 mb-2">Tips for best results:</h3>
              <ul className="space-y-2 text-sm text-gray-600">
                <li className="flex items-start gap-2">
                  <CheckCircle size={16} className="text-green-500 mt-0.5 flex-shrink-0" />
                  Ensure good lighting and avoid shadows
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle size={16} className="text-green-500 mt-0.5 flex-shrink-0" />
                  Keep the camera straight and level with the shelf
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle size={16} className="text-green-500 mt-0.5 flex-shrink-0" />
                  Make sure book spines are clearly visible
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle size={16} className="text-green-500 mt-0.5 flex-shrink-0" />
                  For large shelves, take multiple photos of sections
                </li>
              </ul>
            </div>
          </div>
        )}

        {step === STEPS.PROCESSING && <ProcessingState progress={progress} />}

        {step === STEPS.RESULTS && (
          <div className="space-y-6">
            {/* Original image preview with bounding boxes */}
            {uploadedImageUrl && (
              <div className="relative rounded-lg overflow-hidden bg-gray-100 max-h-[500px] group">
                <img
                  src={uploadedImageUrl}
                  alt="Uploaded shelf"
                  className="w-full h-full object-contain"
                />

                {/* Bounding Box Overlay */}
                <div className="absolute inset-0 pointer-events-none">
                  {detections.map((det) => {
                    const [x1, y1, x2, y2] = det.bbox;
                    const width = (x2 - x1) * 100;
                    const height = (y2 - y1) * 100;
                    const left = x1 * 100;
                    const top = y1 * 100;

                    const isIdentified = !!det.identified_book;
                    const isHighConfidence = det.confidence >= 0.8;

                    // Default to blue for detected but unidentified items (better visibility than gray)
                    let borderColor = 'border-blue-500';
                    let bgColor = 'bg-blue-500/20';

                    if (isIdentified) {
                      if (isHighConfidence) {
                        borderColor = 'border-green-500';
                        bgColor = 'bg-green-500/20';
                      } else {
                        borderColor = 'border-amber-500';
                        bgColor = 'bg-amber-500/20';
                      }
                    }

                    return (
                      <div
                        key={det.id}
                        className={`absolute border-2 ${borderColor} ${bgColor} transition-colors duration-200`}
                        style={{
                          left: `${left}%`,
                          top: `${top}%`,
                          width: `${width}%`,
                          height: `${height}%`,
                        }}
                      >
                        {/* Optional: Tooltip on hover */}
                        <div className="absolute -top-6 left-0 bg-black/75 text-white text-[10px] px-1.5 py-0.5 rounded whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity">
                          {det.identified_book ? det.identified_book.title.slice(0, 20) + (det.identified_book.title.length > 20 ? '...' : '') : 'Unidentified'}
                          <span className="ml-1 opacity-75">({Math.round(det.confidence * 100)}%)</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Detection results */}
            <DetectionResults
              detections={detections}
              imageUrl={uploadedImageUrl}
              onAddSelected={handleAddSelected}
              onAddAll={handleAddAll}
              onEdit={handleEdit}
              isAddingAll={addingAll}
            />

            {/* Done button */}
            {detections.length > 0 && (
              <div className="flex justify-end pt-4 border-t border-gray-100 gap-3">
                <button
                  onClick={() => {
                    setCompletionStats({ added: completionStats.added, failed: completionStats.failed });
                    setStep(STEPS.COMPLETE);
                  }}
                  className="px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-lg text-sm font-medium"
                >
                  Skip Remaining
                </button>
                <button
                  onClick={async () => {
                    setAddingAll(true);
                    // Filter for identified books
                    const validDetections = detections.filter(d => d.identified_book);
                    let newStats = { added: 0, failed: 0 };

                    if (validDetections.length > 0) {
                      try {
                        newStats = await handleAddSelected(validDetections);
                      } catch (err) {
                        console.error("Error saving remaining books:", err);
                      }
                    }

                    setAddingAll(false);
                    setCompletionStats(prev => ({
                      added: prev.added + newStats.added,
                      failed: prev.failed + newStats.failed
                    }));
                    setStep(STEPS.COMPLETE);
                  }}
                  disabled={addingAll}
                  className="inline-flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition-colors"
                >
                  {addingAll ? (
                    <Loader2 size={18} className="animate-spin" />
                  ) : (
                    <CheckCircle size={18} />
                  )}
                  Save All & Finish
                </button>
              </div>
            )}
          </div>
        )}

        {step === STEPS.COMPLETE && (
          <CompletionState
            added={completionStats.added}
            failed={completionStats.failed}
            onScanAnother={handleScanAnother}
            onViewLibrary={() => navigate('/library')}
          />
        )}
      </div>
    </div>
  );
}
