/**
 * Image uploader component with drag-and-drop, preview, and progress.
 */

import React, { useCallback, useState, useRef } from 'react';
import { Upload, Image, X, Camera, Loader2, AlertCircle, CheckCircle } from 'lucide-react';

const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
const MAX_SIZE_MB = 10;

export default function ImageUploader({
  onUpload,
  onRemove,
  multiple = false,
  maxFiles = 10,
  disabled = false,
  uploading = false,
  progress = 0,
  error = null,
  success = null,
}) {
  const [files, setFiles] = useState([]);
  const [previews, setPreviews] = useState([]);
  const [dragActive, setDragActive] = useState(false);
  const [localError, setLocalError] = useState(null);
  const fileInputRef = useRef(null);

  const validateFile = (file) => {
    if (!ACCEPTED_TYPES.includes(file.type)) {
      return `Invalid file type. Please upload JPEG, PNG, or WebP images.`;
    }
    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      return `File too large. Maximum size is ${MAX_SIZE_MB}MB.`;
    }
    return null;
  };

  const processFiles = useCallback((newFiles) => {
    setLocalError(null);
    
    const validFiles = [];
    const newPreviews = [];
    
    for (const file of newFiles) {
      if (!multiple && files.length + validFiles.length >= 1) break;
      if (multiple && files.length + validFiles.length >= maxFiles) break;
      
      const error = validateFile(file);
      if (error) {
        setLocalError(error);
        continue;
      }
      
      validFiles.push(file);
      
      // Create preview URL
      const previewUrl = URL.createObjectURL(file);
      newPreviews.push({
        file,
        url: previewUrl,
        name: file.name,
        size: file.size,
      });
    }
    
    if (validFiles.length > 0) {
      const updatedFiles = multiple ? [...files, ...validFiles] : validFiles;
      const updatedPreviews = multiple ? [...previews, ...newPreviews] : newPreviews;
      
      setFiles(updatedFiles);
      setPreviews(updatedPreviews);
      
      if (onUpload) {
        onUpload(multiple ? updatedFiles : updatedFiles[0]);
      }
    }
  }, [files, previews, multiple, maxFiles, onUpload]);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (disabled || uploading) return;
    
    const droppedFiles = Array.from(e.dataTransfer.files);
    processFiles(droppedFiles);
  }, [disabled, uploading, processFiles]);

  const handleFileSelect = useCallback((e) => {
    const selectedFiles = Array.from(e.target.files);
    processFiles(selectedFiles);
    // Reset input so same file can be selected again
    e.target.value = '';
  }, [processFiles]);

  const handleRemove = useCallback((index) => {
    // Revoke preview URL to free memory
    URL.revokeObjectURL(previews[index].url);
    
    const newFiles = files.filter((_, i) => i !== index);
    const newPreviews = previews.filter((_, i) => i !== index);
    
    setFiles(newFiles);
    setPreviews(newPreviews);
    
    if (onRemove) {
      onRemove(index, newFiles);
    }
  }, [files, previews, onRemove]);

  const handleCameraCapture = useCallback(() => {
    // Trigger file input with camera capture
    if (fileInputRef.current) {
      fileInputRef.current.setAttribute('capture', 'environment');
      fileInputRef.current.click();
      // Remove capture attribute after click
      setTimeout(() => {
        fileInputRef.current?.removeAttribute('capture');
      }, 100);
    }
  }, []);

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const displayError = error || localError;

  return (
    <div className="space-y-4">
      {/* Drop Zone */}
      <div
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        className={`relative border-2 border-dashed rounded-xl p-8 text-center transition-all duration-200 ${
          dragActive
            ? 'border-indigo-500 bg-indigo-50'
            : disabled || uploading
            ? 'border-gray-200 bg-gray-50'
            : 'border-gray-300 hover:border-indigo-400 hover:bg-gray-50 cursor-pointer'
        }`}
        onClick={() => !disabled && !uploading && fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept={ACCEPTED_TYPES.join(',')}
          multiple={multiple}
          onChange={handleFileSelect}
          disabled={disabled || uploading}
          className="hidden"
        />

        {uploading ? (
          <div className="space-y-4">
            <Loader2 className="w-12 h-12 mx-auto text-indigo-500 animate-spin" />
            <div>
              <p className="text-gray-700 font-medium">Processing image...</p>
              <p className="text-sm text-gray-500">Detecting books</p>
            </div>
            {progress > 0 && (
              <div className="w-full max-w-xs mx-auto bg-gray-200 rounded-full h-2">
                <div
                  className="bg-indigo-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-4">
            <div className="w-16 h-16 mx-auto bg-gray-100 rounded-full flex items-center justify-center">
              <Upload className="w-8 h-8 text-gray-400" />
            </div>
            <div>
              <p className="text-gray-700 font-medium">
                {dragActive ? 'Drop your image here' : 'Drag & drop shelf image'}
              </p>
              <p className="text-sm text-gray-500 mt-1">
                or click to browse • JPEG, PNG, WebP up to {MAX_SIZE_MB}MB
              </p>
            </div>
            
            {/* Mobile camera button */}
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                handleCameraCapture();
              }}
              className="inline-flex items-center gap-2 px-4 py-2 bg-indigo-100 text-indigo-700 rounded-lg hover:bg-indigo-200 transition-colors md:hidden"
            >
              <Camera size={20} />
              <span>Take Photo</span>
            </button>
          </div>
        )}
      </div>

      {/* Error Message */}
      {displayError && (
        <div className="flex items-center gap-2 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700">
          <AlertCircle size={20} />
          <span className="text-sm">{displayError}</span>
        </div>
      )}

      {/* Success Message */}
      {success && (
        <div className="flex items-center gap-2 p-3 bg-green-50 border border-green-200 rounded-lg text-green-700">
          <CheckCircle size={20} />
          <span className="text-sm">{success}</span>
        </div>
      )}

      {/* Preview Grid */}
      {previews.length > 0 && (
        <div className={`grid gap-4 ${multiple ? 'grid-cols-2 md:grid-cols-3 lg:grid-cols-4' : ''}`}>
          {previews.map((preview, index) => (
            <div
              key={preview.url}
              className="relative group bg-white border border-gray-200 rounded-lg overflow-hidden"
            >
              <div className={`aspect-[4/3] relative ${!multiple ? 'max-h-96' : ''}`}>
                <img
                  src={preview.url}
                  alt={preview.name}
                  className="w-full h-full object-cover"
                />
                
                {/* Overlay with remove button */}
                <div className="absolute inset-0 bg-black/0 group-hover:bg-black/30 transition-colors flex items-center justify-center">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleRemove(index);
                    }}
                    disabled={uploading}
                    className="opacity-0 group-hover:opacity-100 p-2 bg-white rounded-full shadow-lg hover:bg-red-50 transition-all duration-200"
                  >
                    <X className="w-5 h-5 text-gray-700 hover:text-red-600" />
                  </button>
                </div>
              </div>
              
              {/* File info */}
              <div className="p-2 border-t border-gray-100">
                <p className="text-sm text-gray-700 truncate" title={preview.name}>
                  {preview.name}
                </p>
                <p className="text-xs text-gray-500">
                  {formatFileSize(preview.size)}
                </p>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Multiple file limit indicator */}
      {multiple && (
        <p className="text-sm text-gray-500 text-center">
          {files.length} of {maxFiles} images selected
        </p>
      )}
    </div>
  );
}
