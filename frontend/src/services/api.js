/**
 * API service for ShelfSense backend communication.
 * 
 * Provides typed methods for all API endpoints with error handling.
 */

import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api/v1';

// Create axios instance with defaults
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for auth
api.interceptors.request.use(
  (config) => {
    const apiKey = localStorage.getItem('shelfsense_api_key');
    if (apiKey) {
      config.headers['X-API-Key'] = apiKey;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 429) {
      const retryAfter = error.response.headers['retry-after'];
      error.message = `Rate limited. Please try again in ${retryAfter || 60} seconds.`;
    } else if (error.response?.data?.error?.message) {
      error.message = error.response.data.error.message;
    }
    return Promise.reject(error);
  }
);

// =============================================================================
// Auth API
// =============================================================================

export const authApi = {
  /**
   * Register a new user.
   */
  async signup(userData) {
    const response = await api.post('/auth/signup', userData);
    return response.data;
  },

  /**
   * Login user.
   */
  async login(formData) {
    // fastAPI expects form-data for OAuth2
    const params = new URLSearchParams();
    params.append('username', formData.email);
    params.append('password', formData.password);

    const response = await api.post('/auth/token', params, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    });
    return response.data;
  },

  /**
   * Get current user.
   */
  async getCurrentUser() {
    const response = await api.get('/auth/me', {
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      }
    });
    return response.data;
  },
};

// =============================================================================
// Books API
// =============================================================================

export const booksApi = {
  /**
   * List books with pagination and filters.
   */
  async list({ page = 1, limit = 20, genre, author, readStatus, sortBy, sortOrder } = {}) {
    const params = new URLSearchParams();
    params.append('page', page);
    params.append('limit', limit);
    if (genre) params.append('genre', genre);
    if (author) params.append('author', author);
    if (readStatus) params.append('read_status', readStatus);
    if (sortBy) params.append('sort_by', sortBy);
    if (sortOrder) params.append('sort_order', sortOrder);

    const response = await api.get(`/books?${params}`);
    return response.data;
  },

  /**
   * Get single book by ID.
   */
  async get(bookId) {
    const response = await api.get(`/books/${bookId}`);
    return response.data;
  },

  /**
   * Create a new book.
   */
  async create(bookData) {
    const response = await api.post('/books', bookData);
    return response.data;
  },

  /**
   * Update book.
   */
  async update(bookId, updates) {
    const response = await api.put(`/books/${bookId}`, updates);
    return response.data;
  },

  /**
   * Delete book.
   */
  async delete(bookId) {
    await api.delete(`/books/${bookId}`);
  },

  /**
   * Search books.
   */
  async search({ query, mode = 'hybrid', filters, limit = 20 }) {
    const response = await api.post('/books/search', {
      query,
      mode,
      filters,
      limit,
    });
    return response.data;
  },

  /**
   * Get book by ISBN.
   */
  async getByIsbn(isbn) {
    const response = await api.get(`/books/isbn/${isbn}`);
    return response.data;
  },

  /**
   * Update read status.
   */
  async updateReadStatus(bookId, status) {
    const response = await api.post(`/books/${bookId}/read-status`, { status });
    return response.data;
  },

  /**
   * Rate a book.
   */
  async rate(bookId, rating) {
    const response = await api.post(`/books/${bookId}/rating`, { rating });
    return response.data;
  },
};

// =============================================================================
// Detection API
// =============================================================================

export const detectionApi = {
  /**
   * Detect books from uploaded image.
   */
  async detect(file, options = {}) {
    const formData = new FormData();
    formData.append('file', file);
    if (options.mode) formData.append('mode', options.mode);
    if (options.autoIdentify !== undefined) {
      formData.append('auto_identify', options.autoIdentify);
    }
    if (options.confidenceThreshold) {
      formData.append('confidence_threshold', options.confidenceThreshold);
    }

    const response = await api.post('/detect', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 60000, // Longer timeout for image processing
    });
    return response.data;
  },

  /**
   * Detect books from base64 image.
   */
  async detectBase64(base64Image, options = {}) {
    const response = await api.post('/detect/base64', {
      image_data: base64Image,
      ...options,
    }, {
      timeout: 60000,
    });
    return response.data;
  },

  /**
   * Batch detect from multiple images.
   */
  async detectBatch(files, options = {}) {
    const formData = new FormData();
    files.forEach((file, index) => {
      formData.append(`files`, file);
    });
    if (options.mode) formData.append('mode', options.mode);

    const response = await api.post('/detect/batch', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 120000, // 2 minutes for batch
    });
    return response.data;
  },

  /**
   * Detect and add books to library.
   */
  async detectAndAdd(file, options = {}) {
    const formData = new FormData();
    formData.append('file', file);
    if (options.confidenceThreshold) {
      formData.append('confidence_threshold', options.confidenceThreshold);
    }

    const response = await api.post('/detect/add-to-library', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 90000,
    });
    return response.data;
  },

  /**
   * Check detection job status.
   */
  async getJobStatus(jobId) {
    const response = await api.get(`/detect/job/${jobId}`);
    return response.data;
  },
};

// =============================================================================
// Chat API
// =============================================================================

export const chatApi = {
  /**
   * Send chat message.
   */
  async send(message, conversationId = null) {
    const response = await api.post('/chat', {
      message,
      conversation_id: conversationId,
    });
    return response.data;
  },

  /**
   * Stream chat response using SSE.
   */
  streamMessage(message, conversationId, onChunk, onDone, onError) {
    const url = new URL(`${API_BASE_URL}/chat/stream`, window.location.origin);

    fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
      },
      body: JSON.stringify({
        message,
        conversation_id: conversationId,
      }),
    }).then(async (response) => {
      if (!response.ok) {
        const error = await response.json();
        onError(new Error(error.error?.message || 'Chat request failed'));
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') {
              onDone();
              return;
            }
            try {
              const chunk = JSON.parse(data);

              // Handle error in data stream
              if (chunk.error) {
                onError(new Error(chunk.error));
                return;
              }

              onChunk(chunk);
            } catch (e) {
              console.error('Failed to parse SSE chunk:', e);
            }
          }
        }
      }

      // Fallback: if stream ends without [DONE], treat as done
      onDone();
    }).catch(onError);
  },

  /**
   * List conversations.
   */
  async listConversations({ limit = 20, offset = 0 } = {}) {
    const response = await api.get(`/chat/conversations?limit=${limit}&offset=${offset}`);
    return response.data;
  },

  /**
   * Get conversation by ID.
   */
  async getConversation(conversationId) {
    const response = await api.get(`/chat/conversations/${conversationId}`);
    return response.data;
  },

  /**
   * Delete conversation.
   */
  async deleteConversation(conversationId) {
    await api.delete(`/chat/conversations/${conversationId}`);
  },

  /**
   * Get quick recommendation.
   */
  async quickRecommend(options = {}) {
    const response = await api.post('/chat/quick/recommend', options);
    return response.data;
  },

  /**
   * Get library summary.
   */
  async quickSummary() {
    const response = await api.post('/chat/quick/summary');
    return response.data;
  },
};

// =============================================================================
// Analytics API
// =============================================================================

export const analyticsApi = {
  /**
   * Get library statistics.
   */
  async getStats() {
    const response = await api.get('/analytics/stats');
    return response.data;
  },

  /**
   * Get diversity analysis.
   */
  async getDiversity() {
    const response = await api.get('/analytics/diversity');
    return response.data;
  },

  /**
   * Get recommendations.
   */
  async getRecommendations({ count = 10, includeExploration = true, basedOnBookId } = {}) {
    const params = new URLSearchParams();
    params.append('count', count);
    params.append('include_exploration', includeExploration);
    if (basedOnBookId) params.append('based_on_book_id', basedOnBookId);

    const response = await api.get(`/analytics/recommendations?${params}`);
    return response.data;
  },

  /**
   * Get reading trends.
   */
  async getTrends() {
    const response = await api.get('/analytics/trends');
    return response.data;
  },

  /**
   * Get genre analysis.
   */
  async getGenreAnalysis(genre) {
    const response = await api.get(`/analytics/genre/${encodeURIComponent(genre)}`);
    return response.data;
  },

  /**
   * Get author analysis.
   */
  async getAuthorAnalysis(authorName) {
    const response = await api.get(`/analytics/author/${encodeURIComponent(authorName)}`);
    return response.data;
  },
};

// =============================================================================
// Health API
// =============================================================================

export const healthApi = {
  /**
   * Check API health.
   */
  async check() {
    const response = await api.get('/health');
    return response.data;
  },
};

export default api;
