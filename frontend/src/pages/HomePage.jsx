/**
 * Home page with library overview, quick actions, and recent activity.
 */

import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import {
  BookOpen,
  Camera,
  MessageCircle,
  BarChart2,
  ArrowRight,
  TrendingUp,
  Clock,
  Star,
  Plus,
  Sparkles,
  Loader2,
} from 'lucide-react';
import { analyticsApi, booksApi } from '../services/api';
import BookCard from '../components/BookCard';

function StatCard({ icon: Icon, label, value, change, color = 'indigo' }) {
  const colorClasses = {
    indigo: 'bg-indigo-50 text-indigo-600',
    purple: 'bg-purple-50 text-purple-600',
    green: 'bg-green-50 text-green-600',
    amber: 'bg-amber-50 text-amber-600',
  };

  return (
    <div className="bg-white rounded-xl p-5 border border-gray-100 shadow-sm">
      <div className="flex items-center gap-4">
        <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${colorClasses[color]}`}>
          <Icon size={24} />
        </div>
        <div>
          <p className="text-2xl font-bold text-gray-900">{value}</p>
          <p className="text-sm text-gray-500">{label}</p>
        </div>
      </div>
      {change && (
        <div className="flex items-center gap-1 mt-3 text-sm">
          <TrendingUp size={14} className="text-green-500" />
          <span className="text-green-600">{change}</span>
          <span className="text-gray-400">this month</span>
        </div>
      )}
    </div>
  );
}

function QuickAction({ icon: Icon, title, description, to, color = 'indigo' }) {
  const colorClasses = {
    indigo: 'from-indigo-500 to-indigo-600 group-hover:from-indigo-600 group-hover:to-indigo-700',
    purple: 'from-purple-500 to-purple-600 group-hover:from-purple-600 group-hover:to-purple-700',
    green: 'from-green-500 to-green-600 group-hover:from-green-600 group-hover:to-green-700',
    amber: 'from-amber-500 to-amber-600 group-hover:from-amber-600 group-hover:to-amber-700',
  };

  return (
    <Link
      to={to}
      className="group block p-5 bg-white rounded-xl border border-gray-100 shadow-sm hover:shadow-md transition-all duration-200"
    >
      <div className="flex items-center gap-4">
        <div
          className={`w-12 h-12 rounded-xl bg-gradient-to-br ${colorClasses[color]} flex items-center justify-center shadow-sm`}
        >
          <Icon size={24} className="text-white" />
        </div>
        <div className="flex-1">
          <h3 className="font-semibold text-gray-900 group-hover:text-indigo-600 transition-colors">
            {title}
          </h3>
          <p className="text-sm text-gray-500">{description}</p>
        </div>
        <ArrowRight
          size={20}
          className="text-gray-300 group-hover:text-indigo-500 transform group-hover:translate-x-1 transition-all"
        />
      </div>
    </Link>
  );
}

function RecommendationCard({ book, reason }) {
  return (
    <Link
      to={`/library/${book.id}`}
      className="group flex gap-3 p-3 bg-gray-50 rounded-lg hover:bg-indigo-50 transition-colors"
    >
      <div className="w-10 h-14 bg-gray-200 rounded overflow-hidden flex-shrink-0">
        {book.cover_url ? (
          <img src={book.cover_url} alt={book.title} className="w-full h-full object-cover" />
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <BookOpen className="w-5 h-5 text-gray-400" />
          </div>
        )}
      </div>
      <div className="flex-1 min-w-0">
        <h4 className="font-medium text-gray-900 text-sm truncate group-hover:text-indigo-600">
          {book.title}
        </h4>
        <p className="text-xs text-gray-500 truncate">{book.author}</p>
        <p className="text-xs text-indigo-600 mt-1">{reason}</p>
      </div>
    </Link>
  );
}

export default function HomePage() {
  const [stats, setStats] = useState(null);
  const [recentBooks, setRecentBooks] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);

        // Fetch stats, recent books, and recommendations in parallel
        const [statsData, booksData, recsData] = await Promise.all([
          analyticsApi.getStats(),
          booksApi.list({ limit: 4, sortBy: 'created_at', sortOrder: 'desc' }),
          analyticsApi.getRecommendations({ count: 3 }),
        ]);

        setStats(statsData);
        setRecentBooks(booksData.books || []);
        setRecommendations(recsData.recommendations || []);
      } catch (err) {
        console.error('Failed to fetch home data:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 text-indigo-500 animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Welcome back!</h1>
        <p className="text-gray-500 mt-1">Here's what's happening with your library.</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          icon={BookOpen}
          label="Total Books"
          value={stats?.total_books || 0}
          color="indigo"
        />
        <StatCard
          icon={Clock}
          label="To Read"
          value={stats?.unread_count || 0}
          color="amber"
        />
        <StatCard
          icon={BookOpen}
          label="Currently Reading"
          value={stats?.reading_count || 0}
          color="purple"
        />
        <StatCard
          icon={Star}
          label="Completed"
          value={stats?.read_count || 0}
          color="green"
        />
      </div>

      {/* Quick Actions */}
      <div>
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h2>
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <QuickAction
            icon={Camera}
            title="Scan Shelf"
            description="Add books from a photo"
            to="/scan"
            color="indigo"
          />
          <QuickAction
            icon={MessageCircle}
            title="Ask AI"
            description="Chat about your library"
            to="/chat"
            color="purple"
          />
          <QuickAction
            icon={BarChart2}
            title="Analytics"
            description="View reading insights"
            to="/analytics"
            color="green"
          />
          <QuickAction
            icon={Plus}
            title="Add Book"
            description="Manually add a book"
            to="/library?add=true"
            color="amber"
          />
        </div>
      </div>

      {/* Content Grid */}
      <div className="grid lg:grid-cols-3 gap-6">
        {/* Recently Added */}
        <div className="lg:col-span-2">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-900">Recently Added</h2>
            <Link
              to="/library?sort=recent"
              className="text-sm text-indigo-600 hover:text-indigo-700 font-medium"
            >
              View All
            </Link>
          </div>

          {recentBooks.length > 0 ? (
            <div className="grid sm:grid-cols-2 gap-4">
              {recentBooks.map((book) => (
                <BookCard key={book.id} book={book} variant="list" showActions={false} />
              ))}
            </div>
          ) : (
            <div className="bg-gray-50 rounded-xl p-8 text-center">
              <BookOpen className="w-12 h-12 text-gray-300 mx-auto mb-3" />
              <p className="text-gray-500 mb-4">No books in your library yet</p>
              <Link
                to="/scan"
                className="inline-flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700"
              >
                <Camera size={18} />
                <span>Scan your first shelf</span>
              </Link>
            </div>
          )}
        </div>

        {/* AI Recommendations */}
        <div>
          <div className="flex items-center gap-2 mb-4">
            <Sparkles size={20} className="text-purple-500" />
            <h2 className="text-lg font-semibold text-gray-900">For You</h2>
          </div>

          <div className="bg-white rounded-xl border border-gray-100 p-4 space-y-3">
            {recommendations.length > 0 ? (
              <>
                {recommendations.map((rec, index) => (
                  <RecommendationCard
                    key={index}
                    book={rec}
                    reason={rec.reason}
                  />
                ))}
                <Link
                  to="/chat?prompt=recommend"
                  className="block text-center text-sm text-indigo-600 hover:text-indigo-700 font-medium pt-2"
                >
                  Get more recommendations
                </Link>
              </>
            ) : (
              <div className="text-center py-4">
                <p className="text-sm text-gray-500 mb-3">
                  Add more books to get personalized recommendations
                </p>
                <Link
                  to="/scan"
                  className="text-sm text-indigo-600 hover:text-indigo-700 font-medium"
                >
                  Start adding books
                </Link>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Diversity Score Preview */}
      {stats?.diversity_score && (
        <div className="bg-gradient-to-r from-indigo-500 to-purple-600 rounded-xl p-6 text-white">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold">Library Diversity Score</h3>
              <p className="text-white/80 mt-1">
                Your library spans {stats.unique_genres || 0} genres and{' '}
                {stats.unique_authors || 0} authors
              </p>
            </div>
            <div className="text-right">
              <div className="text-4xl font-bold">
                {stats.diversity_grade || 'B'}
              </div>
              <div className="text-white/80 text-sm">
                {Math.round(stats.diversity_score * 100)}%
              </div>
            </div>
          </div>
          <Link
            to="/analytics"
            className="inline-flex items-center gap-1 mt-4 text-white/90 hover:text-white text-sm font-medium"
          >
            View detailed analysis
            <ArrowRight size={16} />
          </Link>
        </div>
      )}
    </div>
  );
}
