/**
 * Analytics page with library insights, diversity scores, and trends.
 */

import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import {
  BookOpen,
  TrendingUp,
  TrendingDown,
  Minus,
  PieChart,
  BarChart2,
  Star,
  Award,
  ArrowRight,
  Loader2,
  RefreshCw,
  Calendar,
  Users,
  Sparkles,
} from 'lucide-react';
import { analyticsApi } from '../services/api';

// Simple chart components (in production, use recharts or similar)
function DonutChart({ data, centerLabel, centerValue }) {
  const total = data.reduce((sum, d) => sum + d.value, 0);
  let currentAngle = 0;

  const segments = data.map((d, i) => {
    const angle = (d.value / total) * 360;
    const startAngle = currentAngle;
    currentAngle += angle;

    // Calculate SVG arc path
    const startRad = (startAngle - 90) * (Math.PI / 180);
    const endRad = (currentAngle - 90) * (Math.PI / 180);
    const largeArc = angle > 180 ? 1 : 0;

    const x1 = 50 + 40 * Math.cos(startRad);
    const y1 = 50 + 40 * Math.sin(startRad);
    const x2 = 50 + 40 * Math.cos(endRad);
    const y2 = 50 + 40 * Math.sin(endRad);

    return {
      ...d,
      path: `M 50 50 L ${x1} ${y1} A 40 40 0 ${largeArc} 1 ${x2} ${y2} Z`,
    };
  });

  return (
    <div className="relative w-48 h-48 mx-auto">
      <svg viewBox="0 0 100 100" className="w-full h-full">
        {segments.map((seg, i) => (
          <path
            key={i}
            d={seg.path}
            fill={seg.color}
            className="hover:opacity-80 transition-opacity cursor-pointer"
          />
        ))}
        <circle cx="50" cy="50" r="25" fill="white" />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-2xl font-bold text-gray-900">{centerValue}</span>
        <span className="text-xs text-gray-500">{centerLabel}</span>
      </div>
    </div>
  );
}

function BarChartSimple({ data, maxValue }) {
  return (
    <div className="space-y-3">
      {data.map((d, i) => (
        <div key={i}>
          <div className="flex items-center justify-between text-sm mb-1">
            <span className="text-gray-700 truncate">{d.label}</span>
            <span className="text-gray-500">{d.value}</span>
          </div>
          <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
            <div
              className="h-full bg-indigo-500 rounded-full transition-all duration-500"
              style={{ width: `${(d.value / maxValue) * 100}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

function DiversityScoreCard({ score, grade, label, description }) {
  const getGradeColor = (grade) => {
    if (grade.startsWith('A')) return 'text-green-600 bg-green-100';
    if (grade.startsWith('B')) return 'text-blue-600 bg-blue-100';
    if (grade.startsWith('C')) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  return (
    <div className="bg-white rounded-xl p-5 border border-gray-100">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-medium text-gray-900">{label}</h3>
        <span className={`px-2 py-1 rounded-lg font-bold text-lg ${getGradeColor(grade)}`}>
          {grade}
        </span>
      </div>
      <div className="mb-2">
        <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-indigo-500 to-purple-600 rounded-full"
            style={{ width: `${score * 100}%` }}
          />
        </div>
      </div>
      <p className="text-sm text-gray-500">{description}</p>
    </div>
  );
}

function TrendItem({ genre, trend, change }) {
  const TrendIcon = trend === 'increasing' ? TrendingUp : trend === 'decreasing' ? TrendingDown : Minus;
  const trendColor = trend === 'increasing' ? 'text-green-600' : trend === 'decreasing' ? 'text-red-600' : 'text-gray-400';

  return (
    <div className="flex items-center justify-between py-2">
      <span className="text-gray-700">{genre}</span>
      <div className={`flex items-center gap-1 ${trendColor}`}>
        <TrendIcon size={16} />
        <span className="text-sm font-medium">
          {change > 0 ? '+' : ''}{change}%
        </span>
      </div>
    </div>
  );
}

function RecommendationCard({ recommendation }) {
  // Handle both flat structure (current API) and potential nested 'book' object
  const title = recommendation.book?.title || recommendation.title;
  const author = recommendation.book?.author || recommendation.author;
  const coverUrl = recommendation.book?.cover_url;

  return (
    <div className="flex gap-3 p-3 bg-gray-50 rounded-lg hover:bg-indigo-50 transition-colors">
      <div className="w-10 h-14 bg-gray-200 rounded overflow-hidden flex-shrink-0">
        {coverUrl ? (
          <img
            src={coverUrl}
            alt={title}
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <BookOpen className="w-5 h-5 text-gray-400" />
          </div>
        )}
      </div>
      <div className="flex-1 min-w-0">
        <h4 className="font-medium text-gray-900 text-sm truncate">
          {title}
        </h4>
        <p className="text-xs text-gray-500 truncate">
          {author}
        </p>
        <p className="text-xs text-indigo-600 mt-1">{recommendation.reason}</p>
      </div>
    </div>
  );
}

const COLORS = [
  '#6366f1', '#8b5cf6', '#a855f7', '#d946ef', '#ec4899',
  '#f43f5e', '#f97316', '#eab308', '#84cc16', '#22c55e',
  '#14b8a6', '#06b6d4', '#0ea5e9', '#3b82f6',
];

export default function AnalyticsPage() {
  const [stats, setStats] = useState(null);
  const [diversity, setDiversity] = useState(null);
  const [trends, setTrends] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);

      const [statsData, diversityData, trendsData, recsData] = await Promise.all([
        analyticsApi.getStats(),
        analyticsApi.getDiversity(),
        analyticsApi.getTrends(),
        analyticsApi.getRecommendations({ count: 5, includeExploration: true }),
      ]);

      setStats(statsData);
      setDiversity(diversityData);
      setTrends(trendsData);
      setRecommendations(recsData.recommendations || []);
    } catch (err) {
      console.error('Failed to fetch analytics:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 text-indigo-500 animate-spin" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-16">
        <p className="text-red-600 mb-4">{error}</p>
        <button
          onClick={fetchData}
          className="inline-flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg"
        >
          <RefreshCw size={18} />
          Retry
        </button>
      </div>
    );
  }

  // Prepare genre chart data
  const genreChartData = (stats?.genre_distribution || []).slice(0, 8).map((g, i) => ({
    label: g.genre,
    value: g.count,
    color: COLORS[i % COLORS.length],
  }));

  const maxGenreCount = Math.max(...genreChartData.map((d) => d.value), 1);

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Library Analytics</h1>
          <p className="text-gray-500 mt-1">
            Insights and statistics about your reading collection
          </p>
        </div>
        <button
          onClick={fetchData}
          className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg"
        >
          <RefreshCw size={20} />
        </button>
      </div>

      {/* Overview Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white rounded-xl p-5 border border-gray-100">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-indigo-100 rounded-lg flex items-center justify-center">
              <BookOpen className="text-indigo-600" size={20} />
            </div>
            <div>
              <p className="text-2xl font-bold text-gray-900">{stats?.total_books || 0}</p>
              <p className="text-sm text-gray-500">Total Books</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl p-5 border border-gray-100">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center">
              <Users className="text-purple-600" size={20} />
            </div>
            <div>
              <p className="text-2xl font-bold text-gray-900">{stats?.unique_authors || 0}</p>
              <p className="text-sm text-gray-500">Authors</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl p-5 border border-gray-100">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center">
              <PieChart className="text-green-600" size={20} />
            </div>
            <div>
              <p className="text-2xl font-bold text-gray-900">{stats?.unique_genres || 0}</p>
              <p className="text-sm text-gray-500">Genres</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl p-5 border border-gray-100">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-amber-100 rounded-lg flex items-center justify-center">
              <Calendar className="text-amber-600" size={20} />
            </div>
            <div>
              <p className="text-2xl font-bold text-gray-900">
                {stats?.oldest_book_year || 'N/A'} - {stats?.newest_book_year || 'N/A'}
              </p>
              <p className="text-sm text-gray-500">Year Range</p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Grid */}
      <div className="grid lg:grid-cols-3 gap-6">
        {/* Genre Distribution */}
        <div className="lg:col-span-2 bg-white rounded-xl p-6 border border-gray-100">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Genre Distribution</h2>
          <div className="grid md:grid-cols-2 gap-6">
            <DonutChart
              data={genreChartData}
              centerLabel="books"
              centerValue={stats?.total_books || 0}
            />
            <div>
              <BarChartSimple data={genreChartData} maxValue={maxGenreCount} />
            </div>
          </div>
          <div className="flex flex-wrap gap-2 mt-4 pt-4 border-t border-gray-100">
            {genreChartData.map((d, i) => (
              <Link
                key={i}
                to={`/library?genre=${encodeURIComponent(d.label)}`}
                className="flex items-center gap-2 px-2 py-1 rounded hover:bg-gray-50"
              >
                <div
                  className="w-3 h-3 rounded"
                  style={{ backgroundColor: d.color }}
                />
                <span className="text-sm text-gray-600">{d.label}</span>
              </Link>
            ))}
          </div>
        </div>

        {/* Diversity Score */}
        <div className="bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl p-6 text-white">
          <div className="flex items-center gap-2 mb-4">
            <Award size={24} />
            <h2 className="text-lg font-semibold">Diversity Score</h2>
          </div>

          <div className="text-center mb-6">
            <div className="text-6xl font-bold mb-1">
              {diversity?.overall_grade || 'B'}
            </div>
            <div className="text-white/80">
              {Math.round((diversity?.overall_score || 0.7) * 100)}% diverse
            </div>
          </div>

          <div className="space-y-3">
            <div className="flex justify-between text-sm">
              <span className="text-white/80">Genre Diversity</span>
              <span className="font-medium">{diversity?.genre_diversity?.grade || 'B'}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-white/80">Author Diversity</span>
              <span className="font-medium">{diversity?.author_diversity?.grade || 'B'}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-white/80">Temporal Diversity</span>
              <span className="font-medium">{diversity?.temporal_diversity?.grade || 'B'}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Second Row */}
      <div className="grid lg:grid-cols-2 gap-6">
        {/* Reading Trends */}
        <div className="bg-white rounded-xl p-6 border border-gray-100">
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="text-indigo-500" size={20} />
            <h2 className="text-lg font-semibold text-gray-900">Genre Trends</h2>
          </div>

          {trends?.genre_trends && Object.keys(trends.genre_trends).length > 0 ? (
            <div className="divide-y divide-gray-100">
              {Object.entries(trends.genre_trends).slice(0, 6).map(([genre, trend], i) => (
                <TrendItem
                  key={i}
                  genre={genre}
                  trend={trend}
                  change={trend === 'increasing' ? 15 : trend === 'decreasing' ? -10 : 0} // visual placeholder
                />
              ))}
            </div>
          ) : (
            <p className="text-gray-500 text-center py-8">
              Add more books to see trends
            </p>
          )}
        </div>

        {/* Recommendations */}
        <div className="bg-white rounded-xl p-6 border border-gray-100">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Sparkles className="text-purple-500" size={20} />
              <h2 className="text-lg font-semibold text-gray-900">Recommended</h2>
            </div>
            <Link
              to="/chat?prompt=recommend"
              className="text-sm text-indigo-600 hover:text-indigo-700"
            >
              Get more
            </Link>
          </div>

          {recommendations.length > 0 ? (
            <div className="space-y-3">
              {recommendations.map((rec, i) => (
                <RecommendationCard key={i} recommendation={rec} />
              ))}
            </div>
          ) : (
            <p className="text-gray-500 text-center py-8">
              Add more books to get recommendations
            </p>
          )}
        </div>
      </div>

      {/* Diversity Details */}
      {diversity && (
        <div>
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Detailed Diversity Analysis</h2>
          <div className="grid md:grid-cols-3 gap-4">
            <DiversityScoreCard
              score={diversity.genre_diversity?.score || 0.7}
              grade={diversity.genre_diversity?.grade || 'B'}
              label="Genre Diversity"
              description={diversity.genre_diversity?.interpretation || 'Your collection spans multiple genres'}
            />
            <DiversityScoreCard
              score={diversity.author_diversity?.score || 0.7}
              grade={diversity.author_diversity?.grade || 'B'}
              label="Author Diversity"
              description={diversity.author_diversity?.interpretation || 'You read from many different authors'}
            />
            <DiversityScoreCard
              score={diversity.temporal_diversity?.score || 0.7}
              grade={diversity.temporal_diversity?.grade || 'B'}
              label="Temporal Diversity"
              description={diversity.temporal_diversity?.interpretation || 'Your books span multiple decades'}
            />
          </div>

          {/* Recommendations */}
          {diversity.recommendations?.length > 0 && (
            <div className="mt-6 bg-indigo-50 rounded-xl p-6">
              <h3 className="font-medium text-indigo-900 mb-3">Suggestions to improve diversity:</h3>
              <ul className="space-y-2">
                {diversity.recommendations.map((rec, i) => (
                  <li key={i} className="flex items-start gap-2 text-indigo-800">
                    <ArrowRight size={16} className="mt-1 flex-shrink-0" />
                    <span>{rec}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Top Authors */}
      {stats?.top_authors?.length > 0 && (
        <div className="bg-white rounded-xl p-6 border border-gray-100">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Top Authors</h2>
          <div className="grid sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
            {stats.top_authors.slice(0, 5).map((author, i) => (
              <Link
                key={i}
                to={`/library?author=${encodeURIComponent(author.author)}`}
                className="text-center p-4 rounded-lg hover:bg-gray-50 transition-colors"
              >
                <div className="w-16 h-16 mx-auto mb-3 bg-gradient-to-br from-indigo-100 to-purple-100 rounded-full flex items-center justify-center">
                  <span className="text-2xl font-bold text-indigo-600">
                    {(author.author || '?').charAt(0)}
                  </span>
                </div>
                <p className="font-medium text-gray-900 truncate">{author.author}</p>
                <p className="text-sm text-gray-500">{author.count} books</p>
              </Link>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
