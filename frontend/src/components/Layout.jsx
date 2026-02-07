/**
 * Main layout component with navigation and responsive sidebar.
 */

import React, { useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import {
  Home,
  BookOpen,
  Camera,
  MessageCircle,
  BarChart2,
  Menu,
  X,
  Settings,
  Search,
  LogOut,
} from 'lucide-react';

const navItems = [
  { path: '/', icon: Home, label: 'Home' },
  { path: '/library', icon: BookOpen, label: 'Library' },
  { path: '/scan', icon: Camera, label: 'Scan Books' },
  { path: '/chat', icon: MessageCircle, label: 'Chat' },
  { path: '/analytics', icon: BarChart2, label: 'Analytics' },
];

function NavItem({ item, mobile = false, onClick }) {
  const Icon = item.icon;

  return (
    <NavLink
      to={item.path}
      onClick={onClick}
      className={({ isActive }) =>
        `flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 ${isActive
          ? 'bg-indigo-100 text-indigo-700 font-medium'
          : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
        } ${mobile ? 'text-lg' : ''}`
      }
    >
      <Icon size={mobile ? 24 : 20} />
      <span>{item.label}</span>
    </NavLink>
  );
}

function Sidebar({ mobile = false, onClose }) {
  return (
    <div className={`flex flex-col h-full ${mobile ? 'pt-16' : 'pt-6'}`}>
      {/* Logo */}
      {!mobile && (
        <div className="px-4 mb-8">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl flex items-center justify-center">
              <BookOpen className="text-white" size={24} />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">ShelfSense</h1>
              <p className="text-xs text-gray-500">AI Library Manager</p>
            </div>
          </div>
        </div>
      )}

      {/* Navigation */}
      <nav className="flex-1 px-3 space-y-1">
        {navItems.map((item) => (
          <NavItem
            key={item.path}
            item={item}
            mobile={mobile}
            onClick={mobile ? onClose : undefined}
          />
        ))}
      </nav>

      {/* Footer */}
      <div className="px-3 pb-6">
        <button
          onClick={() => {
            localStorage.removeItem('token');
            localStorage.removeItem('shelfsense_api_key');
            window.location.href = '/login';
          }}
          className="flex items-center gap-3 px-4 py-3 w-full text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
        >
          <LogOut size={20} />
          <span>Log out</span>
        </button>
      </div>
    </div>
  );
}

function Header({ onMenuClick, title }) {
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  return (
    <header className="h-16 bg-white border-b border-gray-200 flex items-center justify-between px-4 lg:px-6">
      {/* Mobile menu button */}
      <button
        onClick={onMenuClick}
        className="lg:hidden p-2 text-gray-600 hover:bg-gray-100 rounded-lg"
      >
        <Menu size={24} />
      </button>

      {/* Title - mobile only */}
      <div className="lg:hidden flex items-center gap-2">
        <BookOpen className="text-indigo-600" size={24} />
        <span className="font-semibold text-gray-900">{title || 'ShelfSense'}</span>
      </div>

      {/* Search bar - desktop */}
      <div className="hidden lg:flex items-center flex-1 max-w-xl">
        <div className="relative w-full">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={20} />
          <input
            type="text"
            placeholder="Search your library..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-gray-100 border-0 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:bg-white transition-colors"
          />
        </div>
      </div>

      {/* Search button - mobile */}
      <button
        onClick={() => setSearchOpen(!searchOpen)}
        className="lg:hidden p-2 text-gray-600 hover:bg-gray-100 rounded-lg"
      >
        <Search size={24} />
      </button>

      {/* Mobile search overlay */}
      {searchOpen && (
        <div className="absolute top-16 left-0 right-0 bg-white p-4 border-b border-gray-200 lg:hidden z-40">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={20} />
            <input
              type="text"
              placeholder="Search your library..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              autoFocus
              className="w-full pl-10 pr-4 py-2 bg-gray-100 border-0 rounded-lg focus:ring-2 focus:ring-indigo-500"
            />
          </div>
        </div>
      )}
    </header>
  );
}

export default function Layout({ children }) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const location = useLocation();

  // Get page title from path
  const getPageTitle = () => {
    const item = navItems.find((item) => item.path === location.pathname);
    return item?.label || 'ShelfSense';
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Desktop Sidebar */}
      <aside className="hidden lg:flex lg:fixed lg:inset-y-0 lg:w-64 lg:flex-col bg-white border-r border-gray-200">
        <Sidebar />
      </aside>

      {/* Mobile Sidebar Overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Mobile Sidebar */}
      <aside
        className={`fixed inset-y-0 left-0 w-72 bg-white z-50 transform transition-transform duration-300 ease-in-out lg:hidden ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'
          }`}
      >
        {/* Close button */}
        <button
          onClick={() => setSidebarOpen(false)}
          className="absolute top-4 right-4 p-2 text-gray-600 hover:bg-gray-100 rounded-lg"
        >
          <X size={24} />
        </button>

        {/* Logo */}
        <div className="px-4 pt-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl flex items-center justify-center">
              <BookOpen className="text-white" size={24} />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">ShelfSense</h1>
              <p className="text-xs text-gray-500">AI Library Manager</p>
            </div>
          </div>
        </div>

        <Sidebar mobile onClose={() => setSidebarOpen(false)} />
      </aside>

      {/* Main content */}
      <div className="lg:pl-64 flex flex-col min-h-screen">
        <Header onMenuClick={() => setSidebarOpen(true)} title={getPageTitle()} />
        <main className="flex-1 p-4 lg:p-6">{children}</main>
      </div>
    </div>
  );
}
