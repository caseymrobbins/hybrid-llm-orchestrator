// src/components/Layout.tsx
import React from 'react';
import { Outlet, Link, useLocation } from 'react-router-dom';
import {
  HomeIcon,
  ChartBarIcon,
  CogIcon,
  HeartIcon,
  BoltIcon,
  WifiIcon,
} from '@heroicons/react/24/outline';
import { cn } from '../lib/utils';
import { useWebSocket } from '../contexts/WebSocketContext';
import { ConnectionStatus } from './ConnectionStatus';

const navigation = [
  { name: 'Dashboard', href: '/', icon: HomeIcon },
  { name: 'Metrics', href: '/metrics', icon: ChartBarIcon },
  { name: 'Configuration', href: '/configuration', icon: CogIcon },
  { name: 'Health', href: '/health', icon: HeartIcon },
];

export const Layout: React.FC = () => {
  const location = useLocation();
  const { isConnected } = useWebSocket();

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Sidebar */}
      <div className="fixed inset-y-0 left-0 z-50 w-64 bg-white dark:bg-gray-800 shadow-lg">
        <div className="flex h-16 items-center px-6 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center space-x-2">
            <BoltIcon className="h-8 w-8 text-blue-600" />
            <div className="flex flex-col">
              <h1 className="text-lg font-semibold text-gray-900 dark:text-white">
                LLM Orchestrator
              </h1>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                v1.0.0
              </p>
            </div>
          </div>
        </div>

        <nav className="mt-6 px-3">
          <div className="space-y-1">
            {navigation.map((item) => {
              const isActive = location.pathname === item.href;
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  className={cn(
                    isActive
                      ? 'bg-blue-50 border-blue-500 text-blue-700 dark:bg-blue-900/20 dark:border-blue-400 dark:text-blue-300'
                      : 'border-transparent text-gray-600 hover:bg-gray-50 hover:text-gray-900 dark:text-gray-300 dark:hover:bg-gray-700 dark:hover:text-white',
                    'group flex items-center px-3 py-2 text-sm font-medium border-l-4 transition-colors'
                  )}
                >
                  <item.icon
                    className={cn(
                      isActive
                        ? 'text-blue-500 dark:text-blue-400'
                        : 'text-gray-400 group-hover:text-gray-500 dark:group-hover:text-gray-300',
                      'mr-3 h-5 w-5 transition-colors'
                    )}
                  />
                  {item.name}
                </Link>
              );
            })}
          </div>
        </nav>

        {/* Connection Status */}
        <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-gray-200 dark:border-gray-700">
          <ConnectionStatus />
        </div>
      </div>

      {/* Main content */}
      <div className="pl-64">
        <main className="min-h-screen">
          <Outlet />
        </main>
      </div>
    </div>
  );
};