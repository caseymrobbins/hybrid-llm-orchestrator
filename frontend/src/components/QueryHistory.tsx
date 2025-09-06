// src/components/QueryHistory.tsx
import React from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  ClockIcon,
  ChartBarIcon,
  CurrencyDollarIcon,
  ArrowPathIcon,
} from '@heroicons/react/24/outline';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { getAnalytics } from '../services/api';
import { formatDuration, formatCurrency, truncateText } from '../lib/utils';

interface QueryHistoryItem {
  query: string;
  response: string;
  execution_time: number;
  cost_cents: number;
  modules_used: string[];
  timestamp: string;
  cache_hit_rate: number;
}

export const QueryHistory: React.FC = () => {
  const { data: analytics, isLoading, refetch } = useQuery({
    queryKey: ['analytics', 1], // Last 1 day
    queryFn: () => getAnalytics(1),
    refetchInterval: 60000, // Refresh every minute
  });

  // Mock recent queries for demo (replace with real data from analytics)
  const recentQueries: QueryHistoryItem[] = [
    {
      query: "What are the ethical implications of AI in healthcare?",
      response: "AI in healthcare presents significant ethical considerations...",
      execution_time: 4.2,
      cost_cents: 3,
      modules_used: ["Curiosity", "OpposingOpinion", "Ethics"],
      timestamp: new Date(Date.now() - 300000).toISOString(), // 5 mins ago
      cache_hit_rate: 0.33,
    },
    {
      query: "Explain quantum computing to a 10-year-old",
      response: "Quantum computing is like having a magical computer...",
      execution_time: 2.8,
      cost_cents: 2,
      modules_used: ["Curiosity", "OpposingOpinion", "Ethics"],
      timestamp: new Date(Date.now() - 1800000).toISOString(), // 30 mins ago
      cache_hit_rate: 0.67,
    },
    {
      query: "Benefits of renewable energy vs fossil fuels",
      response: "Renewable energy sources offer numerous advantages...",
      execution_time: 3.5,
      cost_cents: 1,
      modules_used: ["Curiosity", "OpposingOpinion", "Ethics"],
      timestamp: new Date(Date.now() - 3600000).toISOString(), // 1 hour ago
      cache_hit_rate: 0.5,
    },
  ];

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Recent Queries</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse space-y-4">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="h-20 bg-gray-300 rounded"></div>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center space-x-2">
            <ClockIcon className="h-5 w-5" />
            <span>Recent Queries</span>
          </CardTitle>
          <Button
            onClick={() => refetch()}
            variant="ghost"
            size="sm"
            className="flex items-center space-x-1"
          >
            <ArrowPathIcon className="h-4 w-4" />
            <span>Refresh</span>
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {recentQueries.length === 0 ? (
          <div className="text-center py-8 text-gray-500 dark:text-gray-400">
            <ChartBarIcon className="h-12 w-12 mx-auto mb-3 opacity-50" />
            <p>No recent queries found</p>
            <p className="text-sm">Process your first query to see history</p>
          </div>
        ) : (
          <div className="space-y-4">
            {recentQueries.map((item, index) => (
              <div
                key={index}
                className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors"
              >
                <div className="space-y-3">
                  {/* Query */}
                  <div>
                    <div className="text-sm font-medium text-gray-900 dark:text-white">
                      Query
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                      "{truncateText(item.query, 100)}"
                    </div>
                  </div>

                  {/* Response Preview */}
                  <div>
                    <div className="text-sm font-medium text-gray-900 dark:text-white">
                      Response
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                      {truncateText(item.response, 150)}
                    </div>
                  </div>

                  {/* Metadata */}
                  <div className="flex items-center justify-between pt-2 border-t border-gray-100 dark:border-gray-700">
                    <div className="flex flex-wrap items-center gap-2">
                      <Badge variant="outline" className="text-xs">
                        <ClockIcon className="h-3 w-3 mr-1" />
                        {formatDuration(item.execution_time * 1000)}
                      </Badge>
                      
                      <Badge variant="outline" className="text-xs">
                        <CurrencyDollarIcon className="h-3 w-3 mr-1" />
                        {formatCurrency(item.cost_cents)}
                      </Badge>
                      
                      <Badge variant="outline" className="text-xs">
                        🔄 {item.modules_used.length} modules
                      </Badge>
                      
                      <Badge 
                        variant={item.cache_hit_rate > 0.5 ? "default" : "secondary"}
                        className="text-xs"
                      >
                        💾 {(item.cache_hit_rate * 100).toFixed(0)}% cached
                      </Badge>
                    </div>

                    <div className="text-xs text-gray-500 dark:text-gray-400">
                      {new Date(item.timestamp).toLocaleString()}
                    </div>
                  </div>
                </div>
              </div>
            ))}

            {/* View More Button */}
            <div className="text-center pt-4 border-t border-gray-200 dark:border-gray-700">
              <Button variant="outline" size="sm">
                View All History
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};