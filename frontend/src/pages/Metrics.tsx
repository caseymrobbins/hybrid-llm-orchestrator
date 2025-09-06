// src/pages/Metrics.tsx
import React from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  ChartBarIcon,
  CurrencyDollarIcon,
  ClockIcon,
  CpuChipIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
} from '@heroicons/react/24/outline';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Badge } from '../components/ui/badge';
import { getMetrics, getCostBreakdown, getAnalytics } from '../services/api';
import { formatCurrency, formatPercentage, formatDuration, cn } from '../lib/utils';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
} from 'recharts';

const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'];

export const Metrics: React.FC = () => {
  const { data: metrics, isLoading: metricsLoading } = useQuery({
    queryKey: ['metrics'],
    queryFn: getMetrics,
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  const { data: costBreakdown } = useQuery({
    queryKey: ['cost-breakdown'],
    queryFn: getCostBreakdown,
    refetchInterval: 60000, // Refresh every minute
  });

  const { data: analytics } = useQuery({
    queryKey: ['analytics', 7],
    queryFn: () => getAnalytics(7),
    refetchInterval: 300000, // Refresh every 5 minutes
  });

  if (metricsLoading) {
    return (
      <div className="p-6 max-w-7xl mx-auto">
        <div className="animate-pulse space-y-6">
          <div className="h-8 bg-gray-300 rounded w-1/4"></div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-32 bg-gray-300 rounded-lg"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  const totalQueries = metrics?.total_queries || 0;
  const avgLatency = metrics?.avg_latency_ms || 0;
  const cacheHitRate = metrics?.cache_hit_rate || 0;
  const costSaved = costBreakdown?.cost_saved_by_cache || 0;

  // Mock data for charts (replace with real data from analytics)
  const latencyData = Array.from({ length: 24 }, (_, i) => ({
    hour: `${i}:00`,
    latency: Math.random() * 1000 + 200,
    queries: Math.floor(Math.random() * 50) + 10,
  }));

  const modelUsageData = [
    { name: 'Local Models', value: metrics?.local_vs_external.local || 0, color: COLORS[0] },
    { name: 'External APIs', value: metrics?.local_vs_external.external || 0, color: COLORS[1] },
  ];

  const modulePerformanceData = analytics?.module_performance?.map((module: any, index: number) => ({
    name: module.module_name || 'Unknown',
    queries: module.usage_count || 0,
    cost: (module.total_cost_cents || 0) / 100,
    avgLatency: module.avg_latency_ms || 0,
    color: COLORS[index % COLORS.length],
  })) || [];

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          Performance Metrics
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-1">
          Monitor system performance, costs, and usage patterns
        </p>
      </div>

      {/* Key Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Total Queries
              </CardTitle>
              <ChartBarIcon className="h-5 w-5 text-blue-600" />
            </div>
          </CardHeader>
          <CardContent className="pt-0">
            <div className="text-2xl font-bold text-gray-900 dark:text-white">
              {totalQueries.toLocaleString()}
            </div>
            <div className="flex items-center mt-1">
              <ArrowTrendingUpIcon className="h-4 w-4 text-green-600 mr-1" />
              <span className="text-sm text-green-600">+12% vs last week</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Avg Response Time
              </CardTitle>
              <ClockIcon className="h-5 w-5 text-yellow-600" />
            </div>
          </CardHeader>
          <CardContent className="pt-0">
            <div className="text-2xl font-bold text-gray-900 dark:text-white">
              {formatDuration(avgLatency)}
            </div>
            <div className="flex items-center mt-1">
              <ArrowTrendingDownIcon className="h-4 w-4 text-green-600 mr-1" />
              <span className="text-sm text-green-600">-8% improvement</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Cache Hit Rate
              </CardTitle>
              <CpuChipIcon className="h-5 w-5 text-green-600" />
            </div>
          </CardHeader>
          <CardContent className="pt-0">
            <div className="text-2xl font-bold text-gray-900 dark:text-white">
              {formatPercentage(cacheHitRate)}
            </div>
            <div className="flex items-center mt-1">
              <Badge variant={cacheHitRate > 0.5 ? "default" : "secondary"}>
                {cacheHitRate > 0.5 ? "Good" : "Poor"}
              </Badge>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Cost Savings
              </CardTitle>
              <CurrencyDollarIcon className="h-5 w-5 text-green-600" />
            </div>
          </CardHeader>
          <CardContent className="pt-0">
            <div className="text-2xl font-bold text-gray-900 dark:text-white">
              {formatCurrency(costSaved * 100)}
            </div>
            <div className="flex items-center mt-1">
              <span className="text-sm text-gray-600 dark:text-gray-400">
                vs external only
              </span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Latency Over Time */}
        <Card>
          <CardHeader>
            <CardTitle>Response Time Trends (24h)</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={latencyData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="hour" />
                <YAxis />
                <Tooltip 
                  formatter={(value: any) => [formatDuration(value), 'Latency']}
                />
                <Line 
                  type="monotone" 
                  dataKey="latency" 
                  stroke="#3B82F6" 
                  strokeWidth={2}
                  dot={{ fill: '#3B82F6', strokeWidth: 2 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Model Usage Distribution */}
        <Card>
          <CardHeader>
            <CardTitle>Model Usage Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={modelUsageData}
                  cx="50%"
                  cy="50%"
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="value"
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                >
                  {modelUsageData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Module Performance */}
      {modulePerformanceData.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Module Performance Breakdown</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={modulePerformanceData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" />
                <Tooltip 
                  formatter={(value: any, name: string) => {
                    if (name === 'cost') return [formatCurrency(value * 100), 'Cost'];
                    if (name === 'avgLatency') return [formatDuration(value), 'Avg Latency'];
                    return [value, name];
                  }}
                />
                <Bar yAxisId="left" dataKey="queries" fill="#3B82F6" name="Queries" />
                <Bar yAxisId="right" dataKey="avgLatency" fill="#10B981" name="avgLatency" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {/* Cost Analysis */}
      {costBreakdown && (
        <Card>
          <CardHeader>
            <CardTitle>Cost Analysis</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
                <div className="text-sm text-blue-600 dark:text-blue-400">Today's Spend</div>
                <div className="text-xl font-bold text-blue-900 dark:text-blue-100">
                  {formatCurrency(costBreakdown.total_cost_today * 100)}
                </div>
              </div>
              
              <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
                <div className="text-sm text-green-600 dark:text-green-400">Cache Savings</div>
                <div className="text-xl font-bold text-green-900 dark:text-green-100">
                  {formatCurrency(costBreakdown.cost_saved_by_cache * 100)}
                </div>
              </div>
              
              <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
                <div className="text-sm text-purple-600 dark:text-purple-400">Routing Savings</div>
                <div className="text-xl font-bold text-purple-900 dark:text-purple-100">
                  {formatCurrency(costBreakdown.cost_saved_by_routing * 100)}
                </div>
              </div>
              
              <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg">
                <div className="text-sm text-orange-600 dark:text-orange-400">Monthly Projection</div>
                <div className="text-xl font-bold text-orange-900 dark:text-orange-100">
                  {formatCurrency(costBreakdown.projection_monthly * 100)}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};