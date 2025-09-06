// src/pages/Health.tsx
import React from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  HeartIcon,
  ServerIcon,
  CloudIcon,
  ShieldExclamationIcon,
  CheckCircleIcon,
  XCircleIcon,
  ExclamationTriangleIcon,
  ClockIcon,
} from '@heroicons/react/24/outline';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Badge } from '../components/ui/badge';
import { Button } from '../components/ui/button';
import { getHealth } from '../services/api';
import { cn, formatDuration } from '../lib/utils';

interface HealthStatus {
  overall_status: string;
  database: {
    status: string;
    details?: string;
  };
  external_apis: Record<string, string>;
  circuit_breakers: Record<string, {
    state: string;
    failure_count: number;
    last_failure_time?: number;
  }>;
  cache_status: {
    enabled: boolean;
    entries: number;
    hit_rate: number;
    memory_usage_mb: number;
  };
  timestamp: string;
}

const getStatusColor = (status: string) => {
  switch (status.toLowerCase()) {
    case 'healthy':
    case 'ok':
    case 'available':
    case 'connected':
      return 'text-green-600';
    case 'degraded':
    case 'timeout':
    case 'partial':
      return 'text-yellow-600';
    case 'unhealthy':
    case 'error':
    case 'unavailable':
    case 'disconnected':
      return 'text-red-600';
    default:
      return 'text-gray-600';
  }
};

const getStatusIcon = (status: string) => {
  switch (status.toLowerCase()) {
    case 'healthy':
    case 'ok':
    case 'available':
    case 'connected':
      return <CheckCircleIcon className="h-5 w-5 text-green-600" />;
    case 'degraded':
    case 'timeout':
    case 'partial':
      return <ExclamationTriangleIcon className="h-5 w-5 text-yellow-600" />;
    case 'unhealthy':
    case 'error':
    case 'unavailable':
    case 'disconnected':
      return <XCircleIcon className="h-5 w-5 text-red-600" />;
    default:
      return <ClockIcon className="h-5 w-5 text-gray-600" />;
  }
};

export const Health: React.FC = () => {
  const { data: health, isLoading, refetch } = useQuery({
    queryKey: ['health'],
    queryFn: getHealth,
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  if (isLoading) {
    return (
      <div className="p-6 max-w-7xl mx-auto">
        <div className="animate-pulse space-y-6">
          <div className="h-8 bg-gray-300 rounded w-1/4"></div>
          <div className="grid gap-6">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-32 bg-gray-300 rounded-lg"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  const overallStatus = health?.overall_status || 'unknown';
  const lastUpdate = health?.timestamp ? new Date(health.timestamp) : new Date();

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            System Health
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Monitor system components and service availability
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          <Button onClick={() => refetch()} variant="outline" size="sm">
            Refresh
          </Button>
          <Badge variant={
            overallStatus === 'healthy' ? 'default' : 
            overallStatus === 'degraded' ? 'secondary' : 'destructive'
          }>
            {overallStatus.toUpperCase()}
          </Badge>
        </div>
      </div>

      {/* Overall Status */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <HeartIcon className="h-5 w-5" />
              <CardTitle>Overall System Status</CardTitle>
            </div>
            <div className="text-sm text-gray-500 dark:text-gray-400">
              Last updated: {lastUpdate.toLocaleTimeString()}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex items-center space-x-4">
            {getStatusIcon(overallStatus)}
            <div>
              <div className={cn("text-xl font-bold", getStatusColor(overallStatus))}>
                {overallStatus.toUpperCase()}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                {overallStatus === 'healthy' 
                  ? 'All systems operational'
                  : overallStatus === 'degraded'
                  ? 'Some components experiencing issues'
                  : 'System experiencing problems'
                }
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Database Status */}
      {health?.database && (
        <Card>
          <CardHeader>
            <div className="flex items-center space-x-2">
              <ServerIcon className="h-5 w-5" />
              <CardTitle>Database</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                {getStatusIcon(health.database.status)}
                <div>
                  <div className={cn("font-medium", getStatusColor(health.database.status))}>
                    {health.database.status.toUpperCase()}
                  </div>
                  {health.database.details && (
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      {health.database.details}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* External APIs */}
      {health?.external_apis && Object.keys(health.external_apis).length > 0 && (
        <Card>
          <CardHeader>
            <div className="flex items-center space-x-2">
              <CloudIcon className="h-5 w-5" />
              <CardTitle>External APIs</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {Object.entries(health.external_apis).map(([api, status]) => (
                <div key={api} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                  <div className="flex items-center space-x-3">
                    {getStatusIcon(status)}
                    <div>
                      <div className="font-medium text-gray-900 dark:text-white">
                        {api}
                      </div>
                      <div className={cn("text-sm", getStatusColor(status))}>
                        {status.toUpperCase()}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Circuit Breakers */}
      {health?.circuit_breakers && Object.keys(health.circuit_breakers).length > 0 && (
        <Card>
          <CardHeader>
            <div className="flex items-center space-x-2">
              <ShieldExclamationIcon className="h-5 w-5" />
              <CardTitle>Circuit Breakers</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {Object.entries(health.circuit_breakers).map(([name, breaker]) => (
                <div key={name} className="flex items-center justify-between p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
                  <div className="flex items-center space-x-3">
                    <div className={cn(
                      "w-3 h-3 rounded-full",
                      breaker.state === 'closed' ? 'bg-green-500' :
                      breaker.state === 'half-open' ? 'bg-yellow-500' : 'bg-red-500'
                    )} />
                    <div>
                      <div className="font-medium text-gray-900 dark:text-white">
                        {name}
                      </div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">
                        State: {breaker.state} • Failures: {breaker.failure_count}
                      </div>
                    </div>
                  </div>
                  
                  <div className="text-right">
                    <Badge variant={
                      breaker.state === 'closed' ? 'default' :
                      breaker.state === 'half-open' ? 'secondary' : 'destructive'
                    }>
                      {breaker.state.toUpperCase()}
                    </Badge>
                    {breaker.last_failure_time && (
                      <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                        Last failure: {new Date(breaker.last_failure_time * 1000).toLocaleString()}
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Cache Status */}
      {health?.cache_status && (
        <Card>
          <CardHeader>
            <div className="flex items-center space-x-2">
              <ServerIcon className="h-5 w-5" />
              <CardTitle>Cache System</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
                <div className="text-sm text-blue-600 dark:text-blue-400">Status</div>
                <div className="text-xl font-bold text-blue-900 dark:text-blue-100">
                  {health.cache_status.enabled ? 'Enabled' : 'Disabled'}
                </div>
              </div>
              
              <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
                <div className="text-sm text-green-600 dark:text-green-400">Entries</div>
                <div className="text-xl font-bold text-green-900 dark:text-green-100">
                  {health.cache_status.entries.toLocaleString()}
                </div>
              </div>
              
              <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
                <div className="text-sm text-purple-600 dark:text-purple-400">Hit Rate</div>
                <div className="text-xl font-bold text-purple-900 dark:text-purple-100">
                  {(health.cache_status.hit_rate * 100).toFixed(1)}%
                </div>
              </div>
              
              <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg">
                <div className="text-sm text-orange-600 dark:text-orange-400">Memory Usage</div>
                <div className="text-xl font-bold text-orange-900 dark:text-orange-100">
                  {health.cache_status.memory_usage_mb.toFixed(1)} MB
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* System Information */}
      <Card>
        <CardHeader>
          <CardTitle>System Information</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div>
              <div className="font-medium text-gray-900 dark:text-white">Health Check Interval</div>
              <div className="text-gray-600 dark:text-gray-400">30 seconds</div>
            </div>
            <div>
              <div className="font-medium text-gray-900 dark:text-white">Last Health Check</div>
              <div className="text-gray-600 dark:text-gray-400">
                {lastUpdate.toLocaleString()}
              </div>
            </div>
            <div>
              <div className="font-medium text-gray-900 dark:text-white">Auto-refresh</div>
              <div className="text-gray-600 dark:text-gray-400">Enabled</div>
            </div>
            <div>
              <div className="font-medium text-gray-900 dark:text-white">Monitoring</div>
              <div className="text-gray-600 dark:text-gray-400">Real-time</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};