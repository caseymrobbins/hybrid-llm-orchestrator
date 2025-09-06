// src/components/ConnectionStatus.tsx
import React from 'react';
import { WifiIcon, ExclamationTriangleIcon } from '@heroicons/react/24/outline';
import { useWebSocket } from '../contexts/WebSocketContext';
import { useQuery } from '@tanstack/react-query';
import { checkApiHealth } from '../services/api';
import { cn } from '../lib/utils';

export const ConnectionStatus: React.FC = () => {
  const { isConnected: wsConnected } = useWebSocket();
  
  const { data: apiHealthy = false } = useQuery({
    queryKey: ['api-health'],
    queryFn: checkApiHealth,
    refetchInterval: 30000, // Check every 30 seconds
    retry: false,
  });

  const isFullyConnected = wsConnected && apiHealthy;
  const hasPartialConnection = wsConnected || apiHealthy;

  return (
    <div className="space-y-3">
      {/* Main Status */}
      <div className="flex items-center space-x-2">
        {isFullyConnected ? (
          <WifiIcon className="h-5 w-5 text-green-500" />
        ) : hasPartialConnection ? (
          <ExclamationTriangleIcon className="h-5 w-5 text-yellow-500" />
        ) : (
          <WifiIcon className="h-5 w-5 text-red-500" />
        )}
        
        <div className="flex-1">
          <div className={cn(
            "text-sm font-medium",
            isFullyConnected ? "text-green-600 dark:text-green-400" :
            hasPartialConnection ? "text-yellow-600 dark:text-yellow-400" :
            "text-red-600 dark:text-red-400"
          )}>
            {isFullyConnected ? "Connected" : hasPartialConnection ? "Partial" : "Disconnected"}
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-400">
            {isFullyConnected ? "All services online" : 
             hasPartialConnection ? "Limited functionality" : 
             "Services unavailable"}
          </div>
        </div>
        
        <div className={cn(
          "w-2 h-2 rounded-full",
          isFullyConnected ? "bg-green-500" :
          hasPartialConnection ? "bg-yellow-500" :
          "bg-red-500"
        )} />
      </div>

      {/* Detailed Status */}
      <div className="space-y-1 text-xs text-gray-500 dark:text-gray-400">
        <div className="flex justify-between">
          <span>WebSocket:</span>
          <span className={wsConnected ? "text-green-600" : "text-red-600"}>
            {wsConnected ? "Connected" : "Disconnected"}
          </span>
        </div>
        <div className="flex justify-between">
          <span>API:</span>
          <span className={apiHealthy ? "text-green-600" : "text-red-600"}>
            {apiHealthy ? "Available" : "Unavailable"}
          </span>
        </div>
      </div>

      {/* Reconnection Info */}
      {!isFullyConnected && (
        <div className="text-xs text-gray-400 dark:text-gray-500">
          {!wsConnected && "WebSocket will auto-reconnect..."}
          {!apiHealthy && "Checking API availability..."}
        </div>
      )}
    </div>
  );
};