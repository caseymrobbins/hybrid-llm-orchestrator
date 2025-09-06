// src/components/ModuleExecution.tsx
import React from 'react';
import { CheckCircleIcon, ClockIcon, PlayIcon } from '@heroicons/react/24/outline';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { cn } from '../lib/utils';
import { useWebSocket } from '../contexts/WebSocketContext';

interface ModuleStep {
  name: string;
  displayName: string;
  status: 'pending' | 'running' | 'completed' | 'error';
  startTime?: number;
  endTime?: number;
  output?: string;
  model?: string;
  cached?: boolean;
  cost?: number;
}

const defaultModules = [
  { name: 'Curiosity', displayName: 'Curiosity Analysis' },
  { name: 'OpposingOpinion', displayName: 'Opposing Opinion' },
  { name: 'Ethics', displayName: 'Ethical Assessment' },
  { name: 'Synthesis', displayName: 'Final Synthesis' },
];

export const ModuleExecution: React.FC = () => {
  const { messages, currentExecution } = useWebSocket();
  const [modules, setModules] = React.useState<ModuleStep[]>(
    defaultModules.map(m => ({ ...m, status: 'pending' as const }))
  );

  React.useEffect(() => {
    // Reset modules when new query starts
    const queryStartMsg = messages.find(m => m.type === 'query_start');
    if (queryStartMsg && currentExecution.isRunning) {
      setModules(defaultModules.map(m => ({ ...m, status: 'pending' as const })));
    }

    // Update module status based on WebSocket messages
    messages.forEach(message => {
      if (message.type === 'module_start') {
        setModules(prev => 
          prev.map(m => 
            m.name === message.module 
              ? { ...m, status: 'running', startTime: Date.now() }
              : m
          )
        );
      } else if (message.type === 'module_complete') {
        setModules(prev => 
          prev.map(m => 
            m.name === message.module 
              ? { 
                  ...m, 
                  status: 'completed', 
                  endTime: Date.now(),
                  output: message.output?.output,
                  model: message.output?.model_used,
                  cached: message.output?.cached,
                  cost: message.routing?.cost_cents
                }
              : m
          )
        );
      }
    });
  }, [messages, currentExecution.isRunning]);

  if (!currentExecution.isRunning && !modules.some(m => m.status !== 'pending')) {
    return null; // Don't show if no execution in progress and no completed modules
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Module Execution Pipeline</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {modules.map((module, index) => (
            <div
              key={module.name}
              className={cn(
                "flex items-center p-3 rounded-lg border transition-all duration-200",
                module.status === 'completed' && "bg-green-50 border-green-200 dark:bg-green-900/20 dark:border-green-800",
                module.status === 'running' && "bg-blue-50 border-blue-200 dark:bg-blue-900/20 dark:border-blue-800",
                module.status === 'pending' && "bg-gray-50 border-gray-200 dark:bg-gray-800/50 dark:border-gray-700",
                module.status === 'error' && "bg-red-50 border-red-200 dark:bg-red-900/20 dark:border-red-800"
              )}
            >
              {/* Status Icon */}
              <div className="flex-shrink-0 mr-3">
                {module.status === 'completed' && (
                  <CheckCircleIcon className="h-6 w-6 text-green-600 dark:text-green-400" />
                )}
                {module.status === 'running' && (
                  <div className="flex items-center justify-center h-6 w-6">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                  </div>
                )}
                {module.status === 'pending' && (
                  <ClockIcon className="h-6 w-6 text-gray-400" />
                )}
                {module.status === 'error' && (
                  <div className="h-6 w-6 rounded-full bg-red-600 flex items-center justify-center">
                    <span className="text-white text-xs">!</span>
                  </div>
                )}
              </div>

              {/* Module Info */}
              <div className="flex-grow">
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="text-sm font-medium text-gray-900 dark:text-white">
                      {module.displayName}
                    </h4>
                    <div className="flex items-center space-x-2 mt-1">
                      {module.model && (
                        <Badge variant="outline" className="text-xs">
                          {module.model}
                        </Badge>
                      )}
                      {module.cached && (
                        <Badge variant="secondary" className="text-xs">
                          Cached
                        </Badge>
                      )}
                      {module.cost !== undefined && (
                        <Badge variant="outline" className="text-xs">
                          ${(module.cost / 100).toFixed(4)}
                        </Badge>
                      )}
                    </div>
                  </div>

                  {/* Execution Time */}
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    {module.status === 'completed' && module.startTime && module.endTime && (
                      <span>{((module.endTime - module.startTime) / 1000).toFixed(2)}s</span>
                    )}
                    {module.status === 'running' && module.startTime && (
                      <span className="animate-pulse">
                        {((Date.now() - module.startTime) / 1000).toFixed(1)}s
                      </span>
                    )}
                  </div>
                </div>

                {/* Output Preview */}
                {module.output && (
                  <div className="mt-2 p-2 bg-white dark:bg-gray-900 rounded border text-xs">
                    <p className="text-gray-600 dark:text-gray-400 line-clamp-2">
                      {module.output.replace('[CACHED]', '').trim()}
                    </p>
                  </div>
                )}
              </div>

              {/* Connection Line */}
              {index < modules.length - 1 && (
                <div className="absolute left-6 mt-12 w-px h-3 bg-gray-300 dark:bg-gray-600" />
              )}
            </div>
          ))}
        </div>

        {/* Summary */}
        {modules.some(m => m.status === 'completed') && (
          <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
            <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400">
              <span>
                Completed: {modules.filter(m => m.status === 'completed').length}/{modules.length}
              </span>
              <span>
                Total Cost: ${(modules.reduce((sum, m) => sum + (m.cost || 0), 0) / 100).toFixed(4)}
              </span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};