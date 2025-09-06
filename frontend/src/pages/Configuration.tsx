// src/pages/Configuration.tsx
import React from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  CogIcon,
  ServerIcon,
  ShieldCheckIcon,
  ArrowPathIcon,
  CheckCircleIcon,
  XCircleIcon,
} from '@heroicons/react/24/outline';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Badge } from '../components/ui/badge';
import { getConfiguration, getModules, clearCache } from '../services/api';
import { toast } from '../components/ui/use-toast';
import { cn } from '../lib/utils';

export const Configuration: React.FC = () => {
  const queryClient = useQueryClient();

  const { data: config, isLoading: configLoading } = useQuery({
    queryKey: ['configuration'],
    queryFn: getConfiguration,
  });

  const { data: modules, isLoading: modulesLoading } = useQuery({
    queryKey: ['modules'],
    queryFn: getModules,
  });

  const clearCacheMutation = useMutation({
    mutationFn: clearCache,
    onSuccess: () => {
      toast({
        title: "Cache Cleared",
        description: "Semantic cache has been successfully cleared",
      });
      queryClient.invalidateQueries({ queryKey: ['metrics'] });
    },
    onError: (error: any) => {
      toast({
        title: "Cache Clear Failed",
        description: error.message || "Failed to clear cache",
        variant: "destructive",
      });
    },
  });

  const handleClearCache = () => {
    clearCacheMutation.mutate();
  };

  if (configLoading || modulesLoading) {
    return (
      <div className="p-6 max-w-7xl mx-auto">
        <div className="animate-pulse space-y-6">
          <div className="h-8 bg-gray-300 rounded w-1/4"></div>
          <div className="grid gap-6">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="h-48 bg-gray-300 rounded-lg"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            System Configuration
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            View and manage system settings, modules, and preferences
          </p>
        </div>
        
        <Button
          onClick={handleClearCache}
          disabled={clearCacheMutation.isPending}
          variant="outline"
          className="flex items-center space-x-2"
        >
          <ArrowPathIcon className={cn(
            "h-4 w-4",
            clearCacheMutation.isPending && "animate-spin"
          )} />
          <span>Clear Cache</span>
        </Button>
      </div>

      {/* Workflow Configuration */}
      {config && (
        <Card>
          <CardHeader>
            <div className="flex items-center space-x-2">
              <CogIcon className="h-5 w-5" />
              <CardTitle>Workflow Configuration</CardTitle>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium text-gray-900 dark:text-white">Workflow Details</h4>
                <div className="mt-2 space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">Name:</span>
                    <span className="font-medium">{config.workflow.name}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">Version:</span>
                    <span className="font-medium">{config.workflow.version}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">Max Concurrent:</span>
                    <span className="font-medium">{config.workflow.max_concurrent_steps}</span>
                  </div>
                </div>
              </div>
              
              <div>
                <h4 className="font-medium text-gray-900 dark:text-white">Description</h4>
                <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
                  {config.workflow.description}
                </p>
              </div>
            </div>

            <div>
              <h4 className="font-medium text-gray-900 dark:text-white mb-3">Execution Plan</h4>
              <div className="space-y-2">
                {config.workflow.execution_plan.map((step, index) => (
                  <div
                    key={step.module}
                    className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-lg"
                  >
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full flex items-center justify-center text-sm font-medium">
                        {index + 1}
                      </div>
                      <div>
                        <div className="font-medium text-gray-900 dark:text-white">
                          {step.module}
                        </div>
                        {step.dependencies.length > 0 && (
                          <div className="text-sm text-gray-600 dark:text-gray-400">
                            Depends on: {step.dependencies.join(', ')}
                          </div>
                        )}
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      {step.parallel && (
                        <Badge variant="secondary">Parallel</Badge>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Module Configuration */}
      {modules && (
        <Card>
          <CardHeader>
            <div className="flex items-center space-x-2">
              <ServerIcon className="h-5 w-5" />
              <CardTitle>Module Configuration</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <h4 className="font-medium text-gray-900 dark:text-white">
                  Configured Modules ({modules.total_count})
                </h4>
              </div>
              
              <div className="grid gap-4">
                {modules.modules.map((module) => (
                  <div
                    key={module.name}
                    className="flex items-center justify-between p-4 border border-gray-200 dark:border-gray-700 rounded-lg"
                  >
                    <div className="flex-1">
                      <div className="flex items-center space-x-3">
                        <h5 className="font-medium text-gray-900 dark:text-white">
                          {module.aspect}
                        </h5>
                        <Badge variant={module.uses_external ? "default" : "secondary"}>
                          {module.uses_external ? "External" : "Local"}
                        </Badge>
                      </div>
                      
                      <div className="mt-2 space-y-1 text-sm text-gray-600 dark:text-gray-400">
                        <div>
                          <span className="font-medium">Provider:</span> {module.model_info.provider}
                        </div>
                        <div>
                          <span className="font-medium">Model:</span> {module.model_info.model}
                        </div>
                        <div className="flex space-x-4">
                          <span>
                            <span className="font-medium">Temperature:</span> {module.model_info.temperature}
                          </span>
                          <span>
                            <span className="font-medium">Max Tokens:</span> {module.model_info.max_tokens}
                          </span>
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      {module.has_fallback && (
                        <Badge variant="outline">Fallback</Badge>
                      )}
                      <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* System Features */}
      {config && (
        <Card>
          <CardHeader>
            <div className="flex items-center space-x-2">
              <ShieldCheckIcon className="h-5 w-5" />
              <CardTitle>System Features</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Routing Configuration */}
              <div className="space-y-3">
                <h4 className="font-medium text-gray-900 dark:text-white">Intelligent Routing</h4>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-400">Enabled:</span>
                    <div className="flex items-center space-x-1">
                      {config.routing.enabled ? (
                        <CheckCircleIcon className="h-4 w-4 text-green-600" />
                      ) : (
                        <XCircleIcon className="h-4 w-4 text-red-600" />
                      )}
                      <span className="text-sm">{config.routing.enabled ? "Yes" : "No"}</span>
                    </div>
                  </div>
                  
                  {config.routing.enabled && (
                    <>
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-600 dark:text-gray-400">Router Type:</span>
                        <span className="text-sm font-medium">{config.routing.router_type}</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-600 dark:text-gray-400">Strong Model:</span>
                        <span className="text-sm font-medium">{config.routing.strong_model}</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-600 dark:text-gray-400">Weak Model:</span>
                        <span className="text-sm font-medium">{config.routing.weak_model}</span>
                      </div>
                    </>
                  )}
                </div>
              </div>

              {/* Caching Configuration */}
              <div className="space-y-3">
                <h4 className="font-medium text-gray-900 dark:text-white">Semantic Caching</h4>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-400">Enabled:</span>
                    <div className="flex items-center space-x-1">
                      {config.cache.enabled ? (
                        <CheckCircleIcon className="h-4 w-4 text-green-600" />
                      ) : (
                        <XCircleIcon className="h-4 w-4 text-red-600" />
                      )}
                      <span className="text-sm">{config.cache.enabled ? "Yes" : "No"}</span>
                    </div>
                  </div>
                  
                  {config.cache.enabled && (
                    <>
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-600 dark:text-gray-400">Cache Type:</span>
                        <span className="text-sm font-medium">{config.cache.type}</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-600 dark:text-gray-400">Similarity Threshold:</span>
                        <span className="text-sm font-medium">{config.cache.similarity_threshold}</span>
                      </div>
                    </>
                  )}
                </div>
              </div>

              {/* Security Configuration */}
              <div className="space-y-3">
                <h4 className="font-medium text-gray-900 dark:text-white">Security Features</h4>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-400">Prompt Injection Detection:</span>
                    <div className="flex items-center space-x-1">
                      {config.security.prompt_injection_detection ? (
                        <CheckCircleIcon className="h-4 w-4 text-green-600" />
                      ) : (
                        <XCircleIcon className="h-4 w-4 text-red-600" />
                      )}
                      <span className="text-sm">{config.security.prompt_injection_detection ? "Yes" : "No"}</span>
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-400">PII Scrubbing:</span>
                    <div className="flex items-center space-x-1">
                      {config.security.pii_scrubbing ? (
                        <CheckCircleIcon className="h-4 w-4 text-green-600" />
                      ) : (
                        <XCircleIcon className="h-4 w-4 text-red-600" />
                      )}
                      <span className="text-sm">{config.security.pii_scrubbing ? "Yes" : "No"}</span>
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-400">Circuit Breakers:</span>
                    <div className="flex items-center space-x-1">
                      {config.security.circuit_breakers ? (
                        <CheckCircleIcon className="h-4 w-4 text-green-600" />
                      ) : (
                        <XCircleIcon className="h-4 w-4 text-red-600" />
                      )}
                      <span className="text-sm">{config.security.circuit_breakers ? "Yes" : "No"}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};