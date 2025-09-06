// src/pages/Dashboard.tsx
import React, { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { PaperAirplaneIcon, ClockIcon, CpuChipIcon } from '@heroicons/react/24/outline';
import { Button } from '../components/ui/button';
import { Textarea } from '../components/ui/textarea';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Progress } from '../components/ui/progress';
import { Badge } from '../components/ui/badge';
import { useWebSocket } from '../contexts/WebSocketContext';
import { processQuery } from '../services/api';
import { ModuleExecution } from '../components/ModuleExecution';
import { QueryHistory } from '../components/QueryHistory';
import { toast } from '../components/ui/use-toast';

interface QueryResponse {
  response: string;
  module_outputs: Record<string, any>;
  routing_decisions: Array<any>;
  metrics: Record<string, any>;
  execution_time: number;
  interaction_id: number;
}

export const Dashboard: React.FC = () => {
  const [query, setQuery] = useState('');
  const [lastResponse, setLastResponse] = useState<QueryResponse | null>(null);
  const { currentExecution, isConnected } = useWebSocket();

  const queryMutation = useMutation({
    mutationFn: processQuery,
    onSuccess: (data: QueryResponse) => {
      setLastResponse(data);
      setQuery(''); // Clear the input
      toast({
        title: "Query Processed Successfully",
        description: `Completed in ${data.execution_time.toFixed(2)} seconds`,
      });
    },
    onError: (error: any) => {
      toast({
        title: "Query Failed",
        description: error.message || "An unexpected error occurred",
        variant: "destructive",
      });
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    queryMutation.mutate({
      query: query.trim(),
      user_id: 1,
      preferences: {},
    });
  };

  const isProcessing = queryMutation.isPending || currentExecution.isRunning;

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            AI Orchestrator Dashboard
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Process queries through multiple AI models with intelligent routing
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          <Badge variant={isConnected ? "default" : "destructive"}>
            {isConnected ? "Connected" : "Disconnected"}
          </Badge>
        </div>
      </div>

      {/* Query Input */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <CpuChipIcon className="h-5 w-5" />
            <span>Query Processor</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <Textarea
                placeholder="Enter your query here... (e.g., 'What are the ethical implications of AI in healthcare?')"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="min-h-[100px] resize-none"
                disabled={isProcessing}
              />
            </div>
            
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2 text-sm text-gray-500 dark:text-gray-400">
                <ClockIcon className="h-4 w-4" />
                <span>
                  {query.length > 0 ? `${query.length} characters` : 'Type your query...'}
                </span>
              </div>
              
              <Button
                type="submit"
                disabled={!query.trim() || isProcessing}
                className="flex items-center space-x-2"
              >
                <PaperAirplaneIcon className="h-4 w-4" />
                <span>{isProcessing ? 'Processing...' : 'Process Query'}</span>
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>

      {/* Execution Progress */}
      {isProcessing && (
        <Card>
          <CardHeader>
            <CardTitle>Execution Progress</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Overall Progress</span>
                <span>{currentExecution.progress}%</span>
              </div>
              <Progress value={currentExecution.progress} className="w-full" />
            </div>
            
            {currentExecution.currentModule && (
              <div className="flex items-center space-x-2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                <span className="text-sm text-gray-600 dark:text-gray-400">
                  Executing module: <span className="font-medium">{currentExecution.currentModule}</span>
                </span>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Module Execution */}
      <ModuleExecution />

      {/* Results */}
      {lastResponse && (
        <Card>
          <CardHeader>
            <CardTitle>Response</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="prose dark:prose-invert max-w-none">
                <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
                  <p className="whitespace-pre-wrap">{lastResponse.response}</p>
                </div>
              </div>
              
              <div className="flex flex-wrap gap-2 pt-2 border-t border-gray-200 dark:border-gray-700">
                <Badge variant="outline">
                  ⏱️ {lastResponse.execution_time.toFixed(2)}s
                </Badge>
                <Badge variant="outline">
                  🔄 {Object.keys(lastResponse.module_outputs).length} modules
                </Badge>
                <Badge variant="outline">
                  💰 ${(lastResponse.routing_decisions.reduce((sum, d) => sum + (d.cost_cents || 0), 0) / 100).toFixed(4)}
                </Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Query History */}
      <QueryHistory />
    </div>
  );
};