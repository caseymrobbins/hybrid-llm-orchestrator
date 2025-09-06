// src/App.tsx
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from './components/ui/toaster';
import { Layout } from './components/Layout';
import { Dashboard } from './pages/Dashboard';
import { Metrics } from './pages/Metrics';
import { Configuration } from './pages/Configuration';
import { Health } from './pages/Health';
import { WebSocketProvider } from './contexts/WebSocketContext';
import './App.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30000, // 30 seconds
      refetchOnWindowFocus: false,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <WebSocketProvider>
        <Router>
          <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
            <Routes>
              <Route path="/" element={<Layout />}>
                <Route index element={<Dashboard />} />
                <Route path="metrics" element={<Metrics />} />
                <Route path="configuration" element={<Configuration />} />
                <Route path="health" element={<Health />} />
              </Route>
            </Routes>
            <Toaster />
          </div>
        </Router>
      </WebSocketProvider>
    </QueryClientProvider>
  );
}

export default App;