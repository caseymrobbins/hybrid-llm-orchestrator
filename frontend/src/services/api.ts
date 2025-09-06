// src/services/api.ts
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5 minutes for long-running queries
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// Type definitions
export interface QueryRequest {
  query: string;
  user_id?: number;
  preferences?: Record<string, any>;
  modules_override?: string[];
}

export interface QueryResponse {
  response: string;
  module_outputs: Record<string, {
    output: string;
    cached: boolean;
    latency_ms: number;
    model_used: string;
    timestamp: string;
  }>;
  routing_decisions: Array<{
    module: string;
    original_model: string;
    routed_model: string;
    complexity_score: number;
    cost_cents: number;
    cached: boolean;
    latency_ms: number;
  }>;
  metrics: Record<string, any>;
  execution_time: number;
  interaction_id: number;
}

export interface HealthResponse {
  overall_status: string;
  database: Record<string, any>;
  external_apis: Record<string, string>;
  circuit_breakers: Record<string, any>;
  cache_status: Record<string, any>;
  timestamp: string;
}

export interface MetricsResponse {
  total_queries: number;
  avg_latency_ms: number;
  cache_hit_rate: number;
  cost_saved_cents: number;
  local_vs_external: Record<string, number>;
  recent_queries: Array<Record<string, any>>;
}

export interface ModuleInfo {
  name: string;
  aspect: string;
  has_fallback: boolean;
  uses_external: boolean;
  model_info: Record<string, any>;
}

export interface ConfigResponse {
  workflow: {
    name: string;
    description: string;
    version: string;
    execution_plan: Array<{
      module: string;
      dependencies: string[];
      parallel: boolean;
    }>;
    max_concurrent_steps: number;
  };
  routing: {
    enabled: boolean;
    router_type: string;
    strong_model: string;
    weak_model: string;
  };
  cache: {
    enabled: boolean;
    type: string;
    similarity_threshold: number;
  };
  security: {
    prompt_injection_detection: boolean;
    pii_scrubbing: boolean;
    circuit_breakers: boolean;
  };
}

// API functions
export const processQuery = async (request: QueryRequest): Promise<QueryResponse> => {
  const response = await api.post('/process', request);
  return response.data;
};

export const getHealth = async (): Promise<HealthResponse> => {
  const response = await api.get('/health');
  return response.data;
};

export const getMetrics = async (): Promise<MetricsResponse> => {
  const response = await api.get('/metrics');
  return response.data;
};

export const getModules = async (): Promise<{ modules: ModuleInfo[]; total_count: number }> => {
  const response = await api.get('/modules');
  return response.data;
};

export const getConfiguration = async (): Promise<ConfigResponse> => {
  const response = await api.get('/config');
  return response.data;
};

export const getAnalytics = async (days: number = 7): Promise<Record<string, any>> => {
  const response = await api.get(`/analytics?days=${days}`);
  return response.data;
};

export const getCostBreakdown = async (): Promise<{
  total_cost_today: number;
  cost_saved_by_cache: number;
  cost_saved_by_routing: number;
  projection_monthly: number;
}> => {
  const response = await api.get('/costs/breakdown');
  return response.data;
};

export const clearCache = async (): Promise<{ status: string; message: string }> => {
  const response = await api.post('/cache/clear');
  return response.data;
};

export const getCacheStats = async (): Promise<{
  status: string;
  stats: Record<string, any>;
}> => {
  const response = await api.post('/cache/stats');
  return response.data;
};

export const overrideModuleModel = async (
  moduleName: string,
  model: string
): Promise<{ status: string; message: string }> => {
  const response = await api.post(`/modules/${moduleName}/override?model=${model}`);
  return response.data;
};

// Utility function to check API connectivity
export const checkApiHealth = async (): Promise<boolean> => {
  try {
    await api.get('/');
    return true;
  } catch (error) {
    return false;
  }
};