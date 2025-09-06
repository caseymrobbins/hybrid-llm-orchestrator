// src/contexts/WebSocketContext.tsx
import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';

interface WebSocketMessage {
  type: 'query_start' | 'module_start' | 'module_complete' | 'query_complete' | 'heartbeat';
  timestamp: string;
  [key: string]: any;
}

interface WebSocketContextType {
  isConnected: boolean;
  messages: WebSocketMessage[];
  currentExecution: {
    isRunning: boolean;
    currentModule?: string;
    progress: number;
    startTime?: number;
  };
  sendMessage: (message: string) => void;
  clearMessages: () => void;
}

const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined);

interface WebSocketProviderProps {
  children: ReactNode;
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({ children }) => {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [messages, setMessages] = useState<WebSocketMessage[]>([]);
  const [currentExecution, setCurrentExecution] = useState({
    isRunning: false,
    currentModule: undefined as string | undefined,
    progress: 0,
    startTime: undefined as number | undefined,
  });

  const WS_URL = 'ws://localhost:8000/ws';

  useEffect(() => {
    let reconnectTimeout: NodeJS.Timeout;
    let pingInterval: NodeJS.Timeout;

    const connect = () => {
      try {
        const ws = new WebSocket(WS_URL);

        ws.onopen = () => {
          console.log('WebSocket connected');
          setIsConnected(true);
          setSocket(ws);

          // Start ping interval to keep connection alive
          pingInterval = setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) {
              ws.send('ping');
            }
          }, 30000); // Ping every 30 seconds
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            
            // Handle pong response
            if (event.data === 'pong') {
              return;
            }

            // Add message to history
            setMessages(prev => [...prev.slice(-49), data]); // Keep last 50 messages

            // Update execution state based on message type
            switch (data.type) {
              case 'query_start':
                setCurrentExecution({
                  isRunning: true,
                  currentModule: undefined,
                  progress: 0,
                  startTime: Date.now(),
                });
                break;

              case 'module_start':
                setCurrentExecution(prev => ({
                  ...prev,
                  currentModule: data.module,
                }));
                break;

              case 'module_complete':
                setCurrentExecution(prev => ({
                  ...prev,
                  progress: prev.progress + 25, // Assume 4 modules for now
                }));
                break;

              case 'query_complete':
                setCurrentExecution({
                  isRunning: false,
                  currentModule: undefined,
                  progress: 100,
                  startTime: undefined,
                });
                break;
            }
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
          }
        };

        ws.onclose = () => {
          console.log('WebSocket disconnected');
          setIsConnected(false);
          setSocket(null);
          clearInterval(pingInterval);

          // Attempt to reconnect after 3 seconds
          reconnectTimeout = setTimeout(() => {
            console.log('Attempting to reconnect...');
            connect();
          }, 3000);
        };

        ws.onerror = (error) => {
          console.error('WebSocket error:', error);
        };

      } catch (error) {
        console.error('Failed to create WebSocket connection:', error);
        // Retry connection after 5 seconds
        reconnectTimeout = setTimeout(connect, 5000);
      }
    };

    connect();

    return () => {
      clearTimeout(reconnectTimeout);
      clearInterval(pingInterval);
      if (socket) {
        socket.close();
      }
    };
  }, []);

  const sendMessage = (message: string) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(message);
    }
  };

  const clearMessages = () => {
    setMessages([]);
  };

  return (
    <WebSocketContext.Provider
      value={{
        isConnected,
        messages,
        currentExecution,
        sendMessage,
        clearMessages,
      }}
    >
      {children}
    </WebSocketContext.Provider>
  );
};

export const useWebSocket = (): WebSocketContextType => {
  const context = useContext(WebSocketContext);
  if (context === undefined) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
};