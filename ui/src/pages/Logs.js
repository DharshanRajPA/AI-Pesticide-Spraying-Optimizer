import React from 'react';
import { useParams } from 'react-router-dom';
import styled from 'styled-components';

const LogsContainer = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: ${props => props.theme.spacing.lg};
`;

const LogsHeader = styled.div`
  margin-bottom: ${props => props.theme.spacing.lg};
`;

const Title = styled.h1`
  color: ${props => props.theme.colors.text};
  margin-bottom: ${props => props.theme.spacing.sm};
`;

const Subtitle = styled.p`
  color: ${props => props.theme.colors.textSecondary};
  font-size: 1.1rem;
`;

const LogsContent = styled.div`
  background: ${props => props.theme.colors.surface};
  border-radius: ${props => props.theme.borderRadius.lg};
  padding: ${props => props.theme.spacing.lg};
  box-shadow: ${props => props.theme.shadows.sm};
`;

const LogEntry = styled.div`
  padding: ${props => props.theme.spacing.md};
  border-bottom: 1px solid ${props => props.theme.colors.border};
  
  &:last-child {
    border-bottom: none;
  }
`;

const LogTimestamp = styled.span`
  color: ${props => props.theme.colors.textSecondary};
  font-size: 0.9rem;
  margin-right: ${props => props.theme.spacing.md};
`;

const LogMessage = styled.span`
  color: ${props => props.theme.colors.text};
`;

function Logs() {
  const { runId } = useParams();

  // Mock log data - in a real app, this would come from an API
  const mockLogs = [
    { timestamp: '2024-01-15 10:30:15', message: 'System initialized successfully' },
    { timestamp: '2024-01-15 10:30:16', message: 'Loading field data from sensors' },
    { timestamp: '2024-01-15 10:30:18', message: 'Processing pest detection model' },
    { timestamp: '2024-01-15 10:30:22', message: 'Generating spray optimization plan' },
    { timestamp: '2024-01-15 10:30:25', message: 'Plan generated successfully' },
  ];

  return (
    <LogsContainer>
      <LogsHeader>
        <Title>System Logs</Title>
        <Subtitle>Run ID: {runId}</Subtitle>
      </LogsHeader>
      
      <LogsContent>
        {mockLogs.map((log, index) => (
          <LogEntry key={index}>
            <LogTimestamp>{log.timestamp}</LogTimestamp>
            <LogMessage>{log.message}</LogMessage>
          </LogEntry>
        ))}
      </LogsContent>
    </LogsContainer>
  );
}

export default Logs;
