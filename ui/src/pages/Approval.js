import React, { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { 
  ArrowLeft, 
  CheckCircle, 
  XCircle, 
  Clock, 
  User, 
  FileText,
  Shield,
  AlertTriangle,
  Download
} from 'lucide-react';
import toast from 'react-hot-toast';

import { getLogs } from '../services/api';

const ApprovalContainer = styled.div`
  max-width: 800px;
  margin: 0 auto;
  padding: ${props => props.theme.spacing.xl};
`;

const Header = styled.div`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.md};
  margin-bottom: ${props => props.theme.spacing.xl};
`;

const BackButton = styled.button`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
  padding: ${props => props.theme.spacing.sm} ${props => props.theme.spacing.md};
  background: ${props => props.theme.colors.surface};
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.borderRadius.md};
  color: ${props => props.theme.colors.text};
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover {
    background: ${props => props.theme.colors.primaryLight + '10'};
    border-color: ${props => props.theme.colors.primary};
  }
`;

const Title = styled.h1`
  font-size: 2rem;
  font-weight: 600;
  color: ${props => props.theme.colors.text};
  margin: 0;
`;

const StatusCard = styled(motion.div)`
  background: ${props => props.theme.colors.surface};
  border-radius: ${props => props.theme.borderRadius.lg};
  padding: ${props => props.theme.spacing.xl};
  box-shadow: ${props => props.theme.shadows.sm};
  text-align: center;
  margin-bottom: ${props => props.theme.spacing.xl};
`;

const StatusIcon = styled.div`
  width: 80px;
  height: 80px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 auto ${props => props.theme.spacing.lg};
  background: ${props => {
    if (props.status === 'approved') return props.theme.colors.success + '15';
    if (props.status === 'rejected') return props.theme.colors.error + '15';
    return props.theme.colors.warning + '15';
  }};
  color: ${props => {
    if (props.status === 'approved') return props.theme.colors.success;
    if (props.status === 'rejected') return props.theme.colors.error;
    return props.theme.colors.warning;
  }};
`;

const StatusTitle = styled.h2`
  font-size: 1.5rem;
  font-weight: 600;
  color: ${props => props.theme.colors.text};
  margin: 0 0 ${props => props.theme.spacing.sm} 0;
`;

const StatusMessage = styled.p`
  color: ${props => props.theme.colors.textSecondary};
  margin: 0;
`;

const DetailsCard = styled.div`
  background: ${props => props.theme.colors.surface};
  border-radius: ${props => props.theme.borderRadius.lg};
  padding: ${props => props.theme.spacing.lg};
  box-shadow: ${props => props.theme.shadows.sm};
  margin-bottom: ${props => props.theme.spacing.lg};
`;

const DetailsTitle = styled.h3`
  font-size: 1.125rem;
  font-weight: 600;
  color: ${props => props.theme.colors.text};
  margin: 0 0 ${props => props.theme.spacing.md} 0;
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
`;

const DetailsGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: ${props => props.theme.spacing.md};
`;

const DetailItem = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: ${props => props.theme.spacing.sm};
  background: ${props => props.theme.colors.background};
  border-radius: ${props => props.theme.borderRadius.sm};
`;

const DetailLabel = styled.span`
  font-size: 0.875rem;
  color: ${props => props.theme.colors.textSecondary};
`;

const DetailValue = styled.span`
  font-weight: 500;
  color: ${props => props.theme.colors.text};
`;

const CommentsSection = styled.div`
  background: ${props => props.theme.colors.surface};
  border-radius: ${props => props.theme.borderRadius.lg};
  padding: ${props => props.theme.spacing.lg};
  box-shadow: ${props => props.theme.shadows.sm};
  margin-bottom: ${props => props.theme.spacing.lg};
`;

const CommentsTitle = styled.h3`
  font-size: 1.125rem;
  font-weight: 600;
  color: ${props => props.theme.colors.text};
  margin: 0 0 ${props => props.theme.spacing.md} 0;
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
`;

const CommentsText = styled.div`
  background: ${props => props.theme.colors.background};
  border-radius: ${props => props.theme.borderRadius.md};
  padding: ${props => props.theme.spacing.md};
  color: ${props => props.theme.colors.text};
  line-height: 1.6;
  min-height: 100px;
`;

const ActionButtons = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing.md};
  justify-content: center;
`;

const ActionButton = styled(motion.button)`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
  padding: ${props => props.theme.spacing.md} ${props => props.theme.spacing.lg};
  border-radius: ${props => props.theme.borderRadius.md};
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
  border: 1px solid ${props => props.theme.colors.border};
  background: ${props => props.theme.colors.surface};
  color: ${props => props.theme.colors.text};
  
  &:hover {
    background: ${props => props.theme.colors.background};
  }
`;

const WarningBox = styled.div`
  background: ${props => props.theme.colors.warning + '15'};
  border: 1px solid ${props => props.theme.colors.warning + '30'};
  border-radius: ${props => props.theme.borderRadius.md};
  padding: ${props => props.theme.spacing.md};
  margin-bottom: ${props => props.theme.spacing.lg};
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
`;

const InfoBox = styled.div`
  background: ${props => props.theme.colors.info + '15'};
  border: 1px solid ${props => props.theme.colors.info + '30'};
  border-radius: ${props => props.theme.borderRadius.md};
  padding: ${props => props.theme.spacing.md};
  margin-bottom: ${props => props.theme.spacing.lg};
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
`;

function Approval() {
  const { runId } = useParams();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [approvalData, setApprovalData] = useState(null);
  const [error, setError] = useState(null);

  const loadApprovalData = useCallback(async () => {
    try {
      setLoading(true);
      const response = await getLogs(runId);
      setApprovalData(response.data);
    } catch (err) {
      setError(err.message);
      toast.error('Failed to load approval data');
    } finally {
      setLoading(false);
    }
  }, [runId]);

  useEffect(() => {
    loadApprovalData();
  }, [runId, loadApprovalData]);

  const handleDownloadLogs = () => {
    if (approvalData) {
      const blob = new Blob([JSON.stringify(approvalData, null, 2)], {
        type: 'application/json'
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `approval_log_${runId}.json`;
      a.click();
      URL.revokeObjectURL(url);
    }
  };

  if (loading) {
    return (
      <ApprovalContainer>
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '400px' }}>
          <div>Loading approval data...</div>
        </div>
      </ApprovalContainer>
    );
  }

  if (error || !approvalData) {
    return (
      <ApprovalContainer>
        <div style={{ textAlign: 'center', padding: '2rem' }}>
          <AlertTriangle size={48} color="#ef4444" />
          <h2>Error Loading Approval Data</h2>
          <p>{error || 'No approval data available'}</p>
          <BackButton onClick={() => navigate('/')}>
            <ArrowLeft size={16} />
            Back to Home
          </BackButton>
        </div>
      </ApprovalContainer>
    );
  }

  const approval = approvalData.approvals;
  const decision = approval?.decision || 'pending';
  const operatorId = approval?.operator_id || 'Unknown';
  const timestamp = approval?.timestamp || new Date().toISOString();
  const comments = approval?.comments || 'No comments provided';

  const getStatusIcon = () => {
    switch (decision) {
      case 'approved':
        return <CheckCircle size={40} />;
      case 'rejected':
        return <XCircle size={40} />;
      default:
        return <Clock size={40} />;
    }
  };

  const getStatusTitle = () => {
    switch (decision) {
      case 'approved':
        return 'Plan Approved';
      case 'rejected':
        return 'Plan Rejected';
      default:
        return 'Pending Approval';
    }
  };

  const getStatusMessage = () => {
    switch (decision) {
      case 'approved':
        return 'The flight plan has been approved and is ready for execution.';
      case 'rejected':
        return 'The flight plan has been rejected. Please review and modify the plan.';
      default:
        return 'The flight plan is awaiting operator approval.';
    }
  };

  return (
    <ApprovalContainer>
      <Header>
        <BackButton onClick={() => navigate('/')}>
          <ArrowLeft size={16} />
          Back to Home
        </BackButton>
        <Title>Approval Status</Title>
      </Header>

      <StatusCard
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <StatusIcon status={decision}>
          {getStatusIcon()}
        </StatusIcon>
        <StatusTitle>{getStatusTitle()}</StatusTitle>
        <StatusMessage>{getStatusMessage()}</StatusMessage>
      </StatusCard>

      {decision === 'approved' && (
        <InfoBox>
          <Shield size={20} color="#3b82f6" />
          <div>
            <strong>Plan Approved for Execution</strong>
            <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.875rem' }}>
              The flight plan has been approved and can now be executed. All safety checks have been completed.
            </p>
          </div>
        </InfoBox>
      )}

      {decision === 'rejected' && (
        <WarningBox>
          <AlertTriangle size={20} color="#f59e0b" />
          <div>
            <strong>Plan Rejected</strong>
            <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.875rem' }}>
              The flight plan has been rejected. Please review the comments and modify the plan accordingly.
            </p>
          </div>
        </WarningBox>
      )}

      <DetailsCard>
        <DetailsTitle>
          <User size={20} />
          Approval Details
        </DetailsTitle>
        <DetailsGrid>
          <DetailItem>
            <DetailLabel>Run ID</DetailLabel>
            <DetailValue>{runId}</DetailValue>
          </DetailItem>
          <DetailItem>
            <DetailLabel>Decision</DetailLabel>
            <DetailValue style={{ 
              color: decision === 'approved' ? '#10b981' : 
                     decision === 'rejected' ? '#ef4444' : '#f59e0b',
              textTransform: 'capitalize'
            }}>
              {decision}
            </DetailValue>
          </DetailItem>
          <DetailItem>
            <DetailLabel>Operator ID</DetailLabel>
            <DetailValue>{operatorId}</DetailValue>
          </DetailItem>
          <DetailItem>
            <DetailLabel>Timestamp</DetailLabel>
            <DetailValue>{new Date(timestamp).toLocaleString()}</DetailValue>
          </DetailItem>
        </DetailsGrid>
      </DetailsCard>

      <CommentsSection>
        <CommentsTitle>
          <FileText size={20} />
          Operator Comments
        </CommentsTitle>
        <CommentsText>
          {comments}
        </CommentsText>
      </CommentsSection>

      <ActionButtons>
        <ActionButton
          onClick={handleDownloadLogs}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <Download size={16} />
          Download Approval Log
        </ActionButton>
        
        <ActionButton
          onClick={() => navigate(`/results/${runId}`)}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <FileText size={16} />
          View Original Results
        </ActionButton>
      </ActionButtons>
    </ApprovalContainer>
  );
}

export default Approval;
