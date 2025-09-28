import React, { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { 
  ArrowLeft, 
  Map, 
  Clock, 
  Droplets, 
  CheckCircle, 
  AlertTriangle,
  Download,
  Play,
  Pause,
  RotateCcw
} from 'lucide-react';
import toast from 'react-hot-toast';

import { getLogs, plan, approve } from '../services/api';
import LoadingSpinner from '../components/LoadingSpinner';

const PlanContainer = styled.div`
  max-width: 1200px;
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

const PlanGrid = styled.div`
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: ${props => props.theme.spacing.xl};

  @media (max-width: ${props => props.theme.breakpoints.mobile}) {
    grid-template-columns: 1fr;
  }
`;

const MapSection = styled.div`
  background: ${props => props.theme.colors.surface};
  border-radius: ${props => props.theme.borderRadius.lg};
  padding: ${props => props.theme.spacing.lg};
  box-shadow: ${props => props.theme.shadows.sm};
  min-height: 500px;
`;

const MapContainer = styled.div`
  width: 100%;
  height: 400px;
  background: #f0f0f0;
  border-radius: ${props => props.theme.borderRadius.md};
  display: flex;
  align-items: center;
  justify-content: center;
  color: ${props => props.theme.colors.textSecondary};
  margin-bottom: ${props => props.theme.spacing.md};
`;

const WaypointList = styled.div`
  max-height: 300px;
  overflow-y: auto;
`;

const WaypointItem = styled.div`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
  padding: ${props => props.theme.spacing.sm};
  border-radius: ${props => props.theme.borderRadius.sm};
  background: ${props => props.active ? props.theme.colors.primaryLight + '15' : 'transparent'};
  margin-bottom: ${props => props.theme.spacing.xs};
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover {
    background: ${props => props.theme.colors.primaryLight + '10'};
  }
`;

const WaypointIcon = styled.div`
  width: 24px;
  height: 24px;
  border-radius: 50%;
  background: ${props => props.theme.colors.primary};
  color: ${props => props.theme.colors.surface};
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.75rem;
  font-weight: 600;
`;

const WaypointInfo = styled.div`
  flex: 1;
`;

const WaypointName = styled.div`
  font-weight: 500;
  color: ${props => props.theme.colors.text};
`;

const WaypointDetails = styled.div`
  font-size: 0.75rem;
  color: ${props => props.theme.colors.textSecondary};
`;

const DetailsSection = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${props => props.theme.spacing.lg};
`;

const StatsCard = styled.div`
  background: ${props => props.theme.colors.surface};
  border-radius: ${props => props.theme.borderRadius.lg};
  padding: ${props => props.theme.spacing.lg};
  box-shadow: ${props => props.theme.shadows.sm};
`;

const StatsTitle = styled.h3`
  font-size: 1.125rem;
  font-weight: 600;
  color: ${props => props.theme.colors.text};
  margin: 0 0 ${props => props.theme.spacing.md} 0;
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
`;

const StatsGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: ${props => props.theme.spacing.md};
`;

const StatItem = styled.div`
  text-align: center;
`;

const StatValue = styled.div`
  font-size: 1.5rem;
  font-weight: 700;
  color: ${props => props.theme.colors.primary};
  margin-bottom: ${props => props.theme.spacing.xs};
`;

const StatLabel = styled.div`
  font-size: 0.75rem;
  color: ${props => props.theme.colors.textSecondary};
  font-weight: 500;
`;

const DoseList = styled.div`
  max-height: 200px;
  overflow-y: auto;
`;

const DoseItem = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: ${props => props.theme.spacing.sm};
  border-radius: ${props => props.theme.borderRadius.sm};
  background: ${props => props.theme.colors.background};
  margin-bottom: ${props => props.theme.spacing.xs};
`;

const DoseInfo = styled.div`
  font-size: 0.875rem;
  color: ${props => props.theme.colors.text};
`;

const DoseValue = styled.div`
  font-weight: 600;
  color: ${props => props.theme.colors.primary};
`;

const ApprovalSection = styled.div`
  background: ${props => props.theme.colors.surface};
  border-radius: ${props => props.theme.borderRadius.lg};
  padding: ${props => props.theme.spacing.lg};
  box-shadow: ${props => props.theme.shadows.sm};
`;

const ApprovalTitle = styled.h3`
  font-size: 1.125rem;
  font-weight: 600;
  color: ${props => props.theme.colors.text};
  margin: 0 0 ${props => props.theme.spacing.md} 0;
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
`;

const WarningBox = styled.div`
  background: ${props => props.theme.colors.warning + '15'};
  border: 1px solid ${props => props.theme.colors.warning + '30'};
  border-radius: ${props => props.theme.borderRadius.md};
  padding: ${props => props.theme.spacing.md};
  margin-bottom: ${props => props.theme.spacing.md};
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
`;

const ApprovalButtons = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing.md};
  margin-top: ${props => props.theme.spacing.lg};
`;

const ApprovalButton = styled(motion.button)`
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: ${props => props.theme.spacing.sm};
  padding: ${props => props.theme.spacing.md};
  border-radius: ${props => props.theme.borderRadius.md};
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
  border: none;
  
  ${props => props.approve ? `
    background: ${props.theme.colors.success};
    color: ${props.theme.colors.surface};
    
    &:hover {
      background: ${props.theme.colors.success + 'dd'};
    }
  ` : `
    background: ${props.theme.colors.error};
    color: ${props.theme.colors.surface};
    
    &:hover {
      background: ${props.theme.colors.error + 'dd'};
    }
  `}
`;

const ActionButtons = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing.md};
  margin-top: ${props => props.theme.spacing.xl};
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

function Plan() {
  const { runId } = useParams();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [planData, setPlanData] = useState(null);
  const [error, setError] = useState(null);
  const [approving, setApproving] = useState(false);

  const loadPlan = useCallback(async () => {
    try {
      setLoading(true);
      
      // First get the predictions
      const predictionsResponse = await getLogs(runId);
      const predictions = predictionsResponse.data.predictions?.predictions || [];
      
      if (predictions.length === 0) {
        throw new Error('No predictions found for planning');
      }
      
      // Generate flight plan
      const planResponse = await plan({
        predictions: predictions,
        location: {
          latitude: 40.7128,
          longitude: -74.0060
        },
        constraints: {
          max_dose_per_plant: 50,
          max_total_dose: 1000
        }
      });
      
      setPlanData(planResponse.data);
    } catch (err) {
      setError(err.message);
      toast.error('Failed to generate flight plan');
    } finally {
      setLoading(false);
    }
  }, [runId]);

  useEffect(() => {
    loadPlan();
  }, [runId, loadPlan]);

  const handleApprove = async (approved) => {
    try {
      setApproving(true);
      
      await approve({
        run_id: runId,
        decision: approved ? 'approved' : 'rejected',
        operator_id: 'operator_001',
        comments: approved ? 'Plan approved for execution' : 'Plan rejected due to safety concerns'
      });
      
      toast.success(approved ? 'Plan approved successfully' : 'Plan rejected');
      
      if (approved) {
        navigate(`/approval/${runId}`);
      } else {
        navigate('/');
      }
    } catch (err) {
      toast.error('Failed to process approval');
    } finally {
      setApproving(false);
    }
  };

  const handleDownloadMission = () => {
    if (planData?.mavlink_json) {
      const blob = new Blob([JSON.stringify(planData.mavlink_json, null, 2)], {
        type: 'application/json'
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `mission_${runId}.json`;
      a.click();
      URL.revokeObjectURL(url);
    }
  };

  if (loading) {
    return (
      <PlanContainer>
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '400px' }}>
          <LoadingSpinner size={48} />
        </div>
      </PlanContainer>
    );
  }

  if (error || !planData) {
    return (
      <PlanContainer>
        <div style={{ textAlign: 'center', padding: '2rem' }}>
          <AlertTriangle size={48} color="#ef4444" />
          <h2>Error Generating Plan</h2>
          <p>{error || 'No plan data available'}</p>
          <BackButton onClick={() => navigate('/')}>
            <ArrowLeft size={16} />
            Back to Home
          </BackButton>
        </div>
      </PlanContainer>
    );
  }

  const waypoints = planData.waypoints || [];
  const doses = planData.doses || [];
  const totalPesticide = planData.total_pesticide || 0;
  const estimatedTime = planData.estimated_time || 0;
  const requiresApproval = planData.requires_approval || false;

  return (
    <PlanContainer>
      <Header>
        <BackButton onClick={() => navigate(`/results/${runId}`)}>
          <ArrowLeft size={16} />
          Back to Results
        </BackButton>
        <Title>Flight Plan</Title>
      </Header>

      <PlanGrid>
        <MapSection>
          <h2 style={{ margin: '0 0 1rem 0', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Map size={20} />
            Flight Path
          </h2>
          
          <MapContainer>
            <div style={{ textAlign: 'center' }}>
              <Map size={48} color="#9ca3af" />
              <p>Interactive map will be displayed here</p>
              <p style={{ fontSize: '0.875rem' }}>Showing {waypoints.length} waypoints</p>
            </div>
          </MapContainer>
          
          <WaypointList>
            {waypoints.map((waypoint, index) => (
              <WaypointItem key={index} active={index === 0}>
                <WaypointIcon>{index + 1}</WaypointIcon>
                <WaypointInfo>
                  <WaypointName>
                    {waypoint.command === 22 ? 'Takeoff' : 
                     waypoint.command === 21 ? 'Landing' : 
                     `Waypoint ${index + 1}`}
                  </WaypointName>
                  <WaypointDetails>
                    Lat: {waypoint.latitude?.toFixed(6)}, Lon: {waypoint.longitude?.toFixed(6)}
                    {waypoint.altitude && `, Alt: ${waypoint.altitude}m`}
                  </WaypointDetails>
                </WaypointInfo>
              </WaypointItem>
            ))}
          </WaypointList>
        </MapSection>

        <DetailsSection>
          <StatsCard>
            <StatsTitle>
              <Clock size={20} />
              Mission Statistics
            </StatsTitle>
            <StatsGrid>
              <StatItem>
                <StatValue>{waypoints.length}</StatValue>
                <StatLabel>Waypoints</StatLabel>
              </StatItem>
              <StatItem>
                <StatValue>{(estimatedTime / 60).toFixed(1)}m</StatValue>
                <StatLabel>Flight Time</StatLabel>
              </StatItem>
              <StatItem>
                <StatValue>{totalPesticide.toFixed(1)}ml</StatValue>
                <StatLabel>Total Pesticide</StatLabel>
              </StatItem>
              <StatItem>
                <StatValue>{doses.length}</StatValue>
                <StatLabel>Spray Points</StatLabel>
              </StatItem>
            </StatsGrid>
          </StatsCard>

          <StatsCard>
            <StatsTitle>
              <Droplets size={20} />
              Dose Distribution
            </StatsTitle>
            <DoseList>
              {doses.map((dose, index) => (
                <DoseItem key={index}>
                  <DoseInfo>Plant #{index + 1}</DoseInfo>
                  <DoseValue>{dose.toFixed(1)}ml</DoseValue>
                </DoseItem>
              ))}
            </DoseList>
          </StatsCard>

          <ApprovalSection>
            <ApprovalTitle>
              <CheckCircle size={20} />
              Operator Approval
            </ApprovalTitle>
            
            {requiresApproval && (
              <WarningBox>
                <AlertTriangle size={20} color="#f59e0b" />
                <div>
                  <strong>Approval Required</strong>
                  <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.875rem' }}>
                    This plan requires operator approval due to low confidence predictions or regulatory considerations.
                  </p>
                </div>
              </WarningBox>
            )}
            
            <ApprovalButtons>
              <ApprovalButton
                approve={false}
                onClick={() => handleApprove(false)}
                disabled={approving}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                {approving ? <LoadingSpinner size={16} /> : <Pause size={16} />}
                Reject
              </ApprovalButton>
              
              <ApprovalButton
                approve={true}
                onClick={() => handleApprove(true)}
                disabled={approving}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                {approving ? <LoadingSpinner size={16} /> : <Play size={16} />}
                Approve
              </ApprovalButton>
            </ApprovalButtons>
          </ApprovalSection>
        </DetailsSection>
      </PlanGrid>

      <ActionButtons>
        <ActionButton
          onClick={handleDownloadMission}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <Download size={16} />
          Download MAVLink Mission
        </ActionButton>
        
        <ActionButton
          onClick={() => window.open(planData.explainability_url, '_blank')}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <RotateCcw size={16} />
          View Explainability
        </ActionButton>
      </ActionButtons>
    </PlanContainer>
  );
}

export default Plan;
