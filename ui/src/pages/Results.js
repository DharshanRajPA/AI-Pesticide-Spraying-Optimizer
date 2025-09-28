import React, { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { 
  ArrowLeft, 
  CheckCircle, 
  AlertTriangle, 
  Info, 
  Download,
  Eye,
  Zap,
  Shield
} from 'lucide-react';
import toast from 'react-hot-toast';

import { getLogs, getExplainability } from '../services/api';
import LoadingSpinner from '../components/LoadingSpinner';

const ResultsContainer = styled.div`
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

const ResultsGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: ${props => props.theme.spacing.xl};

  @media (max-width: ${props => props.theme.breakpoints.mobile}) {
    grid-template-columns: 1fr;
  }
`;

const ImageSection = styled.div`
  background: ${props => props.theme.colors.surface};
  border-radius: ${props => props.theme.borderRadius.lg};
  padding: ${props => props.theme.spacing.lg};
  box-shadow: ${props => props.theme.shadows.sm};
`;

const ImageContainer = styled.div`
  position: relative;
  border-radius: ${props => props.theme.borderRadius.md};
  overflow: hidden;
  margin-bottom: ${props => props.theme.spacing.md};
`;

const Image = styled.img`
  width: 100%;
  height: auto;
  display: block;
`;

const OverlayCanvas = styled.canvas`
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
`;

const PredictionsSection = styled.div`
  background: ${props => props.theme.colors.surface};
  border-radius: ${props => props.theme.borderRadius.lg};
  padding: ${props => props.theme.spacing.lg};
  box-shadow: ${props => props.theme.shadows.sm};
`;

const SectionTitle = styled.h2`
  font-size: 1.25rem;
  font-weight: 600;
  color: ${props => props.theme.colors.text};
  margin: 0 0 ${props => props.theme.spacing.lg} 0;
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
`;

const PredictionCard = styled(motion.div)`
  background: ${props => props.theme.colors.background};
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.borderRadius.md};
  padding: ${props => props.theme.spacing.md};
  margin-bottom: ${props => props.theme.spacing.md};
`;

const PredictionHeader = styled.div`
  display: flex;
  justify-content: between;
  align-items: center;
  margin-bottom: ${props => props.theme.spacing.sm};
`;

const PredictionTitle = styled.h3`
  font-size: 1rem;
  font-weight: 600;
  color: ${props => props.theme.colors.text};
  margin: 0;
`;

const ConfidenceBadge = styled.div`
  padding: ${props => props.theme.spacing.xs} ${props => props.theme.spacing.sm};
  border-radius: ${props => props.theme.borderRadius.sm};
  font-size: 0.75rem;
  font-weight: 500;
  background: ${props => {
    if (props.confidence >= 0.8) return props.theme.colors.success + '15';
    if (props.confidence >= 0.6) return props.theme.colors.warning + '15';
    return props.theme.colors.error + '15';
  }};
  color: ${props => {
    if (props.confidence >= 0.8) return props.theme.colors.success;
    if (props.confidence >= 0.6) return props.theme.colors.warning;
    return props.theme.colors.error;
  }};
`;

const PredictionDetails = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: ${props => props.theme.spacing.sm};
  font-size: 0.875rem;
  color: ${props => props.theme.colors.textSecondary};
`;

const DetailItem = styled.div`
  display: flex;
  justify-content: space-between;
`;

const SeverityIndicator = styled.div`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.xs};
  margin-top: ${props => props.theme.spacing.sm};
`;

const SeverityBar = styled.div`
  flex: 1;
  height: 4px;
  background: ${props => props.theme.colors.border};
  border-radius: 2px;
  overflow: hidden;
`;

const SeverityFill = styled.div`
  height: 100%;
  width: ${props => (props.severity / 3) * 100}%;
  background: ${props => {
    if (props.severity === 0) return props.theme.colors.success;
    if (props.severity === 1) return props.theme.colors.warning;
    if (props.severity === 2) return props.theme.colors.error;
    return props.theme.colors.error;
  }};
  transition: width 0.3s ease;
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
  border: none;
  
  ${props => props.primary ? `
    background: ${props.theme.colors.primary};
    color: ${props.theme.colors.surface};
    
    &:hover {
      background: ${props.theme.colors.primaryDark};
    }
  ` : `
    background: ${props.theme.colors.surface};
    color: ${props.theme.colors.text};
    border: 1px solid ${props.theme.colors.border};
    
    &:hover {
      background: ${props.theme.colors.background};
    }
  `}
`;

const StatsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: ${props => props.theme.spacing.md};
  margin-bottom: ${props => props.theme.spacing.xl};
`;

const StatCard = styled.div`
  background: ${props => props.theme.colors.surface};
  border-radius: ${props => props.theme.borderRadius.md};
  padding: ${props => props.theme.spacing.lg};
  text-align: center;
  box-shadow: ${props => props.theme.shadows.sm};
`;

const StatValue = styled.div`
  font-size: 2rem;
  font-weight: 700;
  color: ${props => props.theme.colors.primary};
  margin-bottom: ${props => props.theme.spacing.xs};
`;

const StatLabel = styled.div`
  font-size: 0.875rem;
  color: ${props => props.theme.colors.textSecondary};
  font-weight: 500;
`;

function Results() {
  const { runId } = useParams();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  const loadResults = useCallback(async () => {
    try {
      setLoading(true);
      const response = await getLogs(runId);
      setData(response.data);
    } catch (err) {
      setError(err.message);
      toast.error('Failed to load results');
    } finally {
      setLoading(false);
    }
  }, [runId]);

  useEffect(() => {
    loadResults();
  }, [runId, loadResults]);

  const handleGeneratePlan = () => {
    navigate(`/plan/${runId}`);
  };

  const handleViewExplainability = async () => {
    try {
      await getExplainability(runId);
      // Open explainability in new tab or modal
      window.open(`/explain/${runId}`, '_blank');
    } catch (err) {
      toast.error('Failed to load explainability report');
    }
  };

  const drawPredictions = (canvas, predictions, image) => {
    if (!canvas || !predictions || !image) return;

    const ctx = canvas.getContext('2d');
    
    // Set canvas size to match image
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw bounding boxes
    predictions.forEach((pred, index) => {
      const [x, y, width, height] = pred.bbox;
      
      // Choose color based on confidence
      const confidence = pred.confidence;
      let color = '#ef4444'; // red for low confidence
      if (confidence >= 0.8) color = '#10b981'; // green for high confidence
      else if (confidence >= 0.6) color = '#f59e0b'; // yellow for medium confidence
      
      // Draw bounding box
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, width, height);
      
      // Draw label
      ctx.fillStyle = color;
      ctx.font = '14px sans-serif';
      ctx.fillText(
        `${pred.category_id} (${(confidence * 100).toFixed(1)}%)`,
        x, y - 5
      );
    });
  };

  if (loading) {
    return (
      <ResultsContainer>
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '400px' }}>
          <LoadingSpinner size={48} />
        </div>
      </ResultsContainer>
    );
  }

  if (error || !data) {
    return (
      <ResultsContainer>
        <div style={{ textAlign: 'center', padding: '2rem' }}>
          <AlertTriangle size={48} color="#ef4444" />
          <h2>Error Loading Results</h2>
          <p>{error || 'No data available'}</p>
          <BackButton onClick={() => navigate('/')}>
            <ArrowLeft size={16} />
            Back to Home
          </BackButton>
        </div>
      </ResultsContainer>
    );
  }

  const predictions = data.predictions?.predictions || [];
  const totalPredictions = predictions.length;
  const highConfidencePredictions = predictions.filter(p => p.confidence >= 0.8).length;
  const averageConfidence = predictions.reduce((sum, p) => sum + p.confidence, 0) / totalPredictions || 0;

  return (
    <ResultsContainer>
      <Header>
        <BackButton onClick={() => navigate('/')}>
          <ArrowLeft size={16} />
          Back
        </BackButton>
        <Title>Analysis Results</Title>
      </Header>

      <StatsGrid>
        <StatCard>
          <StatValue>{totalPredictions}</StatValue>
          <StatLabel>Total Detections</StatLabel>
        </StatCard>
        <StatCard>
          <StatValue>{highConfidencePredictions}</StatValue>
          <StatLabel>High Confidence</StatLabel>
        </StatCard>
        <StatCard>
          <StatValue>{(averageConfidence * 100).toFixed(1)}%</StatValue>
          <StatLabel>Avg Confidence</StatLabel>
        </StatCard>
        <StatCard>
          <StatValue>{data.predictions?.processing_time?.toFixed(2) || 0}s</StatValue>
          <StatLabel>Processing Time</StatLabel>
        </StatCard>
      </StatsGrid>

      <ResultsGrid>
        <ImageSection>
          <SectionTitle>
            <Eye size={20} />
            Field Image with Detections
          </SectionTitle>
          <ImageContainer>
            <Image
              src={`/api/images/${runId}`}
              alt="Field analysis"
              onLoad={(e) => {
                const canvas = document.getElementById('overlay-canvas');
                drawPredictions(canvas, predictions, e.target);
              }}
            />
            <OverlayCanvas id="overlay-canvas" />
          </ImageContainer>
        </ImageSection>

        <PredictionsSection>
          <SectionTitle>
            <CheckCircle size={20} />
            Detection Results
          </SectionTitle>
          
          {predictions.map((prediction, index) => (
            <PredictionCard
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <PredictionHeader>
                <PredictionTitle>Detection #{prediction.id}</PredictionTitle>
                <ConfidenceBadge confidence={prediction.confidence}>
                  {(prediction.confidence * 100).toFixed(1)}%
                </ConfidenceBadge>
              </PredictionHeader>
              
              <PredictionDetails>
                <DetailItem>
                  <span>Category:</span>
                  <span>Pest #{prediction.category_id}</span>
                </DetailItem>
                <DetailItem>
                  <span>Area:</span>
                  <span>{prediction.area.toFixed(1)} px²</span>
                </DetailItem>
                <DetailItem>
                  <span>Bbox:</span>
                  <span>[{prediction.bbox.map(b => b.toFixed(0)).join(', ')}]</span>
                </DetailItem>
                <DetailItem>
                  <span>Severity:</span>
                  <span>{prediction.severity}/3</span>
                </DetailItem>
              </PredictionDetails>
              
              <SeverityIndicator>
                <span>Severity:</span>
                <SeverityBar>
                  <SeverityFill severity={prediction.severity} />
                </SeverityBar>
                <span>{prediction.severity}/3</span>
              </SeverityIndicator>
            </PredictionCard>
          ))}
          
          {predictions.length === 0 && (
            <div style={{ textAlign: 'center', padding: '2rem', color: '#6b7280' }}>
              <Info size={48} />
              <p>No detections found in this image.</p>
            </div>
          )}
        </PredictionsSection>
      </ResultsGrid>

      <ActionButtons>
        <ActionButton
          primary
          onClick={handleGeneratePlan}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <Zap size={20} />
          Generate Flight Plan
        </ActionButton>
        
        <ActionButton
          onClick={handleViewExplainability}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <Shield size={20} />
          View Explainability
        </ActionButton>
        
        <ActionButton
          onClick={() => window.print()}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <Download size={20} />
          Download Report
        </ActionButton>
      </ActionButtons>
    </ResultsContainer>
  );
}

export default Results;
