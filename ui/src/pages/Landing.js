import React, { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { 
  Upload, 
  Camera, 
  FileText, 
  MapPin, 
  Zap,
  Shield,
  Target,
  ArrowRight
} from 'lucide-react';
import toast from 'react-hot-toast';

import { predict } from '../services/api';
import LoadingSpinner from '../components/LoadingSpinner';
import FilePreview from '../components/FilePreview';

const LandingContainer = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: ${props => props.theme.spacing.xl};
`;

const HeroSection = styled.section`
  text-align: center;
  margin-bottom: ${props => props.theme.spacing.xxl};
`;

const HeroTitle = styled.h1`
  font-size: 3.5rem;
  font-weight: 700;
  color: ${props => props.theme.colors.primary};
  margin-bottom: ${props => props.theme.spacing.lg};
  line-height: 1.2;

  @media (max-width: ${props => props.theme.breakpoints.mobile}) {
    font-size: 2.5rem;
  }
`;

const HeroSubtitle = styled.p`
  font-size: 1.25rem;
  color: ${props => props.theme.colors.textSecondary};
  margin-bottom: ${props => props.theme.spacing.xl};
  max-width: 600px;
  margin-left: auto;
  margin-right: auto;
`;

const FeatureGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: ${props => props.theme.spacing.lg};
  margin-bottom: ${props => props.theme.spacing.xxl};
`;

const FeatureCard = styled(motion.div)`
  background: ${props => props.theme.colors.surface};
  padding: ${props => props.theme.spacing.xl};
  border-radius: ${props => props.theme.borderRadius.lg};
  box-shadow: ${props => props.theme.shadows.sm};
  text-align: center;
  transition: transform 0.2s ease, box-shadow 0.2s ease;

  &:hover {
    transform: translateY(-4px);
    box-shadow: ${props => props.theme.shadows.md};
  }
`;

const FeatureIcon = styled.div`
  width: 64px;
  height: 64px;
  background: ${props => props.theme.colors.primaryLight};
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 auto ${props => props.theme.spacing.md};
  color: ${props => props.theme.colors.surface};
`;

const FeatureTitle = styled.h3`
  font-size: 1.25rem;
  font-weight: 600;
  margin-bottom: ${props => props.theme.spacing.sm};
  color: ${props => props.theme.colors.text};
`;

const FeatureDescription = styled.p`
  color: ${props => props.theme.colors.textSecondary};
  line-height: 1.6;
`;

const UploadSection = styled.section`
  background: ${props => props.theme.colors.surface};
  border-radius: ${props => props.theme.borderRadius.lg};
  padding: ${props => props.theme.spacing.xxl};
  box-shadow: ${props => props.theme.shadows.sm};
  margin-bottom: ${props => props.theme.spacing.xl};
`;

const UploadTitle = styled.h2`
  font-size: 2rem;
  font-weight: 600;
  text-align: center;
  margin-bottom: ${props => props.theme.spacing.xl};
  color: ${props => props.theme.colors.text};
`;

const UploadGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: ${props => props.theme.spacing.xl};

  @media (max-width: ${props => props.theme.breakpoints.mobile}) {
    grid-template-columns: 1fr;
  }
`;

const UploadArea = styled.div`
  border: 2px dashed ${props => props.isDragActive ? props.theme.colors.primary : props.theme.colors.border};
  border-radius: ${props => props.theme.borderRadius.lg};
  padding: ${props => props.theme.spacing.xl};
  text-align: center;
  cursor: pointer;
  transition: all 0.2s ease;
  background: ${props => props.isDragActive ? props.theme.colors.primaryLight + '10' : 'transparent'};

  &:hover {
    border-color: ${props => props.theme.colors.primary};
    background: ${props => props.theme.colors.primaryLight + '05'};
  }
`;

const UploadIcon = styled.div`
  width: 48px;
  height: 48px;
  margin: 0 auto ${props => props.theme.spacing.md};
  color: ${props => props.theme.colors.primary};
`;

const UploadText = styled.p`
  font-size: 1.1rem;
  color: ${props => props.theme.colors.text};
  margin-bottom: ${props => props.theme.spacing.sm};
`;

const UploadSubtext = styled.p`
  font-size: 0.9rem;
  color: ${props => props.theme.colors.textSecondary};
`;

const TextInputArea = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${props => props.theme.spacing.md};
`;

const TextArea = styled.textarea`
  width: 100%;
  min-height: 120px;
  padding: ${props => props.theme.spacing.md};
  border: 2px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.borderRadius.md};
  font-size: 1rem;
  resize: vertical;
  transition: border-color 0.2s ease;

  &:focus {
    border-color: ${props => props.theme.colors.primary};
  }

  &::placeholder {
    color: ${props => props.theme.colors.textSecondary};
  }
`;

const LocationInput = styled.input`
  width: 100%;
  padding: ${props => props.theme.spacing.md};
  border: 2px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.borderRadius.md};
  font-size: 1rem;
  transition: border-color 0.2s ease;

  &:focus {
    border-color: ${props => props.theme.colors.primary};
  }

  &::placeholder {
    color: ${props => props.theme.colors.textSecondary};
  }
`;

const ProcessButton = styled(motion.button)`
  background: ${props => props.theme.colors.primary};
  color: ${props => props.theme.colors.surface};
  padding: ${props => props.theme.spacing.md} ${props => props.theme.spacing.xl};
  border-radius: ${props => props.theme.borderRadius.md};
  font-size: 1.1rem;
  font-weight: 600;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: ${props => props.theme.spacing.sm};
  margin: ${props => props.theme.spacing.xl} auto 0;
  min-width: 200px;
  transition: all 0.2s ease;

  &:hover:not(:disabled) {
    background: ${props => props.theme.colors.primaryDark};
    transform: translateY(-2px);
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

const SafetyNotice = styled.div`
  background: ${props => props.theme.colors.info + '10'};
  border: 1px solid ${props => props.theme.colors.info + '30'};
  border-radius: ${props => props.theme.borderRadius.md};
  padding: ${props => props.theme.spacing.lg};
  margin-top: ${props => props.theme.spacing.xl};
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.md};
`;

const SafetyIcon = styled(Shield)`
  color: ${props => props.theme.colors.info};
  flex-shrink: 0;
`;

const SafetyText = styled.p`
  color: ${props => props.theme.colors.text};
  font-size: 0.9rem;
  line-height: 1.5;
`;

function Landing() {
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState(null);
  const [transcript, setTranscript] = useState('');
  const [location, setLocation] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setSelectedFile(file);
      toast.success('Image uploaded successfully');
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
    },
    maxFiles: 1,
    maxSize: 50 * 1024 * 1024 // 50MB
  });

  const handleProcess = async () => {
    if (!selectedFile) {
      toast.error('Please upload an image first');
      return;
    }

    setIsProcessing(true);
    
    try {
      // Upload image and get prediction
      const response = await predict(selectedFile, transcript);
      
      if (response.data) {
        toast.success('Analysis completed successfully');
        navigate(`/results/${response.data.run_id}`);
      }
    } catch (error) {
      console.error('Processing error:', error);
      toast.error('Failed to process image. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleFileRemove = () => {
    setSelectedFile(null);
    toast.success('Image removed');
  };

  return (
    <LandingContainer>
      <HeroSection>
        <HeroTitle>AgriSprayAI</HeroTitle>
        <HeroSubtitle>
          AI-powered pesticide spraying optimization for precision agriculture.
          Upload an image, add your observations, and get optimized spray plans.
        </HeroSubtitle>
      </HeroSection>

      <FeatureGrid>
        <FeatureCard
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <FeatureIcon>
            <Target size={32} />
          </FeatureIcon>
          <FeatureTitle>Precision Detection</FeatureTitle>
          <FeatureDescription>
            Advanced computer vision detects pests and diseases with high accuracy,
            providing per-plant severity assessments.
          </FeatureDescription>
        </FeatureCard>

        <FeatureCard
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <FeatureIcon>
            <Zap size={32} />
          </FeatureIcon>
          <FeatureTitle>Optimal Dosing</FeatureTitle>
          <FeatureDescription>
            Convex optimization algorithms calculate minimal guaranteed doses
            while ensuring effective pest control.
          </FeatureDescription>
        </FeatureCard>

        <FeatureCard
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <FeatureIcon>
            <Shield size={32} />
          </FeatureIcon>
          <FeatureTitle>Safety First</FeatureTitle>
          <FeatureDescription>
            Human-in-the-loop approval required for low confidence predictions
            and regulatory compliance monitoring.
          </FeatureDescription>
        </FeatureCard>
      </FeatureGrid>

      <UploadSection>
        <UploadTitle>Upload Your Field Image</UploadTitle>
        
        <UploadGrid>
          <div>
            <UploadArea {...getRootProps()} isDragActive={isDragActive}>
              <input {...getInputProps()} />
              <UploadIcon>
                {isDragActive ? <Upload size={48} /> : <Camera size={48} />}
              </UploadIcon>
              <UploadText>
                {isDragActive ? 'Drop the image here' : 'Drag & drop an image or click to browse'}
              </UploadText>
              <UploadSubtext>
                Supports JPG, PNG, TIFF, BMP (max 50MB)
              </UploadSubtext>
            </UploadArea>

            {selectedFile && (
              <FilePreview
                file={selectedFile}
                onRemove={handleFileRemove}
              />
            )}
          </div>

          <TextInputArea>
            <div>
              <UploadIcon>
                <FileText size={48} />
              </UploadIcon>
              <UploadText>Add Your Observations</UploadText>
              <UploadSubtext>
                Describe what you see in the field (optional)
              </UploadSubtext>
            </div>
            
            <TextArea
              value={transcript}
              onChange={(e) => setTranscript(e.target.value)}
              placeholder="e.g., 'I noticed some yellowing leaves on the eastern side of the field. The plants seem to be wilting in patches.'"
            />

            <div>
              <UploadIcon>
                <MapPin size={48} />
              </UploadIcon>
              <UploadText>Field Location</UploadText>
              <UploadSubtext>
                GPS coordinates or field name (optional)
              </UploadSubtext>
            </div>
            
            <LocationInput
              type="text"
              value={location}
              onChange={(e) => setLocation(e.target.value)}
              placeholder="e.g., 'Field A, North 40.7128, West 74.0060'"
            />
          </TextInputArea>
        </UploadGrid>

        <ProcessButton
          onClick={handleProcess}
          disabled={!selectedFile || isProcessing}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          {isProcessing ? (
            <>
              <LoadingSpinner size={20} />
              Processing...
            </>
          ) : (
            <>
              Analyze Field
              <ArrowRight size={20} />
            </>
          )}
        </ProcessButton>

        <SafetyNotice>
          <SafetyIcon size={24} />
          <SafetyText>
            <strong>Safety Notice:</strong> This system requires human approval for all pesticide applications.
            All decisions are logged for audit purposes. Please consult with agricultural experts
            before field deployment.
          </SafetyText>
        </SafetyNotice>
      </UploadSection>
    </LandingContainer>
  );
}

export default Landing;
