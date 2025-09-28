import React from 'react';
import { useParams } from 'react-router-dom';
import styled from 'styled-components';

const ExplainabilityContainer = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: ${props => props.theme.spacing.lg};
`;

const ExplainabilityHeader = styled.div`
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

const ExplainabilityContent = styled.div`
  background: ${props => props.theme.colors.surface};
  border-radius: ${props => props.theme.borderRadius.lg};
  padding: ${props => props.theme.spacing.lg};
  box-shadow: ${props => props.theme.shadows.sm};
`;

const Section = styled.div`
  margin-bottom: ${props => props.theme.spacing.xl};
  
  &:last-child {
    margin-bottom: 0;
  }
`;

const SectionTitle = styled.h2`
  color: ${props => props.theme.colors.text};
  margin-bottom: ${props => props.theme.spacing.md};
  font-size: 1.3rem;
`;

const ExplanationText = styled.p`
  color: ${props => props.theme.colors.text};
  line-height: 1.6;
  margin-bottom: ${props => props.theme.spacing.md};
`;

const FeatureList = styled.ul`
  list-style: none;
  padding: 0;
`;

const FeatureItem = styled.li`
  padding: ${props => props.theme.spacing.sm} 0;
  border-bottom: 1px solid ${props => props.theme.colors.border};
  
  &:last-child {
    border-bottom: none;
  }
`;

const FeatureName = styled.span`
  font-weight: 600;
  color: ${props => props.theme.colors.primary};
  margin-right: ${props => props.theme.spacing.sm};
`;

const FeatureValue = styled.span`
  color: ${props => props.theme.colors.text};
`;

function Explainability() {
  const { runId } = useParams();

  // Mock explanation data - in a real app, this would come from the AI model
  const mockExplanation = {
    modelDecision: "The AI model recommended a targeted spray approach based on pest density analysis and weather conditions.",
    keyFactors: [
      { name: "Pest Density", value: "High concentration in northwest quadrant" },
      { name: "Weather Forecast", value: "Optimal conditions for 48 hours" },
      { name: "Crop Growth Stage", value: "Vulnerable flowering period" },
      { name: "Previous Treatment", value: "No recent pesticide application" },
      { name: "Soil Moisture", value: "Adequate for pesticide absorption" }
    ],
    confidence: "87%",
    alternativeApproaches: [
      "Full field coverage would use 40% more pesticide",
      "Delaying treatment by 2 days could reduce effectiveness by 15%",
      "Organic alternatives would require 3x more frequent applications"
    ]
  };

  return (
    <ExplainabilityContainer>
      <ExplainabilityHeader>
        <Title>AI Decision Explanation</Title>
        <Subtitle>Run ID: {runId}</Subtitle>
      </ExplainabilityHeader>
      
      <ExplainabilityContent>
        <Section>
          <SectionTitle>Model Decision</SectionTitle>
          <ExplanationText>{mockExplanation.modelDecision}</ExplanationText>
          <ExplanationText>
            <strong>Confidence Level:</strong> {mockExplanation.confidence}
          </ExplanationText>
        </Section>

        <Section>
          <SectionTitle>Key Decision Factors</SectionTitle>
          <FeatureList>
            {mockExplanation.keyFactors.map((factor, index) => (
              <FeatureItem key={index}>
                <FeatureName>{factor.name}:</FeatureName>
                <FeatureValue>{factor.value}</FeatureValue>
              </FeatureItem>
            ))}
          </FeatureList>
        </Section>

        <Section>
          <SectionTitle>Alternative Approaches Considered</SectionTitle>
          <FeatureList>
            {mockExplanation.alternativeApproaches.map((alternative, index) => (
              <FeatureItem key={index}>
                <FeatureValue>{alternative}</FeatureValue>
              </FeatureItem>
            ))}
          </FeatureList>
        </Section>
      </ExplainabilityContent>
    </ExplainabilityContainer>
  );
}

export default Explainability;
