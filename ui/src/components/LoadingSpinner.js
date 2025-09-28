import React from 'react';
import styled, { keyframes } from 'styled-components';

const spin = keyframes`
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
`;

const SpinnerContainer = styled.div`
  display: inline-flex;
  align-items: center;
  justify-content: center;
`;

const Spinner = styled.div`
  width: ${props => props.size || 24}px;
  height: ${props => props.size || 24}px;
  border: 2px solid ${props => props.theme.colors.border};
  border-top: 2px solid ${props => props.theme.colors.primary};
  border-radius: 50%;
  animation: ${spin} 1s linear infinite;
`;

const LoadingSpinner = ({ size = 24, ...props }) => {
  return (
    <SpinnerContainer {...props}>
      <Spinner size={size} />
    </SpinnerContainer>
  );
};

export default LoadingSpinner;
