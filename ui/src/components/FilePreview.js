import React from 'react';
import styled from 'styled-components';
import { X, FileImage, File } from 'lucide-react';

const PreviewContainer = styled.div`
  margin-top: ${props => props.theme.spacing.md};
  padding: ${props => props.theme.spacing.md};
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.borderRadius.md};
  background: ${props => props.theme.colors.surface};
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.md};
`;

const FileIcon = styled.div`
  color: ${props => props.theme.colors.primary};
  flex-shrink: 0;
`;

const FileInfo = styled.div`
  flex: 1;
  min-width: 0;
`;

const FileName = styled.p`
  font-weight: 500;
  color: ${props => props.theme.colors.text};
  margin: 0 0 ${props => props.theme.spacing.xs} 0;
  word-break: break-all;
`;

const FileSize = styled.p`
  font-size: 0.875rem;
  color: ${props => props.theme.colors.textSecondary};
  margin: 0;
`;

const RemoveButton = styled.button`
  background: none;
  border: none;
  color: ${props => props.theme.colors.textSecondary};
  cursor: pointer;
  padding: ${props => props.theme.spacing.xs};
  border-radius: ${props => props.theme.borderRadius.sm};
  transition: all 0.2s ease;
  flex-shrink: 0;

  &:hover {
    background: ${props => props.theme.colors.error + '10'};
    color: ${props => props.theme.colors.error};
  }
`;

const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

const FilePreview = ({ file, onRemove }) => {
  const isImage = file.type.startsWith('image/');

  return (
    <PreviewContainer>
      <FileIcon>
        {isImage ? <FileImage size={24} /> : <File size={24} />}
      </FileIcon>
      
      <FileInfo>
        <FileName>{file.name}</FileName>
        <FileSize>{formatFileSize(file.size)}</FileSize>
      </FileInfo>
      
      <RemoveButton onClick={onRemove} title="Remove file">
        <X size={20} />
      </RemoveButton>
    </PreviewContainer>
  );
};

export default FilePreview;
