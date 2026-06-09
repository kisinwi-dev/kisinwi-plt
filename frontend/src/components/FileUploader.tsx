import React, { useRef, useState } from 'react';
import './FileUploader.css';
import { formatBytes } from '../utils/format';
import { ICONS } from '../constants/icons';

interface FileUploaderProps {
  onFileSelect: (file: File | null) => void;
  accept?: string;
  currentFile?: File | null;
}

const FileUploader: React.FC<FileUploaderProps> = ({
  onFileSelect,
  accept = '.zip,.tar,.gz,.rar,.7z',
  currentFile
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      onFileSelect(file);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const file = files[0];
      onFileSelect(file);
    }
  };

  const handleRemove = () => {
    onFileSelect(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const getFileIcon = (fileName: string) => {
    const ext = fileName.split('.').pop()?.toLowerCase();
    if (ext === 'zip' || ext === 'rar' || ext === '7z' || ext === 'tar' || ext === 'gz') {
      return ICONS.fileArchive;
    }
    if (ext === 'jpg' || ext === 'jpeg' || ext === 'png' || ext === 'gif' || ext === 'svg') {
      return ICONS.fileImage;
    }
    if (ext === 'csv' || ext === 'xlsx' || ext === 'xls') {
      return ICONS.fileTable;
    }
    if (ext === 'txt' || ext === 'md') {
      return ICONS.fileText;
    }
    return ICONS.file;
  };

  return (
    <div className="file-uploader">
      {!currentFile ? (
        <div
          className={`drop-zone ${isDragging ? 'dragging' : ''}`}
          onDragEnter={handleDragEnter}
          onDragLeave={handleDragLeave}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileInput}
            accept={accept}
            style={{ display: 'none' }}
          />
          <div className="drop-zone-content">
            <span className="upload-icon"><i className={`fas ${ICONS.upload}`}></i></span>
            <p>Перетащите файл сюда или <span className="browse-link">выберите</span></p>
            <p className="file-hint">Поддерживаются: {accept.split(',').join(', ')}</p>
          </div>
        </div>
      ) : (
        <div className="file-preview">
          <div className="file-info">
            <span className="file-icon"><i className={`fas ${getFileIcon(currentFile.name)}`}></i></span>
            <div className="file-details">
              <p className="file-name">{currentFile.name}</p>
              <p className="file-size">{formatBytes(currentFile.size)}</p>
            </div>
          </div>
          <button className="remove-file" onClick={handleRemove} title="Удалить файл">
            <i className={`fas ${ICONS.close}`}></i>
          </button>
        </div>
      )}
    </div>
  );
};

export default FileUploader;