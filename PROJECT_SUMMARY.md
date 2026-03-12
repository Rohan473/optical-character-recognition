# Project Summary

## Introduction

1. This project is a handwritten and printed document digitization system that converts note images into editable digital text.
2. It uses a React frontend, a FastAPI backend, and MongoDB for storing notes, folders, and OCR metadata.
3. The system combines image preprocessing, OCR, note management, and PDF export in one workflow-oriented application.

## Motivation

1. Manual transcription of handwritten notes is time-consuming, error-prone, and difficult to organize at scale.
2. Students, researchers, and professionals need a faster way to convert physical notes into searchable digital records.
3. A unified platform for OCR, editing, storage, and export reduces friction compared with using separate tools for each step.

## Existing Work

1. Traditional OCR engines such as Tesseract are widely used for printed text extraction and basic scanned document processing.
2. Recent transformer-based models such as TrOCR improve recognition for handwritten text and complex document images.
3. Commercial cloud OCR services provide high accuracy, but they often introduce cost, privacy concerns, and external dependency risks.

## Gaps In Existing Works

1. Many OCR systems perform well on printed text but struggle with noisy handwritten notes, mixed scripts, and uneven page layouts.
2. Several existing solutions focus only on extraction and do not integrate editing, organization, search, and export in one application.
3. Cloud-based solutions may offer strong accuracy, but they are less suitable where local processing, data privacy, or low-cost deployment is important.

## Work Done Till Date

1. A FastAPI backend has been built with OCR endpoints, note CRUD operations, folder management, search, and searchable PDF generation.
2. A React frontend has been developed with upload, preview, editor, library, and note-detail pages for end-to-end user interaction.
3. The OCR pipeline has been improved with TrOCR integration, Tesseract fallback, image preprocessing, engine switching, and image preview support through backend-served files.

## Results Achieved

1. The application can successfully upload note images, preprocess them, extract text, and display the transcription inside the editor workflow.
2. Notes can be stored with metadata such as OCR engine, confidence, timestamps, and image paths, and can be organized in folders.
3. Searchable PDF generation and local OCR fallback behavior are functioning, although handwritten OCR accuracy still has room for further improvement.