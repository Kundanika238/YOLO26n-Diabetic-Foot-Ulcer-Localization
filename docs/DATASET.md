
# Dataset Preparation and Quality Screening

## Overview

The final diabetic foot ulcer (DFU) localization research dataset contains 7,077 clinical foot images with 7,755 annotated ulcer bounding boxes. The dataset was assembled from eight existing sources and standardized for single-class YOLO26n object detection.

This repository does not redistribute the original clinical images or annotations. Access to the underlying images depends on the respective dataset providers and their usage conditions.

## Dataset Sources

The research dataset was assembled using the following eight sources:

1. DFUC2022
2. FUSeg2021
3. DFU Annotation 200
4. Foot Ulcer Detection 741
5. Roboflow DFU
6. SLIIT Kajanan
7. Deteksi Luka Diabet
8. Ulcer Detection

These source names correspond to the datasets documented in the research manuscript. Original dataset publications and provider-specific access conditions should be consulted before reconstruction or redistribution.

## Annotation Format

All retained annotations were standardized to YOLO bounding-box format.

Each annotation record contains:

class_id x_center y_center width height

The bounding-box coordinates are normalized relative to the image dimensions.

The final dataset uses one object class:

- Class ID: 0
- Class name: DFU

The task is ulcer-region localization using bounding boxes, not wound segmentation or clinical diagnosis.

## Dataset Preparation

The preparation workflow consisted of:

1. Collecting the source datasets.
2. Inspecting image and annotation structures.
3. Standardizing the relevant ulcer annotations to one YOLO class.
4. Preparing research training, validation, and test subsets.
5. Checking annotation-file availability and bounding-box validity.
6. Screening for exact duplicates and cross-split image similarity.
7. Retaining the screened research dataset for model training and evaluation.

The initial research draft contained 7,510 images.

After the final screening and exclusion of 433 images, the retained dataset contained 7,077 images.

## Final Dataset Distribution

| Subset | Images | Annotated DFU Boxes |
|---|---:|---:|
| Training | 5,785 | 6,404 |
| Validation | 655 | 681 |
| Testing | 637 | 670 |
| **Total** | **7,077** | **7,755** |

The training subset was used for YOLO26n fine-tuning.

The validation subset was used for monitoring model performance, checkpoint selection, and the separate selective-refinement experiment.

The test subset was reserved for the final single-pass detector evaluation and region-size analysis.

## Quality and Duplicate Screening

The research dataset underwent structural and similarity checks.

### Annotation Integrity

The final structural audit did not identify missing annotation files or invalid bounding-box coordinates in the screened dataset.

These checks verify annotation structure, not the medical correctness of the labeled ulcer regions.

### Exact-Duplicate Screening

SHA-256 file hashes were used to investigate byte-identical images across dataset splits.

No exact cross-split duplicates were found in the final screened dataset.

### Similarity Screening

The initial research review identified 254 cross-split image-similarity candidate pairs.

Following screening, 433 images were excluded.

The retained dataset had no cross-split perceptual-hash candidates at a Hamming-distance threshold of six.

This does not establish that all potentially similar images were eliminated. Other similarity methods or manual inspection could identify further relationships.

## Final Dataset Structure

The screened local dataset follows this directory layout:

DFU_Research_SCREENED_DRAFT/
    data.yaml
    train/
        images/
        labels/
    val/
        images/
        labels/
    test/
        images/
        labels/

The GitHub repository includes an example dataset configuration at:

configs/data.example.yaml

The example path must be replaced with the actual local dataset location before running dataset-dependent scripts.

## Dataset Limitations

Patient-independent separation between training, validation, and test subsets was not verified.

The available audits do not rule out every possible near-duplicate image.

The clinical correctness of the source annotations was not independently assessed by medical experts.

Differences in image acquisition, lighting, backgrounds, and source annotation practices may influence the reported performance.

Consequently, the reported results should not be interpreted as evidence of clinical diagnostic reliability or independent patient-level generalization.

## Dataset Reproducibility

The repository provides research scripts for screening and checking dataset integrity.

Some scripts depend on the original research draft, local audit records, and dataset directory structure. These inputs are not fully distributed in the repository.

Therefore, the final 7,077-image dataset cannot currently be reconstructed from the GitHub files alone.

Users wishing to reconstruct the dataset must obtain the source data under the applicable permissions and recreate the preparation and screening workflow.

## Research Use

The trained YOLO26n model and dataset analyses are intended for research and educational purposes.

This project does not provide a clinically validated medical diagnostic system.
