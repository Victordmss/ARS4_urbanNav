# ARS4 - Urban Navigation
## Localization using poles and signs detected by a lidar

This project has been done in relation with ARS4 course from UTC master (ISC - ARS master's degree).

## Authors
- Victor Demessance
- Fabiola Cornejo Urquieta
- Fadi Ben Ali
  
## Project

Accurate positioning in urban environments remains a complex problem, despite the everincreasing performance of GNSS systems. A 360° vehicle-mounted lidar can be used to detect landmarks accurately geo-referenced on a map. The aim of this project is to study a multi-sensor system that fuses lidar detections on geolocalized landmarks (signs and poles) with a GNSS receiver and kinematic sensors.

## Dataset 

The data correspond to a specific section of a real dataset acquired in Compiègne in 2022. The dataset is divided into two parts: simulated data and real data.

- **Simulated Data:** lidar observations are generated based on a high-definition vector map of Compiègne from 2021, which contains the 2D positions of all poles in the environment. Gaussian noise is added to simulate realistic detections. GNSS observations are derived from reference poses by adding noise as well.
- **Real Data:** This part contains actual lidar detections obtained using two different methods: one based on intensity thresholding and another using machine learning approaches. 

_All data are synchronized to lidar timestamps (approximately 10 Hz), meaning that for any given timestamp, kinematic data, GNSS data (when available), and lidar data can be integrated into the filter. However, some data may be missing for specific timestamps._

## Work 

We propose a modelisation of the system that allows all available sensors to be fused together. Next, we develop an EKF and UKF on both datasets. To apply the filter to real data, we implement an NN data association algorithm to associate lidar measurements with map landmarks. 
All informations about process and results (like figure below) are available in `ARS4_ProjectReport.pdf`

![UKF_results](https://github.com/user-attachments/assets/dca8d9b3-1fdb-4916-98d9-4869d32ab7d4)

## Files

`simulation/` contains all the files about the simulation navigation 
  - `simulation/ekf_with_metrics.m` propose the EKF implementation with some final metrics
  - `simulation/ukj_with_metrics.m` propose the UKF implementation with some final metrics
`real/` contains all the files about real navigation implementation
  - `simulation/ekf_data_association.m` propose the EKF implementation with some final metrics and the NN data association algorithm
