=============================================================================================
Methods for Data-centric Small Satellite Anomaly Detection and Fault Prediction
=============================================================================================

DESCRIPTION:
	This code can be used to calculate predictability metrics indicating how well one set of timestamp events predicts another (most general)
	The specific use case this code was developed for is small satellite fault forecasting using outliers detected earlier in the telemetry. 
	
CONTRIBUTORS: 
	Joseph Melville (1), Michael Lopez (2), Michael Crabtree (2), Joel Harley (1), Seth Lacy (2)

AFFILIATIONS:
	1. University of Florida, SmartDATA Lab, Department of Electrical and Computer Engineering
	2. Air Force Research Lab, Small Sattelite Portfolio, Space Vehicles Directorate

FUNDING SPONSORS:
	U.S. Department of Defence through a Science, Mathematics, and Research for Transformation (SMART) scholarship

FOLDERS:
	None

SCRIPTS:
	functions - includes all the functions neccesary to calculate predictability metrics
	examples - gives examples of how the predictability metrics are calculated

REQUIRMENTS:
	numpy
	sklearn.ensemble.IsolationForest
	sklearn.neighbors.LocalOutlierFactor
	sklearn.decomposition.PCA
	datetime
	time
	matplotlib.pyplot
