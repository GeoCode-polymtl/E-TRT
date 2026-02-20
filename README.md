# Electrical thermal response test (E-TRT)

This repository contains the code to design an electrical thermal response test. This code simulates the electrical response of the pole-pole geoelectrical survey of a thermal response test (E-TRT). The goal is to use the resulting electrical data in conjunction with direct water temperature measurements to recover the thermal conductivity, volumetric capacity, and the linear ratio between electrical conductivity and temperature.

Files ----------------------------------------------------------------------------------------------------

`ETRT_synthetic_case.pynb` : Jupyter notebook showing the electrical model of an E-TRT and how the thermal parameters are estimated. Included a sensitivity analysis.

`ETRT_Varennes_case.pynb`: Jupyter notebook presenting the parameter estimation of the E-TRT to experimental data.

`ETRT_Varennes_data.pynb` : Jupyter notebook processing the experimental data for parameter estimation.

`simpeg_dc_cyl.ipynb` : Jupyter notebook showing the implementation of a DC resistivity forward simulation using SimPEG in cylindrical coordinates.


Folders ---------------------------------------------------------------------------------------------------

01-Synthetic_data : Contains saved files that allow to run `ETRT_synthetic_case.pynb` faster.

02-Varennes_data : Contains the experimental data acquired at the Varennes site, Québec, Canada


