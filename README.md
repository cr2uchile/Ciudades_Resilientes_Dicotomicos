# Ciudades_Resilientes_Dicotomicos
Procesamiento de datos históricos de aerosoles

DATOS-DICOTOMOS-2-5-10_-1988-2016.xlsx
Archivo Excel con valores diarios de Material Particulado MP10 y MP2.5 registrado en 3 estaciones de Santiago (Independencia, Parque Ohiggins y Las Condes). 
Abarca el período de tiempo entre los años 1988 y 2016.

Dicotomicos-2017-2020.xlsx
Archivo Excel con valores diarios de Material Particulado MP10 y MP2.5 registrado en 3 estaciones de Santiago (Independencia, Parque Ohiggins y Las Condes). 
Abarca el período de tiempo entre los años 2017 y 2020.

datasinca.py
Función datasinca cuya finalidad es la descarga directa de la información requerida de la plataforma del Sistema de Información Nacional de Calidad de Aire. 
Es necesario especificar el período de tiempo (aaaa-mm-dd), la sustancia y la región requerida en formato STR.
La función entrega el o los archivos requeridos según la sustancia indicada en un formato de Valores Separados por Coma (.csv) de Excel. 

info_ReadDichotomous.py
Funciones de acceso a archivos .xlsx según el año al que se quiere acceder, el cual se indica en formato STR.
- Función leer_dico0('año') para lectura de años 1988 y 1989.
- Función leer_dico1('año') para lectura de años entre 1990 y 2016.
- Función leer_dico2('año') para lectura de años entre 2017 y 2020.
Posee una función que entrega las series de tiempo entre 1988 y 2020 para el Material Particulado 10 y 2.5 en las estaciones de medición Independencia, Parque Ohiggins y Las Condes.
