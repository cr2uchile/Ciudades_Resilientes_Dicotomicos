
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def datasinca(inicio,fin,eligoSus,R):
    """
    

    Parameters
    ----------
    inicio : STR
        Fecha inicial con formato aaaa-mm-dd. Ej '2020-01-01'
    fin : STR
        Fecha final con formato aaaa-mm-dd. Ej '2020-12-30'
    eligoSus : LIST[STR]
        Una lista con los contaminantes a descargar. Ej ['PM10', 'PM25']
        opciones: ['PM25' , 'PM10' ,'NO2' , 'NOx' , 'SO2' , 'O3' ,'CO' , 'NO'  ] 
    R : STR
        Un Sting con la region a descargar ej. 'RM'
        opciones : 'RM' , 'RV' ,'RVIII', 'RIX', 'RX' , 'RXI'

    Returns 
    -------
    Archivos .csv de la region para los contaminantes selecionados.

    """

    inicio= inicio[2:4] + inicio[5:7] + inicio[8:10]
    fin  = fin[2:4] + fin[5:7] + fin[8:10]
    import requests
 
#    os.chdir('C:\\Users\\Menares\\Desktop\\Tesis\\Data')

    sust = ['PM25' , 'PM10' ,'0003' , '0NOX' , '0001' , '0008' ,'0004' ,'0002'] 
    Nsust = ['PM25' , 'PM10' ,'NO2' , 'NOx' , 'SO2' , 'O3' ,'CO' , 'NO'  ] 
    sust_dic = {'PM25':'PM25','PM10':'PM10', 'NO2':'0003','NOx':'0NOX','SO2':'0001','O3':'0008','CO':'0004','NO':'0002'}
    
    reg = ['RM' , 'RV' ,'RVIII', 'RIX', 'RX' , 'RXI']
    EstacionN = { 'RM' :['Independencia' , 'La Florida', 'Las Condes', 'Parque Ohiggins', 'Pudahuel', 'Cerrillos', 'Cerrillos I', 'El Bosque', 'Cerro Navia', 'Puente Alto', 'Talagante', 'Quilicura', 'Quilicura I'] ,  
    'RV':['Centro Quintero', 'Loncura','Maitenes','Quintero'] , 'RVIII':['INIA Chillan', 'Puren'],  'RIX':  ['Padre Las Casas II', 'Padre Las Casas', 'Las Encinas Temuco', 'Ñielol', 'Museo Ferroviario'], 'RX': ['Osorno'], 'RXI':  ['Coyhaique', 'Coyhaique II']  }
    EstacionC = {'RM':['D11', 'D12', 'D13', 'D14', 'D15', 'D31', 'D16', 'D17', 'D18', 'D27', 'D28', 'D30', 'D29'],
   'RV':['539','547','504','540'] , 'RVIII':['810', '803'], 'RIX': ['902', '903', '901', '905', '904'],  'RX': ['A01'], 'RXI': ['B03', 'B04']  }

    Nreg = len(EstacionN[R])
    
    for i in eligoSus:
        for j in range(Nreg):
            
            
            url = 'http://sinca.mma.gob.cl/cgi-bin/APUB-MMA/apub.tsindico2.cgi?outtype=xcl&macro=./' + R + '/' + EstacionC[R][j] + '/Cal/' + sust_dic[i] + '//' + sust_dic[i] + '.horario.horario.ic&from=' + inicio + '&to=' + fin + '&path=/usr/airviro/data/CONAMA/&lang=esp&rsrc=&macropath=';
            fileName = EstacionN[R][j]+ ' '+ i + '.csv'
            req = requests.get(url)
            file = open(fileName, 'wb')
            for chunk in req.iter_content(100000):
                file.write(chunk)
            file.close()

    
    
    return(url)

