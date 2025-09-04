# Machine-Learning-for-Survival-Prediction-in-Patients-with-Hepatocellular-Carcinoma-

I. 
Data Set


O data set utilizado será o Hepatocellular Carcinoma data set (HCC data set) [1], disponível no 
Kaggle. Este data set contém dados clínicos reais de 165 doentes diagnosticados com CHC, e foi recolhido 
no Centro Hospitalar e Universitário de Coimbra (CHUC), pelo Serviço de Medicina Interna A. 
Esta base de dados contém 49 características – demográficas, de fatores de risco, laboratoriais e de 
sobrevivência – selecionadas de acordo com as Diretrizes de Prática Clínica da EASL – EORTC 
(Associação Europeia para o Estudo do Fígado – Organização Europeia para a Investigação e o Tratamento 
do Cancro), que constituem o estado da arte atual sobre a gestão do CHC [2]. Estas características clínicas – 23 quantitativas e 26 qualitativas –, são consideradas as mais significativas para o processo de decisão 
dos clínicos na escolha das estratégias terapêuticas mais adequadas e na previsão dos seus resultados para 
cada doente. É de notar que nem todos os pacientes possuem informação para a totalidade dos parâmetros, 
sendo que, no total, os dados em falta representam 10,22% de todo o data set, pelo que apenas oito doentes 
têm informação completa em todos os campos (4,85%) [3], [4]. 
Além disso, o data set inclui também um parâmetro de sobrevivência, ao longo de um ano, codificado 
como uma variável binária com valores 0 (doente não sobreviveu) e 1 (doente sobreviveu). 


II. 
Ideia do projeto 


O principal objetivo deste projeto é desenvolver um modelo que preveja a sobrevivência de pacientes 
com carcinoma hepatocelular, utilizando dados de doentes previamente diagnosticados, através da 
implementação de técnicas de Machine Learning. Os dados fornecidos pelo data set envolvem diversos 
indicadores que serão analisados de modo a conseguir prever de forma concreta e concisa a informação a 
avaliar no projeto.  
A escolha prévia deste projeto baseou-se no facto de os dados fornecidos serem relativamente recentes 
e referentes a indivíduos portugueses. Assim, findado o desenvolvimento do modelo, torna-se possível 
obter informação bastante útil, pertinente e atual. 


III. 
Software 


Este projeto será desenvolvido recorrendo maioritariamente a Python, onde se irá procurar 
implementar a técnica de Machine Learning que melhor prevê a sobrevivência, ao longo de um ano, do 
doente com CHC. O modelo implementado criará um classificador binário, ou seja, em que existem apenas 
2 resultados possíveis: 0 (paciente morre) e 1 (paciente vive). 


IV. 
Artigos Relevantes 


a) «A new cluster-based oversampling method for improving survival prediction of hepatocellular      
carcinoma patients», J. Biomed. Inform., vol. 58, pp. 49–59, dez. 2015, doi: 
10.1016/j.jbi.2015.09.012. 
b) H.-Y. Shi et al., «Comparison of Artificial Neural Network and Logistic Regression Models for 
Predicting In-Hospital Mortality after Primary Liver Cancer Surgery», PLOS ONE, vol. 7, n.o 4, p. 
e35781, abr. 2012, doi: 10.1371/journal.pone.0035781. 
c) A. Forner, J. M. Llovet, e J. Bruix, «Hepatocellular carcinoma», The Lancet, vol. 379, n.o 9822, pp. 
1245–1255, mar. 2012, doi: 10.1016/S0140-6736(11)61347-0. 


V. 
Referências 


[1] «HCC survival data set». Acedido: 16 de novembro de 2023. Disponível em: 
https://www.kaggle.com/datasets/mirlei/hcc-survival-data-set 
Mestrado em Bioengenharia – Engenharia Biomédica 
DACO – Diagnóstico Assistido por Computador, 2023/2024 
[2] Anon., European association for the study of the liver, European organisation for research and treatment 
of cancer, EASL-EORTC clinical practice guidelines: management of hepatocellular carcinoma, J. 
Hepatol. 56 (4) 2012, 908–943. 
[3] P. A. Miriam Santos, «HCC Survival». UCI Machine Learning Repository, 2015. doi: 
10.24432/C5TS4S. 
[4] «A new cluster-based oversampling method for improving survival prediction of hepatocellular 
carcinoma patients», J. Biomed. Inform., vol. 58, pp. 49–59, dez. 2015, doi: 10.1016/j.jbi.2015.09.012.
