# InnoLab-AI_Organ_Segmentation


**Environment erstellen (bei Ersterstellung):**
  
   pytorch_env.yml - file herunterladen
  
```
  conda env create -f pytorch_env.yml
  conda activate pytorch_env
```


**Zur Überprüfung ob das environment richtig erstellt wurde:**

```
  conda env list
```


**Environment umbenennen:**
```
conda create --name pytorch_env --clone neuer_test --offline
```
Wenn der Stern bei ```conda env list``` bei _neuer_test_ ist: ```conda deactivate```.<br/>
Nur wenn bei ```conda env list``` das environment _pytorch_env_ existiert:
```
conda env remove --name neuer_test
```


**Environment mit .yml-Datei aktualisieren:**

   pytorch_env.yml - file herunterladen
  
```
  conda env update --prefix ./env --file pytorch_env.yml  --prune 
```
