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
Überprüfe ob das environment erstellt wurde....dann
```
conda env remove --name neuer_test
```

**Environment mit .yml-Datei aktualisieren:**

   pytorch_env.yml - file herunterladen
  
```
  conda env update --prefix ./env --file pytorch_env.yml  --prune 
```
