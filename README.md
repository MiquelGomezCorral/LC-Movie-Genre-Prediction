# LC-Movie-Genre-Prediction
Trabajo de LC


# User project
- Create local env
```bash
 python3.12 -m venv venv
 source venv/bin/activate
 
 # install module
 pip install -e app/
 
 # install requirements
 pip install uv
 uv pip install -r requirements.txt

 # for notebooks
 pip install ipykernel
 python -m ipykernel install --user --name=venv --display-name "Python (venv)"

```

# Train and test models

## Classic models

```python
#Training Logistic Regression
python main.py train

#Testing Logistic Regression
python main.py test
```

## Transformers models

```python
#Training Roberta
python main.py transformer

#Testing roberta
python main.py test_t
```

## Gemini

```python
#Training/Validating
python main.py gemini -v

#Testing
python main.py gemini
```