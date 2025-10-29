Evaluación Llama 3.1 caso MEDDOCAN
    cd code
    python3 evaluate.py brat ner "../../../../Datasets/MEDDOCAN/dev/brat/" "../../../../Modelos/Llama3.1/a) MEDDOCAN/dev_prompt_OneShot/ann/"
    python3 evaluate.py brat ner "../../../../Datasets/MEDDOCAN/test/brat/" "../../../../Modelos/Llama3.1/a) MEDDOCAN/test/ann/"
