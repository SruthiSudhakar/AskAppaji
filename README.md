# Installation

```bash
git clone https://github.com/SruthiSudhakar/AskAppaji.git
conda create --name askappajienv python=3.10
conda activate askappajienv
pip install -r requirements.txt
```

Download the [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) model and change the `model_dir` variable in `askappaji.py` to point to your saved model location.

# Run

```bash
python askappaji.py
```