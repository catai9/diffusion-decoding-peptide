# STAT 946 Project Code
This code is an adapted version of casanovo (https://github.com/Noble-Lab/casanovo) with some code copied from the DINOISER repo (https://github.com/yegcjs/DINOISER/tree/main).

## Setup
Please follow the setup locations in the README of casanovo or on their website (https://casanovo.readthedocs.io/en/latest/).

To compile locally, please also install `flash-attn` if your environment does not have it. 
Example: `pip install flash-attn --no-build-isolation`

The data can be downloaded from https://zenodo.org/records/12587317 
For our purposes, we train using the training and validation files stored in the `mskb_proportional` folder and test using the test file in the `mskb_final` folder.

To run this code, we used two NVIDIA GeForce RTX 4090 GPUs with 25 GB of memory each. 

## Changes
We made changes to the casanovo code, specifically in `casanovo/casanovo/denovo/model.py`, `casanovo/casanovo/denovo/model_runner.py`, and `casanovo/casanovo/config.py`.

We also added new files as detaield below. 
- For the diffusion decoders:
    - `casanovo/casanovo/denovo/decoderDS.py`
    - `casanovo/casanovo/denovo/decoderDM1.py`
    - `casanovo/casanovo/denovo/decoderDM2.py`
- For producing final metrics on the test dataset 
    - `casanovo/casanovo/compute_predict_metrics.py`
- For new config setups: 
    - `casanovo/casanovo/configs`

GitHub Copilot (https://github.com/features/copilot) was used to help with diffusion decoder code design, write code comments, and assist in debugging the code.

## Scripts
The following details the commands used for running training and for experimentation for each of the experiments. Replace the config location file as appropriate.

```
CONFIG_FILE="casanovo/casanovo/configs/config.yaml"
MODEL_SAVE_LOCATION="casanovo"
OUTPUT="config_output"
```

### Training
`python3 -m casanovo.casanovo train data/mgf_data/mskb_proportional/mskb_proportional.train.mgf --validation_peak_path data/mgf_data/mskb_proportional/mskb_proportional.val.mgf --config $CONFIG_FILE --output $OUTPUT | tee $OUTPUT_train.txt`


### Inference on Test Set
`python3 -m casanovo.casanovo sequence data/mgf_data/mskb_final/mskb_final.test.mgf --model casanovo/casanovo/$MODEL_SAVE_LOCATION/best.ckpt --output $OUTPUT_sequence --config $CONFIG_FILE | tee $OUTPUT_sequence.txt`

`python3 -m casanovo.compute_predict_metrics --actual_file data/mgf_data/mskb_final/mskb_final.test.mgf --predicted_file $OUTPUT_sequence  --output_file $OUTPUT_results.csv --plot_file $OUTPUT_prec_cov.png | tee $OUTPUT_results.txt`

If knapsack is used, the `--knapsack_output` argument should be passed in the above call.


## Model Architecture Graphics Creation
To create graphics for the decoder model architecture, graphviz was used. The corresponding files can be found in the `graphics` folder and can be run with `python file_name.py`. Notably, you will need to first run `pip install graphviz`. 

