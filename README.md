# TRACE: Intra-visit Clinical Event Nowcasting via Effective Patient Trajectory Encoding

This is the official implementation of short paper "TRACE: Intra-visit Clinical Event Nowcasting via Effective Patient Trajectory Encoding" (WWW'25)

## Requirement

- Python >= 3.12.4, Pytorch>=4.2.1, scikit-learn>=1.5.1
- install CUDA if you use GPU computation

## Data Preparation
Due to data access restrictions, we cannot directly provide the dataset. Please acquire the data yourself from https://mimic.physionet.org/.

Each patient trajectory is structured as an array of event ids with an additional array of timestamps.
Store the seqeunces of input, ground truths, and the timestamps seperately into three files.

## Sample Usage

`
python main.py --dataset mimic3 --seed 42
`