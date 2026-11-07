# Late Deliveries

This project predicts whether a QuickBite food-delivery order will arrive late. It compares several classification models, selects a probability threshold using the company’s voucher and missed-delivery costs, and saves the final preprocessing-and-model pipeline.

## Setup

```bash
conda env create -f environment.yml
conda activate lab05-late-deliveries
pip install -r requirements.txt
pip install -e .