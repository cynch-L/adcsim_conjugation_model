# Kaggle polish + scorecard (full PyRosetta)

Do not use Colab. Do not generate params on Kaggle.

## Upload these 9 files as one Kaggle Dataset

From this machine:

```
adcsim-project/examples/trop_adc/results_hinge/ADC_DAR1.pdb
adcsim-project/examples/trop_adc/results_hinge/ADC_DAR2.pdb
adcsim-project/examples/trop_adc/results_hinge/ADC_DAR3.pdb
adcsim-project/examples/trop_adc/results_hinge/ADC_DAR4.pdb
adcsim-project/examples/trop_adc/results_hinge/ADC_DAR5.pdb
adcsim-project/examples/trop_adc/results_hinge/ADC_DAR6.pdb
adcsim-project/examples/trop_adc/results_hinge/ADC_DAR7_omit_B229.pdb
adcsim-project/examples/trop_adc/results_hinge/ADC_DAR7_omit_B232.pdb
adcsim-project/examples/trop_adc/runpod/LPP.params
```

Do **not** upload `polished/*_min.pdb`. Do **not** upload the SDF.

## Notebook

File: `adcsim-project/examples/trop_adc/runpod/adcsim_kaggle_polish.ipynb`

1. Kaggle -> Code -> New Notebook.
2. Settings: Internet **ON**. Accelerator **None** (CPU). Persistence **Files**.
3. Add data: attach the dataset with the 9 files.
4. File -> Import notebook -> this ipynb, **or** paste cells.
5. Run all, top to bottom.
6. Download `/kaggle/working/adcsim_kaggle_out.zip`.

Cell 1 installs PyRosetta (~5-10 min, once). Cells 4-5 do the 8 minimizations (~30-40 min). If a cell dies, re-run from cell 4: existing `*_min.pdb` are skipped.

## After download

Send back:

- `score_summary.json`
- the 8 `*_min.pdb`

Those are the polished representatives for the final project summary.
