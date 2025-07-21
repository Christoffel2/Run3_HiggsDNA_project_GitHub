#!/bin/bash

# Prompt the user to specify the process name you want to load and the name of the npy file.
echo "Specify the process name you want to load (Options: Diphoton, DYJets, GJets, QCD, Top, VV, WG, ZG, VBF, ggH, ggH_powheg, ttH, WH_signal, ZH_Zto2L_signal, ZH_Zto2Nu_signal, VH_bkg):"
read sample_name
echo "Specify the name of the npy file:"
read output_filename

python /eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/WH_BDT/WH_BDT_training_variables.py --sample_name "$sample_name" --output_filename "$output_filename"
