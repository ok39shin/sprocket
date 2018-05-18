#!/bin/csh

### 
# Generate Mixing Wav 0 20 40 60 80
# 
# input : org, tar
# output: (data/pair/org-tar/test/org/*.wav)
#
###
set Ratio=(0 20 40 60 80)

foreach r ($Ratio)
    python run_sp_r_mcep0.py -5 $1 $2 $r
    python run_sp_r.py -5 $1 $2 $r
end

