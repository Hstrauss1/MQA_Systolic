accelergy accelergy_input/*.yaml accelergy_input/components/*.yaml -o accelergy_output/mqa_decode_64x64_os -v 1

mv ./accelergy_output/mqa_decode_64x64_os  /Users/hudsons/Code/ScaleSim/MQA_SCALE_SIM/rundir-accelergy/output/accelergy_output_mqa_decode_64x64_os

rm -rf ./accelergy_output

