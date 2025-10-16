python cs224r/scripts/run_hw1.py \
--expert_policy_file cs224r/policies/experts/Ant.pkl \
--env_name Ant-v4 --exp_name bc_ant --n_iter 1 \
--expert_data cs224r/expert_data/expert_data_Ant-v4.pkl \
--video_log_freq -1
--eval_batch_size 5000 \

PYTHONPATH=submit python submit/cs224r/scripts/run_hw1.py \
--expert_policy_file submit/cs224r/policies/experts/Ant.pkl \
--env_name Ant-v4 --exp_name dagger_ant --n_iter 10 \
--do_dagger \
--expert_data submit/cs224r/expert_data/expert_data_Ant-v4.pkl \
--eval_batch_size 5000 \
--video_log_freq -1

PYTHONPATH=. python cs224r/scripts/run_hw1.py \
--expert_policy_file cs224r/policies/experts/Walker2d.pkl \
--env_name Walker2d-v4 \
--exp_name q1_bc_walker_Walker2d-v4 \
--n_iter 1 \
--expert_data cs224r/expert_data/expert_data_Walker2d-v4.pkl \
--eval_batch_size 5000 \
--video_log_freq -1
 
PYTHONPATH=submit python submit/cs224r/scripts/run_hw1.py \
--expert_policy_file submit/cs224r/policies/experts/Walker2d.pkl \
--env_name Walker2d-v4 \
--exp_name q2_dagger_walker_Walker2d-v4 \
--n_iter 10 \
--do_dagger \
--expert_data submit/cs224r/expert_data/expert_data_Walker2d-v4.pkl \
--eval_batch_size 5000 \
--video_log_freq -1

