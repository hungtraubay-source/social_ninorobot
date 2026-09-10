#!/usr/bin/env bash
# Đọc số của run MỚI NHẤT trong ~/social_rl_runs. Chạy được trong lúc đang train.
# Chỉ để theo dõi/shortlist trong lúc train; checkpoint cuối cùng phải chọn bằng
# --eval deterministic theo bốn ngưỡng nghiệm thu trong RUN_RL.txt.
# Các đường cần theo dõi (xem RUN_RL.txt, mục CHẠY MỘT VÒNG TRAIN ĐẦY ĐỦ):
#   outcome/goal lên, outcome/hit_obstacle xuống, mean_peak_intrusion xuống,
#   và mean_peak_intrusion_unseen phải BÁM SÁT mean_peak_intrusion.
python3 -c "
import glob, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
runs = sorted(glob.glob('${HOME}/social_rl_runs/*/RecurrentPPO_1/events*'))
if not runs:
    sys.exit('khong tim thay run nao trong ~/social_rl_runs')
acc = EventAccumulator(runs[-1]); acc.Reload()
print(runs[-1].split('/')[-3])
last = None
for tag in ('outcome/goal', 'outcome/hit_obstacle', 'outcome/timeout',
            'social/mean_peak_intrusion', 'social/mean_peak_intrusion_unseen',
            'social/clear_episodes'):
    if tag in acc.Tags()['scalars']:
        last = acc.Scalars(tag)
        print(f'  {tag:38} {last[-1].value:7.3f}   (dau run {last[0].value:.3f})')
if last:
    print(f'  {\"BUOC HIEN TAI\":38} {last[-1].step:7d}')
"
