@echo off
python main.py "log_name"
python neural_network/optimize_vit.py -event_log "log_name"
python main_adv.py "log_name"
python neural_network/optimize_vit_adv.py -event_log "log_name"