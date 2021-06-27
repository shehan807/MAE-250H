rm main.py.lprof ./Profiling/main_profile.out
kernprof -l main.py
python3 -m line_profiler main.py.lprof > ./Profiling/main_profile.out
vim ./Profiling/main_profile.out


