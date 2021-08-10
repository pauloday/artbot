#!/bin/bash
# you aren't supposed to run multiple processes in docker
# lets see if it works anyway
lt --port 8501 &
streamlit run ./artbot_studio.py
