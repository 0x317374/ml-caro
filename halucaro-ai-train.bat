@echo off
title HaluCaro AI trainning
cd C:\HaluCaro\HaluCaro
call C:\Anaconda\Scripts\activate.bat
call conda activate py3.6v
call python train.py
pause
