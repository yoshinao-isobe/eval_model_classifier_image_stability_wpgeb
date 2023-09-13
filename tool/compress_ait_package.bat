@echo off

@REM define zipfile name
for %%I in ("%~dp0\..") do set FILE_NAME=%%~nI.zip

@REM make zipfile
cd /d %~dp0
powershell Compress-Archive -Force ^
                            -Path ../deploy/container/*  ^
                            -DestinationPath ../%FILE_NAME%