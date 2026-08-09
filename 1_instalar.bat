@echo off
echo ============================================
echo   GD-AI - instalando dependencias (so 1a vez)
echo ============================================
python -m venv venv
call venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Instalacao concluida!
echo Agora rode: 2_calibrar.bat  (uma vez, com o jogo aberto)
echo Depois:     3_iniciar.bat   (pra IA comecar a jogar/aprender)
pause
